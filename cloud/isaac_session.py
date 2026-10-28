"""One capped Isaac Sim GPU session that runs committed DREAM snapshots as numbered jobs.

    python cloud/isaac_session.py plan  --project <p>
    python cloud/isaac_session.py start --project <p> --allowance-usd 6 --yes
    python cloud/isaac_session.py submit --manifest <m> experiments/isaac_g1_kitchen.py -- <args>
    python cloud/isaac_session.py stop  --manifest <m> --yes

Same private-VM design as cloud/launch.py, whose helpers it reuses (resource naming, stockout
fallback, scoped service account, temporary network/NAT, result checksums, cleanup, audit):
the difference is that one cold install serves several jobs instead of one. Safeguards:
- Google deletes the VM at ``max_minutes`` regardless of this process (max-run-duration);
- a per-session ceiling and a cumulative ledger for the continuation, from the config;
- every job is ``git archive HEAD`` of a clean tracked tree, recorded by SHA;
- outputs are retrieved and checksum-verified before the bucket is removed; on failure the
  bucket is preserved and ``stop`` can be rerun.
Cost figures are estimates from list prices and enforced lifetime, not billed amounts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloud.launch import (  # noqa: E402
    allocate,
    audit,
    cleanup,
    gcloud,
    git,
    grant_result_access,
    verify_results,
)

CONFIG = ROOT / "experiments/configs/cloud_isaac_g1_session.json"
WORKLOAD = "isaac-g1-session"
ARCHIVE_PATHS = ["dream", "experiments", "cloud", "third_party", "pyproject.toml", "README.md",
                 "AGENTS.md"]  # fmt: skip


def local_root() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DREAM"
    if root.resolve().is_relative_to(ROOT.resolve()) or "onedrive" in str(root).lower():
        raise ValueError("cloud session outputs must stay outside Git and OneDrive")
    return root


def ledger_path() -> Path:
    return local_root() / "cloud/isaac_ledger.json"


def read_ledger() -> list[dict]:
    path = ledger_path()
    return json.loads(path.read_text()) if path.is_file() else []


def estimate(cfg: dict, minutes: int) -> float:
    r = cfg["cost"]
    hourly = r["vm_usd_per_hour"] + r["disk_usd_per_hour"] + r["nat_gateway_and_ip_usd_per_hour"]
    return round(
        minutes / 60 * hourly
        + r["nat_download_gib"] * r["nat_usd_per_gib"]
        + r["storage_egress_and_reserve_usd"],
        2,
    )


def charged(entry: dict) -> float:
    """What a ledger entry counts against the cap: the reconciled estimate once the session
    ended (actual VM lifetime, full network/storage reserves), else the full estimate."""
    return entry.get("reconciled_usd", entry["estimated_usd"])


def reconcile(entry: dict, cfg: dict) -> float:
    """Estimate from the recorded lifetime (creation to verified deletion); never below the
    reserves, never above the full estimate. Still an estimate, not a bill."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start = time.mktime(time.strptime(entry["started_utc"], fmt))
    end = time.mktime(time.strptime(entry["ended_utc"], fmt))
    minutes = max(0.0, (end - start) / 60)
    return min(entry["estimated_usd"], estimate(cfg, math.ceil(minutes)))


def plan(project: str, cfg: dict, name: str, iap_api_port: int | None = None) -> dict:
    """Inspectable plan; no cloud calls."""
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project):
        raise ValueError("supply a valid --project")
    if not re.fullmatch(r"dream-qwen-isaac-[a-z0-9-]{1,29}", name):
        raise ValueError("run names must be DREAM-specific")
    minutes = cfg["max_minutes"]
    if (
        cfg["machine"] != "g2-standard-8"
        or not isinstance(minutes, int)
        or not 30 <= minutes <= 180
    ):
        raise ValueError("sessions use one reviewed L4 configuration for 30..180 minutes")
    if iap_api_port is not None and not 1024 <= iap_api_port <= 65535:
        raise ValueError("IAP API port must be between 1024 and 65535")
    return {
        "name": name, "workload": WORKLOAD, "project": project, "zone": cfg["zone"],
        "region": "us-central1", "machine": cfg["machine"], "disk_gb": cfg["disk_gb"],
        "max_minutes": minutes, "bucket": name, "network": name, "subnet": name + "-subnet",
        "router": name + "-router", "nat": name + "-nat",
        "account": "dream-" + hashlib.sha256(name.encode()).hexdigest()[:16],
        "image": cfg["image"], "image_project": cfg["image_project"],
        "estimated_usd": estimate(cfg, minutes),
        "per_session_ceiling_usd": cfg["per_session_ceiling_usd"],
        "continuation_cap_usd": cfg["continuation_cap_usd"],
        "cost_note": "estimate, not a billing hard cap; VM lifetime is enforced by Google",
        **({"iap_api_port": iap_api_port, "iap_firewall": name + "-iap"}
           if iap_api_port is not None else {}),
    }  # fmt: skip


def archive(directory: Path) -> tuple[Path, str, str]:
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit first: only a clean committed snapshot can run")
    path = directory / "code.tar.gz"
    git("archive", "--format=tar.gz", "-o", str(path), "HEAD", *ARCHIVE_PATHS)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), git("rev-parse", "HEAD")


NO_VM_RESERVE_USD = 0.01  # minutes of an empty bucket/network; no VM, NAT traffic or disk


def never_allocated(dest: Path) -> bool:
    """True when every VM request was refused, so no VM ever existed (only then)."""
    path = dest / "allocation_attempts.json"
    rows = json.loads(path.read_text()) if path.is_file() else []
    return all(r.get("status") in ("capacity_rejected", "failed") for r in rows)


def settle_failed_start(p: dict, dest: Path, errors: list) -> None:
    """A start that failed before any VM existed stops counting its full estimate."""
    if errors or not never_allocated(dest) or vm_present(p):
        return  # keep the full estimate: something may still be billing
    ledger = read_ledger()
    for entry in ledger:
        if entry["name"] == p["name"]:
            entry.update(ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         retrieved=False, cleanup_errors=[], reconciled_usd=NO_VM_RESERVE_USD,
                         note="start failed before a VM existed; resources removed")  # fmt: skip
    ledger_path().write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def start(p: dict, allowance: float, yes: bool) -> Path:
    with tempfile.TemporaryDirectory(prefix="dream-isaac-check-") as directory:
        archive(Path(directory))  # refuse a dirty tree before creating or charging anything
    spent = sum(charged(e) for e in read_ledger())
    if not yes or not math.isfinite(allowance) or allowance < p["estimated_usd"]:
        raise ValueError("--yes and --allowance-usd covering the estimate are required")
    if allowance > p["per_session_ceiling_usd"]:
        raise ValueError("allowance exceeds the configured per-session ceiling")
    if spent + p["estimated_usd"] > p["continuation_cap_usd"]:
        raise ValueError(f"cumulative estimate {spent:.2f} + this session exceeds the cap")
    state = audit(p["project"])
    quota = {q["metric"]: q["limit"] - q["usage"] for q in state["quotas"]}
    if not state["billing_enabled"] or state["dream_instances"]:
        raise ValueError("billing disabled or another DREAM GPU job is present")
    if quota.get("NVIDIA_L4_GPUS", 0) < 1 or quota.get("CPUS", 0) < 8:
        raise ValueError("insufficient regional CPU/L4 quota")
    dest = local_root() / "cloud_runs" / p["name"]
    dest.mkdir(parents=True, exist_ok=False)
    p = {**p, "git_sha": git("rev-parse", "HEAD"), "allowance_usd": allowance,
         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
    (dest / "manifest.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    project, region = f"--project={p['project']}", f"--region={p['region']}"
    account = f"{p['account']}@{p['project']}.iam.gserviceaccount.com"
    ledger = read_ledger() + [
        {
            "name": p["name"],
            "estimated_usd": p["estimated_usd"],
            "started_utc": p["started_utc"],
            "max_minutes": p["max_minutes"],
        }
    ]
    ledger_path().parent.mkdir(parents=True, exist_ok=True)
    ledger_path().write_text(json.dumps(ledger, indent=2), encoding="utf-8")  # before spending
    try:
        gcloud("storage", "buckets", "create", f"gs://{p['bucket']}", project,
               f"--location={p['region']}", "--uniform-bucket-level-access",
               "--public-access-prevention")  # fmt: skip
        gcloud("iam", "service-accounts", "create", p["account"], project)
        grant_result_access(p["bucket"], account)
        gcloud("compute", "networks", "create", p["network"], project, "--subnet-mode=custom")
        gcloud("compute", "networks", "subnets", "create", p["subnet"], project, region,
               f"--network={p['network']}", "--range=10.77.0.0/28",
               "--enable-private-ip-google-access")  # fmt: skip
        if p.get("iap_api_port") is not None:
            gcloud("compute", "firewall-rules", "create", p["iap_firewall"], project,
                   f"--network={p['network']}", "--direction=INGRESS", "--action=ALLOW",
                   f"--rules=tcp:{p['iap_api_port']}", "--source-ranges=35.235.240.0/20",
                   f"--target-service-accounts={account}", "--enable-logging")  # fmt: skip
        gcloud("compute", "routers", "create", p["router"], project, region,
               f"--network={p['network']}")  # fmt: skip
        gcloud("compute", "routers", "nats", "create", p["nat"], project, region,
               f"--router={p['router']}", "--auto-allocate-nat-external-ips",
               f"--nat-custom-subnet-ip-ranges={p['subnet']}")  # fmt: skip
        with tempfile.TemporaryDirectory(prefix="dream-isaac-") as directory:
            code, checksum, sha = archive(Path(directory))
            gcloud("storage", "cp", str(code), f"gs://{p['bucket']}/code.tar.gz")
            startup = Path(directory) / "startup.sh"
            startup.write_text(git("show", "HEAD:cloud/vm_startup.sh") + "\n", newline="\n")
            allocate(p, sha, checksum, startup, dest)
        gcloud("compute", "instances", "add-iam-policy-binding", p["name"], project,
               f"--zone={p['zone']}", f"--member=serviceAccount:{account}",
               "--role=roles/compute.instanceAdmin.v1")  # fmt: skip
    except BaseException:
        errors = cleanup(p, remove_results=True)
        (dest / "start_cleanup_errors.json").write_text(json.dumps(errors, indent=2))
        settle_failed_start(p, dest, errors)
        raise
    (dest / "manifest.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    print(f"session started; manifest {dest / 'manifest.json'}", flush=True)
    return dest / "manifest.json"


def vm_status(p: dict) -> str | None:
    """The VM's status, or None once it is gone. Control-plane timeouts are retried."""
    for attempt in range(3):
        try:
            rows = json.loads(gcloud("compute", "instances", "list", f"--project={p['project']}",
                                     f"--filter=name={p['name']}", "--format=json"))  # fmt: skip
            return rows[0]["status"] if rows else None
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
    return None


def vm_present(p: dict) -> bool:
    return vm_status(p) is not None


def remote_exists(url: str) -> bool:
    try:
        gcloud("storage", "objects", "describe", url, "--format=value(name)")
        return True
    except subprocess.CalledProcessError:
        return False


def submit(
    manifest: Path, module: str, args: list[str], timeout_s: int, env: str = "isaac"
) -> Path:
    p = json.loads(manifest.read_text())
    if not re.fullmatch(r"experiments/[a-z0-9_]+\.py", module):
        raise ValueError("jobs are experiments/<name>.py")
    jobs = manifest.parent / "jobs"
    jobs.mkdir(exist_ok=True)
    job = str(len(list(jobs.iterdir())) + 1)
    with tempfile.TemporaryDirectory(prefix="dream-isaac-job-") as directory:
        code, checksum, sha = archive(Path(directory))
        spec = {"id": job, "git_sha": sha, "archive_sha256": checksum, "module": module,
                "args": args, "timeout_s": timeout_s, "env": env,
                "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
        (jobs / job).mkdir()
        (jobs / job / "job.json").write_text(json.dumps(spec, indent=2))
        gcloud("storage", "cp", str(code), f"gs://{p['bucket']}/jobs/{job}/code.tar.gz")
        gcloud("storage", "cp", str(jobs / job / "job.json"),
               f"gs://{p['bucket']}/jobs/{job}/job.json")  # fmt: skip
    print(f"submitted job {job} at {sha[:7]}", flush=True)
    deadline = time.monotonic() + timeout_s + 900  # queueing behind install/other jobs
    while not remote_exists(f"gs://{p['bucket']}/out/jobs/{job}/EXIT_CODE"):
        if time.monotonic() > deadline or not vm_present(p):
            raise RuntimeError("job did not finish; run `stop` to retrieve what exists")
        time.sleep(20)
    time.sleep(5)  # the guest syncs the manifest with EXIT_CODE; allow the rsync to finish
    out = jobs / job / "out"
    gcloud("storage", "rsync", "--recursive", f"gs://{p['bucket']}/out/jobs/{job}", str(out))
    verify_results(out)
    print(f"job {job} exit {(out / 'EXIT_CODE').read_text().strip()}; outputs {out}")
    return out


def stop(manifest: Path, yes: bool) -> None:
    if not yes:
        raise ValueError("--yes is required to end the session and clean up")
    p = json.loads(manifest.read_text())
    dest = manifest.parent
    with tempfile.TemporaryDirectory() as directory:
        marker = Path(directory) / "STOP"
        marker.write_text("stop\n")
        try:
            gcloud("storage", "cp", str(marker), f"gs://{p['bucket']}/jobs/STOP")
        except subprocess.CalledProcessError:
            pass  # bucket may already be gone after an interrupted earlier stop
    deadline = time.monotonic() + 600
    while vm_status(p) not in (None, "TERMINATED") and time.monotonic() < deadline:
        time.sleep(20)  # the guest finishes its job, syncs and deletes itself
    if vm_present(p):  # still running past the wait, or stopped without self-deletion
        gcloud("compute", "instances", "delete", p["name"], f"--project={p['project']}",
               f"--zone={p['zone']}")  # fmt: skip
    retrieved = False
    try:
        gcloud("storage", "rsync", "--recursive", f"gs://{p['bucket']}/out", str(dest / "out"))
        verify_results(dest / "out")
        retrieved = True
    finally:
        errors = cleanup(p, remove_results=retrieved)
        ended = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (dest / "cleanup_errors.json").write_text(json.dumps(errors, indent=2))
        (dest / "post_audit.json").write_text(json.dumps(audit(p["project"]), indent=2))
        ledger = read_ledger()
        for entry in ledger:
            if entry["name"] == p["name"]:
                entry["ended_utc"] = ended
                entry["retrieved"] = retrieved
                entry["cleanup_errors"] = errors
                entry["reconciled_usd"] = reconcile(entry, json.loads(CONFIG.read_text()))
        ledger_path().write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(f"cleanup incomplete: {errors}")
    print("session stopped, outputs verified and resources removed", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["plan", "start", "submit", "stop", "ledger"])
    ap.add_argument("--project", default=os.environ.get("DREAM_GCP_PROJECT", ""))
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--allowance-usd", type=float, default=0.0)
    ap.add_argument("--iap-api-port", type=int)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--env", choices=["isaac", "dream"], default="isaac")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("job", nargs="*", help="after --: experiments/<name>.py and its arguments")
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    if args.command in ("plan", "start"):
        name = f"dream-qwen-isaac-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        p = plan(args.project, cfg, name, args.iap_api_port)
        print(json.dumps({**p, "ledger_charged_usd": sum(
            charged(e) for e in read_ledger())}, indent=2), flush=True)  # fmt: skip
        if args.command == "start":
            start(p, args.allowance_usd, args.yes)
    elif args.command == "submit":
        module, *rest = [a for a in args.job if a != "--"]
        submit(args.manifest, module, rest, args.timeout_s, args.env)
    elif args.command == "stop":
        stop(args.manifest, args.yes)
    else:
        print(json.dumps(read_ledger(), indent=2))


if __name__ == "__main__":
    main()
