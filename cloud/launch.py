"""One DREAM GPU pilot, private and capped, with an explicit spending approval gate.

Reuses NERVA's deployment design (cloud/launch.py and vm_startup.sh at 6b05692):
committed archives, private VM + temporary NAT, Google-enforced deletion, result retrieval,
and resource audit. Independently implemented here; NERVA code/resources are not modified.
Deployment identifiers and run manifests stay local; credentials are never archived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/cloud_qwen_pilot.json"


@contextmanager
def single_run_lock(path: Path):
    """Fail closed on simultaneous local launches; never silently reclaim a stale lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"another run or stale lock exists; audit first: {path}") from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def gcloud(*args: str) -> str:
    exe = shutil.which("gcloud.cmd") or shutil.which("gcloud")
    if not exe:
        raise RuntimeError("Google Cloud SDK is required")
    return subprocess.run(
        [exe, *args, "--quiet"], text=True, capture_output=True, check=True, timeout=120
    ).stdout.strip()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def plan(project: str, cfg: dict, name: str) -> dict:
    """Build an inspectable plan. No cloud calls, uploads, IAM changes or resource creation."""
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project):
        raise ValueError("supply a valid --project or DREAM_GCP_PROJECT")
    if not re.fullmatch(r"dream-qwen-[a-z0-9-]{1,35}", name):
        raise ValueError("run names must be DREAM-specific")
    if cfg["machine"] != "g2-standard-8" or cfg["zone"] not in ("us-central1-a", "us-central1-c"):
        raise ValueError("this pilot is restricted to one reviewed L4 configuration")
    minutes = cfg["max_minutes"]
    if not isinstance(minutes, int) or isinstance(minutes, bool) or not 1 <= minutes <= 45:
        raise ValueError("the pilot lifetime must be an integer in 1..45 minutes")
    hours = minutes / 60
    rates = cfg["cost"]
    estimate = hours * (
        rates["vm_usd_per_hour"]
        + rates["disk_usd_per_hour"]
        + rates["nat_gateway_and_ip_usd_per_hour"]
    )
    estimate += rates["nat_download_gib"] * rates["nat_usd_per_gib"]
    estimate += rates["storage_egress_and_reserve_usd"]
    return {
        "name": name,
        "project": project,
        "zone": cfg["zone"],
        "region": "us-central1",
        "machine": cfg["machine"],
        "max_minutes": minutes,
        "bucket": name,
        "network": name,
        "subnet": name + "-subnet",
        "router": name + "-router",
        "nat": name + "-nat",
        "account": "dream-" + hashlib.sha256(name.encode()).hexdigest()[:16],
        "image": cfg["image"],
        "estimated_usd": round(estimate, 2),
        "cost_note": "estimate, not a billing hard cap; VM lifetime is enforced by Google",
    }


def create_command(p: dict, sha: str, archive_hash: str, startup: Path | None = None) -> list[str]:
    account = f"{p['account']}@{p['project']}.iam.gserviceaccount.com"
    metadata = {
        "install-nvidia-driver": "True",
        "dream-bucket": p["bucket"],
        "dream-code-sha": sha,
        "dream-archive-sha256": archive_hash,
        "dream-max-minutes": str(p["max_minutes"]),
    }
    return [
        "compute",
        "instances",
        "create",
        p["name"],
        f"--project={p['project']}",
        f"--zone={p['zone']}",
        f"--machine-type={p['machine']}",
        f"--image={p['image']}",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=100GB",
        "--boot-disk-type=pd-balanced",
        "--boot-disk-auto-delete",
        "--no-address",
        f"--network={p['network']}",
        f"--subnet={p['subnet']}",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        f"--service-account={account}",
        "--scopes=cloud-platform",
        f"--max-run-duration={p['max_minutes']}m",
        "--instance-termination-action=DELETE",
        "--labels=dream=1,purpose=qwen-pilot",
        f"--metadata-from-file=startup-script={startup or ROOT / 'cloud/vm_startup.sh'}",
        "--metadata=" + ",".join(f"{k}={v}" for k, v in metadata.items()),
    ]


def audit(project: str) -> dict:
    """Read-only readiness/inventory. An available quota is not guaranteed zonal capacity."""
    billing = json.loads(gcloud("billing", "projects", "describe", project, "--format=json"))
    region = json.loads(
        gcloud(
            "compute", "regions", "describe", "us-central1", f"--project={project}", "--format=json"
        )
    )
    instances = json.loads(
        gcloud("compute", "instances", "list", f"--project={project}", "--format=json")
    )
    quotas = [q for q in region["quotas"] if q["metric"] in ("CPUS", "NVIDIA_L4_GPUS")]
    inventory = {}
    for kind in ("disks", "addresses", "routers", "networks"):
        rows = json.loads(gcloud("compute", kind, "list", f"--project={project}", "--format=json"))
        inventory[kind] = [r["name"] for r in rows if r["name"].startswith("dream-qwen-")]
    budget_alert = {"status": "unavailable", "authorization": False}
    billing_account = billing.get("billingAccountName", "").removeprefix("billingAccounts/")
    if billing_account:
        try:
            budgets = json.loads(
                gcloud(
                    "billing",
                    "budgets",
                    "list",
                    f"--billing-account={billing_account}",
                    "--format=json",
                )
            )
            budget_alert = {
                "status": "read",
                "authorization": False,
                "nerva": [b.get("amount") for b in budgets if b.get("displayName") == "nerva"],
            }
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass  # budget visibility is distinct from billing being enabled
    return {
        "billing_enabled": billing.get("billingEnabled", False),
        "quotas": quotas,
        "project_instance_count": len(instances),
        "dream_instances": [
            r["name"] for r in instances if r.get("labels", {}).get("dream") == "1"
        ],
        "dream_inventory": inventory,
        "budget_alert": budget_alert,
        "remaining_credit_or_spending_allowance": "not established",
    }


def fetch(p: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    gcloud("storage", "rsync", "--recursive", f"gs://{p['bucket']}/out", str(dest / "out"))
    if not (dest / "out/EXIT_CODE").is_file():
        raise RuntimeError("final output missing; bucket preserved for recovery")
    verify_results(dest / "out")


def verify_results(directory: Path) -> None:
    """Verify stable output files before deleting the remote copy (live logs are excluded)."""
    root = directory.resolve()
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise RuntimeError("output manifest missing; preserve the bucket")
    verified = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if root not in path.parents or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid result manifest path or hash")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"retrieved file checksum differs: {relative}")
        verified.add(path.relative_to(root).as_posix())
    if "EXIT_CODE" not in verified:
        raise RuntimeError("final status not covered by result manifest; preserve the bucket")


def cleanup(p: dict, *, remove_results: bool) -> list[str]:
    """Remove only this manifest's isolated resources; keep results on retrieval failure."""
    errors = []
    project, region = f"--project={p['project']}", f"--region={p['region']}"
    commands = [
        ["compute", "instances", "delete", p["name"], project, f"--zone={p['zone']}"],
        [
            "compute",
            "routers",
            "nats",
            "delete",
            p["nat"],
            f"--router={p['router']}",
            project,
            region,
        ],
        ["compute", "routers", "delete", p["router"], project, region],
        ["compute", "networks", "subnets", "delete", p["subnet"], project, region],
        ["compute", "networks", "delete", p["network"], project],
        [
            "iam",
            "service-accounts",
            "delete",
            f"{p['account']}@{p['project']}.iam.gserviceaccount.com",
            project,
        ],
    ]
    if remove_results:
        commands += [["storage", "rm", "--recursive", f"gs://{p['bucket']}"]]
    for command in commands:
        for attempt in range(3):
            try:
                gcloud(*command)
                break
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or "").lower()
                if any(word in detail for word in ("not found", "not_found", "does not exist")):
                    break
                if attempt < 2 and any(
                    word in detail for word in ("in use", "being used", "in progress")
                ):
                    time.sleep(5)  # resource deletion can take time to release its dependencies
                    continue
                errors.append(" ".join(command[:4]))
                break
            except subprocess.TimeoutExpired:
                errors.append("timeout: " + " ".join(command[:4]))
                break
    return errors


def launch(p: dict, allowance: float, yes: bool) -> None:
    """One foreground run. Approval is external; --yes acknowledges that specific approval."""
    if not yes or not math.isfinite(allowance) or not allowance >= p["estimated_usd"]:
        raise ValueError(
            "approval required: --yes and --allowance-usd covering the printed estimate"
        )
    if git("status", "--porcelain"):
        raise ValueError("commit first: only a clean committed snapshot can run")
    lock = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "DREAM/cloud/active.lock"
    with single_run_lock(lock):
        _launch(p, allowance)


def _launch(p: dict, allowance: float) -> None:
    state = audit(p["project"])
    if not state["billing_enabled"] or state["dream_instances"]:
        raise ValueError("billing disabled or another DREAM GPU job is present")
    quota = {q["metric"]: q["limit"] - q["usage"] for q in state["quotas"]}
    if quota.get("NVIDIA_L4_GPUS", 0) < 1 or quota.get("CPUS", 0) < 8:
        raise ValueError("insufficient regional CPU/L4 quota")
    p = {**p, "git_sha": git("rev-parse", "HEAD"), "allowance_usd": allowance}
    dest = ROOT / "results/cloud_runs" / p["name"]
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "manifest.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    project, region = f"--project={p['project']}", f"--region={p['region']}"
    account = f"{p['account']}@{p['project']}.iam.gserviceaccount.com"
    retrieved = False
    try:
        gcloud(
            "storage",
            "buckets",
            "create",
            f"gs://{p['bucket']}",
            project,
            f"--location={p['region']}",
            "--uniform-bucket-level-access",
            "--public-access-prevention",
        )
        gcloud("iam", "service-accounts", "create", p["account"], project)
        gcloud(
            "storage",
            "buckets",
            "add-iam-policy-binding",
            f"gs://{p['bucket']}",
            f"--member=serviceAccount:{account}",
            "--role=roles/storage.objectAdmin",
        )
        gcloud("compute", "networks", "create", p["network"], project, "--subnet-mode=custom")
        gcloud(
            "compute",
            "networks",
            "subnets",
            "create",
            p["subnet"],
            project,
            region,
            f"--network={p['network']}",
            "--range=10.77.0.0/28",
            "--enable-private-ip-google-access",
        )
        gcloud(
            "compute",
            "routers",
            "create",
            p["router"],
            project,
            region,
            f"--network={p['network']}",
        )
        gcloud(
            "compute",
            "routers",
            "nats",
            "create",
            p["nat"],
            project,
            region,
            f"--router={p['router']}",
            "--auto-allocate-nat-external-ips",
            f"--nat-custom-subnet-ip-ranges={p['subnet']}",
        )
        with tempfile.TemporaryDirectory(prefix="dream-cloud-") as directory:
            archive = Path(directory) / "code.tar.gz"
            git("archive", "--format=tar.gz", "-o", str(archive), "HEAD")
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            gcloud("storage", "cp", str(archive), f"gs://{p['bucket']}/code.tar.gz")
            startup = Path(directory) / "startup.sh"
            startup.write_text(
                git("show", "HEAD:cloud/vm_startup.sh") + "\n", encoding="utf-8", newline="\n"
            )
            gcloud(*create_command(p, p["git_sha"], checksum, startup))
        # Instance-scoped deletion permission, rather than project-wide access to NERVA VMs.
        gcloud(
            "compute",
            "instances",
            "add-iam-policy-binding",
            p["name"],
            project,
            f"--zone={p['zone']}",
            f"--member=serviceAccount:{account}",
            "--role=roles/compute.instanceAdmin.v1",
        )
        deadline = time.monotonic() + (p["max_minutes"] + 5) * 60
        while True:
            rows = json.loads(
                gcloud(
                    "compute",
                    "instances",
                    "list",
                    project,
                    f"--filter=name={p['name']}",
                    "--format=json",
                )
            )
            if not rows:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("VM still present after its scheduled deletion; cleaning up")
            time.sleep(30)
        fetch(p, dest)
        retrieved = True
    finally:
        errors = cleanup(p, remove_results=retrieved)
        (dest / "cleanup_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
        (dest / "post_audit.json").write_text(
            json.dumps(audit(p["project"]), indent=2), encoding="utf-8"
        )
        if errors:
            print("Cleanup needs audit; inspect", dest / "cleanup_errors.json")


def recover(manifest: Path, yes: bool) -> None:
    """Recover one interrupted run. A manifest cannot expand the derived resource names."""
    if not yes:
        raise ValueError("--yes is required to stop and clean up this DREAM run")
    p = json.loads(manifest.read_text())
    # A later pilot's zone must not redirect recovery of an earlier allocation attempt.
    cfg = {**json.loads(CONFIG.read_text()), "zone": p["zone"]}
    expected = plan(p["project"], cfg, p["name"])
    for field in ("zone", "region", "bucket", "network", "subnet", "router", "nat", "account"):
        if p[field] != expected[field]:
            raise ValueError(f"manifest resource mismatch: {field}")
    # Stop only this VM before touching its egress. Missing VM is normal after self-deletion.
    try:
        gcloud(
            "compute",
            "instances",
            "delete",
            p["name"],
            f"--project={p['project']}",
            f"--zone={p['zone']}",
        )
    except subprocess.CalledProcessError as error:
        if "not found" not in (error.stderr or "").lower():
            raise
    retrieved = False
    try:
        fetch(p, manifest.parent)
        retrieved = True
    finally:
        errors = cleanup(p, remove_results=retrieved)
        print(json.dumps({"cleanup_errors": errors, "audit": audit(p["project"])}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["plan", "audit", "run", "recover"])
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--project", default=os.environ.get("DREAM_GCP_PROJECT", ""))
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--allowance-usd", type=float, default=0.0)
    args = ap.parse_args()
    if args.command == "recover":
        if args.manifest is None:
            ap.error("recover requires --manifest")
        recover(args.manifest, args.yes)
        return
    cfg = json.loads(CONFIG.read_text())
    name = f"dream-qwen-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    p = plan(args.project, cfg, name)
    if args.command == "audit":
        print(json.dumps(audit(args.project), indent=2))
    else:
        print(json.dumps(p, indent=2), flush=True)
        if args.command == "run":
            launch(p, args.allowance_usd, args.yes)


if __name__ == "__main__":
    main()
