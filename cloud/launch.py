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
import sys
import tempfile
import time
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "experiments/configs/cloud_qwen_pilot.json"
ZONES = ("us-central1-c", "us-central1-b", "us-central1-a")
WORKLOAD_CONFIGS = {
    "groot-openloop": ROOT / "experiments/configs/cloud_groot_openloop.json",
    "g1-qwen": ROOT / "experiments/configs/cloud_g1_qwen.json",
    "g1-autonomy": ROOT / "experiments/configs/cloud_g1_autonomy.json",
    "qwen-pilot": CONFIG,
    "molmo-openloop": ROOT / "experiments/configs/cloud_molmo_openloop.json",
    "molmo-closedloop": ROOT / "experiments/configs/cloud_molmo_closedloop.json",
    "isaac-t1-walk": ROOT / "experiments/configs/cloud_isaac_t1_walk.json",
    "isaac-t1-stand": ROOT / "experiments/configs/cloud_isaac_t1_stand.json",
}


def run_destination(workload: str, name: str) -> Path:
    """Retrieve recorded jobs outside Git/OneDrive; retain old recovery locations."""
    if workload in ("g1-autonomy", "g1-qwen"):
        root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DREAM/cloud_runs"
        destination = (root / name).resolve()
        if destination.is_relative_to(ROOT.resolve()) or any(
            part.lower().startswith("onedrive") for part in destination.parts
        ):
            raise ValueError("recorded cloud results must stay outside Git and OneDrive")
        return destination
    return ROOT / "results/cloud_runs" / name


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
    # The 41 MiB source upload exceeded 120 s on the development uplink, before
    # VM allocation. Allow a bounded ten-minute transfer; keep RPC limits short.
    timeout = 600 if args[:2] == ("storage", "cp") else 120
    return subprocess.run(
        [exe, *args, "--quiet"], text=True, capture_output=True, check=True, timeout=timeout
    ).stdout.strip()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def grant_result_access(bucket: str, account: str) -> None:
    """Wait narrowly for a newly created identity to propagate to Storage IAM.

    Observed in the walking pilot: account creation succeeded, but the next bucket
    binding rejected that same identity as nonexistent. Never retry a permission denial
    or broaden the role. Four attempts give bounded 5/10/20-second propagation waits.
    """
    for attempt in range(4):
        try:
            gcloud(
                "storage",
                "buckets",
                "add-iam-policy-binding",
                f"gs://{bucket}",
                f"--member=serviceAccount:{account}",
                "--role=roles/storage.objectAdmin",
            )
            return
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or "").lower()
            if (
                attempt == 3
                or account.lower() not in detail
                or "service account" not in detail
                or "does not exist" not in detail
            ):
                raise
            time.sleep(5 * 2**attempt)


def plan(project: str, cfg: dict, name: str) -> dict:
    """Build an inspectable plan. No cloud calls, uploads, IAM changes or resource creation."""
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project):
        raise ValueError("supply a valid --project or DREAM_GCP_PROJECT")
    if not re.fullmatch(r"dream-qwen-[a-z0-9-]{1,35}", name):
        raise ValueError("run names must be DREAM-specific")
    if cfg["machine"] != "g2-standard-8" or cfg["zone"] not in ZONES:
        raise ValueError("this pilot is restricted to one reviewed L4 configuration")
    minutes = cfg["max_minutes"]
    workload = cfg.get("workload", "qwen-pilot")
    if workload not in WORKLOAD_CONFIGS:
        raise ValueError("unsupported workload")
    disk_gb = cfg.get("disk_gb", 100)
    if disk_gb != (200 if workload in ("isaac-t1-walk", "isaac-t1-stand") else 100):
        raise ValueError("unsupported workload disk size")
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
        "workload": workload,
        "project": project,
        "zone": cfg["zone"],
        "region": "us-central1",
        "machine": cfg["machine"],
        "disk_gb": disk_gb,
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
        "dream-workload": p.get("workload", "qwen-pilot"),
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
        f"--boot-disk-size={p.get('disk_gb', 100)}GB",
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
        f"--labels=dream=1,purpose={p.get('workload', 'qwen-pilot')}",
        f"--metadata-from-file=startup-script={startup or ROOT / 'cloud/vm_startup.sh'}",
        "--metadata=" + ",".join(f"{k}={v}" for k, v in metadata.items()),
    ]


def allocate(p: dict, sha: str, checksum: str, startup: Path, dest: Path) -> None:
    """Retry explicit stockouts only, sharing regional resources and never duplicating VMs.

    Three zones are supported by the same regional L4 configuration. Other failures,
    including ambiguous timeouts, require inspection rather than another allocation.
    """
    zones = [p["zone"], *(z for z in ZONES if z != p["zone"])]
    records = []
    for zone in zones:
        p["zone"] = zone
        (dest / "manifest.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
        record = {
            "zone": zone,
            "status": "requested",
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        records.append(record)
        path = dest / "allocation_attempts.json"
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Allocating {p['machine']} in {zone}", flush=True)
        try:
            gcloud(*create_command(p, sha, checksum, startup))
        except subprocess.CalledProcessError as error:
            detail = error.stderr or ""
            capacity = "ZONE_RESOURCE_POOL_EXHAUSTED" in detail.upper()
            record.update(status="capacity_rejected" if capacity else "failed", stderr=detail)
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            if not capacity:
                raise
            for kind in ("instances", "disks"):
                rows = json.loads(
                    gcloud(
                        "compute",
                        kind,
                        "list",
                        f"--project={p['project']}",
                        f"--filter=name={p['name']}",
                        "--format=json",
                    )
                )
                if rows:
                    raise RuntimeError(
                        f"allocation uncertain: {kind} still present; no retry"
                    ) from error
            print(f"No capacity in {zone}; absence verified, trying the next zone", flush=True)
            if zone == zones[-1]:
                raise
        else:
            record["status"] = "allocated"
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            return


def validate_policy_steps(folder: Path, record: dict) -> None:
    """Check stored action/state/physics contracts independently of the guest's pass flag."""
    import numpy as np

    try:
        limits = np.asarray(json.loads((folder / "articulation.json").read_text())["limits_rad"])
        if (
            limits.shape != (23, 2)
            or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])
        ):
            raise ValueError("invalid mechanical joint limits")
        warmup = [json.loads(line) for line in (folder / "warmup.jsonl").read_text().splitlines()]
        rows = warmup + record["rows"]
        if len(warmup) != 50 or record["warmup_steps"] != 50:
            raise ValueError("50 completed standing policy steps required")
        times = np.array([r["state_time_s"] for r in rows])
        steps = np.array([r["physics_step_count"] for r in rows])
        if (
            not np.allclose(np.diff(times), 0.02, atol=1e-6)
            or not np.all(np.diff(steps) == 4)
            or steps[-1] != record["physics_steps"]
        ):
            raise ValueError("physics decimation/timestamps disagree")
        if record["case"] == "stand" and record["rows"][-1]["time_s"] < 4.0:
            raise ValueError("longer standing interval missing")
        dimensions = {
            "joint_position_rad": 23,
            "joint_velocity_rad_s": 23,
            "post_joint_position_rad": 23,
            "post_joint_velocity_rad_s": 23,
            "joint_target_rad": 23,
            "raw_ppo_action": 23,
            "target_limit_excess_rad": 23,
            "score_position_m": 3,
            "orientation_wxyz": 4,
            "body_linear_velocity_m_s": 3,
            "body_angular_velocity_rad_s": 3,
        }
        for row in rows:
            for key, size in dimensions.items():
                value = np.asarray(row[key])
                if value.shape != (size,) or not np.isfinite(value).all():
                    raise ValueError("invalid recorded sensor/action")
            for key in ("joint_position_rad", "post_joint_position_rad"):
                q = np.asarray(row[key])
                if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
                    raise ValueError("measured joint outside mechanical limits")
            if (
                row["stopped"]
                or not row["state_finite"]
                or row["score_position_m"][2] < 0.45
                or not 0 <= row["tilt_rad"] <= math.radians(25)
            ):
                raise ValueError("standing stability failed")
    except (KeyError, ValueError, OSError, TypeError) as error:
        raise RuntimeError(f"physical policy step evidence failed: {error}") from error


def check_pilot(out: Path, workload: str = "qwen-pilot") -> None:
    """A retrieved guest failure is not a pass. Keep preregistered task success report-only."""
    if (out / "EXIT_CODE").read_text().strip() != "0":
        raise RuntimeError("guest failed; inspect preserved outputs before another paid run")
    if workload in ("g1-qwen", "g1-autonomy"):
        from experiments.cloud_g1_qwen import validate_evidence

        try:
            if workload == "g1-autonomy":
                validate_evidence(out / "g1", ROOT / "experiments/configs/g1_autonomy_frozen.json")
            else:
                validate_evidence(out / "g1")
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise RuntimeError(f"G1 inference/physical evidence failed: {error}") from error
        return
    if workload == "groot-openloop":
        import numpy as np

        from experiments.groot_openloop import validate_actions

        try:
            folder = out / "groot/inference"
            summary = json.loads((folder / "summary.json").read_text())
            raw = np.load(folder / "normalized_actions.npy", allow_pickle=False)
            with np.load(folder / "decoded_actions.npz", allow_pickle=False) as actions:
                validate_actions(dict(actions))
            if (
                summary.get("genuine_inference") is not True
                or summary.get("device") != "cuda:0"
                or summary.get("physical_execution") is not False
                or summary.get("actuator_mapping") is not False
                or raw.shape != (1, 40, 132)
                or not np.isfinite(raw).all()
            ):
                raise ValueError("inference contract failed")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f"GR00T inference evidence failed: {error}") from error
        return
    if workload in ("isaac-t1-walk", "isaac-t1-stand"):
        import numpy as np

        cases = ("stand",) if workload == "isaac-t1-stand" else ("stand", "forward", "left")
        for case in cases:
            if (out / case / "failure.json").exists():
                raise RuntimeError("physical walking failure report present")
            path = out / case / "episode.json"
            if not path.is_file():
                raise RuntimeError("physical walking episode missing")
            record = json.loads(path.read_text())
            if (
                record.get("physics_executed") is not True
                or record.get("real_actor_executed") is not True
                or record.get("navgpt_executed") is not False
                or record.get("gate_pass") is not True
                or not record.get("rows")
            ):
                raise RuntimeError("physical walking gate failed")
            rows = record["rows"]
            times = np.array([r.get("time_s", float("nan")) for r in rows])
            tilts = np.array([r.get("tilt_rad", float("nan")) for r in rows])
            final = np.array(record.get("score_final_position_m", []))
            target = np.array(record.get("score_target_m", []))
            if (
                final.shape != (3,)
                or target.shape != (3,)
                or not np.isfinite(final).all()
                or not np.isfinite(target).all()
                or len(times) < 2
                or not np.isfinite(times).all()
                or not np.all(np.diff(times) > 0)
                or not np.isfinite(tilts).all()
                or np.max(tilts) > math.radians(25)
                or np.linalg.norm(final[:2] - target[:2]) >= 0.15
                or not 0 <= record.get("final_speed_m_s", float("inf")) < 0.1
                or record.get("status") != "waypoint reached"
                or not list((out / case).glob("sensor_*.npz"))
                or not (out / case / "sensor_calibration.json").is_file()
            ):
                raise RuntimeError("physical walking measurements failed")
            for frame_path in (out / case).glob("sensor_*.npz"):
                with np.load(frame_path, allow_pickle=False) as frame:
                    rgb, depth = frame["rgb"], frame["depth"]
                    if (
                        rgb.shape != (400, 400, 3)
                        or rgb.dtype != np.uint8
                        or depth.shape != (400, 400)
                        or not np.any(np.isfinite(depth) & (depth > 0))
                    ):
                        raise RuntimeError("physical walking camera evidence invalid")
            validate_policy_steps(out / case, record)
        return
    if workload == "molmo-closedloop":
        import numpy as np

        path = out / "closedloop_summary.json"
        if not path.is_file():
            raise RuntimeError("closed-loop summary missing")
        summary = json.loads(path.read_text())
        tensors = sorted((out / "learned_joint").glob("raw_actions_*.npy"))
        if (
            summary.get("fresh_model_inference") is not True
            or summary.get("at_least_one_verified_target") is not True
            or not tensors
        ):
            raise RuntimeError("closed-loop execution gate failed")
        for tensor in tensors:
            a = np.load(tensor, allow_pickle=False)
            if a.shape != (15, 8) or not np.isfinite(a).all():
                raise RuntimeError("invalid raw closed-loop tensor")
        trace_path = out / "learned_joint/physical_trace.npz"
        episode_path = out / "learned_joint/episode.json"
        if not trace_path.is_file() or not episode_path.is_file():
            raise RuntimeError("closed-loop physical evidence missing")
        trace = np.load(trace_path, allow_pickle=False)["trace"]
        episode = json.loads(episode_path.read_text())
        if (
            trace.ndim != 2
            or trace.shape[1] != 20
            or len(trace) < 2
            or not np.isfinite(trace).all()
            or not np.all(np.diff(trace[:, 0]) > 0)
            or episode.get("classical_fallback") is not False
            or not any(
                s.get("executed") is True
                and s.get("target_reached") is True
                and 0 <= s.get("tracking_error_rad", float("inf")) < 0.01
                for s in episode.get("steps", [])
            )
        ):
            raise RuntimeError("closed-loop physical execution gate failed")
        return
    if workload == "molmo-openloop":
        path = out / "molmo/summary.json"
        if not path.is_file():
            raise RuntimeError("action summary missing")
        summary = json.loads(path.read_text())
        if (
            summary.get("real_model_executed") is not True
            or summary.get("finite") is not True
            or summary.get("shape") != [15, 8]
            or summary.get("execution_permitted") is not False
            or not (out / "molmo/raw_actions.npy").is_file()
        ):
            raise RuntimeError("open-loop inference gate failed")
        import numpy as np

        actions = np.load(out / "molmo/raw_actions.npy", allow_pickle=False)
        if actions.shape != (15, 8) or not np.isfinite(actions).all():
            raise RuntimeError("saved action tensor is invalid")
        return
    if workload != "qwen-pilot":
        raise ValueError("unsupported workload")
    path = out / "pilot/summary.json"
    if not path.is_file():
        raise RuntimeError("pilot summary missing")
    summary = json.loads(path.read_text())
    cfg = json.loads((ROOT / "experiments/configs/vlm_cloud_pilot.json").read_text())
    expected = {(e["seed"], c) for e in cfg["episodes"] for c in ("A_rule_based", "B_vlm")}
    episodes = summary.get("episodes", [])
    observed = {(e.get("seed"), e.get("condition")) for e in episodes}
    if (
        not summary.get("offline_contract", {}).get("valid_named_call")
        or summary.get("false_claim_gate_pass") is not True
        or len(episodes) != len(expected)
        or observed != expected
        or any(e.get("false_claim") is not False for e in episodes)
    ):
        raise RuntimeError("pilot acceptance gate failed or episodes incomplete")


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
    if p.get("workload") == "groot-openloop":
        commands.insert(1, ["secrets", "delete", p["name"], project])
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
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit first: only a clean committed snapshot can run")
    if p.get("workload") == "groot-openloop":
        groot_access_preflight()
    lock = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "DREAM/cloud/active.lock"
    with single_run_lock(lock):
        _launch(p, allowance)


def hf_token_path() -> Path:
    """Return only the login-file path, never its contents or a shell argument token."""
    return Path(
        os.environ.get(
            "HF_TOKEN_PATH",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token",
        )
    )


def groot_access_preflight() -> Path:
    """Check gated access before cloud mutations; do not log the credential or HTTP body."""
    path = hf_token_path()
    if not path.is_file():
        raise RuntimeError("authorized Hugging Face login required before GR00T allocation")
    token = path.read_text().strip()
    try:
        request = urllib.request.Request(
            "https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/"
            "9ce19a195e423419c349abfc86fd07178b230561/config.json",
            headers={"Authorization": "Bearer " + token},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            json.load(response)
    except Exception:
        raise RuntimeError("authorized Cosmos backbone access not established") from None
    return path


def create_g1_budget_alert(p: dict) -> str:
    """Create a project-scoped AUD alert before allocation; never a billing hard cap.

    The existing billing account uses AUD. Five AUD is an alert threshold, not an
    exchange-rate assertion about the separately approved USD workload allowance.
    Failure to configure the requested alerts prevents allocation.
    """
    billing = json.loads(gcloud("billing", "projects", "describe", p["project"], "--format=json"))
    account = billing["billingAccountName"].removeprefix("billingAccounts/")
    project = json.loads(gcloud("projects", "describe", p["project"], "--format=json"))
    result = json.loads(
        gcloud(
            "billing",
            "budgets",
            "create",
            f"--billing-account={account}",
            f"--display-name={p['name']}",
            "--budget-amount=5AUD",
            f"--filter-projects=projects/{project['projectNumber']}",
            "--threshold-rule=percent=0.5",
            "--threshold-rule=percent=0.9",
            "--threshold-rule=percent=1.0",
            "--format=json",
        )
    )
    return result["name"]


def wait_for_guest(p: dict) -> None:
    """Wait for completion; a stopped guest still owns its disk until explicitly deleted."""
    deadline = time.monotonic() + (p["max_minutes"] + 5) * 60
    while True:
        rows = json.loads(
            gcloud(
                "compute",
                "instances",
                "list",
                f"--project={p['project']}",
                f"--filter=name={p['name']}",
                "--format=json",
            )
        )
        if not rows:
            return
        if all(row.get("status") == "TERMINATED" for row in rows):
            # Observed in the first real action-inference run: guest shutdown succeeded,
            # but self-deletion did not. The bucket already holds its final outputs.
            gcloud(
                "compute",
                "instances",
                "delete",
                p["name"],
                f"--project={p['project']}",
                f"--zone={p['zone']}",
            )
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("VM still present after its scheduled deletion; cleaning up")
        time.sleep(30)


def _launch(p: dict, allowance: float) -> None:
    state = audit(p["project"])
    if not state["billing_enabled"] or state["dream_instances"]:
        raise ValueError("billing disabled or another DREAM GPU job is present")
    quota = {q["metric"]: q["limit"] - q["usage"] for q in state["quotas"]}
    if quota.get("NVIDIA_L4_GPUS", 0) < 1 or quota.get("CPUS", 0) < 8:
        raise ValueError("insufficient regional CPU/L4 quota")
    p = {**p, "git_sha": git("rev-parse", "HEAD"), "allowance_usd": allowance}
    dest = run_destination(p["workload"], p["name"])
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "manifest.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    assets = None
    if p.get("workload") == "groot-openloop":
        from experiments.groot_assets import prepare_assets

        assets = (
            Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
            / "DREAM/staging"
            / p["name"]
        )
        cfg = json.loads((ROOT / "experiments/configs/groot_openloop.json").read_text())
        asset_evidence = prepare_assets(assets, cfg)
        (dest / "model_assets.json").write_text(
            json.dumps(asset_evidence, indent=2), encoding="utf-8"
        )
    project, region = f"--project={p['project']}", f"--region={p['region']}"
    account = f"{p['account']}@{p['project']}.iam.gserviceaccount.com"
    retrieved = False
    alert = None
    try:
        if p.get("workload") in ("g1-qwen", "g1-autonomy", "groot-openloop"):
            alert = create_g1_budget_alert(p)
            (dest / "budget_alert.json").write_text(
                json.dumps(
                    {
                        "name": alert,
                        "amount": "5AUD",
                        "thresholds": [0.5, 0.9, 1.0],
                        "scope": "project spend; delayed notifications, not a cap",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
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
        grant_result_access(p["bucket"], account)
        if assets is not None:
            gcloud(
                "storage", "rsync", "--recursive", str(assets), f"gs://{p['bucket']}/model-input"
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
            paths = (
                [
                    "dream",
                    "experiments",
                    "cloud",
                    "third_party",
                    "pyproject.toml",
                    "README.md",
                    "AGENTS.md",
                ]
                if p.get("workload") in ("g1-qwen", "g1-autonomy", "groot-openloop")
                else []
            )
            git("archive", "--format=tar.gz", "-o", str(archive), "HEAD", *paths)
            checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            gcloud("storage", "cp", str(archive), f"gs://{p['bucket']}/code.tar.gz")
            startup = Path(directory) / "startup.sh"
            startup.write_text(
                git("show", "HEAD:cloud/vm_startup.sh") + "\n", encoding="utf-8", newline="\n"
            )
            allocate(p, p["git_sha"], checksum, startup, dest)
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
        wait_for_guest(p)
        fetch(p, dest)
        retrieved = True
    finally:
        if assets is not None:
            (assets / "weights.private.json").unlink(missing_ok=True)
        errors = cleanup(p, remove_results=retrieved)
        if alert:
            account, budget = alert.split("/budgets/")
            try:
                gcloud(
                    "billing",
                    "budgets",
                    "delete",
                    budget,
                    f"--billing-account={account.removeprefix('billingAccounts/')}",
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                errors.append("delete budget alert " + alert)
        (dest / "cleanup_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
        (dest / "post_audit.json").write_text(
            json.dumps(audit(p["project"]), indent=2), encoding="utf-8"
        )
        if errors:
            print("Cleanup needs audit; inspect", dest / "cleanup_errors.json")
    if errors:
        raise RuntimeError("cleanup incomplete; resolve before another paid run")
    check_pilot(dest / "out", p.get("workload", "qwen-pilot"))
    print(
        "Pilot completed; preregistered gates passed. Task success remains report-only.", flush=True
    )


def recover(manifest: Path, yes: bool) -> None:
    """Recover one interrupted run. A manifest cannot expand the derived resource names."""
    if not yes:
        raise ValueError("--yes is required to stop and clean up this DREAM run")
    p = json.loads(manifest.read_text())
    # A later pilot's zone must not redirect recovery of an earlier allocation attempt.
    workload = p.get("workload", "qwen-pilot")
    if workload not in WORKLOAD_CONFIGS:
        raise ValueError("unsupported recovery workload")
    config_path = CONFIG if workload == "qwen-pilot" else WORKLOAD_CONFIGS[workload]
    cfg = {**json.loads(config_path.read_text()), "zone": p["zone"]}
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
    ap.add_argument("--workload", choices=list(WORKLOAD_CONFIGS), default="qwen-pilot")
    args = ap.parse_args()
    if args.command == "recover":
        if args.manifest is None:
            ap.error("recover requires --manifest")
        recover(args.manifest, args.yes)
        return
    cfg = json.loads(WORKLOAD_CONFIGS[args.workload].read_text())
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
