"""Guest side of an Isaac session: run submitted committed snapshots one at a time.

Standard library only (system Python on the VM). A job is gs://<bucket>/jobs/<id>/job.json
plus code.tar.gz from `cloud/isaac_session.py submit`. Only `experiments/<name>.py` with
string arguments is accepted, executed without a shell. Each job's outputs go to
/work/out/jobs/<id> with EXIT_CODE and a SHA-256 manifest, synced before the next job.
The loop ends on a STOP marker or before the VM's enforced deletion deadline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

BUCKET, CAP_MIN, PYTHON = sys.argv[1], int(sys.argv[2]), sys.argv[3]
OUT, WORK = Path("/work/out/jobs"), Path("/work/jobs")
MARGIN_S = 180  # final sync and self-deletion before Google's deadline
SUBS = {"{robot_dir}": "/work/g1_usd", "{policy_cache}": "/work/cache"}


def uptime() -> float:
    return float(Path("/proc/uptime").read_text().split()[0])


def gs(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gsutil", "-q", *args], capture_output=True, text=True, check=check)


def sync(job: str) -> None:
    gs("-m", "rsync", "-r", str(OUT / job), f"gs://{BUCKET}/out/jobs/{job}", check=False)


def manifest(folder: Path) -> None:
    rows = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"):
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(folder)}")
    (folder / "MANIFEST.sha256").write_text("\n".join(rows) + "\n")


def run(job: str) -> None:
    folder, out = WORK / job, OUT / job
    folder.mkdir(parents=True)
    out.mkdir(parents=True)
    gs("cp", f"gs://{BUCKET}/jobs/{job}/job.json", str(folder / "job.json"))
    spec = json.loads((folder / "job.json").read_text())
    log = open(out / "job.log", "w")
    code = 2
    try:
        if not re.fullmatch(r"experiments/[a-z0-9_]+\.py", spec["module"]) or not all(
            isinstance(a, str) for a in spec["args"]
        ):
            raise ValueError("job must be experiments/<name>.py with string arguments")
        gs("cp", f"gs://{BUCKET}/jobs/{job}/code.tar.gz", str(folder / "code.tar.gz"))
        digest = hashlib.sha256((folder / "code.tar.gz").read_bytes()).hexdigest()
        if digest != spec["archive_sha256"]:
            raise ValueError("job archive checksum differs")
        with tarfile.open(folder / "code.tar.gz") as tar:
            try:
                tar.extractall(folder / "DREAM", filter="data")
            except TypeError:  # Python without extraction filters; archive is our own
                tar.extractall(folder / "DREAM")
        args = [spec["module"]]
        for a in spec["args"]:
            a = a.replace("{out}", str(out))
            for key, value in SUBS.items():
                a = a.replace(key, value)
            args.append(a)
        left = CAP_MIN * 60 - MARGIN_S - uptime()
        timeout = min(float(spec.get("timeout_s", left)), left)
        if timeout <= 60:
            raise TimeoutError("not enough VM lifetime left for this job")
        env = {**os.environ, "DREAM_CODE_SHA": spec["git_sha"], "DREAM_CACHE": "/work/cache"}
        (out / "started_utc.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        proc = subprocess.run(
            [PYTHON, *args], cwd=folder / "DREAM", env=env, stdout=log, stderr=subprocess.STDOUT,
            timeout=timeout,
        )  # fmt: skip
        code = proc.returncode
    except subprocess.TimeoutExpired:
        code = 124
    except Exception as error:  # recorded, never hidden
        log.write(f"\nJOB SETUP ERROR: {error!r}\n")
    finally:
        log.close()
        (out / "EXIT_CODE").write_text(f"{code}\n")
        (out / "finished_utc.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        manifest(out)
        sync(job)


def main() -> None:
    done: set[str] = set()
    while CAP_MIN * 60 - MARGIN_S - uptime() > 60:
        if gs("stat", f"gs://{BUCKET}/jobs/STOP", check=False).returncode == 0:
            print("stop marker found", flush=True)
            return
        listed = gs("ls", f"gs://{BUCKET}/jobs/*/job.json", check=False).stdout.split()
        pending = sorted(
            {p.split("/jobs/")[1].split("/")[0] for p in listed} - done, key=lambda j: int(j)
        )
        for job in pending:
            print(f"running job {job}", flush=True)
            done.add(job)
            run(job)
            break
        else:
            time.sleep(15)
    print("session lifetime reached", flush=True)


if __name__ == "__main__":
    main()
