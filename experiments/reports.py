"""Maintain the hierarchical ``Reports/`` directory: one task per development prompt, one run
per experimental run, generated indexes.

Why: every development prompt needs a permanent, checkable record. Task metadata never lists
runs or active dates; both are derived from the files on disk (runs/*/metadata.json and
notes/*.json), and every file name carries a timestamp, so records made in different git
worktrees merge without conflict. Index files are fully regenerated (``index``) and carry no
timestamps, so a conflict in one is resolved by rerunning ``index``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Australia/Sydney")
TODO = "_TO COMPLETE_"
MAX_HASH_BYTES = 500 * 1024 * 1024  # evidence files above this are listed, not hashed
FINAL_TASK = ("completed", "partially_completed", "failed", "blocked", "cancelled")
RUN_STATUS = ("completed", "partial", "failed", "crashed", "timed_out", "interrupted", "cancelled")
SECTIONS = [
    "A. Objective",
    "B. Starting state",
    "C. Work performed",
    "D. What was achieved",
    "E. What was not achieved",
    "F. Passed tests and experiments",
    "G. Failed tests and experiments",
    "H. Improvements",
    "I. Regressions and dis-improvements",
    "J. Comparison against baseline",
    "K. Research integrity and assumptions",
    "L. Cloud resource usage",
    "M. Remaining work",
    "N. Final conclusion",
]
TASK_HEADINGS = ["## Task identification", *(f"## {s}" for s in SECTIONS)]
TASK_KEYS = (
    "task_id slug title submitted created timezone branch start_commit start_dirty "
    "final_commit status main_finding report"
).split()
RUN_KEYS = "run_id task_id title started ended status commit dirty branch configs metrics".split()
CLOUD_KEYS = (
    "provider project region machine_type gpu_type gpu_count session_id job_ids max_lifetime_min "
    "estimated_runtime_min actual_runtime_min estimated_cost_usd billed_cost_usd "
    "checkpoint_status evidence_location upload_status cleanup_status resources_remaining"
).split()
RUN_FIELDS = [  # (report label, metadata key)
    ("Seeds", "seeds"),
    ("Model and checkpoint", "model"),
    ("Simulator and version", "simulator"),
    ("Renderer", "renderer"),
    ("Robot embodiment", "embodiment"),
    ("Robot controller", "controller"),
    ("Hardware", "hardware"),
]
REDACTIONS = [
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    r"AIza[0-9A-Za-z_-]{30,}",
    r"ghp_[0-9A-Za-z]{30,}",
    r"github_pat_\w{30,}",
    r"sk-[0-9A-Za-z_-]{20,}",
    r"tskey-[\w-]{10,}",
    r"ya29\.[\w.-]+",
    r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+",
]


def now() -> dt.datetime:
    return dt.datetime.now(TZ).replace(microsecond=0)


def _git(*args: str) -> str:
    """Output of a git command in the repository, or '' when git fails."""
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def git_state() -> tuple[str, str, bool]:
    """(branch, HEAD commit, dirty) of the repository."""
    return (
        _git("branch", "--show-current"),
        _git("rev-parse", "HEAD"),
        bool(_git("status", "--porcelain")),
    )


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48].strip("-")
    if not slug:
        raise SystemExit(f"slug {text!r} has no usable characters")
    return slug


def redact(text: str) -> tuple[str, int]:
    total = 0
    for pattern in REDACTIONS:
        text, n = re.subn(pattern, "[REDACTED]", text, flags=re.DOTALL)
        total += n
    return text, total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def json_arg(value: str) -> dict:
    """A JSON object given inline or as @file."""
    raw = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise SystemExit("expected a JSON object")
    return data


def reports_root(arg: str | None) -> Path:
    return Path(arg or os.environ.get("DREAM_REPORTS_ROOT") or REPO / "Reports")


# ---------------------------------------------------------------- report text helpers


def set_line(text: str, label: str, value: str) -> str:
    """Replace the '- label: ...' identification line; fail loudly if it is absent."""
    new, n = re.subn(
        rf"^- {re.escape(label)}:.*$", lambda _: f"- {label}: {value}", text, flags=re.M
    )
    if n != 1:
        raise SystemExit(f"report has no unique line '- {label}:'")
    return new


def set_section(text: str, heading: str, body: str) -> str:
    new, n = re.subn(
        rf"(^{re.escape(heading)}\n)(.*?)(?=^## |\Z)",
        lambda m: f"{m.group(1)}\n{body}\n\n",
        text,
        flags=re.M | re.S,
    )
    if n != 1:
        raise SystemExit(f"report has no unique section {heading}")
    return new.rstrip("\n") + "\n"


def todo_count(text: str) -> int:
    """Placeholders left in a report; HTML comments do not count."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S).count(TODO)


def task_report(meta: dict) -> str:
    body = [
        f"# DREAM Research / Development Report: {meta['title']}",
        "",
        "## Task identification",
        f"- Task title: {meta['title']}",
        f"- Task ID: {meta['task_id']}",
        f"- Date and time: {meta['created']}",
        "- Original prompt: [PROMPT.md](PROMPT.md)",
        f"- Git branch: {meta['branch']}",
        f"- Starting commit: {meta['start_commit']}",
        f"- Final commit: {TODO}",
        "- Task status: in progress",
        "",
    ]
    for name in SECTIONS:
        body += [f"## {name}", ""]
        if name.startswith("C."):
            body += [
                "### Implementation",
                TODO,
                "",
                "### Evaluation",
                TODO,
                "",
                "### Runs",
                "Run records: see runs/ and the daily index",
                TODO,
                "",
            ]
            continue
        if name.startswith("I."):
            body += [
                '<!-- If a metric was not measured, write "Not measured", not "No regression". -->'
            ]
        body += [TODO, ""]
    return "\n".join(body)


def run_report(meta: dict) -> str:
    def val(key: str) -> str:
        return str(meta[key]) if meta.get(key) is not None else TODO

    configs = "; ".join(f"{c['path']} ({c['sha256']})" for c in meta["configs"]) or TODO
    command = f"`{meta['command']}`" if meta.get("command") else TODO
    commit = f"{meta['commit']} (dirty)" if meta["dirty"] else meta["commit"]
    lines = [
        f"# Run report: {meta['title']}",
        "",
        f"- Run ID: {meta['run_id']}",
        f"- Parent task: {meta['task_id']}",
        f"- Started: {meta['started']}",
        f"- Ended: {TODO}",
        f"- Git commit: {commit}",
        f"- Branch: {meta['branch']}",
        f"- Config files and SHA-256: {configs}",
    ]
    lines += [f"- {label}: {val(key)}" for label, key in RUN_FIELDS]
    lines += [
        f"- Command: {command}",
        f"- Exit status: {TODO}",
        f"- Completed (yes/no/partial): {TODO}",
        f"- Cases attempted: {TODO}",
        f"- Cases completed: {TODO}",
        "",
    ]
    for head in ("Purpose", "Results", "Regressions observed", "Interruptions and failures"):
        lines += [f"## {head}", ""]
        if head == "Results":
            lines += ["<!-- Passed and failed cases; quantitative metrics. -->"]
        lines += [TODO, ""]
    lines += ["## Evidence", "", TODO, ""]
    if meta["cloud"] is not None:
        lines += ["## Cloud", "", *cloud_lines(meta["cloud"]), ""]
    return "\n".join(lines)


def cloud_lines(cloud: dict) -> list[str]:
    return [f"- {k}: {TODO if cloud.get(k) is None else cloud[k]}" for k in CLOUD_KEYS]


def evidence_entry(path: Path) -> dict:
    """Checksum a file, or a directory (hash of its sorted 'relpath sha256' lines)."""
    path = path.resolve()
    try:
        shown = path.relative_to(REPO).as_posix()
    except ValueError:
        shown = path.as_posix()
    if path.is_file():
        return {"path": shown, "kind": "file", "sha256": sha256_file(path)}
    if not path.is_dir():
        raise SystemExit(f"evidence path does not exist: {path}")
    lines, skipped = [], []
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = f.relative_to(path).as_posix()
        if f.stat().st_size > MAX_HASH_BYTES:
            skipped.append(rel)
        else:
            lines.append(f"{rel} {sha256_file(f)}")
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return {"path": shown, "kind": "dir", "sha256": digest, "skipped": skipped}


# ---------------------------------------------------------------- locating records


def task_dirs(root: Path) -> list[Path]:
    return sorted(m.parent for m in root.glob("????-??-??/???-*/metadata.json"))


def find_task(root: Path, ident: str) -> Path:
    if Path(ident).is_dir() and (Path(ident) / "metadata.json").exists():
        return Path(ident)
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-(\d{3})", ident)
    if m:
        hits = sorted((root / m.group(1)).glob(f"{m.group(2)}-*/metadata.json"))
        if hits:
            return hits[0].parent
    raise SystemExit(f"no such task: {ident}")


def run_dirs(task: Path) -> list[Path]:
    return sorted(m.parent for m in task.glob("runs/*/metadata.json"))


def find_run(task: Path, ident: str) -> Path:
    name = ident.replace("\\", "/").rstrip("/").split("/")[-1]
    if (task / "runs" / name / "metadata.json").exists():
        return task / "runs" / name
    raise SystemExit(f"no such run in {task.name}: {ident}")


def task_info(task: Path) -> dict:
    """Task metadata plus runs and active dates derived from the files on disk."""
    meta = read_json(task / "metadata.json")
    runs = [read_json(r / "metadata.json") for r in run_dirs(task)]
    dates = {meta["created"][:10]}
    for run in runs:
        dates |= {run["started"][:10], *([run["ended"][:10]] if run.get("ended") else [])}
    dates |= {f.stem[:4] + "-" + f.stem[4:6] + "-" + f.stem[6:8] for f in task.glob("notes/*.json")}
    return {"dir": task, "meta": meta, "runs": runs, "dates": dates}


# ---------------------------------------------------------------- commands


def cmd_new_task(root: Path, a: argparse.Namespace) -> None:
    created = now()
    submitted = created
    if a.submitted:
        submitted = dt.datetime.fromisoformat(a.submitted)
        if submitted.tzinfo is None:
            submitted = submitted.replace(tzinfo=TZ)
    day = created.strftime("%Y-%m-%d")
    taken = [int(p.name[:3]) for p in (root / day).glob("[0-9][0-9][0-9]-*")]
    number = max(taken, default=0) + 1
    slug = slugify(a.slug)
    branch, commit, dirty = git_state()
    meta = {
        "task_id": f"{day}-{number:03d}",
        "slug": slug,
        "title": a.title,
        "submitted": submitted.isoformat(),
        "created": created.isoformat(),
        "timezone": "Australia/Sydney",
        "branch": branch,
        "start_commit": commit,
        "start_dirty": dirty,
        "final_commit": None,
        "status": "in_progress",
        "main_finding": None,
        "report": "REPORT.md",
    }
    prompt, redactions = redact(Path(a.prompt_file).read_text(encoding="utf-8"))
    task = root / day / f"{number:03d}-{slug}"
    header = [
        f"# Prompt for task {meta['task_id']}",
        "",
        f"- Task ID: {meta['task_id']}",
        f"- Submitted: {meta['submitted']}",
        f"- Created: {meta['created']}",
        f"- Branch: {branch}",
        f"- Starting commit: {commit}",
        f"- Redactions: {redactions}",
        "",
        "## Original prompt",
        "",
    ]
    task.mkdir(parents=True)
    write_text(task / "PROMPT.md", "\n".join(header) + prompt.rstrip("\n") + "\n")
    write_json(task / "metadata.json", meta)
    write_text(task / "REPORT.md", task_report(meta))
    print(task)


def cmd_run_start(root: Path, a: argparse.Namespace) -> None:
    task = find_task(root, a.task)
    task_meta = read_json(task / "metadata.json")
    started = now()
    stamp = f"{started:%Y%m%d-%H%M%S}-{slugify(a.slug)}"
    name, n = stamp, 1
    while (task / "runs" / name).exists():
        n += 1
        name = f"{stamp}-{n}"
    branch, commit, dirty = git_state()
    configs = [{"path": p, "sha256": sha256_file(Path(p))} for p in a.config]
    cloud = {k: None for k in CLOUD_KEYS} if a.cloud else None
    meta = {
        "run_id": f"{task_meta['task_id']}/{name}",
        "task_id": task_meta["task_id"],
        "title": a.title,
        "started": started.isoformat(),
        "commit": commit,
        "dirty": dirty,
        "branch": branch,
        "configs": configs,
        "command": a.command,
        **{key: getattr(a, key) for _, key in RUN_FIELDS},
        "status": "running",
        "ended": None,
        "exit_status": None,
        "cases_attempted": None,
        "cases_completed": None,
        "metrics": {},
        "evidence": [],
        "cloud": cloud,
    }
    run = task / "runs" / name
    run.mkdir(parents=True)
    write_json(run / "metadata.json", meta)
    write_text(run / "REPORT.md", run_report(meta))
    print(run)


def cmd_run_finish(root: Path, a: argparse.Namespace) -> None:
    run = find_run(find_task(root, a.task), a.run)
    meta = read_json(run / "metadata.json")
    if a.cloud:
        meta["cloud"] = {**(meta["cloud"] or {k: None for k in CLOUD_KEYS}), **json_arg(a.cloud)}
    if a.status == "completed" and meta["cloud"] is not None:
        missing = [k for k in ("cleanup_status", "upload_status") if not meta["cloud"].get(k)]
        if missing:
            raise SystemExit(
                f"cloud run cannot be completed without cloud.{', cloud.'.join(missing)}"
            )
    ended = now()
    meta.update(status=a.status, ended=ended.isoformat(), exit_status=a.exit_status)
    for key in ("cases_attempted", "cases_completed"):
        if getattr(a, key) is not None:
            meta[key] = getattr(a, key)
    if a.metrics:
        meta["metrics"] = {**meta["metrics"], **json_arg(a.metrics)}
    meta["evidence"] += [evidence_entry(Path(p)) for p in a.evidence]
    text = (run / "REPORT.md").read_text(encoding="utf-8")
    shown = lambda v: "Not recorded" if v is None else str(v)  # noqa: E731
    text = set_line(text, "Ended", meta["ended"])
    text = set_line(text, "Exit status", shown(meta["exit_status"]))
    done = {"completed": "yes", "partial": "partial"}.get(a.status, "no")
    text = set_line(text, "Completed (yes/no/partial)", f"{done} ({a.status})")
    text = set_line(text, "Cases attempted", shown(meta["cases_attempted"]))
    text = set_line(text, "Cases completed", shown(meta["cases_completed"]))
    if meta["evidence"]:
        items = []
        for e in meta["evidence"]:
            skip = f"; skipped over 500 MB: {', '.join(e['skipped'])}" if e.get("skipped") else ""
            items.append(f"- {e['path']} ({e['kind']}, sha256 {e['sha256']}{skip})")
        text = set_section(text, "## Evidence", "\n".join(items))
    if meta["cloud"] is not None:
        text = set_section(text, "## Cloud", "\n".join(cloud_lines(meta["cloud"])))
    write_json(run / "metadata.json", meta)
    write_text(run / "REPORT.md", text)


def cmd_note(root: Path, a: argparse.Namespace) -> None:
    task = find_task(root, a.task)
    when = now()
    write_json(
        task / "notes" / f"{when:%Y%m%d-%H%M%S}.json", {"time": when.isoformat(), "text": a.text}
    )


def cmd_amend(root: Path, a: argparse.Namespace) -> None:
    task = find_task(root, a.task)
    report = (find_run(task, a.run) if a.run else task) / "REPORT.md"
    text = report.read_text(encoding="utf-8").rstrip("\n")
    if not re.search(r"^## Amendments$", text, flags=re.M):
        text += "\n\n## Amendments"
    write_text(report, f"{text}\n\n### {now().isoformat()}\n\n{a.text}\n")


def cmd_fill(root: Path, a: argparse.Namespace) -> None:
    """Fill a section or identification line that still holds the placeholder.

    Lets agents complete reports from the command line. Text that is already written is never
    replaced here (revisions go through `amend`), so a filled report cannot be silently edited.
    """
    task = find_task(root, a.task)
    report = (find_run(task, a.run) if a.run else task) / "REPORT.md"
    text = report.read_text(encoding="utf-8")
    value = a.text
    if value.startswith("@"):
        value = Path(value[1:]).read_text(encoding="utf-8").strip()
    if a.section:
        heading = a.section if a.section.startswith("#") else f"## {a.section}"
        level = len(heading) - len(heading.lstrip("#"))
        stop = rf"(?=^#{{1,{level}}} |\Z)"  # next heading of the same or a higher level
        pattern = rf"(^{re.escape(heading)}\n)(.*?){stop}"
        m = re.search(pattern, text, flags=re.M | re.S)
        if m is None or TODO not in m.group(2):
            raise SystemExit(f"{heading} is missing or already filled; use amend to revise")
        # replace only the placeholder: sub-headings, notes and comments in the section stay
        body = m.group(2).replace(TODO, value, 1)
        text = text[: m.start(2)] + body + text[m.end(2) :]
    else:
        m = re.search(rf"^- {re.escape(a.field)}:(.*)$", text, flags=re.M)
        if m is None or TODO not in m.group(1):
            raise SystemExit(f"field '{a.field}' is missing or already filled")
        text = set_line(text, a.field, value)
    write_text(report, text)


def cmd_finish_task(root: Path, a: argparse.Namespace) -> None:
    task = find_task(root, a.task)
    old = {f: (task / f).read_text(encoding="utf-8") for f in ("metadata.json", "REPORT.md")}
    meta = read_json(task / "metadata.json")
    meta.update(
        status=a.status,
        main_finding=a.main_finding,
        final_commit=a.final_commit or git_state()[1],
    )
    text = set_line(old["REPORT.md"], "Final commit", meta["final_commit"])
    text = set_line(text, "Task status", a.status.replace("_", " "))
    write_json(task / "metadata.json", meta)
    write_text(task / "REPORT.md", text)
    problems = check_task(task)
    if problems:
        for f, content in old.items():
            write_text(task / f, content)
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit("finish-task refused; nothing changed")


def table_row(*cells: str) -> str:
    return "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |"


def link(label: str, target: Path, start: Path) -> str:
    return f"[{label}]({Path(os.path.relpath(target, start)).as_posix()})"


def cmd_index(root: Path, _a: argparse.Namespace | None = None) -> None:
    infos = sorted(
        (task_info(t) for t in task_dirs(root)),
        key=lambda i: (i["meta"]["created"], i["meta"]["task_id"]),
        reverse=True,
    )
    head = "| Date | Task ID | Prompt | Status | Main finding | Runs | Report |"
    rows = [head, "|---|---|---|---|---|---|---|"]
    for i in infos:
        m, d = i["meta"], i["dir"]
        rows.append(
            table_row(
                m["created"][:10],
                m["task_id"],
                link(m["title"], d / "PROMPT.md", root),
                m["status"],
                m["main_finding"] or "-",
                str(len(i["runs"])),
                link("REPORT", d / "REPORT.md", root),
            )
        )
    intro = [
        "# DREAM reports",
        "",
        "Generated by `experiments/reports.py index`; do not edit. "
        "Earlier results: [HISTORY.md](HISTORY.md).",
        "",
    ]
    write_text(root / "INDEX.md", "\n".join([*intro, *rows]) + "\n")
    days = {d for i in infos for d in i["dates"]}
    for day in sorted(days):
        out = root / day
        own = [i for i in infos if i["meta"]["created"][:10] == day]
        cont = [i for i in infos if day in i["dates"] and i not in own]
        lines = [
            f"# Reports for {day}",
            "",
            "[Root index](../INDEX.md)",
            "",
            "## Tasks created",
            "",
        ]
        lines += [head.replace("| Date ", "| Date ", 1), "|---|---|---|---|---|---|---|"]
        for i in own:
            m = i["meta"]
            lines.append(
                table_row(
                    day,
                    m["task_id"],
                    link(m["title"], i["dir"] / "PROMPT.md", out),
                    m["status"],
                    m["main_finding"] or "-",
                    str(len(i["runs"])),
                    link("REPORT", i["dir"] / "REPORT.md", out),
                )
            )
        lines += ["", "## Continued on this day", ""]
        lines += [
            f"- {link(i['meta']['task_id'], i['dir'] / 'REPORT.md', out)}: {i['meta']['title']}"
            for i in cont
        ]
        lines += [
            "",
            "## Runs started",
            "",
            "| Run | Task | Status | Started | Report |",
            "|---|---|---|---|---|",
        ]
        for i in infos:
            for r_dir, r in zip(run_dirs(i["dir"]), i["runs"], strict=True):
                if r["started"][:10] == day:
                    lines.append(
                        table_row(
                            r["run_id"].split("/")[-1],
                            r["task_id"],
                            r["status"],
                            r["started"],
                            link("REPORT", r_dir / "REPORT.md", out),
                        )
                    )
        write_text(out / "INDEX.md", "\n".join(lines) + "\n")


def check_task(task: Path) -> list[str]:
    name = task.name
    problems = []
    for f in ("PROMPT.md", "REPORT.md", "metadata.json"):
        if not (task / f).exists():
            problems.append(f"{name}: missing {f}")
    if problems:
        return problems
    meta = read_json(task / "metadata.json")
    problems += [f"{name}: metadata lacks key {k}" for k in TASK_KEYS if k not in meta]
    text = (task / "REPORT.md").read_text(encoding="utf-8")
    lines, at = text.splitlines(), 0
    for heading in TASK_HEADINGS:
        if heading not in lines[at:]:
            problems.append(f"{name}: REPORT.md lacks heading '{heading}' (or it is out of order)")
        else:
            at += lines[at:].index(heading) + 1
    final = meta.get("status") != "in_progress"
    if final and todo_count(text):
        problems.append(f"{name}: final task REPORT.md still has {TODO} markers")
    for run in run_dirs(task):
        rname = f"{name}/runs/{run.name}"
        if not (run / "REPORT.md").exists():
            problems.append(f"{rname}: missing REPORT.md")
            continue
        rmeta = read_json(run / "metadata.json")
        problems += [f"{rname}: metadata lacks key {k}" for k in RUN_KEYS if k not in rmeta]
        if final and rmeta.get("status") == "running":
            problems.append(f"{rname}: still running in a final task")
        if rmeta.get("status") != "running" and todo_count((run / "REPORT.md").read_text("utf-8")):
            problems.append(f"{rname}: finished run REPORT.md still has {TODO} markers")
    return problems


def cmd_check(root: Path, a: argparse.Namespace) -> None:
    tasks = [find_task(root, a.task)] if a.task else task_dirs(root)
    problems = [p for t in tasks for p in check_task(t)]
    print("\n".join(problems) if problems else f"ok: {len(tasks)} task(s) checked")
    if problems:
        raise SystemExit(1)


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root", help="Reports directory (default: $DREAM_REPORTS_ROOT or Reports/)"
    )
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], parents=[common])
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_, parents=[common])
        s.set_defaults(func=func)
        return s

    s = add("new-task", cmd_new_task, "create a task from a prompt file")
    s.add_argument("--slug", required=True)
    s.add_argument("--title", required=True)
    s.add_argument("--prompt-file", required=True)
    s.add_argument("--submitted", help="ISO 8601 time the prompt was submitted")

    s = add("run-start", cmd_run_start, "register an experimental run")
    s.add_argument("--task", required=True, help="task ID or path")
    s.add_argument("--slug", required=True)
    s.add_argument("--title", required=True)
    s.add_argument("--command", help="command line that runs the experiment")
    s.add_argument("--config", action="append", default=[], help="config file (repeatable)")
    for _, key in RUN_FIELDS:
        s.add_argument(f"--{key}")
    s.add_argument("--cloud", action="store_true", help="run uses cloud resources")

    s = add("run-finish", cmd_run_finish, "record how a run ended")
    s.add_argument("--task", required=True)
    s.add_argument("--run", required=True, help="run ID, directory name or path")
    s.add_argument("--status", required=True, choices=RUN_STATUS)
    s.add_argument("--exit-status", type=int)
    s.add_argument("--cases-attempted", type=int)
    s.add_argument("--cases-completed", type=int)
    s.add_argument("--metrics", help="JSON object or @file")
    s.add_argument("--evidence", action="append", default=[], help="file or dir to checksum")
    s.add_argument("--cloud", help="JSON object or @file merged into cloud")

    s = add("note", cmd_note, "add a dated progress note to a task")
    s.add_argument("--task", required=True)
    s.add_argument("--text", required=True)

    s = add("amend", cmd_amend, "append a timestamped amendment to a report")
    s.add_argument("--task", required=True)
    s.add_argument("--run", help="amend this run's report instead of the task's")
    s.add_argument("--text", required=True)

    s = add("fill", cmd_fill, "fill a placeholder section or field (never overwrites text)")
    s.add_argument("--task", required=True)
    s.add_argument("--run", help="fill this run's report instead of the task's")
    target = s.add_mutually_exclusive_group(required=True)
    target.add_argument("--section", help="heading text, e.g. 'Purpose' or '### Evaluation'")
    target.add_argument("--field", help="identification label, e.g. 'Renderer'")
    s.add_argument("--text", required=True, help="text or @file")

    s = add("finish-task", cmd_finish_task, "finalise a task (refused if check fails)")
    s.add_argument("--task", required=True)
    s.add_argument("--status", required=True, choices=FINAL_TASK)
    s.add_argument("--main-finding", required=True)
    s.add_argument("--final-commit", help="default: HEAD")

    add("index", cmd_index, "regenerate Reports/INDEX.md and the daily indexes")
    s = add("check", cmd_check, "validate reports; exit 1 on problems")
    s.add_argument("--task")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = reports_root(args.root)
    args.func(root, args)
    if args.command not in ("check", "index"):
        cmd_index(root)


if __name__ == "__main__":
    main()
