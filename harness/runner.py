from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from harness.backends import hccl_vm
from harness.case_loader import load_case, load_cases_from_dir
from harness.report import write_case_report, write_final_real_run_report, write_methods_report, write_summary
from harness.validator import validate_case


@dataclass
class SkippedResult:
    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    duration_sec: float
    timeout_sec: int
    timed_out: bool = False
    hccl_log_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    dry_run: bool = False
    preflight_ok: bool = True
    preflight_reasons: list[str] | None = None


def run_single(case_path: str | Path, dry_run: bool = False, timeout_sec: int | None = None) -> dict[str, Any]:
    case = load_case(case_path)
    return run_loaded_case(case, dry_run=dry_run, timeout_sec=timeout_sec)


def run_loaded_case(case: dict[str, Any], dry_run: bool = False, timeout_sec: int | None = None) -> dict[str, Any]:
    output_dir = _run_output_dir(case["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"

    if not case.get("enabled", True):
        result = SkippedResult(
            command=[],
            stdout="",
            stderr=case.get("notes", {}).get("reason", "case is disabled"),
            returncode=0,
            duration_sec=0.0,
            timeout_sec=int(timeout_sec or case["backend_config"].get("timeout_sec", 0) or 0),
            timed_out=False,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
    else:
        if case["backend"] != "hccl_vm":
            raise ValueError(f"unsupported backend: {case['backend']}")
        result = hccl_vm.run_case(case, output_dir=output_dir, dry_run=dry_run, timeout_sec=timeout_sec)

    if not stdout_path.exists():
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
    if not stderr_path.exists():
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
    validation = validate_case(case, result, output_dir)
    return write_case_report(case, output_dir, result, validation)


def run_dir(
    case_dir: str | Path,
    dry_run: bool = False,
    summary: bool = False,
    scale: str | None = None,
    timeout_sec: int | None = None,
) -> list[dict[str, Any]]:
    cases = load_cases_from_dir(case_dir)
    if scale:
        cases = [case for case in cases if case.get("scale") == scale]

    reports = [run_loaded_case(case, dry_run=dry_run, timeout_sec=timeout_sec) for case in cases]
    if summary:
        write_summary(reports)
        write_methods_report(cases, reports)
        config = hccl_vm.load_hccl_vm_config()
        commands = [_report_command(item) for item in reports]
        preflight_reasons = []
        if not dry_run:
            seen = set()
            for item in reports:
                for reason in item.get("preflight_reasons", []):
                    if reason not in seen:
                        preflight_reasons.append(reason)
                        seen.add(reason)
        write_final_real_run_report(reports, config, commands, preflight_reasons)
    return reports


def summarize(case_roots: list[str | Path] | None = None) -> Path:
    roots = case_roots or [Path("cases")]
    cases: list[dict[str, Any]] = []
    for root in roots:
        path = Path(root)
        if path.exists():
            if path.is_dir():
                cases.extend(load_cases_from_dir(path))
            else:
                cases.append(load_case(path))
    reports = None
    summary_path = Path("outputs/reports/summary.json")
    if summary_path.exists():
        try:
            reports = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reports = None
    return write_methods_report(cases, reports)


def _run_output_dir(case_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path("outputs") / "runs" / f"{timestamp}_{case_name}"


def _report_command(report: dict[str, Any]) -> str:
    command = report.get("command") or []
    return " ".join(str(part) for part in command) if command else f"SKIP {report.get('case_name', '')}"
