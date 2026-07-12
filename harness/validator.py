from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_case(case: dict[str, Any], result: Any, output_dir: str | Path | None = None) -> dict[str, Any]:
    if not case.get("enabled", True):
        return {
            "status": "SKIP",
            "fail_reasons": [case.get("notes", {}).get("reason") or "case is disabled"],
        }

    if getattr(result, "dry_run", False):
        return {"status": "DRY_RUN", "fail_reasons": []}

    expect = case.get("expect", {})
    fail_reasons: list[str] = []
    preflight_reasons = getattr(result, "preflight_reasons", []) or []
    if not getattr(result, "preflight_ok", True):
        fail_reasons.extend(preflight_reasons)

    if getattr(result, "timed_out", False):
        timeout_sec = getattr(result, "timeout_sec", None)
        fail_reasons.append(f"case timed out after {timeout_sec} seconds")
        fail_reasons.append("可能原因：sudo 凭据不存在、HCCL-VM 需要 TTY、HCCL-VM 启动或运行阶段卡住")

    expected_returncode = expect.get("returncode", 0)
    if result.returncode != expected_returncode:
        fail_reasons.append(f"returncode {result.returncode} != expected {expected_returncode}")

    log_text = (result.stdout or "") + "\n" + (result.stderr or "")
    hccl_log_path = getattr(result, "hccl_log_path", None)
    if hccl_log_path and Path(hccl_log_path).exists():
        log_text += "\n" + Path(hccl_log_path).read_text(encoding="utf-8", errors="replace")

    if getattr(result, "preflight_ok", True) and not getattr(result, "timed_out", False):
        for pattern in expect.get("required_log_patterns", []) or []:
            if pattern not in log_text:
                fail_reasons.append(f"required log pattern missing: {pattern}")

        for pattern in expect.get("forbidden_log_patterns", []) or []:
            if pattern in log_text:
                fail_reasons.append(f"forbidden log pattern found: {pattern}")

    return {
        "status": "FAIL" if fail_reasons else "PASS",
        "fail_reasons": fail_reasons,
    }
