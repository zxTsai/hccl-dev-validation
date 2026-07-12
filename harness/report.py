from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_DIR = Path("outputs/reports")


def write_case_report(
    case: dict[str, Any],
    output_dir: str | Path,
    backend_result: Any,
    validation: dict[str, Any],
) -> dict[str, Any]:
    output = Path(output_dir)
    report = {
        "case_name": case["name"],
        "case_path": case.get("_path", ""),
        "op": case["op"],
        "rank_size": case["rank_size"],
        "dtype": case["dtype"],
        "count": case["count"],
        "scale": case["scale"],
        "backend": case["backend"],
        "enabled": case["enabled"],
        "status": validation["status"],
        "returncode": getattr(backend_result, "returncode", None),
        "timeout_sec": getattr(backend_result, "timeout_sec", None),
        "timed_out": bool(getattr(backend_result, "timed_out", False)),
        "command": getattr(backend_result, "command", []),
        "stdout_path": getattr(backend_result, "stdout_path", None) or str(output / "stdout.log"),
        "stderr_path": getattr(backend_result, "stderr_path", None) or str(output / "stderr.log"),
        "hccl_log_path": getattr(backend_result, "hccl_log_path", None),
        "report_path": str(output / "report.json"),
        "fail_reasons": validation["fail_reasons"],
        "duration_sec": round(float(getattr(backend_result, "duration_sec", 0.0)), 3),
        "dry_run": bool(getattr(backend_result, "dry_run", False)),
        "preflight_ok": bool(getattr(backend_result, "preflight_ok", True)),
        "preflight_reasons": getattr(backend_result, "preflight_reasons", []) or [],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_summary(reports: list[dict[str, Any]]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = REPORT_DIR / "summary.json"
    summary_md = REPORT_DIR / "summary.md"
    summary_json.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# HCCL-VM Harness Summary",
        "",
        "| Case | Op | Ranks | dtype | count | scale | status | timeout_sec | timed_out | fail_reasons | stdout | stderr | hccl_log | report |",
        "| --- | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for item in reports:
        reasons = "; ".join(item.get("fail_reasons", []))
        row = {key: _md_escape(str(item.get(key, ""))) for key in item.keys()}
        for key in [
            "case_name",
            "op",
            "rank_size",
            "dtype",
            "count",
            "scale",
            "status",
            "timeout_sec",
            "timed_out",
            "stdout_path",
            "stderr_path",
            "hccl_log_path",
            "report_path",
        ]:
            row.setdefault(key, "")
        lines.append(
            "| {case_name} | {op} | {rank_size} | {dtype} | {count} | {scale} | {status} | {timeout_sec} | {timed_out} | {reasons} | {stdout_path} | {stderr_path} | {hccl_log_path} | {report_path} |".format(
                reasons=_md_escape(reasons),
                **row,
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_json, summary_md


def write_methods_report(cases: list[dict[str, Any]], reports: list[dict[str, Any]] | None = None) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_by_case = {item["case_name"]: item for item in reports or []}
    target = REPORT_DIR / "communication_primitives_test_methods.md"
    lines = [
        "# 通信原语测试方法汇总",
        "",
        "| 通信原语 | Case 文件 | HCCL-VM 配置文件 | rank_size | dtype | count | scale | 测试方法 | 预期行为 | 校验方式 | 当前状态 | 备注 |",
        "| ---- | ------- | ------------ | --------: | ----- | ----: | ----- | ---- | ---- | ---- | ---- | -- |",
    ]
    for case in sorted(cases, key=lambda x: (x["op"], x["rank_size"], x["scale"], x["name"])):
        runtime_report = report_by_case.get(case["name"], {})
        status = runtime_report.get("status")
        if not status:
            status = "ENABLED" if case.get("enabled", True) else "DISABLED"
        notes = case.get("notes", {})
        reason = notes.get("reason", "")
        if runtime_report.get("fail_reasons"):
            reason = "; ".join(runtime_report["fail_reasons"])
        lines.append(
            "| {op} | {case_file} | {case_config} | {rank_size} | {dtype} | {count} | {scale} | {method} | {expected} | {check} | {status} | {reason} |".format(
                op=_md_escape(str(case["op"])),
                case_file=_md_escape(str(case.get("_path", ""))),
                case_config=_md_escape(str(case["backend_config"]["case_config"])),
                rank_size=case["rank_size"],
                dtype=_md_escape(str(case["dtype"])),
                count=case["count"],
                scale=_md_escape(str(case["scale"])),
                method=_md_escape(str(notes.get("method", ""))),
                expected=_md_escape(str(notes.get("expected_behavior", ""))),
                check=_md_escape(str(notes.get("check_method", ""))),
                status=_md_escape(str(status)),
                reason=_md_escape(str(reason)),
            )
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_final_real_run_report(
    reports: list[dict[str, Any]],
    config: dict[str, Any],
    commands: list[str],
    preflight_reasons: list[str] | None = None,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = REPORT_DIR / "final_real_run_report.md"
    status_counts = Counter(item.get("status", "UNKNOWN") for item in reports)
    op_counts: dict[str, Counter] = defaultdict(Counter)
    scale_counts: dict[str, Counter] = defaultdict(Counter)
    for item in reports:
        op_counts[item.get("op", "")][item.get("status", "UNKNOWN")] += 1
        scale_counts[item.get("scale", "")][item.get("status", "UNKNOWN")] += 1

    enabled = sum(1 for item in reports if item.get("enabled"))
    disabled = sum(1 for item in reports if not item.get("enabled"))
    preflight_reasons = preflight_reasons or []
    sudo_required = bool((config.get("preflight") or {}).get("sudo_required", config.get("sudo_required", False)))

    lines = [
        "# 最终真实执行报告",
        "",
        f"- 执行时间: {datetime.now().isoformat(timespec='seconds')}",
        f"- HCCL-VM root: `{config.get('hccl_vm_root', '')}`",
        f"- install_dir: `{config.get('install_dir', '')}`",
        f"- CANN home: `{config.get('cann_home', '')}`",
        f"- sudo_required: `{sudo_required}`",
        f"- preflight: `{'PASS' if not preflight_reasons else 'FAIL'}`",
        f"- sudo -n true: `{'PASS/NOT_REQUIRED' if not preflight_reasons else 'FAIL'}`",
        "",
    ]
    all_enabled_blocked = bool(preflight_reasons) and bool(reports) and all(
        (not item.get("enabled")) or (not item.get("preflight_ok", True)) for item in reports
    )
    if preflight_reasons and (not reports or all_enabled_blocked):
        lines.extend(
            [
                "真实执行未开始，因为当前非交互环境没有满足 preflight 条件。",
                "请在交互终端执行 `sudo -v` 后重新运行 harness。",
                "",
            ]
        )
    if preflight_reasons:
        lines.append("## Preflight 失败原因")
        lines.extend(f"- {reason}" for reason in preflight_reasons)
        lines.append("")

    lines.extend(["## 执行命令", ""])
    lines.extend(f"- `{cmd}`" for cmd in commands)
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            f"- 总 case 数: {len(reports)}",
            f"- enabled case 数: {enabled}",
            f"- disabled case 数: {disabled}",
            f"- PASS: {status_counts.get('PASS', 0)}",
            f"- FAIL: {status_counts.get('FAIL', 0)}",
            f"- SKIP/DISABLED: {status_counts.get('SKIP', 0) + status_counts.get('DISABLED', 0)}",
            "",
            "## 按通信原语统计",
            "",
            "| Op | PASS | FAIL | SKIP |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for op, counter in sorted(op_counts.items()):
        lines.append(f"| {op} | {counter.get('PASS', 0)} | {counter.get('FAIL', 0)} | {counter.get('SKIP', 0)} |")

    lines.extend(["", "## 按 Scale 统计", "", "| Scale | PASS | FAIL | SKIP |", "| --- | ---: | ---: | ---: |"])
    for scale, counter in sorted(scale_counts.items()):
        lines.append(f"| {scale} | {counter.get('PASS', 0)} | {counter.get('FAIL', 0)} | {counter.get('SKIP', 0)} |")

    lines.extend(
        [
            "",
            "## Case 结果",
            "",
            "| Case | Op | Scale | Status | timeout_sec | timed_out | stdout | stderr | hccl_log | report | fail_reasons |",
            "| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in reports:
        lines.append(
            "| {case_name} | {op} | {scale} | {status} | {timeout_sec} | {timed_out} | {stdout_path} | {stderr_path} | {hccl_log_path} | {report_path} | {reasons} |".format(
                reasons=_md_escape("; ".join(item.get("fail_reasons", []))),
                **{k: _md_escape(str(v)) for k, v in item.items()},
            )
        )

    fail_reports = [item for item in reports if item.get("status") == "FAIL"]
    if fail_reports:
        lines.extend(["", "## FAIL 详情", ""])
        for item in fail_reports:
            lines.extend(
                [
                    f"### {item['case_name']}",
                    f"- 原因: {'; '.join(item.get('fail_reasons', []))}",
                    f"- stdout: `{item.get('stdout_path')}`",
                    f"- stderr: `{item.get('stderr_path')}`",
                    f"- hccl_log: `{item.get('hccl_log_path')}`",
                    f"- report: `{item.get('report_path')}`",
                    "",
                ]
            )

    disabled_reports = [item for item in reports if not item.get("enabled")]
    if disabled_reports:
        lines.extend(["", "## Disabled/SKIP 原因", ""])
        for item in disabled_reports:
            lines.append(f"- {item['case_name']}: {'; '.join(item.get('fail_reasons', []))}")

    lines.extend(
        [
            "",
            "## 下一步建议",
            "",
            "- 对 FAIL case 先查看对应 `report.json`、`stderr.log`、`stdout.log` 和 `hccl_test.log`。",
            "- 若真实执行未开始，先在交互终端执行 `sudo -v`。",
            "- AllToAllV 与 SendRecv 仍需确认稳定参数或真实可运行方式后再启用。",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
