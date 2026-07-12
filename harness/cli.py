from __future__ import annotations

import argparse
import sys

from harness.case_loader import CaseLoadError
from harness.runner import run_dir, run_single, summarize


def cmd_run(args: argparse.Namespace) -> int:
    report = run_single(args.case, dry_run=args.dry_run, timeout_sec=args.timeout_sec)
    _print_case_report(report)
    return 0 if report["status"] in {"PASS", "SKIP"} else 1


def cmd_run_dir(args: argparse.Namespace) -> int:
    reports = run_dir(
        args.case_dir,
        dry_run=args.dry_run,
        summary=args.summary,
        scale=args.scale,
        timeout_sec=args.timeout_sec,
    )
    for report in reports:
        _print_case_report(report)
    if args.summary:
        print("summary.json: outputs/reports/summary.json")
        print("summary.md: outputs/reports/summary.md")
    return 0 if all(item["status"] in {"PASS", "SKIP"} for item in reports) else 1


def cmd_summarize(args: argparse.Namespace) -> int:
    path = summarize(args.paths or None)
    print(f"methods report: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="HCCL code validation harness driven by HCCL-VM",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one case")
    run_parser.add_argument("case", help="case YAML path")
    run_parser.add_argument("--dry-run", action="store_true", help="render commands without running HCCL-VM")
    run_parser.add_argument("--timeout-sec", type=int, help="override case execution timeout in seconds")
    run_parser.set_defaults(func=cmd_run)

    dir_parser = subparsers.add_parser("run-dir", help="run all cases under a directory")
    dir_parser.add_argument("case_dir", help="case directory")
    dir_parser.add_argument("--dry-run", action="store_true", help="render commands without running HCCL-VM")
    dir_parser.add_argument("--summary", action="store_true", help="write outputs/reports/summary.json and summary.md")
    dir_parser.add_argument("--scale", choices=["small", "basic", "medium"], help="run only cases with this scale")
    dir_parser.add_argument("--timeout-sec", type=int, help="override each case execution timeout in seconds")
    dir_parser.set_defaults(func=cmd_run_dir)

    summarize_parser = subparsers.add_parser("summarize", help="write communication primitive test method report")
    summarize_parser.add_argument("paths", nargs="*", help="case file or directory paths, defaults to cases/")
    summarize_parser.set_defaults(func=cmd_summarize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        exit_code = args.func(args)
    except (CaseLoadError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        exit_code = 2
    sys.exit(exit_code)


def _print_case_report(report: dict) -> None:
    reasons = "; ".join(report.get("fail_reasons", []))
    suffix = f" ({reasons})" if reasons else ""
    print(f"{report['status']}: {report['case_name']} [{report['op']} {report['rank_size']}p {report['scale']}]{suffix}")
    print(f"  report: {report.get('report_path')}")


if __name__ == "__main__":
    main()
