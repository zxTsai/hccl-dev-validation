from __future__ import annotations

import json
import os
import signal
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BackendResult:
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


def load_hccl_vm_config(config_path: str | Path = "configs/hccl_vm.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"HCCL-VM config does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"HCCL-VM config must be a mapping: {path}")
    return data


def load_case_config(case_config_path: str | Path) -> dict[str, Any]:
    path = Path(case_config_path)
    if not path.exists():
        raise FileNotFoundError(f"HCCL-VM case config does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"HCCL-VM case config must be a JSON object: {path}")
    return data


def build_command(
    case: dict[str, Any],
    config: dict[str, Any] | None = None,
    case_config: dict[str, Any] | None = None,
) -> list[str]:
    config = config or load_hccl_vm_config()
    case_config_path = case["backend_config"]["case_config"]
    case_config = case_config or load_case_config(case_config_path)
    context = _build_context(config, case, case_config)
    return [part.format(**context) for part in config.get("command_template", [])]


def run_case(
    case: dict[str, Any],
    output_dir: str | Path,
    dry_run: bool = False,
    config_path: str | Path = "configs/hccl_vm.yaml",
    timeout_sec: int | None = None,
) -> BackendResult:
    config = load_hccl_vm_config(config_path)
    case_config_path = Path(case["backend_config"]["case_config"])
    case_config = load_case_config(case_config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    command = build_command(case, config=config, case_config=case_config)
    script = build_shell_script(config, case, case_config, output)
    (output / "command.txt").write_text(" ".join(shlex.quote(x) for x in command) + "\n\n" + script, encoding="utf-8")
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    stdout_path.touch()
    stderr_path.touch()
    timeout = _effective_timeout(config, case, case_config, timeout_sec)

    if dry_run:
        stdout_path.write_text(script, encoding="utf-8")
        return BackendResult(
            command=command,
            stdout=script,
            stderr="",
            returncode=0,
            duration_sec=0.0,
            timeout_sec=timeout,
            timed_out=False,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            dry_run=True,
        )

    preflight_reasons = preflight(config, case_config)
    if preflight_reasons:
        message = "\n".join(preflight_reasons) + "\n"
        stderr_path.write_text(message, encoding="utf-8")
        return BackendResult(
            command=command,
            stdout="",
            stderr=message,
            returncode=2,
            duration_sec=0.0,
            timeout_sec=timeout,
            timed_out=False,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            dry_run=False,
            preflight_ok=False,
            preflight_reasons=preflight_reasons,
        )

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in config.get("env", {}).items()})

    start = time.monotonic()
    returncode = 0
    timed_out = False
    proc: subprocess.Popen[str] | None = None
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr_file:
        try:
            proc = subprocess.Popen(
                [config.get("shell", "bash"), "-lc", script],
                cwd=_install_bin_dir(config),
                text=True,
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                start_new_session=True,
            )
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            stderr_file.write(
                f"\ncase timed out after {timeout} seconds.\n"
                "possible causes: missing sudo credential, HCCL-VM needs a TTY, or HCCL-VM startup/runtime is stuck.\n"
            )
            stderr_file.flush()
            _terminate_process_group(proc)
            _cleanup_after_timeout(config)
    try:
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        stdout = ""
        stderr = ""
    duration = time.monotonic() - start

    hccl_log_path = _copy_hccl_test_log(config, case_config, output)
    return BackendResult(
        command=command,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        duration_sec=duration,
        timeout_sec=timeout,
        timed_out=timed_out,
        hccl_log_path=str(hccl_log_path) if hccl_log_path else None,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        preflight_ok=True,
        preflight_reasons=[],
    )


def preflight(config: dict[str, Any], case_config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    install_dir = _install_dir(config)
    hccl_vm_bin = install_dir / "bin" / "hccl-vm"
    cann_home = Path(config["cann_home"])
    hccl_test_path = cann_home / "tools" / "hccl_test" / "bin" / case_config["hccl_test_bin"]

    required_paths = [
        ("hccl-vm binary", hccl_vm_bin),
        ("CANN set_env.sh", cann_home / "set_env.sh"),
        ("hccl_test binary", hccl_test_path),
    ]
    for label, path in required_paths:
        if not path.exists():
            reasons.append(f"{label} does not exist: {path}")

    if _sudo_required(config):
        sudo_checks = [
            (["sudo", "-n", str(hccl_vm_bin), "--help"], f"Grant NOPASSWD for {hccl_vm_bin}."),
            (["sudo", "-n", "rm", "--version"], "Grant NOPASSWD for /usr/bin/rm."),
        ]
        for sudo_check_cmd, hint in sudo_checks:
            try:
                proc = subprocess.run(sudo_check_cmd, text=True, capture_output=True, timeout=5)
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                reasons.append(f"sudo preflight failed: {exc}")
            else:
                if proc.returncode != 0:
                    detail = (proc.stderr or proc.stdout or "").strip()
                    reasons.append(
                        f"sudo preflight failed for: {' '.join(sudo_check_cmd)}. {hint} "
                        "Or run sudo -v in an interactive terminal."
                        + (f" sudo output: {detail}" if detail else "")
                    )

    return reasons


def build_shell_script(
    config: dict[str, Any],
    case: dict[str, Any],
    case_config: dict[str, Any],
    output_dir: str | Path,
) -> str:
    context = _build_context(config, case, case_config)
    hccl_test_args = " ".join(shlex.quote(str(x)) for x in case_config.get("args", []))
    hccl_test_cmd = (
        f"mpirun --allow-run-as-root --oversubscribe -np {context['np']} "
        f"{shlex.quote(context['hccl_test_path'])} {hccl_test_args} > {shlex.quote(context['hccl_test_log'])} 2>&1"
    )
    grep_pattern = case_config.get("summary_grep", "check_result|success|\\[error\\]|execute failed|comm is nullptr")
    grep_cmd = f"grep -E '{grep_pattern}' {context['hccl_test_log']} | tail -20 || true"

    lines = [
        "set -euo pipefail",
        f"source {shlex.quote(context['set_env'])} >/dev/null 2>&1",
        f"export ASCEND_HOME_PATH={shlex.quote(context['cann_home'])}",
        "export HCCL_DFS_CONFIG=cluster_heartbeat:off",
        f"export HCCL_OP_EXPANSION_MODE={shlex.quote(context['mode'])}",
        f"export RANK_TABLE_FILE={shlex.quote(context['rank_table_file'])}",
        'export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/devlib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
        _rm_rf_command(config, [context["topology_output_dir"]]),
        _rm_rf_command(config, config.get("cleanup_shm", [])),
        f"rm -f {shlex.quote(context['hccl_test_log'])}",
        f"cd {shlex.quote(context['bin_dir'])}",
        "{",
        '  echo "hccl-vm plugin install @runner"',
        f'  echo "hccl-vm mock-comm {context["mock_comm"]}"',
        f"  echo {shlex.quote(hccl_test_cmd)}",
        f"  echo {shlex.quote(grep_cmd)}",
        '  echo "exit"',
        f"}} | {shlex.quote(context['hccl_vm_bin'])} start {shlex.quote(context['topology'])}",
    ]
    return "\n".join(lines) + "\n"


def _build_context(config: dict[str, Any], case: dict[str, Any], case_config: dict[str, Any]) -> dict[str, Any]:
    install_dir = _install_dir(config)
    bin_dir = install_dir / "bin"
    cann_home = Path(config["cann_home"])
    hccl_test_bin = case_config["hccl_test_bin"]
    result_log = case_config.get("result_log", f"log_{case['name']}.txt")
    return {
        "case_config": case["backend_config"]["case_config"],
        "hccl_vm_root": config["hccl_vm_root"],
        "install_dir": str(install_dir),
        "bin_dir": str(bin_dir),
        "hccl_vm_bin": str(bin_dir / "hccl-vm"),
        "hccl_vm": str(bin_dir / "hccl-vm"),
        "cann_home": str(cann_home),
        "set_env": str(cann_home / "set_env.sh"),
        "rank_table_file": str(install_dir / "data" / "ranktable.json"),
        "topology": case_config.get("topology", config.get("topology", "ascend950_cluster_32_server_normal.yaml")),
        "mode": case_config.get("mode", config.get("mode", "AIV")),
        "mock_comm": case_config.get("mock_comm", _mock_comm_for_rank(config, case["rank_size"])),
        "np": case_config.get("np", case["rank_size"]),
        "hccl_test_bin": hccl_test_bin,
        "hccl_test_path": str(cann_home / "tools" / "hccl_test" / "bin" / hccl_test_bin),
        "hccl_test_log": str(bin_dir / result_log),
        "topology_output_dir": str(install_dir / "config" / "network" / "cluster" / Path(case_config.get("topology", config.get("topology", "ascend950_cluster_32_server_normal.yaml"))).stem),
    }


def _install_dir(config: dict[str, Any]) -> Path:
    if config.get("install_dir"):
        return Path(config["install_dir"])
    return Path(config["hccl_vm_root"]) / "hccl_vm_install"


def _install_bin_dir(config: dict[str, Any]) -> str:
    return str(_install_dir(config) / "bin")


def _sudo_required(config: dict[str, Any]) -> bool:
    preflight_config = config.get("preflight", {}) or {}
    return bool(preflight_config.get("sudo_required", config.get("sudo_required", False)))


def _mock_comm_for_rank(config: dict[str, Any], rank_size: int) -> str:
    rank_map = config.get("mock_comm_by_rank", {})
    key = str(rank_size)
    if key not in rank_map:
        raise ValueError(f"no mock_comm configured for rank_size={rank_size}")
    return str(rank_map[key])


def _effective_timeout(
    config: dict[str, Any],
    case: dict[str, Any],
    case_config: dict[str, Any],
    override: int | None,
) -> int:
    if override is not None:
        return int(override)
    return int(case["backend_config"].get("timeout_sec", case_config.get("timeout_sec", config.get("timeout_sec", 120))))


def _cleanup_args(paths: list[str]) -> str:
    # Keep simple glob patterns expandable while still quoting ordinary paths.
    parts = []
    for path in paths:
        if any(ch in path for ch in "*?["):
            parts.append(path)
        else:
            parts.append(shlex.quote(path))
    return " ".join(parts)


def _rm_rf_command(config: dict[str, Any], paths: list[str]) -> str:
    args = _cleanup_args(paths)
    if not args:
        return "true"
    prefix = "sudo -n " if _sudo_required(config) else ""
    return f"{prefix}rm -rf {args} 2>/dev/null || true"


def _terminate_process_group(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=5)


def _cleanup_after_timeout(config: dict[str, Any]) -> None:
    patterns = config.get("timeout_cleanup_process_patterns", ["hccl-vm start"])
    for pattern in patterns:
        subprocess.run(["pkill", "-f", str(pattern)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _copy_hccl_test_log(config: dict[str, Any], case_config: dict[str, Any], output: Path) -> Path | None:
    result_log = case_config.get("result_log")
    if not result_log:
        return None
    source = _install_dir(config) / "bin" / result_log
    if not source.exists():
        return None
    target = output / "hccl_test.log"
    target.write_text(source.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return target
