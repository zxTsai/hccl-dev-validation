from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.backends import hccl_vm
from harness.backends.hccl_vm import build_command, build_shell_script, preflight, run_case
from harness.case_loader import CaseLoadError, load_case
from harness.cli import build_parser
from harness.report import write_methods_report, write_summary
from harness.runner import run_dir
from harness.validator import validate_case


class Result:
    def __init__(
        self,
        returncode=0,
        stdout="",
        stderr="",
        hccl_log_path=None,
        dry_run=False,
        timeout_sec=10,
        timed_out=False,
        preflight_ok=True,
        preflight_reasons=None,
    ):
        self.command = ["fake"]
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.hccl_log_path = hccl_log_path
        self.duration_sec = 0.01
        self.dry_run = dry_run
        self.timeout_sec = timeout_sec
        self.timed_out = timed_out
        self.preflight_ok = preflight_ok
        self.preflight_reasons = preflight_reasons or []


def test_case_loader_loads_valid_case():
    case = load_case("cases/smoke/allreduce_2p.yaml")
    assert case["name"] == "allreduce_2p"
    assert case["enabled"] is True
    assert case["scale"] == "small"


def test_case_loader_reports_missing_fields(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\n", encoding="utf-8")
    with pytest.raises(CaseLoadError, match="missing required field"):
        load_case(bad)


def test_hccl_vm_backend_builds_command():
    case = load_case("cases/smoke/allreduce_2p.yaml")
    case_config = json.loads(Path(case["backend_config"]["case_config"]).read_text(encoding="utf-8"))
    config = {
        "hccl_vm_root": "/opt/hccl_vm",
        "install_dir": "/opt/hccl_vm/hccl_vm_install",
        "cann_home": "/opt/Ascend/cann",
        "topology": "topo.yaml",
        "command_template": ["{hccl_vm_bin}", "start", "{topology}"],
        "mock_comm_by_rank": {"2": "112"},
    }
    command = build_command(case, config=config, case_config=case_config)
    assert command == ["/opt/hccl_vm/hccl_vm_install/bin/hccl-vm", "start", "ascend950_cluster_32_server_normal.yaml"]
    script = build_shell_script(config, case, case_config, tmp_path := Path("/tmp"))
    assert "hccl-vm mock-comm 112" in script
    assert "all_reduce_test" in script


def test_hccl_vm_backend_keeps_runtime_command_direct_when_sudo_preflight_is_required():
    case = load_case("cases/smoke/allreduce_2p.yaml")
    case_config = json.loads(Path(case["backend_config"]["case_config"]).read_text(encoding="utf-8"))
    config = {
        "hccl_vm_root": "/opt/hccl_vm",
        "install_dir": "/opt/hccl_vm/hccl_vm_install",
        "cann_home": "/opt/Ascend/cann",
        "topology": "topo.yaml",
        "command_template": ["{hccl_vm_bin}", "start", "{topology}"],
        "mock_comm_by_rank": {"2": "112"},
        "preflight": {"sudo_required": True},
    }
    command = build_command(case, config=config, case_config=case_config)
    assert command == [
        "/opt/hccl_vm/hccl_vm_install/bin/hccl-vm",
        "start",
        "ascend950_cluster_32_server_normal.yaml",
    ]
    script = build_shell_script(config, case, case_config, tmp_path := Path("/tmp"))
    assert "sudo -n rm -rf /opt/hccl_vm/hccl_vm_install/config/network/cluster/ascend950_cluster_32_server_normal" in script
    assert "} | /opt/hccl_vm/hccl_vm_install/bin/hccl-vm start ascend950_cluster_32_server_normal.yaml" in script


def test_validator_passes_with_required_patterns(tmp_path: Path):
    case = load_case("cases/smoke/allreduce_2p.yaml")
    log = tmp_path / "hccl.log"
    log.write_text(
        "2 rank ready, start runner\nPlugin [runner] exited successfully\nShell exited. Host shutting down\ncheck_result:\nsuccess\n",
        encoding="utf-8",
    )
    validation = validate_case(case, Result(hccl_log_path=str(log)))
    assert validation["status"] == "PASS"


def test_validator_fails_when_required_pattern_missing():
    case = load_case("cases/smoke/allreduce_2p.yaml")
    validation = validate_case(case, Result(stdout="no success marker"))
    assert validation["status"] == "FAIL"
    assert any("required log pattern missing" in x for x in validation["fail_reasons"])


def test_validator_fails_when_forbidden_pattern_found():
    case = load_case("cases/smoke/allreduce_2p.yaml")
    validation = validate_case(
        case,
        Result(
            stdout="rank ready, start runner\nPlugin [runner] exited successfully\nShell exited. Host shutting down\ncheck_result:\nsuccess\n[ERROR]"
        ),
    )
    assert validation["status"] == "FAIL"
    assert any("forbidden log pattern found" in x for x in validation["fail_reasons"])


def test_validator_ignores_harmless_warning_and_failed_word():
    case = load_case("cases/smoke/allreduce_2p.yaml")
    text = "\n".join(
        [
            "rank ready, start runner",
            "Plugin [runner] exited successfully",
            "Shell exited. Host shutting down",
            "check_result:",
            "success",
            "target file already exists, will overwrite: topo.json",
            "[error][main.cc][envInit] envInit failed",
            "No Env enable exception dump",
            "aclrtSetExceptionInfoCallback set nullptr success",
            "aclmdlRICaptureThreadExchangeModeImpl: exchange capture mode failed",
            "cleaned 0 shared memory files in /dev/shm, failed 0",
        ]
    )
    validation = validate_case(case, Result(stdout=text))
    assert validation["status"] == "PASS"


def test_validator_error_uppercase_fails():
    case = load_case("cases/smoke/allreduce_2p.yaml")
    validation = validate_case(
        case,
        Result(stdout="rank ready, start runner\nPlugin [runner] exited successfully\nShell exited. Host shutting down\ncheck_result:\nsuccess\n[ERROR] bad"),
    )
    assert validation["status"] == "FAIL"


def test_validator_timeout_fails():
    case = load_case("cases/smoke/allreduce_2p.yaml")
    validation = validate_case(case, Result(timed_out=True, returncode=124))
    assert validation["status"] == "FAIL"
    assert any("timed out" in x for x in validation["fail_reasons"])


def test_run_dir_skips_disabled_case_in_dry_run():
    reports = run_dir("cases/primitives", dry_run=True, scale="small")
    disabled = [r for r in reports if r["case_name"] == "sendrecv_2p_small"]
    assert disabled and disabled[0]["status"] == "SKIP"


def test_run_dir_filters_by_scale_in_dry_run():
    reports = run_dir("cases/primitives", dry_run=True, scale="basic")
    assert reports
    assert {r["scale"] for r in reports} == {"basic"}


def test_report_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    reports = [{"case_name": "c", "op": "AllReduce", "rank_size": 2, "dtype": "fp32", "count": 1, "scale": "small", "status": "PASS", "fail_reasons": [], "stdout_path": "out", "stderr_path": "err"}]
    summary_json, summary_md = write_summary(reports)
    assert summary_json.exists()
    assert summary_md.exists()


def test_cli_accepts_timeout_sec_for_run_and_run_dir():
    parser = build_parser()
    args = parser.parse_args(["run", "cases/smoke/allreduce_2p.yaml", "--dry-run", "--timeout-sec", "600"])
    assert args.timeout_sec == 600
    args = parser.parse_args(["run-dir", "cases/primitives", "--summary", "--timeout-sec", "900"])
    assert args.timeout_sec == 900


def test_summarize_writes_methods_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    case = load_case(Path(__file__).parents[1] / "cases/smoke/allreduce_2p.yaml")
    path = write_methods_report([case])
    assert path.exists()
    assert "AllReduce" in path.read_text(encoding="utf-8")


def test_disabled_case_is_shown_as_disabled_in_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    case = load_case(Path(__file__).parents[1] / "cases/primitives/sendrecv_2p_small.yaml")
    path = write_methods_report([case])
    text = path.read_text(encoding="utf-8")
    assert "DISABLED" in text
    assert "sendrecv_2p_small" in text


def test_preflight_sudo_required_fails_without_credential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    install = tmp_path / "vm" / "hccl_vm_install"
    (install / "bin").mkdir(parents=True)
    (install / "bin" / "hccl-vm").write_text("#!/bin/sh\n", encoding="utf-8")
    cann = tmp_path / "cann"
    (cann / "tools" / "hccl_test" / "bin").mkdir(parents=True)
    (cann / "set_env.sh").write_text("", encoding="utf-8")
    (cann / "tools" / "hccl_test" / "bin" / "all_reduce_test").write_text("", encoding="utf-8")

    class Proc:
        returncode = 1
        stdout = ""
        stderr = "sudo: a password is required"

    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return Proc()

    monkeypatch.setattr(hccl_vm.subprocess, "run", fake_run)
    reasons = preflight(
        {"install_dir": str(install), "hccl_vm_root": str(tmp_path / "vm"), "cann_home": str(cann), "preflight": {"sudo_required": True}},
        {"hccl_test_bin": "all_reduce_test"},
    )
    assert reasons
    assert "sudo" in reasons[-1]
    assert calls == [
        ["sudo", "-n", str(install / "bin" / "hccl-vm"), "--help"],
        ["sudo", "-n", "rm", "--version"],
    ]


def test_dry_run_does_not_execute_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = load_case(Path(__file__).parents[1] / "cases/smoke/allreduce_2p.yaml")
    cfg = json.loads(Path(case["backend_config"]["case_config"]).read_text(encoding="utf-8"))
    cfg_path = tmp_path / "case.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    case["backend_config"]["case_config"] = str(cfg_path)

    install = tmp_path / "vm" / "hccl_vm_install"
    (install / "bin").mkdir(parents=True)
    cann = tmp_path / "cann"
    config_path = tmp_path / "hccl_vm.yaml"
    config_path.write_text(
        f"name: hccl_vm\nhccl_vm_root: {tmp_path / 'vm'}\ninstall_dir: {install}\ncann_home: {cann}\ncommand_template: [\"{{hccl_vm_bin}}\", \"start\", \"{{topology}}\"]\nmock_comm_by_rank: {{\"2\": \"112\"}}\npreflight: {{sudo_required: true}}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(hccl_vm, "preflight", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preflight should not run")))
    result = run_case(case, tmp_path / "out", dry_run=True, config_path=config_path, timeout_sec=600)
    assert result.dry_run is True
    assert result.timeout_sec == 600


def test_run_case_timeout_writes_logs_and_marks_timed_out(tmp_path: Path):
    case = load_case(Path(__file__).parents[1] / "cases/smoke/allreduce_2p.yaml")
    cfg = json.loads(Path(case["backend_config"]["case_config"]).read_text(encoding="utf-8"))
    cfg_path = tmp_path / "case.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    case["backend_config"]["case_config"] = str(cfg_path)

    install = tmp_path / "vm" / "hccl_vm_install"
    bin_dir = install / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / "hccl-vm"
    fake.write_text("#!/bin/sh\necho start\nsleep 5\n", encoding="utf-8")
    fake.chmod(0o755)
    cann = tmp_path / "cann"
    (cann / "tools" / "hccl_test" / "bin").mkdir(parents=True)
    (cann / "set_env.sh").write_text("", encoding="utf-8")
    (cann / "tools" / "hccl_test" / "bin" / "all_reduce_test").write_text("", encoding="utf-8")
    config_path = tmp_path / "hccl_vm.yaml"
    config_path.write_text(
        f"name: hccl_vm\nshell: bash\nhccl_vm_root: {tmp_path / 'vm'}\ninstall_dir: {install}\ncann_home: {cann}\ntimeout_sec: 30\ncommand_template: [\"{{hccl_vm_bin}}\", \"start\", \"{{topology}}\"]\nmock_comm_by_rank: {{\"2\": \"112\"}}\npreflight: {{sudo_required: false}}\ncleanup_shm: []\ntimeout_cleanup_process_patterns: []\n",
        encoding="utf-8",
    )
    result = run_case(case, tmp_path / "out", dry_run=False, config_path=config_path, timeout_sec=1)
    assert result.timed_out is True
    assert result.returncode == 124
    assert Path(result.stdout_path).exists()
    assert Path(result.stderr_path).exists()
    assert "timed out" in Path(result.stderr_path).read_text(encoding="utf-8")
