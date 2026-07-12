# HCCL Development Validation Harness

这是一个用于 HCCL 代码开发验证的轻量级 harness 工程。它不实现 HCCL，也不实现 HCCL-VM；它负责读取 case、调用本机 HCCL-VM 仿真工具、执行 `hccl_test` 二进制、采集日志、判断 PASS/FAIL/SKIP，并生成统一报告。

当前工程默认使用：

- HCCL-VM: `/home/czx/code/hccl_vm`
- HCCL-VM install: `/home/czx/code/hccl_vm/hccl_vm_install`
- CANN: `/home/czx/cann/cann0625/cann`
- 后端配置文件: `configs/hccl_vm.yaml`

## 1. 快速上手

进入工程目录：

```bash
cd /home/czx/code/hccl_dev_validation
```

安装 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

先跑 harness 自身单元测试：

```bash
python -m pytest
```

预期结果：

```text
19 passed
```

检查 HCCL-VM/CANN 路径配置：

```bash
sed -n '1,120p' configs/hccl_vm.yaml
```

如果 HCCL-VM 或 CANN 路径不同，只改 `configs/hccl_vm.yaml`，不要改 Python 源码。

## 2. 配置 sudo

真实运行 HCCL-VM 前，harness 会做 sudo 预检，并在启动前清理 HCCL-VM 生成目录和 `/dev/shm` 残留。推荐给当前用户配置最小免密 sudo。

执行：

```bash
sudo visudo -f /etc/sudoers.d/hccl-dev-validation
```

写入一行：

```sudoers
czx ALL=(root) NOPASSWD: /home/czx/code/hccl_vm/hccl_vm_install/bin/hccl-vm, /usr/bin/rm
```

保存退出后检查语法：

```bash
sudo visudo -cf /etc/sudoers.d/hccl-dev-validation
```

预期结果：

```text
/etc/sudoers.d/hccl-dev-validation: parsed OK
```

再检查免密 sudo：

```bash
sudo -n /home/czx/code/hccl_vm/hccl_vm_install/bin/hccl-vm --help >/dev/null 2>&1 && echo HCCL_VM_SUDO_OK
sudo -n rm --version >/dev/null 2>&1 && echo RM_SUDO_OK
```

预期结果：

```text
HCCL_VM_SUDO_OK
RM_SUDO_OK
```

注意：`czx ALL=(root) ...` 这一行不是 shell 命令，不能直接粘到终端执行；它只能写入 sudoers 文件。

## 3. 先做 dry-run

dry-run 只生成命令和脚本，不启动 HCCL-VM，适合检查 case 配置是否正确。

单个 smoke case：

```bash
python -m harness run cases/smoke/allreduce_2p.yaml --dry-run --timeout-sec 600
```

批量 dry-run：

```bash
python -m harness run-dir cases/primitives/ --dry-run --summary --timeout-sec 600
```

每个 case 会生成一个输出目录：

```text
outputs/runs/<timestamp>_<case_name>/
```

重点看：

```bash
cat outputs/runs/<timestamp>_<case_name>/command.txt
```

## 4. 跑真实 smoke

先跑一个最小真实用例：

```bash
python -m harness run cases/smoke/allreduce_2p.yaml --timeout-sec 600
```

预期结果：

```text
PASS: allreduce_2p [AllReduce 2p small]
```

如果失败，先看输出中的 `report:` 路径，然后查看：

```bash
cat outputs/runs/<run_dir>/report.json
cat outputs/runs/<run_dir>/stdout.log
cat outputs/runs/<run_dir>/stderr.log
cat outputs/runs/<run_dir>/hccl_test.log
```

## 5. HCCL Change Validation Workflow

For daily development, use this loop: change HCCL code, build a package, install it into the CANN tree used by this harness, then run the harness.

Set paths first:

```bash
export HCCL_SRC=/home/czx/code/hccl_master/hccl
export CANN_HOME=/home/czx/cann/cann0625/cann
```

Build the HCCL package:

```bash
cd ${HCCL_SRC}
bash build.sh --pkg --full
```

Install the generated `.run` package. Use the actual file under `build_out/`:

```bash
cd ${HCCL_SRC}
./build_out/cann-hccl_*.run --full --install-path=${CANN_HOME%/cann} --quiet
```

Confirm the harness points to the same CANN tree:

```bash
grep -n "cann_home" /home/czx/code/hccl_dev_validation/configs/hccl_vm.yaml
```

Then run smoke first, followed by larger scales:

```bash
cd /home/czx/code/hccl_dev_validation
python -m harness run cases/smoke/allreduce_2p.yaml --timeout-sec 600
python -m harness run-dir cases/primitives/ --scale small --summary --timeout-sec 600
python -m harness run-dir cases/primitives/ --scale basic --summary --timeout-sec 600
python -m harness run-dir cases/primitives/ --scale medium --summary --timeout-sec 900
```

If the change touches shared communication code, finish with the full suite:

```bash
python -m harness run-dir cases/primitives/ --summary --timeout-sec 900
python -m harness summarize
```

A local helper script also exists at `/home/czx/script/A5_hccl_st_test/build_hccl_test.sh`. Check the HCCL source path and install path inside that script before using it.

## 6. 按规模跑 primitives

建议按 small、basic、medium 逐步扩大。

small：

```bash
python -m harness run-dir cases/primitives/ --scale small --summary --timeout-sec 600
```

basic：

```bash
python -m harness run-dir cases/primitives/ --scale basic --summary --timeout-sec 600
```

medium：

```bash
python -m harness run-dir cases/primitives/ --scale medium --summary --timeout-sec 900
```

全量：

```bash
python -m harness run-dir cases/primitives/ --summary --timeout-sec 900
python -m harness summarize
```

当前已验证的全量结果是：

```text
PASS: 17
SKIP: 1
FAIL: 0
```

3 个 SKIP 是当前未默认启用项：

- `sendrecv_2p_small`

## 7. 查看报告

汇总报告：

```bash
cat outputs/reports/summary.md
```

通信原语测试方法汇总：

```bash
cat outputs/reports/communication_primitives_test_methods.md
```

最终真实执行报告：

```bash
cat outputs/reports/final_real_run_report.md
```

单 case 报告目录结构：

```text
outputs/runs/<timestamp>_<case_name>/
  command.txt      # harness 生成的 HCCL-VM 执行脚本
  stdout.log       # HCCL-VM stdout
  stderr.log       # HCCL-VM stderr
  hccl_test.log    # hccl_test 二进制日志
  report.json      # 结构化执行结果
```

`outputs/` 已在 `.gitignore` 中忽略，不要提交大日志。

## 8. 常用命令

查看 CLI 帮助：

```bash
python -m harness --help
python -m harness run --help
python -m harness run-dir --help
```

跑单个 case：

```bash
python -m harness run cases/smoke/allreduce_2p.yaml --timeout-sec 600
```

跑目录：

```bash
python -m harness run-dir cases/primitives/ --summary --timeout-sec 900
```

只跑某个规模：

```bash
python -m harness run-dir cases/primitives/ --scale small --summary --timeout-sec 600
python -m harness run-dir cases/primitives/ --scale basic --summary --timeout-sec 600
python -m harness run-dir cases/primitives/ --scale medium --summary --timeout-sec 900
```

重新生成方法汇总：

```bash
python -m harness summarize
```

## 9. 当前覆盖范围

已默认启用并通过真实验证：

| Op | Scale | Rank |
| --- | --- | --- |
| AllReduce | small/basic/medium | 2/4/8 |
| AllGather | small/basic/medium | 2/4/8 |
| Broadcast | small/basic | 2/4 |
| Reduce | small/basic | 2/4 |
| ReduceScatter | small/basic/medium | 2/4/8 |
| AllToAll | basic/medium | 4/8 |
| AllToAllV | basic/medium | 4/8 |

暂未默认启用：

| Op | Case | 原因 |
| --- | --- | --- |
| SendRecv | `sendrecv_2p_small` | 当前 `hccl_test/bin` 下没有明确 send/recv 或 sendrecv 测试二进制 |

## 10. AllToAllV 参数格式

`alltoallv_test` 的命令行参数和其它 `hccl_test` 算子一致，不需要在命令行传 `sendCounts/sdispls/recvCounts/rdispls`。这些数组由测试程序内部构造。

4p 示例：

```bash
mpirun --allow-run-as-root --oversubscribe -np 4 \
  /home/czx/cann/cann0625/cann/tools/hccl_test/bin/alltoallv_test \
  -b 1048576 -e 1048576 -d fp32 -w 0 -n 1 -c 1
```

8p 示例：

```bash
mpirun --allow-run-as-root --oversubscribe -np 8 \
  /home/czx/cann/cann0625/cann/tools/hccl_test/bin/alltoallv_test \
  -b 4194304 -e 4194304 -d fp16 -w 0 -n 1 -c 1
```

HCCL API 层的 AllToAllV 需要 `sendCounts`、`sdispls`、`recvCounts`、`rdispls`，但当前 `alltoallv_test` CLI 不暴露这些参数。

## 11. Case 文件结构

每个用例由两部分组成。

YAML case：

```text
cases/primitives/allreduce_4p_basic.yaml
```

负责描述：

- case 名称
- 通信原语
- rank 数
- dtype
- count
- scale
- 是否启用
- 校验规则

后端 JSON：

```text
case_configs/primitives/allreduce_4p_basic_vm.json
```

负责描述：

- HCCL-VM topology
- mock-comm 编号
- `mpirun -np`
- `hccl_test` 二进制名
- `hccl_test` 参数
- 结果日志名

新增 case 时，先复制一个相近的 YAML 和 JSON，再改 op、rank、dtype、count、args、mock-comm 和校验规则。

## 12. PASS/FAIL/SKIP 判断

PASS 需要同时满足：

- 进程返回码符合预期，通常是 `0`
- 没有 timeout
- required log pattern 都存在
- forbidden log pattern 都不存在

常见 required pattern：

```text
rank ready, start runner
Plugin [runner] exited successfully
Shell exited. Host shutting down
check_result:
success
```

常见 forbidden pattern：

```text
[ERROR]
execute failed
RESULT: FAILED
Segmentation fault
core dumped
comm is nullptr
```

注意：普通 `failed` 不作为全局失败关键字，因为真实 HCCL-VM 日志里存在 harmless warning，例如 `exchange capture mode failed`。

## 13. 常见问题

### sudo 预检失败

现象：

```text
sudo preflight failed
```

处理：

```bash
sudo -n /home/czx/code/hccl_vm/hccl_vm_install/bin/hccl-vm --help
sudo -n rm --version
```

如果失败，重新检查 `/etc/sudoers.d/hccl-dev-validation`。

### HCCL-VM 拓扑生成失败

现象：

```text
generate_cluster_topo.sh failed
Permission denied
```

原因通常是之前用 root 生成过 HCCL-VM 拓扑文件，普通用户删不掉。当前 harness 已在启动前自动执行 sudo 清理；如果仍然失败，手动清理：

```bash
sudo rm -rf /home/czx/code/hccl_vm/hccl_vm_install/config/network/cluster/ascend950_cluster_32_server_normal
```

### 缺少 libhccl.so

现象：

```text
error while loading shared libraries: libhccl.so
```

通常是没有通过 HCCL-VM runner 环境执行，或 `LD_LIBRARY_PATH` 没带上 CANN 路径。优先使用 harness 命令，不要直接裸跑 `hccl_test`。

### case FAIL 后如何定位

按顺序看：

```bash
cat outputs/runs/<run_dir>/report.json
cat outputs/runs/<run_dir>/stderr.log
cat outputs/runs/<run_dir>/stdout.log
cat outputs/runs/<run_dir>/hccl_test.log
```

先判断是：

- harness/preflight 问题
- HCCL-VM 启动问题
- `hccl_test` 参数问题
- HCCL 算子真实失败
- validator 误判

## 14. 工程边界

当前 harness 不做这些事：

- 不实现 HCCL
- 不实现 HCCL-VM
- 不管理多个 CANN 包
- 不提供 Web UI
- 不区分 AIV/AICPU 多后端策略
- 不自动提交日志或大文件

当前目标是让 HCCL 开发者能用一组稳定命令快速完成 smoke、small、basic、medium 规模的功能验证。
