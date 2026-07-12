# AGENTS.md

本项目是 HCCL 代码开发验证 harness。它读取 case YAML，通过当前唯一 backend `hccl_vm` 调用外部 HCCL-VM 工具，收集日志并生成报告。

协作规则：

- 当前 backend 只有 `hccl_vm`。
- 当前不引入 AIV/AICPU 分支逻辑。
- 当前不管理多个 CANN 包。
- 不实现 HCCL。
- 不实现 HCCL-VM。
- 不引入 Web UI。
- 不做过度抽象，优先保持代码直接、可读、可维护。
- 所有运行输出必须写入 `outputs/`。
- Python 源码不得硬编码本机 HCCL-VM/CANN 绝对路径；这些路径只能来自 `configs/hccl_vm.yaml`。
- 如果新增 case，必须同步新增对应 `case_configs/..._vm.json`，并写清楚 `notes.method`、`notes.expected_behavior`、`notes.check_method`。
- enabled case 必须能真实运行，或在真实运行失败时明确记录 FAIL 和日志路径。
- disabled case 必须写清楚 `notes.reason`。
- 不要把 dry-run 结果当成真实 PASS。
- 非交互环境下必须避免卡在 sudo 密码输入；真实执行前使用配置控制的 preflight。
- 不要提交 `outputs/` 下的大日志。
