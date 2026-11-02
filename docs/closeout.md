# 本地基线收尾与清理预览

日期：2026-11-02。用户明确确认清单后，已完成限定范围的清理。

## 基线与验证

- 原始候选提交：`c22f8a9`，标签 `ipc-reproduction-baseline-20261102`。
- 收尾后标签：`ipc-reproduction-baseline-clean-20261102`。
- 基线表示源码检查点，不代表中速场景或阶段 1 已验收。
- 基线收尾仅简化 QP 残差计算，CPU/CUDA 各 8 个问题与候选逐项一致。
- 本次集中管理执行配置与恢复策略，没有修改 QP 数学问题和制动公式。
- 两种配置的固定输入 QP、控制输出与恢复状态均与 `66432fa` 一致。
- 本次单元测试 20 项通过；lint 通过。
- 已检查理想和真实两条中速 CLI：前者到达，后者制动后碰撞；不改写历史导航统计。
- 冻结的场景数值见 `baseline-evidence.json`，原始数据仍在本地 `results/`。

## 事实面状态

| 事实面 | 状态 | 依据或范围 |
| --- | --- | --- |
| 代码 | changed-and-verified | 执行配置统一；固定输入等价、20 项单测、两条 CLI 和 lint |
| 部署运行态 | not-applicable | 没有部署服务；本次未发现本项目测试进程 |
| 文档 | changed-and-verified | 更新配置、恢复策略、8 个录制的来源和数据保留方式 |
| 规则 | changed-and-verified | 新增项目 AGENTS.md，引用已接受 ADR |
| 跨任务记忆 | out-of-scope | 生成记忆只读，本次不写入 |
| 工作区删除 | changed-and-verified | 用户确认后，准确删除下列 8 项；证据和运行依赖保留 |

## 机械盘点

跳过 `.venv/` 和 `.git/` 的内容，盘点源码、文档、生成物与证据。
共枚举 1925 个文件或文件链接。文档列表：

- `AGENTS.md`
- `CONTEXT.md`
- `README.md`
- `docs/adr/0001-native-ipc-boundary.md`
- `docs/adr/0002-ac-mpc-role.md`
- `docs/adr/0003-execution-profiles.md`
- `docs/closeout.md`
- `docs/design.md`
- `docs/execution-profiles.md`
- `docs/provenance.md`
- `docs/validation.md`

<a id="待确认的删除清单"></a>

## 已完成的删除清单

大小为文件内容逻辑大小，不保证等于释放的磁盘空间。

| 路径 | 逻辑大小 | 删除理由 |
| --- | ---: | --- |
| `.pytest_cache` | 1.2 KiB | Python 或工具缓存，可重建 |
| `.ruff_cache` | 3.7 KiB | Python 或工具缓存，可重建 |
| `scripts/__pycache__` | 10.1 KiB | Python 或工具缓存，可重建 |
| `tests/__pycache__` | 38.6 KiB | Python 或工具缓存，可重建 |
| `src/isaaclab_ipc/__pycache__` | 55.9 KiB | Python 或工具缓存，可重建 |
| `src/isaaclab_ipc/assets/__pycache__` | 35.2 KiB | Python 或工具缓存，可重建 |
| `src/isaaclab_ipc/tasks/__pycache__` | 21.6 KiB | Python 或工具缓存，可重建 |
| `build/forest-voxels.npy` | 13017.2 KiB | 一次性体素探索中间文件；当前构建和运行入口均不读取 |

用户已明确确认本清单；已删除 8 项，文件逻辑大小合计约 12.87 MiB。
本地执行记录为 `results/cleanup-execution-contract.json`。

## 必须保留

- `results/`：原版对照、所有失败试次、首个失败 QP、诊断与历史日志。
- `recordings/`：8 个用户要求的 RRD。
- `build/forest/`：森林任务直接使用的地图和射线网格。
- `build/native/`、已编译 `.so`：本地构建审计记录和运行依赖。
- `.venv/`：当前可用运行环境。
- `results/source-snapshot-20261102.zip`：Git 建仓前的历史快照。
- `results/recording-producers/`：已存档的实际录制工具；当前没有稳定录制 CLI。

## 遗留与范围外

- `pending`：中速真实配置为 0/10，GPU 前端和传感精确等价未完成。
- `changed-and-verified`：架构候选 1 已完成；`ExecutionProfile` 统一执行语义和恢复策略选择。
- `out-of-scope`：未实施其他架构候选，不抽取 Planner 状态机或合并两个执行器。
- `out-of-scope`：远端发布、正式 wheel 分发、AC-MPC 训练和外部参考仓库改动。
- 历史 ROS 日志内的 `latest` 链接指向容器 `/work`；保留原始链接，不把该链接当主机上的有效入口。
- 架构 HTML 已发送给系统默认程序打开，并已通过 Chrome 截图审阅。
  图表的 CDN 依赖不属于源码运行依赖。

本次只删除已确认的 8 项。测试结果、失败记录和录制文件仍全部保留。
