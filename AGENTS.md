# Isaac Lab IPC

## 项目
- 本分支的 AC-MPC 主线采用完整原生 PathCover、28N 逐节点成本与 10D CTBR；见 ADR-0008。
- 复用原生 TurboMPC CUDA/cuDSS、Isaac Lab/Newton 和 MARSIM/Flightmare。历史 IPC 代码保留为回归参考。
- 必须先读 `CONTEXT.md`、`docs/adr/` 和 `docs/execution-profiles.md`。
- 参考源码必须使用 `reference_repos/IPC/hku-mars`，不得误用上层同名接触物理项目。

## 运行与检查
- 技术栈为 Python 3.12、Isaac Lab 3、Newton/MJWarp、Warp、Torch、TurboMPC CUDA/cuDSS。OSQP 仅用于独立数值测试。
- 依赖必须使用本项目 `.venv` 和 `uv.lock`；安装与运行见 `README.md`。
- 原生 PathCover 使用 `scripts/build_pathcover_native.py`；历史 IPC A* 和单走廊扩展使用 `scripts/build_native.py`。
- 单测命令：`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q`。
- 代码检查：`uv run ruff check src scripts tests`。

## 目录与约定
- `src/isaaclab_ipc/` 保存规划、QP、物理资产和标准任务注册。
- `scripts/` 保存构建、场景测试和独立数值检查入口。
- 新 AC-MPC 的配置与结果见 `docs/acmpc-native-validation.md`；历史 IPC 执行约定见 `docs/execution-profiles.md`。
- 训练与评测使用 `scripts/acmpc.py`；MARSIM 正式评测使用 `scripts/run_marsim_ros2.py`；Newton 回归使用 `scripts/evaluate_native.py`。
- `docs/validation.md` 只保存首次候选的历史结果。
- `results/` 与 `recordings/` 为本地证据，不进入 Git；必须保留失败记录。
- 参考仓库必须只读；不得修改其他项目的环境或停止其训练。
- 删除必须先提交清单，用户确认后才能执行。

## 历史 IPC 执行约定
- IPC 执行语义统一从 `execution.py` 的 `ExecutionProfile` 获取，不得在数值参数字典中添加恢复策略开关。
- `real` 默认使用真实速度、Newton 电机力和制动重规划；`ideal` 保留速度置零与位置直接赋值。
- 必须分开报告两种配置，不能把理想轨迹计入 Newton 成功率。
- 必须保留 100 Hz 控制周期和 0.1 s MPC 预测步长的区别。
- 历史中速真实配置为 0/10；四项奖励复测的固定成本为 1/10、学习候选为 2/10，仍未解决中高速近障碍。完整 120 episode 配对矩阵见 `docs/reward.md`，不得表述为阶段 1 验收通过。
- IPC 默认 SFC 为 PathCover/RISP，保留 `[A,-b]` 接口和配置的安全半径。

## 原生 AC-MPC 约定
- 使用 10D CTBR、4D 总推力/角速度控制；N 计入当前成本节点，物理转移数为 N-1。
- PPO 读取独立保存的状态、参考、SFC 和热启动控制；更新时不执行前端。
- 使用完整原生 PathCover；SFC 保留 `[A,-b]` 接口，安全距离只处理一次。
- Newton 策略为 10 Hz、电机反馈与物理为 300 Hz；MARSIM 执行为 100 Hz、MPC 为 10 Hz。
- 按原生轨迹段时间分配 SFC；不接入历史 jerk 恢复或重复执行轨迹积分。
- 分别报告仿真时间频率、墙钟规划延迟、训练结果和正式 MARSIM 结果。

## 共享物理与感知

- 真实配置默认 Mid-360 随机方向窗口：20,000 rays、10 Hz、40 m、机体原点轴对齐。
- 官方 MultiMesh 管理环境网格与位姿；ADR-0006 允许为共享显存融合射线查询，不得绕过环境隔离或修改安装包。
- 位姿/速度使用 ArticulationData，加速度使用 PVA，事件与录制使用官方 Manager。
- ContactSensor 统计物理接触，完整 PCD 仍独立定义森林几何碰撞；迁移结果见 `docs/official-components.md`。
