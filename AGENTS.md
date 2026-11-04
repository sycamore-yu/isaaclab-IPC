# Isaac Lab IPC

## 项目
- 正式算法仅有固定成本 IPC 与学习成本 AC-MPC；共享 IPC 前端、PathCover/RISP 和原生 TurboMPC CUDA jerk 求解器。
- CTBR 是 AC-MPC 的可选动力学模型；SUPER 仅作外部对比，不属于项目算法路径。
- 必须先读 `CONTEXT.md`、`docs/adr/` 和 `docs/execution-profiles.md`。
- 参考源码必须使用 `reference_repos/IPC/hku-mars`，不得误用上层同名接触物理项目。

## 运行与检查
- 技术栈为 Python 3.12、Isaac Lab 3、Newton/MJWarp、Warp、Torch、TurboMPC CUDA/cuDSS。OSQP 仅用于独立数值测试。
- 依赖必须使用本项目 `.venv` 和 `uv.lock`；安装与运行见 `README.md`。
- 几何扩展必须使用 `scripts/build_native.py` 构建固定 IPC A* 和 vendored PathCover/RISP。
- 单测命令：`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q`。
- 代码检查：`uv run ruff check src scripts tests`。

## 目录与约定
- `src/isaaclab_ipc/` 保存规划、QP、物理资产和标准任务注册。
- `scripts/` 保存构建、场景测试和独立数值检查入口。
- `docs/execution-profiles.md` 是当前配置与结果的权威说明。
- 训练与评测使用 `scripts/acmpc.py`；MARSIM 正式评测使用 `scripts/run_marsim_ros2.py`；Newton 回归使用 `scripts/evaluate_native.py`。
- `docs/validation.md` 只保存首次候选的历史结果。
- `results/` 与 `recordings/` 为本地证据，不进入 Git；必须保留失败记录。
- 参考仓库必须只读；不得修改其他项目的环境或停止其训练。
- 删除必须先提交清单，用户确认后才能执行。

## 当前状态与下一步
- 执行语义必须统一从 `execution.py` 的 `ExecutionProfile` 获取，不得在数值参数字典中添加恢复策略开关。
- `real` 默认使用真实速度、Newton 电机力和制动重规划；`ideal` 保留速度置零与位置直接赋值。
- 必须分开报告两种配置，不能把理想轨迹计入 Newton 成功率。
- 必须保留 100 Hz 控制周期和 0.1 s MPC 预测步长的区别。
- 历史中速真实配置为 0/10；四项奖励复测的固定成本为 1/10、学习候选为 2/10，仍未解决中高速近障碍。完整 120 episode 配对矩阵见 `docs/reward.md`，不得表述为阶段 1 验收通过。
- AC-MPC 实现、训练命令和验证结果见 `docs/acmpc.md`；使用 IPC 9 维状态、三轴 jerk 和线性模型。
- 学习任务保留独立的状态、SFC 和恢复快照；PPO 更新不得修改在线 Planner。
- 后续工作为中速失败排查、CPU 几何前端加速。

- 默认 SFC 为 PathCover/RISP；必须保留 `[A,-b]` 接口和配置的安全半径，不得重新引入旧走廊运行后端。

- 真实配置默认 Mid-360 随机方向窗口：20,000 rays、10 Hz、40 m、机体原点轴对齐。
- 官方 MultiMesh 管理环境网格与位姿；ADR-0006 允许为共享显存融合射线查询，不得绕过环境隔离或修改安装包。
- 位姿/速度使用 ArticulationData，加速度使用 PVA，事件与录制使用官方 Manager。
- ContactSensor 统计物理接触，完整 PCD 仍独立定义森林几何碰撞；迁移结果见 `docs/official-components.md`。
