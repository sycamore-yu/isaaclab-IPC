# Isaac Lab IPC

## 项目
- 本项目复刻 hku-mars 的四旋翼 Integrated Planning and Control。
- 必须先读 `CONTEXT.md`、`docs/adr/` 和 `docs/execution-profiles.md`。
- 参考源码必须使用 `reference_repos/IPC/hku-mars`，不得误用上层同名接触物理项目。

## 运行与检查
- 技术栈为 Python 3.12、Isaac Lab 3、Newton/MJWarp、Warp、Torch 和 OSQP。
- 依赖必须使用本项目 `.venv` 和 `uv.lock`；安装与运行见 `README.md`。
- 几何扩展必须使用 `scripts/build_native.py` 从固定 revision 构建。
- 单测命令：`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q`。
- 代码检查：`uv run ruff check src scripts tests`。

## 目录与约定
- `src/isaaclab_ipc/` 保存规划、QP、物理资产和标准任务注册。
- `scripts/` 保存构建、场景测试和独立数值检查入口。
- `docs/execution-profiles.md` 是当前配置与结果的权威说明。
- `docs/validation.md` 只保存首次候选的历史结果。
- `results/` 与 `recordings/` 为本地证据，不进入 Git；必须保留失败记录。
- 参考仓库必须只读；不得修改其他项目的环境或停止其训练。
- 删除必须先提交清单，用户确认后才能执行。

## 当前状态与下一步
- 执行语义必须统一从 `execution.py` 的 `ExecutionProfile` 获取，不得在数值参数字典中添加恢复策略开关。
- `real` 默认使用真实速度、Newton 电机力和制动重规划；`ideal` 保留速度置零与位置直接赋值。
- 必须分开报告两种配置，不能把理想轨迹计入 Newton 成功率。
- 必须保留 100 Hz 控制周期和 0.1 s MPC 预测步长的区别。
- 中速真实配置仍为 0/10；不得将本地基线提交表述为阶段 1 验收通过。
- 后续工作为中速失败排查、GPU 前端和传感等价；AC-MPC 留在阶段 2。
