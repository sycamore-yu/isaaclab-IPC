# Isaac Lab IPC

阶段 1 候选包已能运行。**成功率尚未达到原版水平**：
Newton 中速突现障碍测试仍失败。
历史结果见 [场景验证](docs/validation.md)。
新执行配置与复测见 [执行配置](docs/execution-profiles.md)。

## 安装与运行

需要 Linux、Python 3.12、CUDA GPU、`uv`、G++ 和 Eigen3 头文件。
依赖版本由 `uv.lock` 固定；安装不会修改其他项目的环境。

```bash
uv sync
REF=/home/tong/tongworkspace/reference_repos/IPC/hku-mars
uv run python scripts/build_native.py --upstream "$REF"

env -u PYTHONPATH uv run python scripts/evaluate_native.py \
  --stride 10 --distance 1 --num-envs 10 --output results/my-slow-run
```

输出目录必须不存在，避免覆盖历史结果。
`records.json` 保存每个 episode 的结果。
`trace-*.npz` 保存状态、控制量和间距。
`stage-times-ms.npy` 保存分段耗时。
QP 失败时保存首个失败问题，供独立求解器复核。

默认使用 CPU OSQP 作为数值基线。
加上 `--solver gpu` 使用批量 CUDA graph ADMM。
单环境测试宜保留默认；64 个 QP 的批处理已有加速证据。
A*、EMVP 和 QP 组装目前仍在 CPU。

## 原版森林场景

先从原 PCD 创建感知网格。碰撞判定使用完整 PCD。

```bash
REF=/home/tong/tongworkspace/reference_repos/IPC/hku-mars
MAP="$REF/mars_uav_sim/uav_simulator/map_generator/resource/rc_forest.pcd"
env -u PYTHONPATH uv run python scripts/build_forest.py \
  --pcd "$MAP" --output build/forest
env -u PYTHONPATH uv run python scripts/evaluate_native.py \
  --forest build/forest --stride 6 --num-envs 10 --output results/my-forest-run
```

森林感知使用 MiniCF 扫描方向和 Warp 首命中射线。
射线查询对象为 0.1 m 体素表面。
该方式不同于 MARSIM 的点云平面插值。
森林碰撞在每个物理子步检查，发生碰撞即终止。
树木不提供接触反弹；避障场景的墙和障碍物由 MJWarp 处理接触。

## Isaac Lab 接口

包通过 `isaaclab.tasks` entry point 注册 2 个 `DirectRLEnv`：

- `Isaac-IPC-Avoid-Direct`
- `Isaac-IPC-Forest-Direct`，需要先生成 `build/forest`。

```python
import gymnasium as gym
import isaaclab_ipc.tasks
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-IPC-Avoid-Direct", device="cuda:0", num_envs=16)
env = gym.make("Isaac-IPC-Avoid-Direct", cfg=cfg)
```

动作是 `[specific_thrust, body_rate_x, body_rate_y, body_rate_z]`。
单位分别为 m/s² 和 rad/s，不是归一化电机命令。
观测为 `[position, velocity, acceleration, rotation_matrix]`，共 18 维。
奖励为目标距离减少量，加到达奖励 10，减碰撞惩罚 10。
到达与碰撞返回 `terminated`；35 s 超时返回 `truncated`。
标准环境自动 reset；场景测试关闭自动 reset，以保留终止状态。

AC-MPC actor、可微 QP 和训练留在阶段 2。
当前环境接口不等于已经完成 AC-MPC 训练接入。

## 复现检查

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
env -u PYTHONPATH uv run python scripts/smoke_sim.py
bash scripts/build_oracle.sh
env -u PYTHONPATH uv run python scripts/check_qp.py --output results/my-qp-check
```

原版 Docker 对照使用已有镜像 `ipc-hku-mars:noetic`：

```bash
env -u PYTHONPATH uv run python scripts/run_reference.py \
  --reference /home/tong/tongworkspace/reference_repos/IPC/hku-mars \
  --repeats 10 --output results/my-reference
```

原仓库只读挂载，所有输出留在本项目。
共享 GPU 上宜串行启动测试进程；本轮出现过显存不足。
来源和分发限制见 [来源记录](docs/provenance.md)。

## 选择理想配置或真实配置

`--profile real` 为默认配置。
QP 不可行时，真实配置停止沿用旧 jerk 序列，改为制动并请求 A* 重新规划。
恢复期间仍读取实际速度，待速度低于 0.2 m/s、加速度低于 0.5 m/s²，
且导航 QP 求解成功后恢复导航。
制动目标加速度为 `-2v`，受原加速度界限制。
jerk 受原 jerk 界限制，机体仍通过 Newton 电机力运动。
制动控制不能保证在突现障碍出现后仍有足够停车距离。

`--profile ideal` 复刻原避障测试：MPC 输入速度置零，
执行器直接采用预测位置，姿态按预测加速度计算。
理想执行器不启动 Newton 积分，也不把这种位置设置当成刚体飞行。
修改前的上一序列回退测试和结果保留在历史证据中。

```bash
# 理想配置：中速突现障碍，一次 episode。
env -u PYTHONPATH uv run python scripts/evaluate_native.py \
  --profile ideal --stride 20 --distance 1 --output results/my-ideal-medium

# 默认真实配置：制动并重新规划。
env -u PYTHONPATH uv run python scripts/evaluate_native.py \
  --stride 20 --distance 1 --num-envs 10 --output results/my-real-medium

# 同一原始森林地图，将目标延长至 (32,0,1) m。
env -u PYTHONPATH uv run python scripts/evaluate_native.py \
  --forest build/forest --stride 6 --goal-x 32 --max-steps 9000 \
  --output results/my-forest-32m
```

此前 8 m 森林目标来自本地 `repro/forest_monitor.py` 的测试设置。
上游 `IPC/launch/ipc.launch` 接收外部目标，并不限制只飞 8 m。
地图没有树木实例标签，路线长度不能直接换算为经过的树木数。
