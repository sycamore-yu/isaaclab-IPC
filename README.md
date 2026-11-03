# Isaac Lab IPC

Newton/MJWarp IPC 和 CUDA 可微 MPC 的 PPO 训练已能运行。
**成功率尚未达到原版水平**：中速突现障碍测试仍失败。
历史结果见 [场景验证](docs/validation.md)。
新执行配置与复测见 [执行配置](docs/execution-profiles.md)。
共享森林 BVH、紧凑 Mid-360 与规划缓存已默认启用；实测与边界见 [规模优化](docs/scale-optimization.md)。

## 安装与运行

需要 Linux、Python 3.12、CUDA GPU、`uv`、G++、Eigen3 和 Qhull 头文件（`libqhull-dev`）。
依赖版本由 `uv.lock` 固定；安装不会修改其他项目的环境。

```bash
uv sync
uv run python scripts/prepare_mid360.py
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
该基线入口的 A*、PathCover/RISP 和 QP 组装仍在 CPU。
下述 AC-MPC 入口将 QP 组装和求解放在 GPU。

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

真实配置默认使用 Mid-360 随机方向窗口与官方 `MultiMeshRayCasterCfg`。
Newton 使用项目内的紧凑实现，保留官方目标与位姿管理，共享相同森林几何。
射线查询对象为 0.1 m 体素表面。
该方式不同于 MARSIM 的点云平面插值。
森林碰撞在每个物理子步检查，发生碰撞即终止。
树木不提供接触反弹；避障场景的墙和障碍物由 MJWarp 处理接触。

## 输入是点云和机体状态

森林和突现障碍真实配置均输入模拟激光雷达点云。
扫描模式为 **Mid-360 随机方向窗口**，每帧 20,000 条射线，10 Hz，测距 40 m。
传感器位于机体原点、轴对齐；官方 MultiMesh 管理多环境与移动障碍网格。
森林查询原 PCD 的 0.1 m 体素表面，突现障碍查询当前墙体和障碍物网格。
理想配置继续使用原来的合成点云，不启动传感器或 Newton。
当前没有深度相机、深度图或图像编码器。

规划器还读取当前位置、速度、加速度、姿态和目标点。
位置、速度和姿态来自仿真真值。
位姿、速度读取官方 `ArticulationData`；世界系净加速度来自官方 `PvaCfg`，包含重力与接触响应。
传感器、事件与记录迁移的行为边界和实测结果见 [官方组件迁移](docs/official-components.md)。
当前尚未接入 SLAM、VIO 或真实状态估计器。

```text
点云 → 最近 10 帧聚合 / 0.2 m 体素 → A* → PathCover/RISP SFC
状态 + 目标 + 参考轨迹 + SFC → MPC → 推力 / 机体角速度 → Newton
```

AC-MPC 成本网络读取由状态、目标、参考轨迹、SFC 和恢复标记组成的特征。
网络不直接读取原始点云或相机图像。

## Isaac Lab 接口

包通过 `isaaclab.tasks` entry point 注册基础物理任务：

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

## CUDA AC-MPC 训练

学习任务保留 IPC 的 9 维状态、三轴 jerk 和线性模型。
成本网络通过 CUDA 可微硬约束 QP 接入标准 RSL-RL PPO。

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  uv run python -u scripts/acmpc.py train \
  --num-envs 4 --iterations 32 --output results/my-acmpc
```

任务为 `Isaac-IPC-Avoid-ACMPC-Direct` 和 `Isaac-IPC-Forest-ACMPC-Direct`。
训练、恢复、播放、约束语义和验证结果见 [AC-MPC 使用说明](docs/acmpc.md)。
MPC 学习任务使用进度、时间、平滑、终端四项奖励；权重和新一轮调试见 [奖励与重训](docs/reward.md)。

默认走廊为 PathCover/RISP，保留原安全距离和 IPC 走廊时序。
设计、结果和剩余测试见 [走廊替换说明](docs/pathcover.md)。

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

`--profile real` 为默认配置，使用实际速度、Newton 电机力和制动重规划。
`--profile ideal` 使用速度置零与位置直接赋值，不启动 Newton。
恢复阈值、执行差异和测试结果统一见 [执行配置](docs/execution-profiles.md)。

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

## 本地 Git 基线与数据

原始候选快照标签为 `ipc-reproduction-baseline-20261102`。
收尾后的基线标签为 `ipc-reproduction-baseline-clean-20261102`。
基线名称表示可恢复的源码版本，不表示阶段 1 已验收。

Git 保存源码、配置、锁文件、测试和文档。
`results/`、`recordings/`、`build/` 和 `.venv/` 保留在本地，不进入 Git。
复制 Git 仓库不会复制这些数据；需要完整交接时必须同时备份结果和录制。
录制目录与对应配置见 [执行配置](docs/execution-profiles.md#rerun-录制)。

SUPER High-speed / Dense 场景、参数对应与录制命令见 [SUPER 场景](docs/super-benchmarks.md)。


## SUPER Dense 增量前端

SUPER 场景默认使用 GPU 增量占据栅格，保留原有 IPC A*、PathCover/RISP、Newton 与 CUDA MPC。
实现、耗时、验证边界见 [增量前端说明](docs/gpu-incremental-front.md)。
