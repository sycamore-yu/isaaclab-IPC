# 用学习成本驱动 IPC

默认走廊已改为 PathCover/RISP，见 [走廊说明](pathcover.md)。
本页已完成的训练、场景结果及 RRD 均来自切换前的走廊版本。

AC-MPC actor 接收 IPC 状态、参考轨迹和安全飞行走廊，输出推力与机体角速度。
PPO 通过 CUDA 可微 QP 更新成本网络。预测模型沿用 IPC：9 维位置、速度、加速度，3 维 jerk，15 步，每步 0.1 s。
Newton/MJWarp 继续以 300 Hz 积分，控制频率保持 100 Hz。

## 训练、恢复和评估

避障训练默认交替使用低速和中速参考采样间隔。
每个环境保持自己的规划器、地图历史和恢复状态。
所有输出目录必须不存在。

```bash
# 4 个环境，32 次 PPO 更新，每次每环境采样 64 步。
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  uv run python -u scripts/acmpc.py train \
  --num-envs 4 --iterations 32 --output results/my-acmpc

# 恢复网络、优化器和随机数状态；环境从新 episode 开始。
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  uv run python -u scripts/acmpc.py train \
  --checkpoint results/my-acmpc/checkpoint.pt \
  --num-envs 4 --iterations 16 --output results/my-acmpc-resume

# 学习成本：确定性动作，同一中速突现障碍场景。
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  uv run python -u scripts/acmpc.py evaluate \
  --checkpoint results/my-acmpc/checkpoint.pt --num-envs 1 \
  --output results/my-acmpc-medium 'env.reference_strides=[20]'

# 固定 IPC 成本：同一 CUDA 求解器、恢复策略和物理。
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  uv run python -u scripts/acmpc.py evaluate --fixed-cost --num-envs 1 \
  --output results/my-ipc-medium 'env.reference_strides=[20]'

# 森林 32 m。地图生成方法见 README。
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  uv run python -u scripts/acmpc.py evaluate \
  --task Isaac-IPC-Forest-ACMPC-Direct \
  --checkpoint results/my-acmpc/checkpoint.pt --num-envs 1 \
  --output results/my-acmpc-forest-32m env.goal_x=32 env.episode_length_s=90
```

入口使用 Isaac Lab 的 `resolve_task_config`，接受 `env.*` 和 `agent.*` Hydra 参数。
注册任务为 `Isaac-IPC-Avoid-ACMPC-Direct`、`Isaac-IPC-Forest-ACMPC-Direct`，均提供 `rsl_rl_cfg_entry_point`。
运行器和 PPO 损失来自 RSL-RL；`MPCPPO.act` 只将本次求解状态交给环境。
直接调用 actor 时，必须在 `env.step(action)` 前调用 `env.record_mpc_result(actor.last_result)`。
PPO 更新只重算保存的样本，不修改在线规划器。

`checkpoint.pt` 保存完整 actor、critic、优化器和恢复信息。
播放也加载完整 MPC actor。JIT/ONNX 导出会明确报错，避免导出成只有成本网络的策略。
`records.json` 保存完成的 episode，`metrics.json` 保存参数更新或评估汇总，TensorBoard 保存训练曲线。
评估的 `timing-ms.npy` 两列为 actor 时间和整步时间，单位 ms。

## 录制带 SFC 的森林路线

在森林评估命令后添加 `--record`，即可保存实际飞行和 SFC。
录制要求 1 个森林环境，文件路径必须不存在。
Rerun 是可选依赖，训练无需安装。

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  uv run --extra recording python -u scripts/acmpc.py evaluate \
  --task Isaac-IPC-Forest-ACMPC-Direct --num-envs 1 \
  --checkpoint results/acmpc-development/resume/checkpoint.pt \
  --output results/my-forest-sfc --record recordings/my-forest-sfc.rrd \
  env.goal_x=32 env.episode_length_s=90
```

蓝线显示实际飞行，白线显示参考轨迹，紫线显示已采用的 QP 预测轨迹。
青色和橙色线框直接由该步 QP 使用的 SFC 半空间相交得到。
每个线框标注适用的预测步，每步为 0.1 s。
制动时改为显示红色单步预测点；SFC 无效时清除旧线框。
状态面板显示求解、制动和前端失败次数。
位姿以 100 Hz 录制；走廊以 10 Hz 和每次控制模式变化时录制。
录制开销计入完整 step 耗时。

本轮 32 m 文件为：

- 固定成本：`recordings/forest_32m_fixed_cost_sfc.rrd`。
- 学习成本：`recordings/forest_32m_learned_cost_sfc.rrd`。

两次录制分别在 17.83 s、15.50 s 到达目标。
RRD 已通过解码检查，并已检查跟随机位下的实际 SFC 显示。
评估记录保存在 `results/acmpc-forest-sfc/{fixed,learned}/`。

## 7 个学习输出保持基线可恢复

成本网络为两层 64 单元的 MLP，输出 7 个值：

- 前 4 个值分别缩放位置、速度、加速度和 jerk 成本，倍率为 `exp(log(2) * tanh(z))`，范围为 0.5–2。
- 后 3 个值添加三轴位置成本的线性项。
  系数为 `b = 0.2 * tanh(z)`，按基线位置权重构造，单位为 m。

末层初始化为零，因此初始 `H、g` 与固定 IPC 成本相同。
位置权重倍率为 `w_p` 时，上述线性项的等效参考偏移为 `b / w_p`。
SFC、速度、加速度和 jerk 限制保持固定。
参考轨迹和走廊时序由原 IPC 前端生成；成本网络不能放宽约束。
固定状态下，改变成本不会恢复空的可行域。

## CUDA 前向与隐式反向

`rl/qp.py` 使用 GPU 批量 ADMM，每 32 次迭代复用一次 CUDA Graph。
活动约束的等式精化结果，须通过原始约束残差、对偶残差和乘子符号检查后才能使用。
重复平面由特征分解处理，不需要删除原 SFC 行。
反向通过活动约束零空间计算成本导数，QP 组装和平坦性转换也保留 Torch 梯度。
状态、参考轨迹、SFC 为条件输入；A*、PathCover/RISP 和 Newton 不参加反传。

求解器使用 float64，原始单位下的最大约束残差阈值为 `1e-5`。
对偶残差阈值为 `1e-5 + 1e-5 * max(|HU|, |g|, |Aᵀy|)`。
最多迭代 4096 次。未满足残差阈值时，返回求解失败，交给真实配置的恢复策略。
失败样本的成本导数为零。
约束切换点的导数采用当前活动集；恰好触边且乘子为零的行采用非活动侧导数。

每个预测步最多保存 32 个 SFC 平面，用有效掩码表示空位。
超过容量时任务报错；通过 `env.max_planes` 扩大容量后重新运行，不丢弃平面。
网络特征使用固定比例缩放；求解器数据保留物理单位和完整精度。
每个控制步创建独立观测，RSL-RL 保存状态、参考、平面、掩码和恢复标记。
当前存储两组走廊及 `corridor_index`，QP 组装时无损还原 15 步约束；详见 [规模优化](scale-optimization.md)。
CUDA Graph 工作区最多缓存 4 种批次形状，淘汰不改变已保存样本的反向梯度。
前向每次从固定初值求解，不依赖跨样本 warm start。

## 探索和恢复

正常求解时，训练从 MPC 命令均值附近采样高斯动作，4 个分量的标准差均为 0.03。
动作按物理单位传入 Newton；电机限幅仍由控制器处理。
确定性评估使用均值。

QP 失败或尚未满足恢复退出条件时，actor 输出确定性制动命令。
该分支使用概率质量为 1 的制动动作，熵为零，成本梯度为零。
PPO 保留恢复样本用于价值学习。求解模式切换时，不把制动动作当作普通高斯样本。
制动公式和退出阈值与 `execution.py` 共用，恢复状态由每个环境的 Planner 保存。

走廊生成失败或返回非有限 SFC 平面时，学习任务标记该环境的 `sfc_valid=False`。
该环境跳过 QP，执行同一制动策略，并请求重规划。
其余环境照常求解。下一步必须取得有效 SFC 后才有机会退出制动。
该处理拒绝整份无效走廊，不删除个别平面来求解。
`frontend_failures` 与 `qp_failures` 分开记录。

成功条件沿用真实 IPC 场景统计：到达、未碰撞、预测命令未越界。
另行记录采样推力越界步数、实际速度/加速度越界步数、QP 失败和制动次数。
预测约束与实际飞行分别检查；加噪动作不享有 MPC 均值的约束满足性质。
`maximum_qp_primal_residual` 包含失败的求解尝试；失败解不会作为导航命令执行。

## CA-AC-MPC 来源与适配

参考 [CA-AC-MPC 论文](https://arxiv.org/html/2605.29155v1)和
[作者源码 `14aa349`](https://github.com/prisma-lab/CA-AC-MPC/tree/14aa349bb1f509c529bf3662c7feb72db68f1cb7)。
项目采用其成本网络—可微 MPC—PPO 结构。
求解层将 iLQR 替换为 IPC 的线性硬约束 QP。
CUDA Graph 承载前向批量计算，GPU 线性代数承载隐式反向。
作者的 10 状态、4 输入非线性模型留待后续接入。
本次未复制作者仓库源码。

A*、PathCover/RISP 和地图管理仍在 CPU。端到端耗时包括这些步骤和物理积分。
本地开发证据位于 `results/acmpc-development/`；实测结果见下节。

## 验证范围

开发检查使用 `tests/test_acmpc.py`：

- 一般不等式与重复活动平面的解析解、有限差分成本梯度。
- 不可行问题的失败返回和零梯度。
- IPC 全量 QP 组装、严格 OSQP 前向对照、平坦性命令和成本网络梯度。
- 保存的走廊快照与确定性制动分支。
- 无效 SFC 的拒绝、混合批次隔离和有效样本的反传。

共享恢复函数另由原有执行配置和 Planner 测试检查。
训练采用避障场景的 4 个环境，低速与中速各 2 个。
森林评估检验该避障训练策略的场景迁移。

## 实测结果

### 训练、保存和恢复已运行

日期为 2026-11-02，结果根目录为 `results/acmpc-development/`。
先运行 32 次 PPO 更新，再恢复检查点运行 9 次。
每次采样 `4 × 64` 步，共 10496 条环境转移。
续训的成本网络最大参数变化为 `0.0058361`。
保存并重载后，同一观测的动作检查通过，容差为 `1e-6`。

本轮比较使用 `results/acmpc-development/resume/checkpoint.pt`。
该策略来自避障训练，未用森林训练更新。
初次 32 次更新完成后，episode JSON 导出因 NumPy 布尔值失败。
该段 episode 明细缺失；检查点、TensorBoard 和失败日志保留。
布尔值转换和逐 episode 原子写入已修复。
续训记录了 4 个完成的中速 episode，均碰撞。

### 同场景比较：中速失败仍存在

两列均采用真实速度、Newton/MJWarp、电机力和制动重规划。
固定成本列使用零初始化成本网络，和学习列共用 CUDA QP。
每格运行 1 个确定性 episode，种子均为 0。
表内时间为仿真时间，最小间距在每个物理子步计算。

| 场景 | 固定成本 | 学习成本 | 固定 / 学习最小间距 |
| --- | --- | --- | --- |
| 低速近障碍 | 到达，6.54 s | 到达，6.60 s | 0.131 / 0.114 m |
| 中速近障碍 | 碰撞，2.34 s | 碰撞，2.35 s | -0.003 / -0.014 m |
| 原森林 8 m | 到达，3.49 s | 到达，3.52 s | 0.412 / 0.408 m |
| 原森林 32 m | 到达，17.83 s | 到达，15.50 s | 0.139 / 0.106 m |

8 个 episode 的预测命令越界、实际速度/加速度越界和确定性推力越界均为 0。
中速两列分别执行 21、22 个制动步后碰撞。
32 m 两列分别触发 5、2 次恢复，执行 476、234 个制动步。
学习成本在该次长路线中更快，但最小间距也更小。
这组检查验证检查点在物理环境中的运行；成功结果与固定成本相同。

32 m 学习列首次在 `x=26.423 m` 处因 SFC 非有限平面中断。
当时状态和参考轨迹均有限，7 个预测步共享的一个平面为 NaN。
加入上述前端恢复后，仅重跑受影响的长路线。
重跑记录 1 次前端失败和 1 次 QP 失败，最终到达目标。
原始中断日志和失败观测仍保留。

录制时再次捕获了同一无效 SFC，已定位到原 EMVP 的切平面公式。
候选走廊端点为 `(30.694484, -0.072905, 1.0)` m。
该端点距最近点云点为 0.256355 m，小于 SFC 安全半径 0.3 m。
这是前方走廊的端点，当时无人机仍在 `x=26.423 m` 附近。

原 `findTangentPlaneOfSphere` 计算 `sqrt(d²-r²)`。
本例根号内为 `-0.02428199 m²`，因此输出 NaN。
保存输入后，独立调用原几何代码仍产生同一坏平面。
诊断副本只增加了数值打印，生产几何代码保持原版。
输入和打印分别见 `results/acmpc-forest-sfc/invalid-corridor-input.npz`、`tangent-diagnostic.log`。

当前恢复处理拒绝该份走廊，跳过 QP，制动并重规划。
上述现象来自历史 EMVP 版本；当前默认走廊已替换为 PathCover/RISP。

比较证据位于 `comparison/`：

- 低速：`low-fixed/`、`low-learned/`。
- 中速：`medium-fixed/`、`medium-learned/`。
- 森林 8 m：`forest-fixed/`、`forest-learned/`。
- 森林 32 m：`forest32-fixed/`、`forest32-learned-recovery/`。
- 原始中断与诊断：`forest32-learned.log`、`forest32-learned-diagnostic/`。

历史 OSQP 基线和原版 Docker 统计保留在 [执行配置](execution-profiles.md)。
本轮固定成本列的 CUDA 求解容差更严格，与历史 OSQP 结果分别记录。

### 检查与性能边界

共享执行和 Planner 的针对性检查为 18 项通过，见 `targeted.log`。
最终 4 项学习专项测试通过，见 `frontend-recovery-test-final.log`。
其中 3 项为先前检查的重跑，新增 1 项保护无效 SFC 恢复。
Ruff 和 `git diff --check` 通过。

32 m 学习列实测 actor 平均耗时 13.37 ms，完整 step 为 28.34 ms。
这是共享 GPU 上的单环境结果，未达到 100 Hz 墙钟实时控制。
仿真控制周期仍为 0.01 s。A*、PathCover/RISP 和地图管理仍在 CPU。
本轮未运行长时间收敛、多种子统计、全量回归或独占 GPU 吞吐测试。

### 高速近障碍补测

沿用原 `fast-near` 场景：`ref_dis=42`，触发距离为 0.5 m。
固定成本与同一学习检查点各运行 1 个确定性 episode，种子为 0。

| 策略 | 结果 | 仿真时间 | 制动控制步 | 最小间距 |
| --- | --- | ---: | ---: | ---: |
| 固定成本 | 碰撞 | 1.44 s | 4 | -0.0276 m |
| 学习成本 | 碰撞 | 1.44 s | 4 | -0.0256 m |

两次均出现 4 次 QP 求解失败，随后执行制动并碰撞。
两次前端失败、预测命令越界、实际状态越界和推力越界计数均为 0。
这次高速失败与森林的无效 SFC 不同。
数据位于 `results/acmpc-high-near/{fixed,learned}/records.json`。

## 当前传感与历史结果

真实任务现默认使用 Mid-360 / 官方 MultiMesh，状态由 ArticulationData / PVA 提供，
轨迹与 Rerun 由 RecorderManager 管理。使用方式和迁移结果见 [官方组件迁移](official-components.md)。
本页此前的训练、对比和 32 m 录制均使用切换前的输入；未用新传感器重训。
当前实际加速度统计包含接触响应，历史电机合力加速度不包含接触响应。
