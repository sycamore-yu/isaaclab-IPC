# 理想配置、真实配置与长森林测试

当前执行决策见 [ADR-0003](adr/0003-execution-profiles.md)。
本页维护执行配置含义、恢复逻辑及走廊切换前的场景结果和录制索引。
当前默认走廊为 PathCover/RISP，结果见 [走廊说明](pathcover.md)。
真实配置现默认 Mid-360 / MultiMesh；本页旧结果保留原传感输入，新配对结果见 [官方组件迁移](official-components.md)。

## MPC 输入的速度是当前速度

MPC 用当前位置、当前速度和当前加速度预测后续轨迹。
其中的速度是“现在如何运动”，不是“希望飞多快”。
例如当前速度为 3 m/s，预测间隔为 0.1 s，
位置预测中的速度项就是前进 0.3 m。
`vx_max` 是速度上限，`ref_dis` 是参考路径采样间隔，含义不同。

## 只有 2 个公开执行配置

- `real`：默认配置。读取实际速度，用 Newton 电机力推进。
- `ideal`：复刻原避障测试。把 MPC 输入速度置零，直接采用规划位置。

执行约定统一由 `src/isaaclab_ipc/execution.py` 的 `ExecutionProfile` 定义。
`Config.profile` 固定选择其中一种；恢复策略由 `profile.recovery` 推导。
`Planner.prepare` 在该约定下处理速度输入，评估入口据此选择执行器。
两种执行器保持独立，不把理想执行器包装成 Newton 物理。

```python
from isaaclab_ipc.mpc import Config

cfg = Config.load("avoid", profile="ideal")
assert cfg.profile == "ideal"
assert cfg.profile.recovery == "previous_sequence"
```

`Config.load` 的参数和 CLI 的 `--profile real|ideal` 保持不变。
结果 JSON 中的 `profile` 也仍为字符串。
`Config.profile` 不能原地改写；切换执行约定时重新创建配置与规划器。
`Config.params` 只保留算法参数，数值调参方式保持不变。
旧的 `execution_profile`、`recovery`、`perfect_simu` 执行元数据不再存入该字典。
直接构造含这些执行键的配置会报错，不能再用字典覆盖恢复策略。
上游 YAML 中的 `perfect_simu` 只作源码记录，加载时移除，由执行约定取代。

理想执行器由 `scripts/evaluate_ideal.py` 提供。
统一入口为 `scripts/evaluate_native.py --profile ideal`。
理想执行器不启动 Newton，避免把位置赋值误记为刚体飞行。
该配置限于原来的 10 m 避障场景，单次调用运行 1 个 episode。
`run_ideal_campaign.py` 可重复运行这套理想配置。

原版故障回退逻辑仍保留在 `Planner.consume` 中。
理想配置继续使用上一条 jerk 序列，便于原版对照。
真实配置改用下述恢复逻辑，历史基线结果保持不变。

## 真实配置的制动与恢复

1. 导航 QP 失败后，进入持续制动状态。
2. 请求 A* 按最新地图和位置重新规划。
3. 制动目标加速度为 `-2v`，限制在原加速度界内。
4. 由当前加速度计算 jerk，并限制在原 jerk 界内。
5. 保持制动，直到实际速度低于 0.2 m/s、实际加速度低于 0.5 m/s²。
6. 导航 QP 同时求解成功时，退出制动并再次请求规划。

首次求解失败也能进入制动，不再依赖已存在的历史解。
机体仍受电机响应、推力饱和和刚体动力学影响。
停车距离不足时，该策略仍可能碰撞。

## 森林路线可以长于 8 m

8 m 是本地复现脚本选择的目标，不是地图或算法的长度上限。
上游 `IPC/launch/ipc.launch` 接收外部目标。
本地原版历史记录也包含到达 `(16,0,1)` m 的测试。
32 m 测试将同一地图的目标扩展为 `(32,0,1)` m，超时设为 90 s。

原版长路线证据位于 `results/reference-forest-32m/`。
原版实际路程为 32.34 m，已到达目标。
对完整 PCD 的折线间距下界为 0.200 m，使用 0.25 m 机体半径。
地图没有树木实例标签，不能把路线长度换算为经过的树木数。

复测数据目录：

- `results/ideal-profile-retest/`：理想配置 4 组避障测试，每组 10 次。
- `results/real-brake-low/`：默认真实配置低速 10 环境。
- `results/real-brake-medium/`：默认真实配置中速 10 环境。
- `results/real-brake-forest/`：默认真实配置森林 10 环境。
- `results/real-forest-32m/`：默认真实配置的 32 m 森林路线。

本轮结果与历史基线分开保存，未重定义历史成功率。

## 走廊切换前的场景结果

| 测试 | 理想配置 | 默认真实配置 |
| --- | ---: | ---: |
| 低速近障碍 | 10/10 | 10/10 |
| 中速近障碍 | 10/10 | 0/10 |
| 高速远障碍 | 0/10 | 本轮未重测 |
| 高速近障碍 | 0/10 | 本轮未重测 |
| 原森林 8 m | 不适用 | 10/10 |

中速真实配置的每个环境均触发 1 次恢复，执行 21 个制动控制步后碰撞。
代表环境的向前速度从约 3.575 m/s 降至 3.035 m/s。
制动已经产生减速，但首次求解失败后约 0.21 s 即碰撞，未能停住。
当前结果不支持“制动重规划已解决中速失败”。

32 m 森林路线：原版和 Newton 各运行 1 次，均到达目标。
Newton 在 13.62 s 仿真时间内到达，未触发恢复，没有命令越界。
完整 PCD 的物理子步最小间距为 0.237 m。
这 2 次是长路线功能检查，不代表长路线成功率统计。

## Rerun 录制

本地 `recordings/` 保存下列 8 个基线 RRD，均已通过解码检查。

| 路线 | 原版 Docker | Newton |
| --- | --- | --- |
| 低速突现障碍 | `low_original.rrd` | `low_newton.rrd` |
| 中速突现障碍 | `medium_original.rrd` | `medium_newton.rrd` |
| 森林 8 m | `forest_original.rrd` | `forest_newton.rrd` |
| 森林 32 m | `forest_32m_original.rrd` | `forest_32m_newton.rrd` |

前 6 个文件录于恢复策略修改前；32 m Newton 录制使用当前真实配置。
32 m 两端均到达目标，Newton 录制中未触发恢复。
录制是独立的新运行，不能替代上表的固定任务统计。
原版只记录到达目标，不从录制中的到达状态推断命令界与全局碰撞验收。

这 8 个基线录制由一次性工具生成。
当时使用的生产脚本已存档在本地 `results/recording-producers/`。
森林静态地图仅裁剪显示范围；实际传感点云和机体状态来自运行过程。
Git 不包含 RRD 和原始结果；唯一数据副本不能作为清理候选。

另有 2 个带 SFC 的 32 m 森林 RRD：

- `forest_32m_fixed_cost_sfc.rrd`：固定成本，CUDA QP，Newton 真实配置。
- `forest_32m_learned_cost_sfc.rrd`：学习成本，同一 CUDA QP 和物理配置。

两次均到达目标。文件显示实际 SFC、参考轨迹、预测轨迹和制动状态。
这类录制使用 `scripts/acmpc.py evaluate --record`。
命令和结果见 [AC-MPC 录制说明](acmpc.md#录制带-sfc-的森林路线)。

## 执行配置重构检查

本次集中配置选择、速度输入规则和恢复策略选择，没有改变制动公式或 QP 数学问题。
固定输入下，理想与真实配置的 QP、控制输出和恢复状态均与 `66432fa` 一致。
两条中速 CLI 冒烟测试分别保存在：

- `results/execution-contract-ideal/`：理想配置到达，无命令越界。
- `results/execution-contract-real/`：真实配置触发 1 次恢复、21 个制动步后碰撞。

这些检查覆盖重构后的入口与执行选择，不替代既有成功率统计。
