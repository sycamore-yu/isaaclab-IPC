# IPC 原生复刻：设计调查

复刻基准和 RL 角色已确认。
先完成 IPC 基线，再开发 AC-MPC 扩展。
阶段 1 候选包已能运行，场景对照已完成。
当前验收差距见 [场景验证](validation.md)。

## 已确定的目标

- 使用 Isaac Lab 标准外部包和 RL 工作流。
- 使用 Newton 后端的 MJWarp 求解器。
- 复刻 hku-mars/IPC，优先复用 Isaac Lab 生态。
- 将控制计算批量放到 GPU，避免逐环境 Python 控制循环。
- 复用 cuRobo/Warp 时保留原算法语义。
- IPC 基线以固定发布代码为准，论文差异单独记录。
- RL 学习 MPC 成本，策略梯度穿过 MPC。

## 原始方法需要保留什么

上游代码固定为 `0f959ca7995e71a1fcc4942bd4a86869ab6ee975`。
调查日期为 2026-11-02。

| 环节 | 冻结代码的行为 | 源码位置 |
| --- | --- | --- |
| 输入 | 世界系点云、odometry、IMU、目标点 | `IPC/src/planner.cpp:397-478` |
| 地图 | 最近 10 帧点云，局部裁剪，0.2 m voxel | `planner.cpp:424-455` |
| 路径 | 局部 A*、视线简化、路径插值 | `planner.cpp:259-375` |
| SFC | 凸多面体，平面约束为 `n·p+d≤0` | `planner.cpp:215-256` |
| MPC | 9 维状态 `[p,v,a]`，3 维输入 jerk | `IPC/include/mpc.cpp:140-190` |
| 约束 | 各轴速度、加速度、jerk 界及预测位置 SFC 硬约束 | `mpc.cpp:13-42,252-330` |
| 求解 | OSQP-Eigen，每次调用新建 solver | `mpc.cpp:53-77` |
| 时序 | 控制 100 Hz；预测 15 步，每步 0.1 s | `IPC/config/param*.yaml` |
| 输出 | 微分平坦变换、推力模型、body-rate/throttle | `planner.cpp:195-204`、`planner.h:221-249,312+` |
| 求解失败 | 继续使用上一条 jerk 序列的对应片段 | `planner.cpp:178-193` |

100 Hz 控制周期与 0.1 s 预测步长不能合并。
原始 MPC 的线性预测模型与仿真刚体模型也需分别保留。

原始来源：[planner](https://github.com/hku-mars/IPC/blob/0f959ca7995e71a1fcc4942bd4a86869ab6ee975/IPC/src/planner.cpp)、[MPC](https://github.com/hku-mars/IPC/blob/0f959ca7995e71a1fcc4942bd4a86869ab6ee975/IPC/include/mpc.cpp)、[参数](https://github.com/hku-mars/IPC/blob/0f959ca7995e71a1fcc4942bd4a86869ab6ee975/IPC/config/param.yaml)。

## 发布代码与论文存在差异

| 项目 | 论文 | 发布代码 |
| --- | --- | --- |
| 地图记忆 | 栅格记录最近命中时间，超过阈值后遗忘 | 保留最近 10 帧点云 |
| 位置权重 | `Rp=2000` | 主配置 `Rp=1000` |
| 末端速度、加速度权重 | 200 | 主配置 1000 |
| 速度界 | 实机段落为 8 m/s | 主配置为 10 m/s |

论文来源：[liu2023ipc.pdf](https://github.com/hku-mars/IPC/blob/0f959ca7995e71a1fcc4942bd4a86869ab6ee975/paper/liu2023ipc.pdf)。
参数比较涉及论文实机设置和发布主配置，不能代替场景逐项核对。

原避障仿真使用 `perfect_tracker`。
其回调直接赋值规划位置，见 `perfect_drone_model.hpp:58-70`。
Newton 刚体环境会增加角速度跟踪、电机响应和接触效应。
算法数值复刻与真实动力学导航结果因此需要分别验收。

数值疑点已核对：OSQP-Eigen 0.7.0 丢弃下三角非对角项。
当前 Hessian 保留原求解器实际接收的上三角语义。
理想配置首次求解失败时明确报错，不读取未初始化历史解。
真实配置的故障恢复按 [ADR-0003](adr/0003-execution-profiles.md) 执行。
证据和其余差异见 [来源记录](provenance.md)。

## 原生包结构

### 物理和控制分开

当前配置为 `NewtonCfg(solver_cfg=MJWarpSolverCfg())`。
MJWarp 是 Newton 的求解器，不是另一套并列后端。
参见 [Isaac Lab physics backends](https://docs.nvidia.com/isaaclab/concepts/physics-backends)。

本机 contrib `Multirotor` 直接导入 PhysX articulation。
不能仅切换 preset 就用作 Newton 无人机。
当前采用 Newton articulation 适配和状态力回调。
该适配复用已有四电机资产，并保留官方 `Thruster`。
官方 `Thruster` 电机模型使用 Torch。
机体状态由 Newton 积分，控制器只提交力和力矩。

### 批量计算保留硬约束

先建立同输入的 CPU 数值参考，再验证批量 GPU QP。
当前默认使用 CPU OSQP，GPU QP 由 `--solver gpu` 显式选择。
GPU 求解失败时不会静默转到 CPU。
GPU 路径需覆盖局部地图、A*、SFC 和 MPC。
仅加速 MPC 不能证明端到端吞吐提高。

cuRobo 当前为 V2，公共 API 已重写。
旧版 `WorldCollision` 示例不能直接视作 V2 接口。
V2 提供自定义 rollout、梯度求解器和粒子求解器。
官方文档没有提供 IPC 硬约束 QP 的直接替代入口。
具体复用范围以碰撞查询和独立数值检查为起点。
求解器选择仍需以原问题的约束残差和控制输出为依据。

依据：[V2 说明](https://github.com/NVlabs/curobo)、[求解器](https://nvlabs.github.io/curobo/latest/concepts/optimization_solver.html)、[自定义问题](https://nvlabs.github.io/curobo/latest/guides/optimization_problem.html)。

### 使用标准包和 RL 入口

采用外部模板的 `src/` 布局及 `isaaclab.tasks` entry point。
任务注册遵循官方外部包入口，当前评估使用本项目 CLI。
AC-MPC 训练和播放尚未接入；后续复用官方工作流。
不复制一套训练平台。

Direct workflow 适合自定义 IPC 批量状态。
官方 `DirectRLEnvWarp` 可减少主循环的数据转换。
该入口当前标记 experimental，需要独立 reset/step 验证。
RL 采用下文的 AC-MPC 角色。
观测、动作、奖励和终止条件需要与导航任务一致。
`terminated`、`truncated` 和随机种子需要保留标准语义。

包包含任务注册、环境、控制器、地图和规划器。
测试范围覆盖数值参考、异步 reset、失败路径和 RL 入口。
配置和文档与同一套接口一起交付。

## 本机可复用依赖

本机 Isaac Lab 源码位于：
`/home/tong/tongworkspace/simulation_dev/isaac/IsaacLab-3.0`。
源码版本为 3.0.0，revision 为 `adc6c81c840acc03cff0340e518595431ffb19a1`。

以下版本来自邻近 NavRL 环境的安装元数据。
IPC 项目现有独立 `.venv`，解析版本由 `uv.lock` 固定。

| 组件 | 观察到的版本 |
| --- | --- |
| Isaac Lab | 3.0.0 |
| Newton | 1.6.1 |
| Warp | 1.17.0 |
| MuJoCo / MJWarp | 3.12.0 |
| Torch | 2.11.0+cu130 |
| RSL-RL | 5.5.1 |

Isaac Lab 源码声明 Newton 1.6.0，邻近安装为 1.6.1。
当前锁文件采用 Newton 1.6.1。
本机 cuRobo V2 参考源码 revision 为：
`78fd485fa82d9b9a063fb4985e371814587e666a`。
位置为 `/home/tong/tongworkspace/reference_repos/curobo-v2`。
复用源码不等于目标环境已安装或验证依赖。

接口证据均相对于 Isaac Lab 源码目录：

- `tools/template/templates/external/pyproject.toml.jinja`
- `source/isaaclab/isaaclab/cli/__init__.py:35-67`
- `source/isaaclab_contrib/isaaclab_contrib/assets/multirotor/multirotor.py:17-18`
- `source/isaaclab/isaaclab/utils/wrench_composer.py:229-248`
- `source/isaaclab_newton/isaaclab_newton/assets/rigid_object/rigid_object.py:110-164`
- `docs/source/concepts/warp_environments.rst`

调查时仅有 1 张 RTX 4090，总显存 24564 MiB。
6 个 NavRL Python 进程占用 19558 MiB，GPU 利用率为 100%。
该快照不代表后续空闲容量。性能测试前需要重新查看占用。

## 验收需要分开回答的问题

1. 数学复刻：同输入下比较预测矩阵、目标值、首条控制量和约束残差。
2. 物理集成：验证悬停、角速度响应、碰撞、独立环境 reset 和状态更新。
3. 导航行为：记录每场景的到达、碰撞、超时和求解失败。
4. GPU 性能：分别记录感知、地图、规划、QP、物理和完整 step 的耗时。
5. RL 支持：先验证环境接口，再验收可微 MPC actor 的训练和恢复。

论文突现障碍矩阵为 6 档距离 × 20 档速度 × 10 次。
每方法共 1200 次，完整自动运行器未随代码提供。
原测试还使用 xy 平面 clearance，不能代替 3D 物理碰撞指标。
是否执行完整矩阵，以及 RL 是否要求收敛，由验收讨论确定。
本轮未选择吞吐数字、训练预算或通过率阈值。

## 已确认的设计决策

用户于 2026-11-02 选择发布代码作为基准。
用户指定 RL 类似 AC-MPC，允许复杂 RL 留到下一阶段。
由此采用分阶段交付，不再采用独立动作策略或残差控制方案。

### AC-MPC 接入方式

原方法将神经成本映射与可微 MPC 组成 actor。
网络输出各预测步的二次项 `Q` 和线性项 `p`。
MPC 输出作为动作分布均值，PPO 损失通过 MPC 回传。
部署时使用均值，物理环境不需要可微。
原 diffMPC 只支持控制输入界，不支持状态约束。
来源：[论文 v8，III-F/III-G](https://arxiv.org/html/2306.09852v8#S3.SS6)。

作者公开代码固定为 `c59e53aec11c1fffa8b69d99b0ee7879ba7ccb28`。
`training_modules/mlp_mpc_policy.py:91-206` 展示成本和动作构造。
代码包含模型、策略模块及子模块，未提供完整仿真训练工程。
`mlp_mpc_policy.py:4` 还保留 `<YOUR_ACMPC_FOLDER>` 路径。
仓库许可证为 GPL-3.0；引用方法与复制源码需要分别处理。
来源：[作者仓库](https://github.com/uzh-rpg/acmpc_public/tree/c59e53aec11c1fffa8b69d99b0ee7879ba7ccb28)。

固定子模块为：

- `mpc.pytorch@63732fa85ab2a151045493c4e67653210ca3d7ff`
- `stable-baselines3-acmpc@152c353863d3b05fb5feed4deb37b952bb4beb7b`

`mpc/mpc.py:58-68,123-143` 是带控制 box 约束的 iLQR。
直接替换 IPC QP 会丢失 SFC、速度和加速度状态约束。
第二阶段需要支持这些线性不等式的可微 QP。

本项目建议采用以下结构：

```text
观测 ──> 神经成本映射 ──> H、g ──> 可微 IPC QP ──> 控制动作均值
状态、A*、SFC ──────────────────> 固定约束
PPO 损失 ──反传──> 动作均值 ──> QP ──> 神经成本映射
Newton/MJWarp ──交互──> 轨迹、奖励 ──> critic 与 PPO
```

QP 保留 `min 0.5 UᵀHU + gᵀU`、`l ≤ AU ≤ u`。
预测 15 步、每步 3 维 jerk，共 45 个决策变量。
学习成本仅改变目标函数，不改变固定状态下的可行域。
首版成本参数化优先使用正权重构造二次项。
具体形式需保持与基线矩阵构造可比较。

反传范围限定为连续 QP 及输出控制变换。
A* 和 SFC 构造不反传；采样时保存状态和走廊快照。
PPO 更新使用同一快照，避免在新地图上重算旧样本的动作概率。
QP 的活动约束切换与退化点需要单独检查梯度。

若仅把 MPC 放在 `env.step()`，策略梯度不会穿过求解器。
若让 PPO 输出成本参数并把参数当动作，动作概率的定义也会改变。
这两种接法均不能替代已选定的 AC-MPC 角色。

训练期采样可能使动作离开 MPC 均值满足的约束。
因此评估采样动作和确定性部署动作时，需要分别记录约束违规。
不能从 QP 可行推断带噪动作或真实飞行安全。

### 复用 RSL-RL 的 actor 和数据存储接口

本机 RSL-RL 5.5.1 支持用 `actor.class_name` 加载项目内模型。
保留官方 `OnPolicyRunner` 和 PPO，只扩展 actor 与可微 QP。
完整 observation `TensorDict` 会被标准 rollout storage 保存。
状态、SFC 平面及有效掩码可用固定形状张量传递。
成本网络输入可归一化，求解器数据保留物理单位。

rollout 在 `torch.inference_mode()` 中调用 actor。
PPO update 则从保存的观测重算动作分布，再反传损失。
求解器需要同时支持无梯度前向和有梯度更新。

storage 的实际复制发生在 `env.step()` 之后。
下一步不得原地覆盖上一份观测中的 SFC 数据。
用连续 2 步不同走廊的样本验证快照保存和打乱后的对应关系。
JIT/ONNX 导出需能包含 MPC；普通 MLP 导出结果不能代替完整策略。

源码证据位于邻近 NavRL 环境的 `site-packages`：

- `isaaclab_rl/rsl_rl/rl_cfg.py:25,265-285,343-350`
- `rsl_rl/algorithms/ppo.py:128-140,229-249,427-461`
- `rsl_rl/storage/rollout_storage.py:142-149,180,225-251`
- `rsl_rl/runners/on_policy_runner.py:79-105`

上述接口已静态核对，尚未测试项目 actor 的运行兼容性。

### 阶段与完成条件

1. **IPC 基线**：完成标准包、感知到控制、Newton 刚体环境和 GPU 批量计算。
   用原始数值参考核对 QP，再验证导航、reset 和分段耗时。
   阶段内提供标准 RL 环境接口，不把普通 PPO 当作 AC-MPC 成果。
2. **AC-MPC 扩展**：先验证硬约束 QP 前向等价和梯度。
   再接入自定义 actor，完成 PPO 更新、保存、恢复与播放。
   使用同场景、同种子比较 IPC 基线与学习版本。

第二阶段不要求更换 Newton，也不把 cuRobo 默认轨迹优化器当 QP。
MPVE、学习动力学和可微 A* 不纳入首版扩展。
实际训练预算和导航通过率在运行前确定。

## 来源许可与验证状态

上游根目录未发现许可证，`IPC/package.xml` 的 license 为 `TODO`。
`emvp.hpp` 单文件包含 MIT 声明。
该声明不能代表整个仓库的许可。
本地构建按固定 revision 提取源码。
项目暂不分发生成源码和二进制；发布前需明确授权。
证据：[package.xml](https://github.com/hku-mars/IPC/blob/0f959ca7995e71a1fcc4942bd4a86869ab6ee975/IPC/package.xml)。

已检查 IPC、AC-MPC、RSL-RL 源码及原论文和官方文档。
数值、物理和原版场景证据见 [场景验证](validation.md)。
AC-MPC 训练尚未运行。
