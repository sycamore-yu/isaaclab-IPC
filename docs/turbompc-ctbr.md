# TurboMPC 的 AC-MPC CTBR 模型与接口

当前主线采用 [ADR-0008](adr/0008-native-acmpc-pathcover.md) 的逐节点 `28N` 成本和原生 PathCover。下文保留旧七参数 CTBR、精修器和 IPC 前端的历史证据。旧接口、默认界和旧验收状态不适用于新主线。

## 范围与已完成验证

本阶段在已披露的统一 IR2 配置 M4、IPC SFC 独立数值检查和真实低速闭环之后开始。最近一次通过的前置为提交 `403fc2ddb19e753ec959771cd1a4fa593328ce72` 的 [M4 final-v8 汇总](../results/turbompc-port/m4-ir2-accepted-final-v8.json)：同 CUDA12、H15、float64、B=1/16/64/256 的400组成对测量及数值/梯度检查通过，`m4_gate_passed=true`。其 A 端明确是上游 `7050a4f` 加三行 `IR_N_STEPS=2` 配置，B 端也使用 IR2；未修正原版 A0 的 `original_baseline_passed=false` 与全部原始失败记录继续保留。前置 SFC 结果见 [turbompc-sfc.md](turbompc-sfc.md)。本页四旋翼模型、成本头和结果消费者使用独立 CTBR 契约。

提交 `50858a03` 的 [M4 final-v7 原始报告](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v7.json) 已完成400组成对测量：全部数值、梯度、来源守卫及其余性能项通过，但 B16 的 P50 为 `43.412735/37.506053 = 1.157486 > 1.1`，因此该版本的 M4 未通过。当前冻结提交 `403fc2ddb19e753ec959771cd1a4fa593328ce72` 新增编译路径的通用矩阵向量乘积融合表达，已通过新的正式 M4 final-v8；历史失败不改写。模型 v6 的116个数值样本和240步模型闭环、Newton v6 的600步真实物理复验，以及 PPO v3 的一次实际更新独立审计均已完整通过。真实导航 v2 出现 raw QP 失败，其保存状态的原走廊问题已被独立可达性证书排除，受控终止后标为 `FAILED_INCOMPLETE`。各域使用本轮独立报告，未沿用历史 v5 的成功标志。

生产提交 `29563a5` 的 `CTBRPolisher` 已加入有界原 KKT 残差校正，保留 raw ADMM 结果，并使用精修后的乘子和活动基进行隐式反向。27项精修器 CPU、13项 Actor/env CPU检查通过。[完整 CTBR v5](../results/turbompc-port/ctbr-acceptance-cu130-v5.json) 的116个开环/差分求解及四例各60步、共240个模型闭环求解全部通过；[Newton v5](../results/turbompc-port/ctbr-newton-cu130-v5/report.json) 的200步悬停与400步位置阶跃全部通过。v1…v4 的原始失败、部分轨迹和单点诊断继续保留，不以新结果改写历史。

[训练 v2](../results/turbompc-port/ctbr-train-smoke-v2/metrics.json) 已实际完成16步 rollout、一次 PPO 更新及 checkpoint 重载，旧 route 审计通过；当时尚未保存有效 MPC 成本 VJP 的增强证据。后续 `403fc2dd` 的 [PPO v3 独立验收](../results/turbompc-port/ctbr-train-smoke-v3-acceptance.json) 已通过43项检查，确认真实成本反向和参数更新，范围仍仅为一次训练更新。提交 `b45fd01` 的生产反向校正已通过实际 Newton state9 的29个成本差分求解，以及四个原固定案例的116个求解；两组均通过原成本 FD 门槛和追加的实际反向 KKT 原系统残差检查。真实 low 导航 v1 已出现求解失败，随后受控中止，标为 `FAILED_INCOMPLETE`；最后完整日志为第1000步，终局和总失败次数未知。v2 的新失败及原问题不可行证据单独保留。模型闭环、Newton 跟踪、含 SFC 的实际导航与训练学习证据分别判定，**上述旧导航版本不支持完整 M5 通过**。

独立参考调节候选 `3f9cd34` 已另外完成 [PPO v4 一次更新审计](../results/turbompc-port/ctbr-train-smoke-v4-acceptance.json)，但 [真实导航 v3](../results/turbompc-port/ctbr-real-low-v3-acceptance.json) 在第391次求解输入处被全控制可达性证书证明原新走廊不可行，390步已执行后受控中断。第391次求解没有返回状态，终局未知；该候选仍为失败/未完成，不能沿用模型或训练通过标志判定导航成功。

最新参考余量策略 V2 的冻结候选 `1a71d69` 已通过 [导航 v4 独立验收](../results/turbompc-port/ctbr-real-low-v4-acceptance.json)：原 seed0、固定成本与场景协议完整执行1076步，在原目标半径内结束，碰撞、约束违反、求解/前端失败、恢复和制动计数均为0。同一候选的 [PPO v5 一次更新](../results/turbompc-port/ctbr-train-smoke-v5-acceptance.json) 也通过。结合来源相同的核心组件证据，[最终组件索引 v2](../results/turbompc-port/native-port-completion-v2.json) 的七个限定验证域均通过，`qualified_migration_acceptance_passed=true`。这一结论保留共同 IR2 的前提、未修改原版 A0 的 M4 失败、旧导航失败，以及单 seed 低速导航和单次 PPO 更新的范围；不表示未见动态障碍的通用安全保证或墙钟100Hz能力。

已测前端现已顺序集成到verified分支，代码提交 `282a3fa92c8b738e520ec331a081b61a84d5ce92`。144项冻结来源、155项运行/测试文件和CUDA12/13相关构建产物逐字核对一致，分支差异只有文档；核心实际403、前端实际1a71的原执行身份继续保留。[集成一致性报告](../results/turbompc-port/native-port-integrated-source-v1.json)

## 实现边界

| 文件 | 职责 |
| --- | --- |
| [quadrotor.py](../src/isaaclab_ipc/turbompc/quadrotor.py) | 作者离散 CTBR 映射、精确自动微分、坐标与单位适配 |
| [ctbr.py](../src/isaaclab_ipc/turbompc/ctbr.py) | 固定物理配置、七成本语义、完整 SFC/速度/输入约束、hover 初猜 |
| [constraints.py](../src/isaaclab_ipc/turbompc/constraints.py) | 真实非线性路径函数、原界检查、活动集合、二阶导与终端 mask |
| [solver.py](../src/isaaclab_ipc/turbompc/solver.py) | Torch SQP 与 CUDA KKT 隐式反向，接收可选固定路径约束和精修器 |
| [ctbr_polish.py](../src/isaaclab_ipc/turbompc/ctbr_polish.py) | 原物理问题的有界 KKT/Newton 精修、六项证书、独立精修乘子和活动基 |
| [ctbr_actor.py](../src/isaaclab_ipc/rl/ctbr_actor.py) | 原生求解调用、成本网络、无效行隔离、执行接受与恢复 |
| [ctbr_env.py](../src/isaaclab_ipc/tasks/ctbr_env.py) | Isaac Lab 快照、127 维特征、CTBR 结果消费者与实际命令检查 |
| [check_ctbr_acceptance.py](../scripts/turbompc/check_ctbr_acceptance.py) | 独立 NumPy 映射、作者值对照、双步长七成本 FD、模型闭环 |

运行路径不需要 JAX。原版 JAX 仅用于前置基准，不参与 CTBR Actor 前向或反向。模型没有替换 Newton 的刚体、电机或控制器积分；CTBR 直接预测给定机体系角速度的运动，实际角速度仍由 [RateController](../src/isaaclab_ipc/assets/rate_control.py) 跟踪。

## 作者来源与精确离散式

Canonical 来源是 `prisma-lab/CA-AC-MPC`，固定提交 `14aa349bb1f509c529bf3662c7feb72db68f1cb7` 的 `differentialMPCPerformance/dynamics/drone_ts.py` 中 `DroneDynamics.forward`。源文件 SHA256 为 `090d99a39f2cfccf98ae872486f01fbe6a0976c2a3bf499550b7b34379759246`。实现为独立 Torch 代码；保存的作者 fixture 是前向输入/输出数据，不是复制的算法源码。[fixture 来源记录](../tests/data/ca_acmpc_forward_reference.json)

定义状态与控制：

\[
x_k=(p_k,q_k,v_k)\in\mathbb R^{10},\qquad
u_k=(F_k,\omega_k)\in\mathbb R^4.
\]

`q=(qw,qx,qy,qz)` 为 WXYZ，`p/v` 用世界坐标轴，`F` 为总推力 N，`omega` 为机体系 rad/s。单步为：

\[
\begin{aligned}
p_{k+1} &= p_k + \Delta t\,v_k,\\
\widetilde q_{k+1} &= q_k\otimes
\left(\cos\theta,\; \frac{\Delta t}{2}\operatorname{sinc}(\theta)\omega_k\right),
\quad \theta=\frac{\Delta t}{2}\|\omega_k\|,\\
q_{k+1} &= \frac{\widetilde q_{k+1}}{\|\widetilde q_{k+1}\|+10^{-8}},\\
a_k &= \frac{F_k}{m}\,r(q_k)-g e_3,\\
v_{k+1} &= v_k+\Delta t\,a_k.
\end{aligned}
\]

这里 `sinc(theta)=sin(theta)/theta`，零点取解析极限。原始 Hamilton 旋转的第三列是：

\[
r(q)=\begin{bmatrix}
2(q_xq_z+q_wq_y)\\
2(q_yq_z-q_wq_x)\\
q_w^2-q_x^2-q_y^2+q_z^2
\end{bmatrix}.
\]

以下细节属于被测试的数学契约：

- 位置是 Euler 更新，不能额外加入 `dt²*a/2`；速度使用旧姿态 `q[k]` 计算加速度。
- 四元数采用右乘 `q ⊗ exp`，并且每一步都除以 `norm+1e-8`。这不是仅在外部输入边界执行的保护，也不是严格投影到单位球。
- SQP 中间状态可以偏离单位四元数流形，因此第三分量不能改成单位四元数才等价的 `1-2*qx²-2*qy²`；这种替换会改变离流形值、Jacobian 和 Hessian。
- 作者 `get_jacobian` 的 Euler 近似及遗漏归一化导数，不用作精确导数 oracle；作者 traced forward 的单位四元数假设也不替代 canonical `forward`。
- 原作者指数映射使用 `norm(omega)+1e-8`。这里明确改为解析 sinc，小角用四次项 Taylor 表达式，避免零角速度处的二阶导未定义；其余离散语义保留。两分支都保护零点，以免 `torch.where` 的未选分支产生非有限反向值。

作者固定 96 组 fixture 在原质量 `.752 kg`、fixture 记录的实际重力浮点值、`dt=.1` 下对照，最大前向差 `7.322420447763989e-12`。生产物理参数使用下节统一配置；不能把原参数与生产参数的差异计为算法误差。[模型测试](../tests/test_turbompc_ctbr.py) [fixture](../tests/data/ca_acmpc_forward_reference.npz)

## 物理参数、坐标与动作

| 项目 | CTBR 契约 |
| --- | --- |
| 质量、重力 | `.75 kg`、`9.81 m/s²`，与当前资产/控制器一致 |
| 预测 | `H=15`，`dt=.1 s` |
| 实际控制/物理时步 | 由真实任务记录；与预测步长分别报告，不把 `.1 s` 当成控制周期 |
| 位置 | 世界坐标轴，减 `env_origins` 后与该环境 IPC 地图/SFC 对齐 |
| 线速度 | 世界系，不转到机体系 |
| 姿态 | Isaac Lab 3 XYZW 显式重排为 WXYZ |
| 角速度 | 控制和新增观测为机体系 rad/s |
| 优化推力 | 总推力 N；hover=`m*g=7.3575 N` |
| 环境动作第一维 | `F/m`，单位 m/s²；其余三维保持机体系 rad/s |
| 总推力界 | `[.840857142857, 9.459642857143] N`，来自导航 specific-thrust 界乘质量 |
| specific-thrust 界 | `[.08*9.81/.7, .9*9.81/.7]`，约 `[1.121142857,12.612857143] m/s²` |
| 机体系角速度界 | `±[1.5,1.5,.1] rad/s` |
| 速度界 | 每轴 `[-10,10] m/s` |
| 真实加速度界 | 下界 `[-20,-20,-10]`，上界 `[20,20,20] m/s²` |

状态适配函数 `state_from_isaac(pose_xyzw, world_twist, env_origins)` 输出 float64 十维状态，保留测得四元数数值，不隐式归一化或裁剪状态。`body_rates_from_isaac` 仅为世界角速度到机体系的坐标转换归一化输入姿态；它不向 CTBR 状态额外添加角速度自由度。外部状态/参考必须有限且满足 `abs(norm(q)-1)<=1e-4`，而内部 SQP 的 raw-Hamilton 离流形行为仍保留。[坐标适配](../src/isaaclab_ipc/turbompc/quadrotor.py) [环境快照](../src/isaaclab_ipc/tasks/ctbr_env.py)

角速度 `1.5/1.5/.1` 是本项目控制器的保守选择，不是作者默认 `10/10/4`。选择考虑当前 `I=.012`、增益 45 和 hover 附近各轴力矩余量，不保证所有总推力及同时多轴命令都不会使电机饱和。Newton、控制器和 CTBR 质量在环境构造时核对，运行记录另存质量、重力、物理总推力容量、预测/控制/物理时步。[资产](../src/isaaclab_ipc/assets/quadrotor.py) [控制器](../src/isaaclab_ipc/assets/rate_control.py)

`controls_to_actions(controls, mass)` 只将总推力除以质量。正常优化动作直接执行 CTBR 控制，不再经 jerk/flatness 转换；失败恢复仍可调用原 IPC 制动器，其输出显式限制在 CTBR 动作范围。浮点32执行边界的处理见 Actor，不能将输出类型转换造成的舍入越界当作增加物理余量。

## Builder API、参考与初猜

```python
from isaaclab_ipc.turbompc import ADMMConfig, SQPConfig, TorchMPC
from isaaclab_ipc.turbompc.ctbr import CTBRConfig, CTBRQPBuilder
from isaaclab_ipc.turbompc.ctbr_polish import CTBRPolisher

builder = CTBRQPBuilder(CTBRConfig()).to(device)
data, guess = builder(obs, costs)
model = builder.dynamics()
sqp_config = SQPConfig(
    max_iter=60, tolerance=1e-7, linesearch="merit",
    compile_kernels=True, require_qp_convergence=True,
)
admm_config = ADMMConfig(max_iter=32768, eps_abs=1e-8, eps_rel=0.0)
solver = TorchMPC(
    model, sqp_config, admm_config,
    constraints=builder.path_constraints(),
    polisher=CTBRPolisher(max_iter=4, tolerance=1e-11, max_linear_refinements=2),
)
result = solver.solve(data, guess)
accepted = (result.status == 0) & (result.admm.status == 0)
physical_plan = result.controls[:, :-1]  # [B,15,4]
# 只执行 accepted 行；其他行按任务恢复规则处理。
# 全部使用及 backward 完成后调用 solver.close()。
```

示例是生产精修路径的配置；`TorchMPC` 的通用默认仍为 `polisher=None`。旧报告、显式 `--no-polish` 重现与当前精修结果分别记录，不能把新配置挂到旧报告。`model(z[...,14])` 返回 `x_next[...,10]`，`model.acceleration(z[...,14])` 返回世界系净加速度 `[...,3]`。二者均支持批量；`linearize` 和 `lagrangian_hessian` 使用实际离散映射的自动微分。

| `obs` 字段 | 形状/含义 |
| --- | --- |
| `ctbr_state` | `[B,10]`，测得的状态 |
| `ctbr_reference` | `[B,15,10]`，每个未来状态的参考 |
| `planes`、`plane_mask` | 与 IPC 保存的 `[A,-b]` 快照一致；稠密时为 `[B,15,faces,4]` |
| `corridor_index`（紧凑时） | `[B,15]` int64；两个平面块，索引 `-1/0/1` |
| `costs` | 单独传入 `[B,7]`；省略时为零成本头 |

`reference_from_ipc(reference_pva, yaw_target)` 保留 IPC 参考的世界位置和速度，以 yaw 构造直立姿态 `[cos(yaw/2),0,0,sin(yaw/2)]`；IPC 加速度不是 CTBR 状态。独立模型审计也可直接提供可实现的非直立 `ctbr_reference`。Builder 从初态开始，逐参考将四元数对齐到连续同半球，该离散上下文选择不依赖学习成本。

返回 `MPCInputs` 的节点数为 16，每节点变量维度 14。初猜在所有物理控制节点施加 hover，然后通过同一非线性模型精确 rollout；它满足初态/动力学等式，不宣称必然满足所有走廊。Builder 不读取在线 Planner，也不裁剪物理初态速度。公开缓冲区包括 `hover_control`、`control_lower/upper`、`action_lower/upper`、`state_weights`、`input_weights`、速度/加速度界和 `acceleration_stage_mask`。

## 七个成本输出的新语义

网络输出顺序固定为 `[position, velocity, attitude, input, bias_x, bias_y, bias_z]`。前四项 `c_i` 映射为 `s_i=exp(log(2)*tanh(c_i))`，后三项映射为 `b=.2*tanh(c[4:7])`。默认基础权重：

| 分组 | 权重 | 参考 |
| --- | --- | --- |
| 位置 | `Qp=10`，三轴相同 | IPC 世界位置 |
| 速度 | `Qv=2`，三轴相同 | IPC 世界线速度 |
| 姿态 | `Qq=4`，四个分量相同 | 已对齐半球的 WXYZ 参考 |
| 输入 | `R=diag(.2,.1,.1,.2)` | `[m*g,0,0,0]` |
| 终端状态 | 四组中的状态权重统一乘 5 | 最后一个状态参考 |

目标在 `x1…xH` 计状态跟踪误差，在 `u0…uH-1` 计输入相对 hover 的误差；`x0` 没有额外状态成本。位置还增加 `-a_k*Qp*bᵀ p_k`，其中非终端 `a_k=1`、终端 `a_H=5`。该偏置乘基础位置权重，等价参考位移是 `b/s_position`，不是直接把 `b` 加到已经缩放的参考。成本的 `1/2` 因子、正定输入权重和零交叉块 `E` 均由独立物理目标对照测试。

`uH` 只是 TurboMPC 节点布局中的终端占位控制：Hessian 为单位矩阵、线性项为 `-hover_control`，独立最优值为 hover。它不进入 15 条执行控制，不施加输入或加速度约束。这里的成本和终端设置是本项目显式集成配置，不宣称等同于作者所有训练默认参数。[Builder](../src/isaaclab_ipc/turbompc/ctbr.py) [目标与梯度测试](../tests/test_turbompc_ctbr.py)

## SFC、速度和真实加速度约束

SFC 保持 `A*p<=b` 的 `[A,-b]` 存储约定。时间槽 k 作用于 `x[k+1]`；保留每个平面、原时序、重复面、非单位法向及既有安全半径。Builder 不去重、不截断到 64 面、不额外收缩，也不改变 `pack_corridors` 已执行的行顺序。无效 NaN padding 在算术前屏蔽；有效非有限平面报错。[存储接口](../src/isaaclab_ipc/corridor_storage.py)

线性行按速度三行、输入四行、全部 SFC 面排列。速度只约束 `x1…xH`；总推力和 body rates 只约束 `u0…uH-1`。不把初态状态或 `uH` 占位项额外放入物理可行域。

加速度使用三维真实非线性函数 `a(q,F)`，通过 `PathConstraints(function, lower, upper, stage_mask)` 接入；mask 在 `k=0…H-1` 为真，终端为假。QP 使用该函数的局部 Jacobian 和平移边界；可行性、活动集合和 normal-cone 检查使用真实函数及原始物理界。隐式反向同时包含动力学与活动路径约束的完整 Lagrangian Hessian，未活动乘子先屏蔽。通用 solver 的线性几何梯度对应原线性行；生产 CTBR 精修路径进一步限定为下节的成本参数梯度。

这一约束保留了导航的物理速度/加速度界。旧 jerk 输入界不是四旋翼 CTBR 角速度或总推力界，不能声称仍以同一意义保留。非线性路径函数的物理常量和边界固定，不提供这些常量的学习梯度。[路径实现](../src/isaaclab_ipc/turbompc/constraints.py) [路径回归](../tests/test_turbompc_constraints.py)

## 生产精修的成本梯度与状态契约

`CTBRPolisher.validate_inputs()` 在 numerical detach 之前检查原始输入。只允许成本 `D/E/q` 的梯度；`initial_state/G/lower/upper` 必须固定，任何 `requires_grad=True` 都明确报错。原因是首个预测位置 `p1=p0+dt*v0` 与控制无关，某些 SFC 行可与初态/动力学等式相关，单个物理参数的乘子导数并不唯一。成本网络仍通过七个成本头产生 `D/q` 并得到隐式 VJP，不能把这一范围写成物理初态或几何参数也已通过可微验收。[精修器](../src/isaaclab_ipc/turbompc/ctbr_polish.py)

截至提交 `b45fd01`，精修只处理外层 status=0 且 raw QP status=0 的样本；提交 `50858a03` 新增下文的显式迭代上限收尾选项。每个 eligible 行独立使用 B1 的生产 `SparseAdjoint/cuDSS`，最多四次完整 Hessian Newton。原始平面、输入/速度/加速度界及终端 mask 全部保留。仅对有精确代数证书的首位置行调整 KKT 基，并把相应对偶贡献转移到等式乘子；这不是删除原物理约束，也不使用 primal regularizer。

每个 Newton 线性解必须先通过原 KKT 的 primal/equality/inequality 三块绝对残差 `<=1e-11`。提交 `29563a5` 在同一个 `qp/H/active` 和原 RHS 下计算 `r=b-K*delta`，再解 `K*correction=r` 并累积方向，默认最多追加两次；`max_linear_refinements` 只允许0、1、2，0保留先前不校正的行为。每次都对原 RHS 重算原 KKT 残差，不用正则化后的系统或仅校正方程的残差替代。最多是每个 Newton 步1+2次原生数值分解/求解，确实增加工作量，不宣称免费复用，也不增加 SQP/ADMM 或四次 Newton 的预算。

接受精修还要求六个原问题无穷范数均 `<=1e-11`：stationarity、初态/动力学等式、原不等式、normal cone、从真实初态独立展开全部控制后的原约束违反、以及该 rollout 与多重射击状态的差距。线性校正通过不能直接标记完整成功；任何一项失败都不能以单项物理违反小于 `1e-5` 替代。

`MPCResult.admm` 保存未改写的原始 QP 状态、轨迹、残差和乘子；`MPCResult.polish` 独立保存以下13个字段。令 `T=H+1=16`，`m` 为线性及非线性约束总行数，`J=max_iter`、`R=max_linear_refinements`。

| polish 字段 | 形状及语义 |
| --- | --- |
| `trajectory` | `[B,T,14]`；接受的精修轨迹，失败/未尝试时保留输入轨迹 |
| `lambda0` | `[B,10]`，初态等式乘子 |
| `lambda_dyn` | `[B,H,10]`，动力学等式乘子 |
| `y` | `[B,T,m]`，原不等式乘子 |
| `active_mask` | `[B,T,m]`；接受后的 KKT 活动基，失败/未尝试行全为 false |
| `status` | `[B]`；0接受、1未尝试、2失败 |
| `iterations` | `[B]`；实际尝试的 Newton 步数 |
| `errors` | `[B,6]`；上述六证书，失败时是最后评估候选的诊断 |
| `linear_residuals` | `[B,J,3]`；每个 Newton 步最终判定使用的原 KKT 三块残差 |
| `failure_reason` | `[B]`；独立原因码，不能仅由 raw QP 状态推断 |
| `prefix_rows_removed` | `[B]`；经精确首位置证书从 KKT 基排除的行数 |
| `linear_residual_history` | `[B,J,R+1,3]`；初次求解与每次校正后的原 KKT 残差，未尝试位置为 infinity |
| `linear_refinement_steps` | `[B,J]`；额外尝试的原生求解次数，包含发生异常的尝试 |

失败/未尝试行的乘子同样保留输入原值；精修失败把原本成功的外层状态设为4，不改 raw QP 状态。失败时全 false 的返回 `active_mask` 不是实际尝试过的活动基，诊断需要从保存的原输入和对偶重建，不能据此推断当时没有活动约束。

隐式反向使用接受后的轨迹、精修乘子与 certified 活动基，重新形成原问题的完整 Lagrangian Hessian；失败行沿用零梯度规则，并在混合批次的 KKT 组装前隔离非有限数据。外层返回误差对应实际返回的数据；失败的 `polish.errors` 是最后一次评估候选的诊断，不表示该候选已被执行。[核心集成测试](../tests/test_turbompc_polish_integration.py) [精修器测试](../tests/test_turbompc_ctbr_polish.py)

提交 `b45fd01` 为 CTBR 的实际隐式反向另加 `max_adjoint_refinements=4`：按原上游梯度的无穷范数缩放 RHS，并在同一原 KKT 上最多追加四次残差校正；最终还要对未缩放的真实 RHS 验证相对残差 `<=1e-11`。初次求解或某次校正的小残差不能替代最终累积返回解的检查。它不改变上述13个前向 `polish` 字段；反向诊断另存在 `last_adjoint_diagnostics`，其范围仅是最后一次调用，不能当成每个 `MPCResult` 的历史。该可选路径不启用时保留通用求解行为，最新 M4 按其既定原配置复验。[生产实现](../src/isaaclab_ipc/turbompc/ctbr_polish.py)

## Actor、checkpoint 与执行消费者

注册任务为 `Isaac-IPC-Avoid-TurboMPC-CTBR-Direct` 与 `Isaac-IPC-Forest-TurboMPC-CTBR-Direct`，使用 `CTBRTrainCfg`、`CTBRMPCActor` 和 `CTBRMPCEnv`。策略特征为旧 124 维加三维测得机体系角速度，新增部分按配置的 rate limits 缩放；求解器状态和物理单位不缩放。[注册](../src/isaaclab_ipc/tasks/__init__.py) [训练配置](../src/isaaclab_ipc/rl/config.py)

checkpoint 除参数形状外保存明确的 CTBR contract、127 维特征数、状态/控制顺序、七成本分组、成本语义版本和完整 `CTBRConfig`。旧 jerk 头即使也有七个输出，不能用作 CTBR checkpoint；同为 127 输入和七输出的旧 Actor state dict 也被拒绝。质量或语义版本不匹配的保存状态同样拒绝。内存和磁盘重放不读取或修改实时 Planner。

Actor 仅把 `sfc_valid` 行交给原生 Builder/求解器，并显式传入上述 `CTBRPolisher`。执行接受条件为 **外层 status=0 且 raw QP status=0**；精修失败的 outer4 同样拒绝执行，raw QP 预算耗尽不改写成成功。无效行或未接受行进入现有恢复链，返回确定性 point-mass 制动动作，成本梯度为零，并保存各自原始状态。外层、raw QP 与 polish 状态必须分别报告。

CTBR 结果有 `model="ctbr"`、十维状态、`solution[B,60]` 和 `trajectory[B,16,14]`。环境拒绝旧 45 维 jerk 方案；执行消费者调用 `Planner.consume_recovery` 更新恢复状态，不将 15×4 控制交给 `consume_state` 的 jerk 解析器。命令、状态和 SFC 诊断仍使用各自明确的原始单位。

环境根据**实际执行/采样动作**重新计算 `F=mass*action[0]`、raw-Hamilton 加速度和下一步速度，然后统计物理违规。Actor 的 `last_result.applied_control/predicted/predicted_acceleration` 来自均值动作；随机 rollout 之后另行采样，不能把这些均值诊断称为实际随机控制。环境检查使用真正传入 `env.step()` 的 actions，启用的 `FlightRecorder` 从 `controller.command` 记录实际动作。Gaussian 探索不是 MPC 可行性的保证；均值安全、实际采样导致速度或加速度超界的两个 CPU 回归明确要求计为违规。[执行消费](../src/isaaclab_ipc/tasks/ctbr_env.py) [实际动作记录](../src/isaaclab_ipc/tasks/recorders.py)

## 已有验证及保留的失败

| 范围 | 结果与对应证据 |
| --- | --- |
| 模型/Builder CPU | 23/23；作者 96 样本、独立 NumPy Hamilton、零/近零/nonzero Jacobian 与 Hessian-vector FD、四元数离流形语义、七成本、完整 SFC、终端 mask、checkpoint。运行结果记录在 [集成摘要](../results/turbompc-port/ctbr-integration-evidence-v1.json)，用例见 [模型测试](../tests/test_turbompc_ctbr.py)。 |
| PathConstraints/core 回归 | 8/8，32.47 s；实际为六项新路径约束检查加两项既有线性/非线性回归，包含独立 dense KKT、双步长 FD、eager/compiled 对照、失败行隔离、原线性几何梯度映射。[原日志](../results/turbompc-port/path-constraints-cu130-v1.log) |
| Actor/env CPU | 8/8，9.58 s；无效行跳过两种求解器、零梯度、旧/错配置 checkpoint、127 特征及动态 SFC 空间、快照坐标/所有权、实际采样违规、结果消费者、SQP0/raw1 恢复路由。[原日志](../results/turbompc-port/ctbr-actor-cpu-v2.log) [测试](../tests/test_turbompc_ctbr_actor.py) |
| Actor CUDA | 两项通过，11.94 s；固定实际 PathCover、goal=.4 m、有效 B1，原生前后向/物理控制/网络梯度/磁盘快照重放/混合无效行/close-reopen。[v2 会话结果摘要](../results/turbompc-port/ctbr-actor-cu130-v2.json) |
| CTBR 独立数值 v1 | 四个固定样例，4 个基点加 112 个成本扰动，全部 116 个求解完成；双步长 `1e-3/5e-4` 各自检查全部七个成本坐标，活动集不变、真实原界及原始状态门槛通过；最大成本梯度相对误差 `4.759414636059366e-7`。[完整报告](../results/turbompc-port/ctbr-acceptance-cu130-v1.json) |
| CTBR 模型闭环 v1 | `model_closed_loop_gate_passed=false`，尝试 12 个样例步求解，执行 8 个样例控制步后停止，总 `passed=false`；不能以开环通过替代闭环结论。后续独立报告保留本轮失败。 |

数值 v1 四例为 hover、位置阶跃、可实现姿态轨迹、末端活动 SFC。最后一例的**开环 wall 只约束末端**，模型闭环预先改为每个预测时间的 wall；不能将两种作用域混写。116 个开环样本没有筛选或替换。每个样例的梯度是固定全时域控制损失的七成本 VJP，不能描述为整个最优轨迹 Jacobian 或任意点的可微性证明。

Actor 首轮两项 GPU 测试失败也保留：[v1 摘要](../results/turbompc-port/ctbr-actor-cu130-v1.json)。当时错误地要求首个 pitch-rate 对成本非零，而固定样例的输出为 `[9.3928747,0,1.5,0]`，pitch 正在 `1.5 rad/s` 硬上界。随后同一输入的完整 4×7 VJP 诊断证明该行最大值约 `4.73e-22`、完整 Jacobian 全有限且最大值为 `1.4123696`；roll 对 `bias_y` 的导数为 `-1.1374019`。[GPU 进程直接写出的诊断](../results/turbompc-port/ctbr-actor-vjp-cu130-v1.json)

修正测试检查完整动作 Jacobian 的非零性及饱和分量的零导数，没有更换输入、物理界或求解参数。该两项 GPU 结果使用 `ADMM max_iter=16384/eps_abs=1e-8/eps_rel=0`、`SQP max_iter=30/tolerance=1e-7/merit`，eager 执行。之后严格 raw-status 接受规则另经上述 CPU 路由测试；后续预算变化应以新的报告为准。

证据范围说明：Actor GPU v1/v2 JSON 是成功观察到的会话结果摘要，不是完整原始 pytest 日志；后续重读旧进程被连接器拒绝，v1 中 `terminal_output` 保存的是该重读拒绝。原观察结果、限制和哈希另记于 [集成证据摘要](../results/turbompc-port/ctbr-integration-evidence-v1.json)，没有为恢复日志重跑 GPU，也没有覆盖历史文件。独立数值报告和 VJP 诊断由各自实际进程直接落盘。

### 闭环停止条件诊断与修复

四个固定模型场景各计划 60 个控制步。任一行求解失败时，整组在执行该控制前停止；因此其他行的短前缀不能算作完整闭环成功。

| 运行 | 配置与保留的失败 |
| --- | --- |
| `ctbr-acceptance-cu130-v1` | SQP30 / ADMM16384。第3次求解的全时域 wall 行外层已达标，末 QP residual=`1.06976e-8 > 1e-8`，raw status=1；全部116个开环/FD样本通过。 |
| `ctbr-acceptance-cu130-v2` | ADMM 上限32768，其余相同。越过旧失败点，第4次求解 raw QP成功，但30轮后 stationarity=`3.85982e-7 > 1e-7`；全部116个开环/FD样本再次通过。 |
| 固定第4步 SQP60 重放 | 前30轮与原轨迹最大差 `9.69e-12`，第43轮达到 `9.73e-8`；尾段实际 alpha=0.1，误差每轮约乘0.899。独立复算确认最后一轮所有候选均未通过 Armijo，原版最低有限 merit fallback 选择0.1，并非数值停滞。 |
| 随后 SQP60 模型闭环诊断 | 第5次求解外层第9轮即满足 `1e-7`，但末 QP primal=`1.62161e-8 > 1e-8`。旧成功规则提前返回，剩余51次预算未使用；该行仍被验收拒绝。 |

这是内外层成功契约不一致，不能继续只增加预算。新增显式 `require_qp_convergence=True` 后，CTBR 在真实外层 KKT、normal cone、末 QP状态全部满足时才成功；否则继续已有 warm state。默认 False 保留原版与 M4 行为。32项停止契约/验收 gate CPU 回归通过，包含预算耗尽与零梯度；日志为 `results/turbompc-port/sqp-strict-stop-cpu-v1.log`。生产 CTBR 预算为 SQP60 / ADMM32768，绝对容差仍为 `1e-7 / 1e-8`，ADMM 相对容差仍为0。

每轮原始 JSON/NPZ及执行脚本保留；单个诊断通过不能替代最终完整模型闭环。最终完整验收需启用相同的严格停止选项。

### 真实物理质量检查修复

`ctbr-newton-cu130-v1/report.json` 保留首次初始化失败。它在执行控制前停止：旧检查把 base link 的 `0.746 kg` 与整机 `0.75 kg` 比较，漏掉四个各 `0.001 kg` 的固定 rotor link。没有发生飞行或物理跟踪失败。

当前通过 Newton 公开的 `robot.data.body_mass.torch` 读取 `[num_instances,num_bodies]`，按每个 articulation 求和，并记录全部 body 名称和 link 质量。USD、控制器、模型和物理参数均未修改。形状、逐环境总量、非有限值和失败关闭清理的五项新增CPU检查通过；最终 Actor/env 13项CPU检查整体通过（9.66s），日志为 `build/turbompc-ctbr-actor-cpu-final-v1.log`。

### Newton 悬停与跨场景重置

[Newton v2 原始报告](../results/turbompc-port/ctbr-newton-cu130-v2/report.json) 的源码前后检查通过。实际 articulation 总质量为 `0.7499999404 kg`；物理时步 `1/300 s`、控制时步 `.01 s`，预测时步仍为 `.1 s`。所有动作经 RateController、官方 Thruster 和 Newton 刚体执行，没有用预测轨迹回写真实状态。

该次悬停完成 200/200 个控制步，最大速度 `3.36964e-7 m/s`、预测物理约束违反量 `8.00939e-14`、实际状态界违反量0，未碰撞。float32 状态中的高度保持1m，记录的位置误差约 `6.05e-17 m`；这反映该对称悬停及存储精度，不能解释为一般物理跟踪精度。求解 P50/P95 为 `15.399/16.807 ms`，控制接口的100Hz不能据此宣称墙钟实时100Hz。

位置阶跃场景尚未开始时，第二次 `reset()` 抛出 inference tensor 原地写入错误。根因是通用 `IPCEnv._get_rewards()` 用新计算的 inference tensor 替换 `previous_distance`；reset 在 inference 模式之外写回该缓存。修复为 `previous_distance.copy_(distance)`，保留初始化时的普通张量及其存储，奖励公式不变。

新增回归先精确复现同一异常，修复后 `tests/test_rewards.py` 三项全过（3.02s）。它调用真实 `reset_flight()`，检查三环境中 `[0,2]` 的部分重置、未重置行、奖励数值和张量所有权。修复提交为 `f4b35ba`；原 v2 失败报告与轨迹保留，不把已完成的悬停扩展为位置阶跃成功。报告 SHA256 为 `ff1247cfc8c4e637f7804f1456c88b7d34d392a98390ea35b15e141f6519b94c`。

### 未精修核心的历史 Newton 悬停与位置阶跃（v3）

[Newton v3 报告](../results/turbompc-port/ctbr-newton-cu130-v3/report.json) 在当时未加入精修的求解器 `50b7298` 上完成，`passed=true`、源码前后一致。模型、任务及原门槛保持；使用 SQP60 / ADMM32768、严格 raw QP 收敛和编译区域。第二场景实际完成 reset 并执行全部400个动作，覆盖前轮重置失败。该成功按其原源码保留，生产精修路径的 v4 另行记录。

| 固定场景 | 实际完成步数 | 终端位置误差 | 终端速度 | 最大预测原约束违反 | 实际状态界违反 | 求解 P50 / P95 |
| --- | --- | --- | --- | --- | --- | --- |
| 悬停 `(0,0,1)` | 200 / 200 | `6.05e-17 m` | `2.416e-7 m/s` | `8.1204e-14` | 0 | `22.117 / 33.497 ms` |
| 位置阶跃 `(.5,.2,1.1)` | 400 / 400 | `3.75825e-4 m` | `7.62819e-4 m/s` | `3.29941e-8` | 0 | `34.140 / 74.806 ms` |

两场景所有接受动作都要求 SQP 与末 QP 状态为0；没有碰撞，最小物理净空均为 `.75 m`，实际四元数范数最大误差 `1.23613e-7`。悬停的近零位置误差仍仅代表对称场景的 float32 记录。全部600步合计6s仿真时间，执行墙钟145.15s（包括首次编译及运行开销）；控制频率100Hz不等于墙钟实时100Hz。测试运行于共享GPU，空场景只禁用了两个扫描事件，真实刚体、推力器、RateController 和采样质量检查保留。它验证模型到真实执行的单位、坐标与跟踪链，不能替代带激光雷达和 SFC 的导航。

原始轨迹保存每步实际状态、机体系角速度、世界加速度、执行动作、电机总推力、求解时间及完整预测；没有用预测状态覆盖 Newton。报告 SHA256 为 `0bf8d0af34233a4019849a619b31ee139159998e5d673b35fe24d254659360c7`，轨迹 NPZ 为 `ec86811634cfccb6efda9386aa2639b9600f5f6df437d8dc5ec9b937a5c5a67c`，执行脚本为 `9da3162e445fd5d66a8bf6043a0b02e293f12b7c4a109eb29487462a286d77d2`。

## 实际导航与 PPO 训练审计

`scripts/turbompc/run_ctbr_pipeline.py` 在同一进程运行原有 `scripts/acmpc.py`，只增加执行来源记录。它要求新输出目录，运行前后记录源码、传感器射线、几何扩展、原生扩展和构建清单；结束时检查实际加载的私有 cuDSS、CUDA13、项目模块路径及无 JAX 模块。`execution_provenance_passed` 仅表示执行及来源一致，不能替代 `records.json` 的导航结果或 `training-audit.json` 的训练结果。

训练使用 `--audit-training` 时，分别记录 rollout、PPO update、checkpoint reload 阶段的 Actor batch 和 Planner/env 调用。update/reload 必须只使用保存的观测；审计在修改 Planner、推进环境、重置或记录执行结果之前拒绝这些调用。重载比较使用同一份克隆观测，成本网络参数还必须全部有限且至少一个发生非零变化。route、求解器成功、实际梯度和导航结果各自保留；参数增量不能单独证明有效 MPC 行参与了学习。

首次最小训练运行 `ctbr-train-smoke-v1` 使用1个环境、16个 rollout 步、1次更新、两个 mini-batch 和1个 epoch。它在构造 SimulationContext 的第一个 CUDA 重力张量时因显存不足退出，尚未完成环境初始化，也未执行 rollout、反向或更新。启动瞬间 GPU 仅剩262MiB；本任务当时的 CTBR 数值进程约642MiB，主要显存占用来自其他共享任务。没有终止其他任务或将失败改写为通过，后续改为串行安排本任务的新 GPU 进程并检查容量，不要求 GPU 空闲。

[训练 v1 来源报告](../results/turbompc-port/ctbr-train-smoke-v1-provenance.json) 中 `execution_provenance_passed=false`、`sources_unchanged=true`；[审计报告](../results/turbompc-port/ctbr-train-smoke-v1/training-audit.json) 为 `passed=false`、`counts=null`，明确失败发生在审计开始前。两个 SHA256 分别为 `0c69ec265be17e6fa7969ec6598c3f88f9feff8a39eb8384c8f1530bfd98501e` 和 `7b31bf397c0ecb9a9dd3cff24d5cced4980c6fb80d7f45f0f9efd82e3696bba0`。

### 训练 v2 已执行的范围与日志限制

[训练 v2 指标](../results/turbompc-port/ctbr-train-smoke-v2/metrics.json) 记录同一最小协议：1个环境、16步 rollout、1次 PPO 更新、2个 mini-batch、1个 epoch。`cost_network_max_parameter_change=0.00019958116172347218`，所有检查到的参数变化有限，`checkpoint_action_match=true`，`last_iteration=0` 是这一次从0开始编号的迭代，已完成 episode 数为0。来源报告的退出码为0，源码及库来源检查通过。[v2 来源](../results/turbompc-port/ctbr-train-smoke-v2-provenance.json)

| v2 阶段 | 实际 route 记录 |
| --- | --- |
| rollout | Actor前向16次、`record_mpc_result/env_step/planner_context/planner_consume_recovery` 各16次，更新点云1次，未重置 |
| update | PPO update 1次、Actor前向2次；所有 Planner/env 修改计数为0 |
| checkpoint reload | Actor前向2次；所有 Planner/env 修改计数为0，动作重载匹配 |

旧 [training-audit.json](../results/turbompc-port/ctbr-train-smoke-v2/training-audit.json) 的 `passed=true` 只认证这些 route 约束。旧 `Control/solved=0`、`braking=1` 日志不能证明16步都求解失败：已核对安装的 RSL logger 按引用保存每一步 `extras["log"]`，而原 MPC reward 代码反复更新同一字典，最终值会污染之前的统计。修复仅在每步写 reward/control 日志之前复制字典，奖励和控制公式未改；使用真实安装 logger 的历史统计回归覆盖该问题。v2 原日志与文件不重写，也不能从旧聚合值反推出有效求解行数。

增强 `TrainingAudit` 使用 schema 2：保留旧 `passed/route_passed` 语义，另存每阶段各次 Actor 的 solved/braking/SFC-valid、native row 映射、outer/raw 状态与残差。在实际 PPO backward 中，MLP 输出 tensor hook 记录七成本 VJP，按有效 native solved/非制动、制动、失败和无效 SFC 行分类；MLP 参数 hook 记录裁剪前的有限性、非零性和范数。`native_learning_gradient_observed` 要求这些真实梯度证据，不能由 parameter delta 替代。计数是前向/重放求值次数，不是独立环境样本数。[实现](../scripts/acmpc.py)

每阶段第一条未解结果独占保存观测、实际成本输出、Actor 结果及完整 native dataclass；当前 `MPCResult.polish` 的13个字段会递归保留，包括活动基、每次线性残差、追加求解次数及失败原因，便于精确重放。首条未解可含无效 SFC；若没有 native 求解，快照明确保存 `None`。29项 CPU 回归通过（5.69s），覆盖失败快照、不修改 Planner、不改变梯度/优化结果、真实 logger 历史统计及参数变化不能代替当前梯度的情形。本次记录时，**增强审计的实际训练 v3 尚未运行**。[测试](../tests/test_acmpc_training_audit.py) [29项日志](../build/turbompc-training-audit-v4.log)

| 训练证据 | SHA256 |
| --- | --- |
| v2 指标 | `a37dfbe029c5ef1fd51f99d5f420c11196cda308841bd5101ee244a8fcda63d5` |
| v2 旧 route 审计 | `dd07de04af8db2a70fe5508cac57c9d162ad809f3d889b81e637ec1f63e33c5b` |
| v2 来源报告 | `26cc2d6021981621278e1d3296395ab1ac138774b4554efc929221c6e4eb0db8` |
| schema 2 的29项 CPU 日志 | `1f9907a1c8bf667a931242bc7befdec5e7b765fd061c3edf1d80d1c819fdecf2` |

## 未精修核心的 CTBR 数值与模型闭环复核（v3）

[完整 v3 报告](../results/turbompc-port/ctbr-acceptance-cu130-v3.json) 在正式 common-IR2 M4 final-v4 通过之后运行，使用当时未加入精修的核心 `50b7298b6fdecfd923d3566705ed68584f9c01f8`。本轮 `ctbr_numerical_gate_passed=true`，但 `model_closed_loop_gate_passed=false`、总体 `passed=false`。它没有替换 v1、v2 或先前单点诊断的失败记录，也没有把 Newton 空场景通过当作带 SFC 模型闭环通过。

运行保持预先固定的四例、15步预测、float64、七个成本坐标及 `1e-3 / 5e-4` 双步长差分。配置为 `compile_kernels=true`、SQP60、容差 `1e-7`、ADMM32768、绝对容差 `1e-8`、相对容差0，并严格要求末次原始 QP 状态为0。开环墙面只作用末端节点；模型闭环墙面在每个未来节点要求 `x <= .2 m`。没有筛掉失败或活动集不稳定的扰动，没有修改样本、预算或门槛。

四例全部116次开环求解完成并通过。所有名义及扰动求解的 SQP、原始 QP 状态均为0；所有扰动的活动集检查通过。

| 固定案例 | 成本梯度相对 L2 误差，步长 `1e-3` | 步长 `5e-4` | 数值门槛 |
| --- | --- | --- | --- |
| hover | `3.37173e-7` | `4.58287e-8` | 通过 |
| position_step | `4.75894e-7` | `2.39815e-7` | 通过 |
| attitude_trajectory | `3.15468e-7` | `2.35105e-7` | 通过 |
| terminal_sfc | `3.58453e-7` | `1.07263e-7` | 通过 |

全部116次开环样本的最大物理界违反为 `8.73600e-10`，独立动力学残差 `7.58405e-10`，独立 stationarity 残差 `9.41977e-8`，四元数范数误差 `1.00225e-8`。CPU 对照的 NumPy/作者原 forward 最大误差分别为 `5.55112e-17 / 4.98962e-13`，两步长精确离散 Jacobian 差分最大绝对误差不超过 `5.00001e-11`。

模型闭环计划为四例各60步，共240次求解。实际完成36次求解、各例执行8步后，第九次求解（零起始 index8）的 `terminal_sfc` 未满足严格原始 QP 收敛要求，harness 按原定失败即停止规则退出，未执行失败控制。其余三例该次 SQP、原始 QP 状态均为0；未完成60步，因此不能宣布任一案例通过完整闭环门槛。

| 失败位置的检查 | 实际值 |
| --- | --- |
| SQP / 原始 QP 状态 | `1 / 1` |
| SQP / 末 QP 迭代数 | `60 / 32768` |
| 原始 QP primal / dual 残差 | `1.0009510859054771e-8 / 2.457347176374965e-9` |
| 原始 QP primal / dual 停止容差 | `1e-8 / 1e-8` |
| SQP stationarity / 等式 / 不等式残差 | `2.45736e-9 / 1.00095e-11 / 1.00095e-8` |
| 独立 SFC 物理违反 | `1.0009510637010166e-8` |

该失败的 primal 残差比固定 `1e-8` 停止容差高约 `0.0951%`。虽然物理违反量低于 `1e-5` 验收界，严格 raw-QP gate 仍然失败，不能用物理门槛代替停止条件。已经执行的32个独立 NumPy plant 动作均无 SFC、速度或加速度界违反；动作界最大违反 `8.70407e-10`，首步预测与 plant 的最大差异 `6.94117e-11`。这些部分轨迹仅用于定位，不能替代剩余闭环步骤。

执行从 `2026-11-03 10:57:56 UTC` 至 `11:18:25 UTC`，与其他 GPU 工作共享设备；没有以运行时间作为本轮通过标准。日志观测中第八次 B4 求解约4分49秒，第九次约6分49秒。正式 M4 的线性基准通过不代表非线性 CTBR 已达到实时控制速度。

源码前后、执行脚本、实际原生扩展与 cuDSS 构建来源检查均通过；结束后再次计算源码与库哈希仍一致。结果无异常或关闭错误。原始数据及执行副本完整保留：

| v3 证据 | SHA256 |
| --- | --- |
| [JSON 报告](../results/turbompc-port/ctbr-acceptance-cu130-v3.json) | `f8c5dce70e00b51f472000db3876b408132733e65c23e123b70ae701d45dc594` |
| [全部样本与部分闭环 NPZ](../results/turbompc-port/ctbr-acceptance-cu130-v3.npz) | `926781c856a4f09fd5ebe5ad18502c4ebf2ab9b543270cfdfc4459fc47fbd0e6` |
| [实际执行脚本](../results/turbompc-port/ctbr-acceptance-cu130-v3.harness.py) | `8831a7dcc057ace1483882124ae57c7f0aa6539236bab69972ab135b0f346843` |
| [完整日志](../results/turbompc-port/ctbr-acceptance-cu130-v3.log) | `1a6ebbc70aae56572896ec8f5954f7dfa17450460f05ff8df507e62ad2ed476e` |
| 最终 `solver.py` | `ed21a70911fa200e6b57af91e69e6f67e99860af3037670735c37a0618540d29` |

### v3 递归边界误差的精确诊断

第9次失败不是残差日志滞后：CUDA 在第32768轮强制重算 primal/dual，`1.0009510859e-8 > 1e-8` 必须保留为 raw failure。从保存的 `model_loop_states[3,8]` 独立复算得到 `p_x=.19999998998048185`、`v_x=2.0049147904377396e-7`，所以任何当前控制都给出 `p_x+.1*v_x-.2=1.0029629738e-8`。这一步位置不受当前控制影响，增加当前迭代不能改变其精确可行性。

前一接受解的第二个预测点超墙 `9.9990336566e-9`，仍在内层容差附近；真实状态加执行控制的独立推进与该预测相差 `3.0596081224e-11`，来自初态与动力学射击缺口传播。这使下一轮固定首位置超过内层门槛。保存的 state7/control7 精确重建 state8，误差为0。另一个已保存样本还显示 raw candidate 与线搜索后轨迹的认证范围不同：index4 的 raw primal 为 `9.9115e-9`，返回轨迹的 SFC 违反量为 `1.0729e-8`，后者按外层 `1e-7` 通过。raw成功并不自动保证执行后的下一问题可行。

修复方向是在原问题上进行有界的 KKT/Newton 数值修正，并从真实初态独立展开控制、检查全部原始约束。不能只改首控制、裁剪状态或改变几何平面，也不能沿用修正前的乘子计算隐式梯度。

CPU 对照先处理保存的 index7，再独立处理全部 index0–7，每个状态固定四次完整 Lagrangian Hessian Newton。完整 touching 集合中的首墙行虽然距边界约 `1e-8`，但真实固定首位置严格在内侧，不能被强设为活动等式；它与初态/首段动力学行存在精确代数恒等式。只将该误选行从 KKT 等式基中排除，原始不等式始终保留检查。其余七个状态没有删行。最坏 KKT 条件数约 `9.053e7`，受约束方向的 Hessian 最小特征值约 `.10045`。

八个独立保存状态全部修正成功：最大 stationarity `7.1202e-15`、等式残差 `2.4373e-16`；独立 NumPy 15步 rollout 的最大 SFC 违反量 `2.7756e-16`，速度/控制/加速度界违反均为0。index7 首总推力修正约 `1.04056e-5 N`，pitch-rate 修正约 `1.29958e-6 rad/s`。原 NPZ 未保存每步对偶，CPU 对照使用独立最小二乘恢复其起点，不能将这些乘子称为原 CUDA 保存值。它们仍是八个历史状态上的诊断，不能代替使用修正控制递归生成的新闭环。

| 数值修正诊断 | 结果与 SHA256 |
| --- | --- |
| CPU index7 | 通过；`ctbr-polish-cpu-v1.json`：`354b5129225a21a1d95335724303bdc23217be075cb5fc2a10423a82dbe643c7` |
| CPU index0–7 | 8/8通过；`ctbr-polish-cpu-v2.json`：`2bcf99382afd9fd0a2b18b3c3c4d8b2178f04fad0643d1474a583ea73b84adb2` |
| CPU 八点数组 | `ctbr-polish-cpu-v2.npz`：`645240990a308c35c6ba70c1087509c6394acfd105155a022b193e8c95b3d828` |
| GPU v1 | 未通过；四次原KKT残差约 `6.66e-9…2.56e-8`，最终等式残差 `2.56e-8`；`ctbr-polish-cu130-v1.json`：`1c4599baa8217b1fffd7300b0052c705fb156afbd2192c143890d03285c42876` |

GPU v1 复用相同原问题、活动基与四次 Newton，但为固定符号结构存入397²个 CSR 元素，包括零值；这与生产 `SparseAdjoint` 的时间块稀疏结构不同。源码与实际库来源检查通过，这一失败仍保留，不能把很小的 stationarity 单项当成完整 KKT 解精度。后续对生产稀疏模式的 GPU v2 检查、实际精修 v4 失败及完整 v5 成功分别记录如下，历史单点证据本身仍不等于闭环成功。

## 生产精修的局部检查与完整 v4 结果

### 接线与生产稀疏结构检查

| 检查 | 已完成范围 |
| --- | --- |
| 精修器 CPU | 16/16，4.20s；[日志](../build/turbompc-ctbr-polish-cpu-v2.log) 与 [源码前后记录](../build/turbompc-ctbr-polish-cpu-v2.json) 保留，源码一致。 |
| solver/core 精修集成 CPU | 6/6，36.95s；eager/compiled、独立解析非线性问题和 dense KKT、双步长 FD、精修乘子/活动基进入 VJP、失败行隔离、raw 缓冲区不变、detach 前输入梯度检查、默认 None 行为。[日志](../build/turbompc-polish-integration-v1.log) |
| 编译路径 CUDA 基点 | 四个固定基点全部通过，`base_diagnostic_passed=true`；此运行不做112个扰动或模型闭环，完整数值/闭环标志仍为 false。[base GPU 报告](../results/turbompc-port/ctbr-polish-base-cu130-v1.json) |
| 生产稀疏结构 GPU v2 | 保存的历史 index7 单点、四次 Newton 通过；使用实际 `SparseAdjoint` 的640维时间块结构和非活动乘子 identity 行，与397维独立活动基 CPU 系统对照。[GPU v2 报告](../results/turbompc-port/ctbr-polish-cu130-v2.json) |

稀疏 GPU v2 的矩阵在符号排列及非活动行扩充后与原问题差为0，RHS精确相同；四次原系统线性残差最大为 `1.4281938120852893e-16`，与 CPU 同系统解的最大差 `7.34933e-14`。最终 stationarity `1.85182e-15`、等式残差 `2.22045e-16`、原不等式和独立 rollout 违反为0，rollout/shooting 差 `9.05634e-16`。源码、实际库及构建来源一致，仍使用既有 IR2，没有新增 C++ 正则项。该历史 index7 起点乘子来自独立最小二乘恢复，不能称为原 ADMM 保存乘子；后面的 v4 数组则已保存实际 raw 与精修对偶。

| 局部验证证据 | SHA256 |
| --- | --- |
| 16项 CPU 日志 | `34240935329e4aec698c015b3c125ba5dfde900d9fd2d422f62844eea4b54833` |
| 6项 core 集成日志 | `6bad8063ea4fb95f684fdb93ee40aac40b0e4813f665fcc9f639dd3ca5fcad41` |
| base GPU JSON | `caddc0195b4a9dbae2df01d5f8c21006ddb47444ca7a2950bbe107f055f2e5cc` |
| base GPU NPZ | `1bad58d18a917b8b3301d5e23e91438465682c66094be4e4c19ef970976dca48` |
| 稀疏 GPU v2 JSON | `8b0aa7f58c9f6e814b99b9a92abb6715cc62a2e92d4d3d550463573235974313` |
| 稀疏 GPU v2 NPZ | `4a97e80072f5f9b57693b5a440d65a41604e61b983509167e2ba7e675df1b06f` |

### CTBR v4：116个数值样本通过，模型闭环未通过

[v4 完整报告](../results/turbompc-port/ctbr-acceptance-cu130-v4.json) 在 common-IR2 M4 final-v5 通过后使用生产精修路径运行，保持四个固定案例、H15、float64、七成本双步长 `1e-3/5e-4`、SQP60/`1e-7`、ADMM32768/`1e-8`/相对容差0、严格 raw 收敛及 `compile_kernels=true`。额外精修预算为4、六证书及原线性系统门槛均为 `1e-11`。116/116个开环与扰动求解全部通过，包括真实原约束、活动集稳定和成本 VJP；两步长最大梯度相对 L2 误差分别为 `3.5943964832e-7`、`8.9882716959e-8`，仍按原 `1e-3` 门槛验收。

模型闭环完成40/240次样本求解，四例分别执行9步，共36个实际 NumPy plant 控制步。第9次 wall 求解成功越过 v3 的旧失败点；第10次（零起始 index9）`terminal_sfc` 返回 outer4/raw0，整组在执行该控制前按原协议停止。其余三行该次通过也不等于完成60步，四例完整闭环标志仍全为 false。

| wall index9 证据 | 保存值 |
| --- | --- |
| outer / raw QP / polish 状态 | `4 / 0 / 2` |
| 精修失败原因 | `5: original_kkt_residual_failed` |
| 首次 Newton 原线性 residual：primal | `4.2719453554441273e-11 > 1e-11` |
| 同次 equality / inequality residual | `5.448340310150233e-16 / 5.448340876520556e-16` |
| 此次移除的固定首位置基行 | `prefix_rows_removed=0` |

原始 QP 已收敛，但首个 Newton 方向没有通过原 KKT 线性证书，因此精修失败并保留输入轨迹/原乘子，没有执行不合格方向。不能把 raw0 或某些物理残差足够小改写为精修成功。`ctbr_numerical_gate_passed=true`、`model_closed_loop_gate_passed=false`、总 `passed=false`；源码、执行脚本、实际原生库/cuDSS 构建来源检查通过，`error=null`。

报告记录的 Git HEAD 仍为 `50b7298`，具体运行版本由逐文件哈希限定：`solver.py=36cce8c6f0246dd4da1f3b5b4c842438da2aca78ef218a90acd9e2f7a7287bad`，`ctbr_polish.py=e5edf628a8b5fe3d651150c9de41da692c3a8debf43eae2924efe29dfd9cba0d`，`adjoint.py=93a73930bf4141f6d1c798d9a699cb30c68a4a629ccba7ef83757922b6ab7537`；不能仅用 HEAD 把当前精修结果与旧未精修 v3 混为一版。

### Newton v4：悬停通过，阶跃在 index9 拒绝执行

[Newton v4 报告](../results/turbompc-port/ctbr-newton-cu130-v4/report.json) 使用相同生产精修设置及真实 RateController、Thruster、Newton 刚体链。悬停完成200/200步，预测检查最大残差 `8.17170e-14`、实际状态界违反0；位置阶跃只执行9/400步。第10次求解（index9）为 outer4/raw0，独立预测检查最大残差 `5.855871343385388e-12`，返回 stationarity `9.825141888963884e-8`，未执行失败控制。源码前后一致，但总 `passed=false`；旧未精修 v3 的400步阶跃成功不覆盖本次失败。

原 v4 报告未保存完整单次精修原因码，后续单点重放才确认它同样是 reason5，证据见下一节。它与模型 wall 是两个不同初态/问题，不能直接把其中一个的矩阵或残差用于解释另一个。共享 GPU 上的运行时间仅作观察，不是这次闭环通过门槛，也没有据此宣称墙钟100Hz。

| v4 与前置 M4 证据 | SHA256 |
| --- | --- |
| [M4 final-v5 限定配置汇总](../results/turbompc-port/m4-ir2-accepted-final-v5.json) | `9b8579a137216569139cd29388d3fba2af1b9b3ce92dbc883ec3f1a9efb32526` |
| [CTBR v4 JSON](../results/turbompc-port/ctbr-acceptance-cu130-v4.json) | `dbcc451166bd0eac917291bdbde367f1186e4c02d93b972922baa5bbe58e74b4` |
| [CTBR v4 NPZ](../results/turbompc-port/ctbr-acceptance-cu130-v4.npz) | `21139f2b286b29fa3b9c63bc0a6b1c7c98273055df7264b9122736108e9c2b6f` |
| [CTBR v4 执行脚本](../results/turbompc-port/ctbr-acceptance-cu130-v4.harness.py) | `75b1134bc72379df16f188a2ef4815b2df6de4ed4c222415073aebf646fa3955` |
| [Newton v4 JSON](../results/turbompc-port/ctbr-newton-cu130-v4/report.json) | `e779300a129248b51977cabae1ffe3504537d6895ecc744c62510918d3b31fdb` |
| [Newton v4 NPZ](../results/turbompc-port/ctbr-newton-cu130-v4/traces.npz) | `ee40ecc224e4d36ea7c770bac1d038b86cefc596c5a5377502c4ba83e6805769` |
| [Newton v4 执行脚本](../results/turbompc-port/ctbr-newton-cu130-v4/harness.py) | `5464b7c2c4c452df217c7a1c72c19d01f44151f86dfbe3c3a9c389ac7e999478` |

### v4 后续原 KKT 单点诊断：生产修复前的历史证据

[Newton index9 重放](../results/turbompc-port/ctbr-newton-polish-step9-cu130-v1.json) 使用保存的同一初态、target 和零成本头，复现 outer4/raw0、polish2/reason5；与原失败预测轨迹最大差 `1.687538997430238e-14`。捕获实际生产544维 KKT、原 RHS 和解后，外部诊断仅对同一原矩阵计算 `r=b-K*delta`，再解 `K*correction=r` 并累积方向。原系统残差由 `3.1173137006557125e-9`，经一次校正降至 `4.69231127875e-12`，第二次降至 `3.77293118072e-14`；独立 CPU 矩阵乘积确认相同结果。源码、库和原报告/数组未改动。该历史诊断没有回写生产解或执行完整阶跃；后续生产实现及 v5 完整闭环证据另见下节。

[模型 wall index9 的 CPU 重建](../results/turbompc-port/ctbr-wall-index9-kkt-cpu-v1.json) 使用 v4 保存的实际 raw `lambda0/lambda_dyn/y`，没有最小二乘恢复。展开点是精修失败后保留的 `outer_z`，它与原 ADMM `raw_x` 最大差 `6.028162271065712e-7`，因此必须在 outer_z 重建原 QP。实际活动基含 stage2…15 的14个 wall，首位置去行数为0；约束 Jacobian 为174×224、数值满行秩，条件数约 `2.95357e6`，原活动 KKT 条件数约 `1.46054e13`，数值秩会随判断阈值变化。

该点第二位置面的推力系数 `kappa=dt²*n·body_z(q0)/mass=2.3931508668839717e-6`，明确非零，不能按控制无关冗余行删除。CPU 对原生产结构的直接解最大线性残差为 `8.881784197001252e-16`。这是保存矩阵的独立数值对照，不是新的 MPC 或闭环运行；没有修改原面、活动基或 primal regularizer。

当时提出的方案为原系统最多两次外部残差校正，原 `1e-11` 线性门槛和全部六项 `1e-11` 证书保持。随后模型 wall 的生产 CSR 诊断将原残差 `2.64123e-11` 经一次校正降到 `9.34274e-16`；应用这一次方向后 stationarity 仍为 `1.08939e-5`，说明不能用线性成功代替完整 Newton 证书。该方案之后在提交 `29563a5` 成为生产前向实现，并完成下节 v5 复验；有效训练 VJP 及新增反向精度问题仍单独处理。所有 v1…v4 失败文件保留。[wall GPU 校正](../results/turbompc-port/ctbr-wall-index9-correction-cu130-v1.json) [一次方向应用的六证书](../results/turbompc-port/ctbr-wall-index9-certificate-cpu-v1.json)

| 单点诊断证据 | SHA256 |
| --- | --- |
| Newton index9 JSON | `22d1bf0598c969c12e2bc0d5990811f504723a37b5051e8ee3e83dc701d05afb` |
| [Newton index9 NPZ](../results/turbompc-port/ctbr-newton-polish-step9-cu130-v1.npz) | `3ac395aa583c9659810895d5402a39033b25fc26890dd42d7657245ee098ff62` |
| wall index9 CPU JSON | `cafdc1918a088f55301caa800fe6548a17212782715d071ef3330e5d2be03ce7` |
| [wall index9 CPU NPZ](../results/turbompc-port/ctbr-wall-index9-kkt-cpu-v1.npz) | `90989f1db43626beb0202f03877e34b813cb7df1e8bfc0bb25525db3a8a26061` |

## 有界前向校正的完整 v5 结果

本节均对应生产提交 `29563a55179067605d59e0cfe2e68e2889d263f7`，前置仍是明确采用共同 IR2 的 M4 final-v5。前向校正没有修改模型、平面、物理界、数据、SQP/ADMM 容差或迭代预算；[v4/v5 协议比较](../results/turbompc-port/ctbr-acceptance-cu130-v5.protocol-comparison.json) 对案例、门槛、FD步长、SQP、ADMM、H、dt、物理配置逐项相等。新增配置仅是每个 Newton 步最多两次原 KKT 残差校正。

生产模块27项 CPU回归通过（4.67s），包含0/1/2的精确调用上限、固定矩阵及原 RHS、原生输出缓冲复用、失败回滚、raw数据不变，以及线性通过而非线性证书失败时必须继续拒绝。13项 Actor/env CPU回归通过（10.10s，两个CUDA测试未执行）；未把CPU结果表述为重跑Actor GPU。[27项日志](../build/turbompc-ctbr-polish-cpu-v3.log) [源码记录](../build/turbompc-ctbr-polish-cpu-v3.json) [13项 Actor 日志](../build/turbompc-ctbr-refinement-actor-cpu-v1.log)

### CTBR v5：116个数值样本与完整240步模型闭环通过

[正式 v5 JSON](../results/turbompc-port/ctbr-acceptance-cu130-v5.json) 的 `passed`、`ctbr_numerical_gate_passed`、`model_closed_loop_gate_passed` 均为 true。116个开环/扰动求解全部完成，四个案例各真实执行60步，共240个闭环求解；合计356个样本的 outer/raw QP/polish 状态全部为0，没有筛选或提前结束。开环仍仅末端 wall，闭环仍在每个未来节点施加原 wall。

| v5 固定案例 | FD相对L2，`h=1e-3` | FD相对L2，`h=5e-4` | 闭环实际步数 | 最终位置误差 | 最终速度误差 |
| --- | --- | --- | --- | --- | --- |
| hover | `3.42024e-7` | `8.55708e-8` | 60/60 | `4.38158e-10 m` | `8.34888e-15 m/s` |
| position_step | `3.52744e-7` | `8.81470e-8` | 60/60 | `0.00652466 m` | `6.66159e-7 m/s` |
| attitude_trajectory | `3.42032e-7` | `8.53934e-8` | 60/60 | `0.00640393 m` | `0.00132893 m/s` |
| terminal_sfc | `3.59454e-7` | `8.98956e-8` | 60/60 | `0.00389987 m` | `5.51498e-9 m/s` |

全部116个开环样本的最大物理违反为0，独立动力学残差 `1.24807e-14`，独立 stationarity `3.37486e-10`。240个模型样本的最大物理违反 `2.77556e-17`、独立动力学残差 `4.57548e-14`、独立 stationarity `1.12646e-10`。完整精修六证书均小于原 `1e-11`；其中最大 stationarity `9.84824e-12`、等式 `4.57548e-14`、精确rollout违反 `8.88178e-16`、shooting gap `4.09591e-13`。

实际共有282个 Newton 步，最多每个样本2步；仅其中两步各追加一次线性校正，共2次额外原生求解。所有初始及校正历史中的最大原 KKT 残差为 `1.25011e-10`，最终判定残差最大 `2.07308e-13`。原始/精修乘子、活动基、六证书、全部线性残差历史及计数均保存于 [NPZ](../results/turbompc-port/ctbr-acceptance-cu130-v5.npz)；[CPU汇总](../results/turbompc-port/ctbr-acceptance-cu130-v5.summary-v2.json) 只读取这些完成后的文件，没有重跑GPU或改变原始报告。

正式源码、扩展16个运行文件、HEAD、作者模型、M4 prerequisite及实际native/cuDSS来源检查均通过。[观察器](../results/turbompc-port/ctbr-acceptance-cu130-v5.progress-summary.json) 完整记录60次开始/结束，等待子进程退出后才读取JSON，`observer_error=null`；旧v4观察器提前读未写完JSON的错误仍保留在旧证据中，不是solver错误。v5启动时可用显存9608MiB，没有等待GPU空闲。运行从 `12:50:45` 至 `13:12:37 UTC`，与后半段Newton验证实际并行；最慢一步观察墙钟126.44s，时间仅用于运行记录，不构成实时或吞吐通过结论。[执行记录](../results/turbompc-port/ctbr-acceptance-cu130-v5.execution.json) [并行开始](../results/turbompc-port/ctbr-acceptance-cu130-v5.execution-context.json) [并行结束](../results/turbompc-port/ctbr-acceptance-cu130-v5.execution-context-end.json)

### Newton v5：生产精修路径的600步真实物理验证通过

[Newton v5 报告](../results/turbompc-port/ctbr-newton-cu130-v5/report.json) 的 `passed=true`，原200步悬停和400步位置阶跃全部实际执行。600个保存的polish状态均为0、六证书均不超过 `1e-11`。仍使用 `.75 kg`、`9.81 m/s²`、预测 `.1s`、控制 `.01s`、物理 `1/300s`，通过原RateController、Thruster和Newton刚体执行，没有回写预测状态。

| 固定物理案例 | 实际步数 | 终端位置误差 | 终端速度 | 最大预测原约束违反 | 实际状态界违反 | 最小净空 |
| --- | --- | --- | --- | --- | --- | --- |
| hover `(0,0,1)` | 200/200 | `6.05030e-17 m` | `2.41595e-7 m/s` | `7.91948e-14` | 0 | `.75 m` |
| position_step `(.5,.2,1.1)` | 400/400 | `3.75825e-4 m` | `7.62822e-4 m/s` | `1.30904e-12` | 0 | `.75 m` |

源码20项、HEAD、实际库及M4前置来源检查通过；总墙钟149.82s，运行期间与CTBR模型验证及外部GPU工作共享资源。悬停近零位置误差仍只代表该对称案例和float32记录；此空场景保留真实物理链但禁用两个扫描事件，不能代替带传感器/SFC的真实导航，也不能称为墙钟100Hz。[完整物理轨迹](../results/turbompc-port/ctbr-newton-cu130-v5/traces.npz) [来源报告](../results/turbompc-port/ctbr-newton-cu130-v5.execution.json)

### 历史 state9 成本FD通过，追加反向 KKT 精度检查失败

[state9 成本FD报告](../results/turbompc-port/ctbr-newton-state9-cost-fd-cu130-v1.json) 使用保存的真实Newton第9步初态、原target和同一成本语义，完成1基点加28个扰动，共29次求解。全部outer/raw/polish状态和原物理约束通过，活动集稳定；七成本AD与FD相对L2分别为 `2.30824e-4`、`2.30681e-4`，均小于原 `1e-3`，两步长FD彼此差 `2.73112e-7`。因此 `cost_fd_gate_passed=true`。

同一进程捕获真正 `backward()` 的原 KKT，独立CSR乘积确认 primal残差 `0.002198706423835302`、等式残差 `4.77936e-8`，相对原RHS的无穷范数残差 `0.0256009`。它未满足追加的 `1e-11` 绝对原系统诊断，`actual_adjoint_original_residual_passed=false`、该诊断报告总 `passed=false`。原成本FD通过不能覆盖这个额外失败；前向的有界Newton校正也不能自动校正另一次反向线性求解。

[保存矩阵的GPU校正诊断](../results/turbompc-port/ctbr-state9-adjoint-correction-cu130-v1.json) 未重跑MPC或改变生产返回值。原RHS及按固定RHS范数归一化的两种求解，在各追加3次原系统校正后，相对RHS残差分别达到 `2.29252e-12 / 2.13328e-12`。这是反向修复的数值依据，不等同于生产接线、全样本梯度或训练验证通过。提交 `29563a5` 的生产反向尚未采用该修复；后续接线和复验另行记录。

| v5及新增诊断证据 | SHA256 |
| --- | --- |
| CTBR v5 JSON | `39cdffe1ff4e9887b0f76cfcf184a8e93c0c37c888cd9321a0a6fbfa7588db24` |
| CTBR v5 NPZ | `e1bda154004071edf83f32663b444fda06d46e18fa31d82828a0087343c847a5` |
| CTBR v5实际harness | `c578a3164ef6698344da3e4523b57e487365c5cc8cbfcafecf0e707dd7d69ea7` |
| CTBR v5执行来源 | `f885d306ee22a3d7f18f452823634c6e54ecbda2cf63637e1a1000643bc7c5de` |
| Newton v5 JSON | `83399536b5135c5cd88f6c04695410524ede21e12e1defbdf41d2bba11cc15c1` |
| Newton v5 NPZ | `8ee13472e43787315edb126cb2a1706da0c70b16d3d9f37e207038c4cc63914c` |
| Newton v5执行来源 | `f5829fe32a03440300c0b157943f0aafc4ef2e5784d3c1df1dc5b9b74db566ab` |
| state9实际成本FD/反向残差诊断 | `a11696114e81d0eee142f6e62b711fe927f4b12082b571d3a7e3d68bafdec9ad` |
| state9保存矩阵校正诊断 | `d81628481fd973cc96bd2adf63dfa0478f58fdf6d7528e5107000ef714b3b0b0` |
| 27项精修器CPU日志 | `ad3e0ea1b08531f0058a9616b8a406f3d41df453280de1f69ea7cc9a77efe9ab` |
| 13项Actor/env CPU日志 | `764191529d5fff936102e542b1bfe59e1c0bf38af92e376ef242d1367cdd337e` |
| 生产前向精修模块 | `a9fb766fdd7da88ec50da80dead9f42457013efdf6d407711b2237841f17fa70` |

## b45fd01 的生产反向复验与真实导航

### 实际反向返回解与七成本 FD 复验通过

以下两份新报告均绑定 `b45fd01f124f647712e0463553a39c3e270dc9b0` 和共同 IR2 的 M4 final-v6。原输入、七成本语义、双步长 `1e-3/5e-4`、SQP60、ADMM32768、物理界及原 `1e-3` 梯度门槛保持不变；完整前向精修仍为四次 Newton、每次最多两次原系统校正。每个差分样本都必须真正完成原 outer/raw/polish 与物理检查，没有筛样。

| 新复验 | 完成求解数 | AD/FD 相对 L2，步长 `1e-3 / 5e-4` | 最终实际反向原 CSR 相对 RHS 残差 |
| --- | ---: | --- | ---: |
| [真实 Newton state9](../results/turbompc-port/ctbr-newton-state9-certified-fd-cu130-v1.json) | 29/29 | `3.64255e-7 / 9.11537e-8` | `2.74951e-15` |
| [四个原固定案例](../results/turbompc-port/ctbr-nominal-certified-adjoint-cu130-v1.json) | 116/116 | 最大 `3.59410e-7 / 8.98519e-8` | 最大 `7.75620e-15` |

两份报告的 `passed` 与 `actual_adjoint_original_residual_passed` 均为 true；state9 的 `cost_fd_gate_passed=true`，四例的 `ctbr_numerical_gate_passed=true`。state9 两步长 FD 彼此相对差 `2.73101e-7`；四例最大 `2.69558e-7`。实际 backward 捕获的独立 CSR 乘积给出未缩放绝对残差分别为 `2.36139e-16`、最大 `6.66134e-16`，均通过追加精度检查。state9 实际使用两次反向校正，四例的批量反向使用一次；默认允许最多四次并不表示每次都执行四次。首次未校正求解的较大残差仍完整保存，不改写为最终残差。

运行前后源码、HEAD、原输入、实际库、构建清单、M4 prerequisite 的全部守卫通过。两份报告明确限定为反向数值复验，**没有重跑240步模型闭环或600步 Newton 闭环**；前述 v5 闭环证据仍绑定 `29563a5` 和其记录的运行源码，不能把旧报告重新标成 `b45fd01` 闭环通过。[state9 数组](../results/turbompc-port/ctbr-newton-state9-certified-fd-cu130-v1.npz) [四例数组](../results/turbompc-port/ctbr-nominal-certified-adjoint-cu130-v1.npz)

### 真实 low 导航 v1：已失败且未完成

[v1 汇总](../results/turbompc-port/ctbr-real-low-v1-summary.json) 明确记录 `status=FAILED_INCOMPLETE`。标准 `run_ctbr_pipeline.py` 启动 `Isaac-IPC-Avoid-TurboMPC-CTBR-Direct`，固定成本、1环境、1个 episode、seed0、`reference_strides=[10]`、`goal_x=10.0`、`obstacle_distance=1.0`、`episode_length_s=35.0`；完整原传感器、SFC 与恢复链保留。启动时显存余量3554MiB，只检查容量，不要求 GPU 空闲。运行期间与独立 state9/四例数值复验及外部任务实际共享 GPU，墙钟时间不作通过标准。

本轮首次保存的未解样本为 outer status1、SQP60，stationarity `1.428115505852645e-7 > 1e-7`；raw QP status0，原 QP 已收敛。由于生产资格规则要求外层先成功，polish status1、iterations0，**精修未尝试**。因此 history 中的 infinity 是未尝试占位，不能解释成精修线性求解或 Newton 失败。[原失败快照](../results/turbompc-port/ctbr-real-low-v1/first-unsolved-seed-0.pt) [独立 CPU 核对](../results/turbompc-port/ctbr-real-low-v1-first-failure-independent.json)

独立 NumPy/complex-step 复得同一 stationarity，最大项在 stage15 的 vx；原射击等式约 `3.51008e-12`，SFC 违反约 `1.99108e-10`，精确 rollout 的 SFC 违反约 `1.99859e-10`，速度、输入及加速度原界违反为0。首个不可控位置 `p1=p0+0.1*v0` 对全部19个有效面严格可行，最小原始行 slack `.434463315`。原 raw.x 配同一对偶的非线性 stationarity 仍为 `1.16010e-7 > 1e-7`，不能把 raw 点替换为已成功的外层解。[首位置几何核算](../results/turbompc-port/ctbr-real-low-v1-first-failure-geometry.json)

在保留上述失败之后，独立 [CUDA 手工资格诊断](../results/turbompc-port/ctbr-nav-first-unsolved-polish-cu130-v1.json) 仅对保存的同一输入、outer 轨迹及原 raw 对偶调用现有精修器，显式强制 accepted mask；未重跑 SQP/ADMM、未改变生产返回值。1次 Newton、0次追加线性校正便使六证书最大值达到 `9.76996e-15`；[独立 CPU 后端交叉检查](../results/turbompc-port/ctbr-nav-first-unsolved-polish-cu130-v1.cpu-crosscheck.json) 的精修轨迹最大差 `1.77636e-15`。这证明该单点有界收尾有数值依据，**不是生产导航通过**：报告明确 `acceptance_gate_passed=false`，原 episode 失败保持。

原运行继续到最后完整日志第1000步，位置 `[5.53675318,-.00411817,.49353582] m`；第200、400、700、800、900步检查点有恢复制动。这些只是每100步日志，不能当成总失败或总恢复计数。确认已有门槛失败并保存独立诊断后，仅对本任务 PID1066700 发一次受控 SIGINT，标准 finally 正常关闭资源。child exit130、外层 launcher exit1；运行终止于 `2026-11-03 13:58:58 UTC`，没有生成完整 `records.json/metrics.json`。目标最终是否到达、终局碰撞及累计求解失败/恢复/约束违反数量均未知，不以检查点推算，也不伪造完整 episode。[完整原日志](../results/turbompc-port/ctbr-real-low-v1.log) [中止请求](../results/turbompc-port/ctbr-real-low-v1-interrupt-request.json)

最终 [标准来源报告](../results/turbompc-port/ctbr-real-low-v1-provenance.json) 保留72项源码前后一致、实际库与模块路径七项检查全部通过，HEAD 与 M4 prerequisite 哈希未变；同时诚实保留 `execution_completed=false`、`execution_provenance_passed=false` 与 `KeyboardInterrupt()`。来源一致不能覆盖任务未完成。全部 PT、日志、execution、并行记录及诊断均保留；下一版必须使用独立输出，按原 seed0/目标/场景重新完成验证，不沿用本轮成功标志。[执行 sidecar](../results/turbompc-port/ctbr-real-low-v1-execution.json)

| b45复验与导航证据 | SHA256 |
| --- | --- |
| M4 final-v6 | `2ecb230460447fffbe3912b4a4fd943cee65bfef86f7b66378fb6f258abe04b3` |
| state9 certified JSON | `ea21cd623adc5b646299825aecf8157f948d1beb416d6eea0583f4d843eddc44` |
| state9 certified NPZ | `d6c2c91e46dd430971600d57565f3a4ff30f23edaa8044e4a387ddbce5688db4` |
| 四例 certified JSON | `a907bcd4b79f6d27dab59033ccac18507010ba774e00ccd2c50ed8273c4e2ddd` |
| 四例 certified NPZ | `a48671c0a1735f707564a0e7fc93606d4e089e19d74b08ee8459d8a4e5653252` |
| nav v1 FAILED/INCOMPLETE 汇总 | `1c6ded495abe223b4a76e756f430fb94fee11da67b3b5877c450c7151624147d` |
| nav v1 标准来源报告 | `103ee4ab80cf34834c3dae66a2cf63f124f66f424713824b5a94abaa74a33f35` |
| nav v1 完整日志 | `10af70166e5d0ee3f9720fdbc3b81a6717071f8417a9269c2bbf6588be192cdc` |
| nav v1 首失败原 PT | `5ba1b7cc7fe1157d318b99904aa364b620da8f7b6c4eb473b66fa6cd3e9c3270` |
| nav 首失败手工资格 CUDA 诊断 | `26e79e508fa5bfe9dccdeb63cb61fcee162b9eb5aec306c298e632af3c3dbc6b` |

## 50858a03 的迭代上限收尾选项

真实导航 v1 的保存样本表明，有限且末 QP 已收敛的 SQP 迭代上限候选，可以在原问题上通过既有有界 Newton 收尾得到更严格证书。提交 `50858a03b59f71f5082d177beeafe7d4052d3267` 因此加入显式 `CTBRPolisher(finish_iteration_limit=...)`；精修器通用默认值为 false，CTBR Actor 的 `polish_finish_iteration_limit` 默认值为 true，并在 `solver_metadata()["polish"]["finish_iteration_limit"]` 记录实际值。该选项只改变哪些候选可以进入原精修阶段，不把迭代上限状态直接当成成功。[精修器](../src/isaaclab_ipc/turbompc/ctbr_polish.py) [Actor 配置](../src/isaaclab_ipc/rl/ctbr_actor.py)

启用时，除原 status0/raw0 行外，只额外允许 **SQP status1、raw QP status0、轨迹/原外层误差/路径 normal cone 均有限**的行尝试收尾。raw QP 未收敛、非有限值及其他 SQP 失败状态不会由此获得资格。Newton 仍最多4次，每次仍最多2次原 KKT 残差校正；原 SQP60、ADMM32768、全部容差、物理界与活动基规则不变。六项完整原问题证书仍须全部 `<=1e-11`，并在返回前再次检查实际返回轨迹的原 SQP/路径误差。只有这些检查均通过才能把最终 `status` 设为0；收尾失败返回状态4并遵守原零梯度规则。[资格与返回状态实现](../src/isaaclab_ipc/turbompc/solver.py)

原始各阶段结果分开保留，禁止用最后成功状态覆盖原始失败诊断：

| 字段 | 保留的阶段 |
| --- | --- |
| `MPCResult.sqp_status` | 精修之前的原 SQP 状态，使用 clone 保存 |
| `MPCResult.sqp_errors` | 精修之前的原 SQP 误差，使用 clone 保存 |
| `MPCResult.admm` | 未改写的原 raw QP 轨迹、状态、残差和乘子 |
| `MPCResult.polish` | 原13项精修诊断，包括每次 Newton/线性校正与六证书 |
| `MPCResult.status/errors` | 实际返回轨迹经完整检查后的最终状态与误差 |

未配置精修器时，新增 `sqp_status/sqp_errors` 为 None，通用路径不因该选项改变。标准 CLI 的失败快照递归保存 native dataclass，因此新增阶段字段也进入快照；历史 v1 仍对应未启用此选项的 `b45fd01`，不重写其原 outer status1。本节记录 `50858a03` 当时的实现及 CPU 检查范围：该提交的 final-v7 因 B16 P50 超限失败，当时没有启动导航 v2。后续 v2 仅在 `403fc2dd` 的 M4 final-v8 通过后，按原 seed0、goal10、场景/事件/35秒 episode 和独立输出启动，结果见下节；不能把选项实现或保存样本通过当作完整导航通过。

## 403fc2dd 的编译路径与重新验证范围

提交 `403fc2ddb19e753ec959771cd1a4fa593328ce72` 只把 `_matvec` 的编译中、同设备 strided float64、精确相等收缩维度路径改为 `(a * x.unsqueeze(-2)).sum(-1)`，使 Inductor 可以与调用处的其他运算融合。eager 以及其他 dtype、布局、设备或不符合条件的 rank/shape，仍执行原 `matmul` 表达式；错误的 singleton 收缩维度不会因乘法广播而被意外接纳。该改动不包含此前负收益的 native preparation 候选，不改变 CUDA/Gram 公式、求解预算、门槛或物理数据。[实现](../src/isaaclab_ipc/turbompc/solver.py) [局部数学回归](../tests/test_turbompc_matvec.py)

生产 solver SHA256 为 `314e9d672b93161be2bfdfdfb855c5d68a7db07d6baac9bc731e78d16131581c`。隔离候选的22项局部检查与17项既有 scalar/selection 检查共39项 CPU 测试通过，包含高精度独立数值期望、非连续和广播输入、空维度、非法输入与非有限值、编译 vmap 的一二阶导，以及实际 CPU Inductor 的非线性状态/梯度回归；独立来源检查确认相对 `50858a03` 只有 `_matvec` 函数体改变。[CPU结果](../build/turbompc/matvec-candidate/build/matvec-cpu-v1.json) [范围报告](../build/turbompc/matvec-candidate/build/matvec-scope-v1.json)

该表达在实数算术下等价，但编译后的浮点乘加和归约顺序可以改变舍入，不能声称 bitwise 一致。历史 CTBR v5 的116个数值样本、240步模型闭环和 Newton v5 的600步真实物理验证都实际启用了编译；因此不能用 eager 路径未变来继承这些闭环结果。正式 M4 final-v8 已完成新的400组成对测量，四批所有数值、梯度与性能门槛通过；仍须以新输出完整复验原非线性数值/模型、Newton、实际导航及训练协议，所有历史失败和 v5 来源保持原样。

以下为 [正式 final-v8](../results/turbompc-port/m4-ir2-accepted-final-v8.json) 的完整前向加反向延迟，每批100组、每端10次 warmup，单位 ms：

| Batch | A P50 | B P50 | B/A P50 | A P95 | B P95 | B/A P95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.429631 | 22.614397 | 0.604184 | 67.588478 | 41.632506 | 0.615970 |
| 16 | 54.734140 | 46.115457 | 0.842536 | 93.856179 | 62.204904 | 0.662768 |
| 64 | 54.331588 | 47.959563 | 0.882720 | 97.339104 | 65.238639 | 0.670220 |
| 256 | 69.467033 | 62.112384 | 0.894128 | 123.414796 | 82.603221 | 0.669314 |

全部计时样本的最坏 `u0` 差为 `6.55032e-15`，成本/初态梯度相对误差分别不超过 `5.97646e-10 / 3.55820e-10`，原物理违反不超过 `4.52922e-13`。328项源码和10项产物守卫通过，汇总 SHA256 为 `9fd2a6e8d5e160b8ed6cc2eff3428e755bd9646fd4908d192c2d984e3eeb9e41`。A 仅指明确披露的共同 IR2 角色，A0仍失败；共享 GPU 的本轮线性验收不代表非线性 CTBR 实时能力。

### 模型 v6：116个数值样本与240步模型闭环全部通过

[完整 v6 报告](../results/turbompc-port/ctbr-acceptance-cu130-v6.json) 在 `403fc2dd` 的实际编译路径上重跑四个原固定案例：hover、position_step、attitude_trajectory、terminal_sfc。原116个基点/七成本双步长差分求解全部完成，四例各60步、共240个模型闭环求解全部完成，`ctbr_numerical_gate_passed=true`、`model_closed_loop_gate_passed=true`。执行记录逐项确认案例、阈值、FD步长、SQP/ADMM、horizon/dt、物理模型及精修配置与原协议一致；没有更换较易样本、减少步数或过滤失败。[执行与协议对照](../results/turbompc-port/ctbr-acceptance-cu130-v6.execution.json)

本轮保留默认 `CTBRPolisher()` 的 `finish_iteration_limit=False`、SQP60/`1e-7`/strict raw0、ADMM32768/绝对`1e-8`/相对0、最多4次 Newton 与每次最多2次原线性残差校正。全部356个求解的最终/raw/polish状态均为0；实际最大 SQP 轮数43、末次 raw 迭代数16775。独立 NumPy 前向与原生模型最大差 `5.55112e-17`，作者相同物理参数前向最大差 `4.98962e-13`，模型 Jacobian 双步长检查通过。七成本 AD/FD 在 `h=1e-3 / 5e-4` 下最坏相对误差分别为 `3.594094484e-7 / 8.985265557e-8`，两步长最坏相对差 `2.695567687e-7`；活动集合检查全部通过。

| 模型 v6 案例 | 完整步数 | 最终位置误差 m | 最终速度误差 m/s | 末段位置/速度 RMS | 末段姿态 RMS rad |
| --- | --- | --- | --- | --- | --- |
| hover | 60/60 | `4.38156e-10` | `1.24345e-15` | `4.38159e-10 / 6.83447e-15` | 0 |
| position_step | 60/60 | `.00652465612` | `6.66159e-7` | `.00652381439 / 3.16988e-6` | `7.14945e-7` |
| attitude_trajectory | 60/60 | `.00640393139` | `.00132892962` | `.00644793217 / .00111709165` | `.00395246685` |
| terminal_sfc | 60/60 | `.00389986566` | `5.51498e-9` | `.00389985826 / 2.69844e-8` | 0 |

全部数值/模型求解的六项精修证书最坏值依次为：stationarity `9.65815e-12`、等式 `4.57035e-14`、原不等式 `2.77556e-17`、normal cone `2.77556e-17`、精确 rollout 违反 `1.27676e-15`、shooting gap `4.09227e-13`，均满足原 `1e-11`。独立数值 Jacobian 复算 stationarity 最大 `3.37486e-10`；独立 NumPy plant 的物理违反最大 `2.77556e-17`，与预测首步的状态差最大 `1.77244e-14`。四元数范数约 `1e-8` 的偏差与保留的逐步 `norm+1e-8` 一致。

本轮从 `15:16:03` 至 `16:09:49 UTC`，使用共享 GPU，包含四例全60步及全部原始 raw/refined 保存数据；不按其墙钟速度作通过标准。报告明确 plant 为独立 NumPy 离散 CTBR，末案例模型闭环在所有未来节点施加原 `x≤.2` 墙；开环 FD 仍使用原仅终端墙。它不包含 Newton、电机或真实碰撞场景，不代替下一节真实物理跟踪或失败的导航验收。

9项执行结果检查和4项执行守卫全部为真，绑定同一 M4 final-v8/HEAD403，实际 native/cuDSS 和源码前后哈希通过。报告 SHA256 `7140cf4377eed83c314b996c4ba7af7d5e9fc3949c6ae5a4d8c541e8c42e3bc6`，[完整 NPZ](../results/turbompc-port/ctbr-acceptance-cu130-v6.npz) `08a35b1e7cb326b941a31f35c6e5c6766bcfd27353dd68231613ae06d66aa6d4`，执行记录 `0a72f341fee704d6731f3f9418a3dd00b8a1d7de345adf1c121f940951fb2e0d`。v1…v5 和所有单点失败诊断仍原样保留。

### Newton v6：新编译路径的600步真实物理复验通过

本轮原样执行 [check_ctbr_newton.py](../scripts/turbompc/check_ctbr_newton.py)，其 SHA256 `65972e1f453170071542f3aab6a906f3c9ced9da8f05f56e34663ca6149f7efb` 与 v5 保存的 harness 完全一致：固定 seed2026/2027、200步 hover 与400步 position_step、默认 compile、原 SQP60/ADMM32768、原四步 Newton 与最多两次线性校正。它使用 `CTBRPolisher()` 的通用默认 `finish_iteration_limit=False`，不静默改成 Actor 的 true；这一轮验证真实 RateController/Thruster/Newton 跟踪，不覆盖新增 true 资格路径。后者须由公共 solve 保存样本及实际 CTBR Actor 导航/PPO 单独验证。

[Newton v6 报告](../results/turbompc-port/ctbr-newton-cu130-v6/report.json) 的 `passed=true`，两例各自完整完成200/400步，无预测状态回写。独立 [CPU结果汇总](../results/turbompc-port/ctbr-newton-cu130-v6.summary.json) 逐项确认模型、阈值、SQP、ADMM、精修配置、控制/物理步长及场景设置与 v5 相同；600条实际物理 trace、600条预测及每条13项精修资料均完整，600个polish状态全部为0。

| Newton v6 案例 | 实际步数 | 最终位置误差 | 最终速度 | 最大预测原约束违反 | 实际状态界违反 | 最小净空 |
| --- | --- | --- | --- | --- | --- | --- |
| hover | 200/200 | `6.05030e-17 m` | `2.41595e-7 m/s` | `7.99680e-14` | 0 | `.75 m` |
| position_step | 400/400 | `3.75825e-4 m` | `7.62822e-4 m/s` | `8.79040e-13` | 0 | `.75 m` |

全部600个精修结果的六证书最大值依次为：stationarity `9.58414e-12`、等式 `8.79040e-13`、原不等式0、normal cone 0、精确rollout违反0、shooting gap `8.79033e-13`，全部满足原 `1e-11`。完整 [NPZ](../results/turbompc-port/ctbr-newton-cu130-v6/traces.npz) 和 [执行来源](../results/turbompc-port/ctbr-newton-cu130-v6.execution.json) 保存了原harness、实际加载库、源码前后哈希、HEAD和M4绑定；328项M4源码、10项产物及额外运行文件守卫全部通过，没有沿用 v5 的历史源码差异例外。

运行自 `2026-11-03 15:14:12` 至 `15:18:15 UTC`，启动显存余量4667MiB，与外部任务及后半段模型验证实际共享 GPU，不要求空闲。结果仅验证规定样例的物理跟踪，不能据此宣称含传感器/SFC导航通过或墙钟100Hz。新报告 SHA256 `3ff2a904da92df3344a966ec88700dd1666a430e33228fc585cb903785e692f5`，NPZ `b6672bbee5aa732926ccd35d00a4f14b0da74e121d6400b943e19a4f8302b01f`，执行来源 `a57fb0e0d7fd8f12b2c13a097d08d978859ea84dcf593d7bfe46514e74e0a762`。

### 新资格路径：原导航失败点的公共 solve 与七成本 FD 通过

[公共 solve 的29次复验](../results/turbompc-port/ctbr-nav-first-unsolved-certified-cu130-v1.json) 直接使用导航 v1 的保存初态、全部物理约束与成本值，在 `403fc2dd` 上调用真实公共求解接口，启用 `finish_iteration_limit=True`；没有手工替换 accepted mask。builder七个输入块及初猜逐项精确相等，原 SQP60、ADMM32768、四步 Newton、六证书 `1e-11` 与双步长成本 FD 门槛不变。

基点仍保留原 SQP status1、60轮、stationarity `1.428117686e-7`，随后1次 Newton 将最终/raw/polish状态全部置于实际通过的0；原 `sqp_status/sqp_errors` 未被成功状态覆盖。全部29个样本中25个原 SQP status1、4个原 status0，每个样本均用1次 Newton，六证书最大值 `2.46470e-14`，独立物理违反最大 `8.88178e-16`。29次调用均逐字段确认原 raw 的11项数据未改写，活动集稳定，最小非活动间隔 `.01850346`。[独立完成核对](../results/turbompc-port/ctbr-nav-first-unsolved-certified-cu130-v1.execution-completion.json)

七成本 AD/FD 在 `h=1e-3 / 5e-4` 下相对L2误差分别为 `3.76204e-7 / 9.40522e-8`，两步长差 `2.82151e-7`；实际反向使用1次原系统校正，独立1568维 CSR 复算的最终绝对残差 `3.14744e-16`、相对原 RHS 残差 `3.66475e-15`。全部16项运行源码、HEAD、输入、库、构建清单和 M4 v8 来源守卫通过。JSON SHA256 `a662e04de39f43e6effaf322d9550636a0c98d3df8b340982a7809b25e15306d`，NPZ `1ab18da46300fab1c8ce1cd595a3e5d18cef6ab981687d5ea81c564a705d8668`。这是保存失败点的生产资格和梯度复验，报告明确 `navigation_acceptance_passed=false`，不能代替完整导航或改写 v1 失败。

### PPO v3：一次实际更新与冻结快照重放通过

[PPO v3 独立验收](../results/turbompc-port/ctbr-train-smoke-v3-acceptance.json) 在 `403fc2dd`、M4 final-v8 和实际 CTBR Actor 上执行1环境、16步 rollout、1次 PPO iteration、2个 minibatch、1个 epoch。43项检查全部通过。实际调用为 rollout 的16次 B1、update 的2次 B8、reload 的2次 B1，共34行前向；全部行 SFC 有效、原生求解接受、无 braking，最终和 raw 状态均为0。update/reload 阶段的 `record_mpc_result`、环境 step/reset、Planner context/cloud/recovery 调用数全部为0，冻结快照反向与重载没有重复推进规划器。[分相原始审计](../results/turbompc-port/ctbr-train-smoke-v3/training-audit.json)

两次真实损失反向分别保存完整 `8×7` 成本 VJP，16行成本梯度均有限且非零；12个 MLP 参数 hook 均有限，其中8个非零。记录的最大参数改变量为 `2.001340762944892e-4`。独立审计逐值复算成本 VJP，并核对 checkpoint 的12个 optimizer state 均在 step2、与保存的 `model_0` 逐 tensor 相同；同一份克隆观测上的确定性动作重载比较通过 `atol=1e-6, rtol=0`。[训练原始 metrics](../results/turbompc-port/ctbr-train-smoke-v3/metrics.json)

本轮最大 stationarity 为 `5.92593e-12`，原等式误差 `2.48246e-13`，原不等式和 normal cone 误差均为0；raw ADMM 的原/对偶残差最大分别为 `7.13209e-9 / 2.67955e-9`。Actor 的 `finish_iteration_limit=True`、七成本/127维 checkpoint 契约、原预算和容差均保留；72项项目源码、实际加载库、HEAD 与 M4 来源守卫通过。[执行来源](../results/turbompc-port/ctbr-train-smoke-v3-provenance.json)

这些结果证明一次真实 PPO 更新确实经过原生求解器并产生可学习的成本梯度，**不代表训练后策略或导航已通过**：本轮完成 episode 数为0。完整参数梯度张量、初始参数及重载前后动作数组未持久化；参数 hook、delta 和动作相等属于来源固定的进程内断言，不能声称已独立重算这些全部差值。完整成本 VJP 可逐值独立复算。成功行保存的是最终/raw 诊断，不含完整 pre-polish 轨迹，不能由迭代计数倒推收尾晋升次数。验收 JSON SHA256 为 `dfdee181b6b89a317767923fc87cf7ae226e47a9c42b6452118d35ac24857a2d`。

### 真实 low 导航 v2：原协议失败且未完成

Newton v6 完成后，标准 CLI 于 `2026-11-03 15:20:13 UTC` 启动 [nav v2 独立记录](../results/turbompc-port/ctbr-real-low-v2-launch.json)，固定成本、1环境、预期1个完整 episode、seed0、stride10、goal10、obstacle1、35秒，以及原传感器/SFC/恢复链全部保留。启动命令与 v1 的比较确认只有输出路径改变；运行版本绑定 `403fc2dd` 和 M4 final-v8，实际 Actor 配置默认 `finish_iteration_limit=True`，保留原 SQP/raw/polish 各阶段。启动显存余量3390MiB，与模型验证、随后公共29次求解和 PPO 更新存在记录在案的共享 GPU 时段；本轮不做延迟验收。

[派生验收](../results/turbompc-port/ctbr-real-low-v2-acceptance.json) 明确 `navigation_acceptance_passed=false`、`status=FAILED_INCOMPLETE`。第300步是最后一个完整的100步日志检查点，位置约 `[4.58058,-.00396168,.998924] m`，该点 solved=true/braking=false；随后保存首个求解失败快照，快照没有精确 episode 步号。已观察到原零失败门槛被触发并完成原问题不可行证书后，仅向核实过命令、启动时间和工作目录的本轮 PID 发送一次 SIGINT。标准 wrapper 在 finally 中关闭资源、保存来源报告，child exit130、外层 launcher exit1，于 `16:01:59 UTC` 结束。[受控终止记录](../results/turbompc-port/ctbr-real-low-v2-interrupt-request.json) [完整日志](../results/turbompc-port/ctbr-real-low-v2.log)

`finalized=true` 仅表示终止证据已落盘；`completed=false`、`episode_complete=false`、`completed_episodes=null`。本轮没有生成终局 `records.json/metrics.json`，因此 goal、collision、timeout、累计失败/恢复次数及实际物理违规总数均为未知，不能写成成功、0次或完整 episode。72项项目源码、328项 M4 源码、10项产物、HEAD 与实际加载库守卫全部通过；原标准报告的 `execution_completed=false` 和 `execution_provenance_passed=false` 仍按中断事实保留，未用来源守卫通过冒充运行完成。[原 provenance](../results/turbompc-port/ctbr-real-low-v2-provenance.json) [M4 与执行绑定](../results/turbompc-port/ctbr-real-low-v2-execution.json)

#### 首个失败的阶段与物理问题

[首个原始 PT](../results/turbompc-port/ctbr-real-low-v2/first-unsolved-seed-0.pt) 的 SHA256 为 `65599edcb1b3759766794c8229a5322914cf4bb7c65471b2fc89708a18618d0f`，保留初态、七个 QP 输入块、全部 SFC 面、原外层/raw 轨迹与乘子、原阶段状态和精修占位诊断。其位置为 `[6.14574242,-.00549539,1.00017715] m`，世界速度为 `[1.69878960,-.04410963,.03749004] m/s`，WXYZ 四元数为 `[.99731284,.05531625,-.04799078,.00205034]`。原两走廊分别有22/14个有效面，原15步赋值是前6步走廊0、后9步走廊1。[只读解析](../results/turbompc-port/ctbr-real-low-v2-first-failure-quickparse.json)

| 保存阶段 | 实际诊断 |
| --- | --- |
| 原 SQP | status1，60轮；stationarity `8.53386e5`、等式误差 `8.51283e-5`、原不等式违反 `.0470510171` |
| raw ADMM | status1，32768轮；原/对偶残差 `.0469854547 / .562072678`，原容差均 `1e-8` |
| polish | status1、reason1、0次 Newton，即未尝试；六证书与线性历史中的 infinity 是占位值 |
| Actor | SFC valid=true、solved=false、braking=true；请求恢复动作，不接受该失败预测 |

这与 v1 的 raw0、略超 stationarity 的失败不同：新样本 raw QP 未收敛，按公开资格规则不会进入 finisher。没有发生精修线性系统失败，不能把 infinity 占位误读成 Newton/KKT 误差。最大失败约束是 `x5` 上的走廊0第18面（G row25）；原外层、raw 与从真实初态独立精确 rollout 的同面违反分别为 `.0470510171 / .0469854547 / .0475565449`。这是未接受的预测违反，不能直接宣称执行物理状态撞障碍。[独立几何核算](../results/turbompc-port/ctbr-real-low-v2-first-failure-geometry.json)

现有 CLI 字段 `maximum_qp_primal_residual` 在 CTBR 路径表示返回轨迹的原等式/不等式误差最大值，名称沿用旧接口；它不是 raw ADMM primal residual。本轮缺少终局累积记录，仅首快照同时保存了这两种量，派生报告分别解释，未从一个量推断另一个量。

#### 可达性证书与两走廊 membership

CPU 核算先排除了首步固定前缀矛盾：`p1=p0+.1*v0` 的22个原面全部严格满足，最小归一化内侧距离 `.332972164 m`；对 `p2`、首步真实加速度和速度界求交后，允许总推力区间仍为完整的 `[.840857142857143,9.459642857142859] N`。这只证明前两步的必要条件没有矛盾，不能证明后续可行。[前缀有理数证书](../results/turbompc-port/ctbr-real-low-v2-prefix-cpu-v1.json)

随后对所有原允许控制构造独立姿态外包络。令 `f=R_unit(q)^T n`、`L=|fx|+|fy|`、`Z=|fz|`，在原机体系角速度盒 `±[1.5,1.5,.1] rad/s` 内，有 `D⁺L≤.1L+3Z`、`D⁺Z≤1.5L`。非负比较系统的矩阵指数用精确有理数 Taylor 和严格正尾界包住，结合原正推力端点、原四元数范数及 `p5=p0+5dt*v0+dt²(4a0+3a1+2a2+a3)` 得到：原面法向上 `n·p5≥5.130211719718089`，但原上界仅为 `5.096769235324367`，所以任意允许控制都至少违反原行 `.03344248439372146`，归一化距离下界 `.0440113753 m`。[stage5 全控制必要界](../results/turbompc-port/ctbr-real-low-v2-stage5-reachability-cpu-v1.json)

该证明保留原 raw-Hamilton 加速度和逐步 `norm+1e-8`，没有把保存四元数替换为单位四元数；小角多项式的等效角不大于原旋转角，仍在原角速度盒内。速度/加速度约束只在构造外包络时省略，扩大可达集合后仍不可行，因此加入这些原约束不能恢复可行性。证明范围是保存二进制输入对应的数学离散模型，显示用归一化距离另用 float64；没有运行局部 NLP、变更原数据或把数值局部失败当成不可行证明。

为判断是否仅走廊切换过晚，独立对 reference、外层/raw 轨迹和两条精确 rollout 的每个未来位置，计算两个走廊全部原有效面的最大 `n·p+b`。下表列关键节点；负值表示该点在对应走廊全部面内，完整15节点/所有轨迹见 [membership 报告](../results/turbompc-port/ctbr-real-low-v2-first-failure-membership.json)。

| 节点 | reference 对旧/新走廊最大行值 | 精确 outer rollout 对旧/新走廊最大行值 | rollout membership |
| --- | --- | --- | --- |
| x3 | `-.336630 / -.356111` | `-.032582 / +.136198` | 仅旧走廊 |
| x4 | `-.314120 / -.482051` | `+.031396 / +.185028` | 两者都不属于 |
| x5 | `-.214436 / -.512216` | `+.047557 / +.190412` | 两者都不属于 |
| x6 | `-.063585 / -.411365` | `+.005212 / +.134464` | 两者都不属于 |
| x7 | `+.087267 / -.310513` | `-.104053 / +.002269` | 仅旧走廊 |

reference 在 x3…x6 位于交集，不能据此认定这些位置在原动态边界下可达。进一步遍历两走廊全部22+14个面，得到 **x4 对两走廊的联合不可达证书**：旧走廊第18面的原行违反下界 `.0283741012`，新走廊第11面的下界 `.0354123864`。因此在这个保存状态下，即使允许任意改变旧/新走廊时序，x4 也无法进入二者并集；只做 retiming 不能修复当前原问题。其余没有得到正违反下界的节点标为 UNKNOWN，不能称为可行。[两走廊全控制外包络](../results/turbompc-port/ctbr-real-low-v2-corridor-reachability-cpu-v1.json)

以上诊断没有修改生产状态、几何、约束、样本、预算或门槛。该轮导航结果仍为失败/未完成，当时尚未验证后续规划修复。派生验收 SHA256 `5fcbea047b1e756ef22d497b70e127072c2b9f43df13914cda95ddbbe0d46c0e`，stage5 证书 `fa002572dcef9fde8e4ccb667ab79feb3296fdab6e80c9fc0009037caf804625`，两走廊证书 `213975d6920a3ade9aab1e1e57c6e3ad0c50a883caa1ec4221cdeb86498650cb`。原 v1/v2 失败及所有原始文件继续保留，不能作为完整 M5 通过的证据。

## 独立参考调节候选 3f9cd34：训练通过，导航 v3 未完成且不可行输入已确认

候选分支 `feat/turbompc-ctbr-reference` 的提交 `3f9cd34719b54b937c13731e022cbc37c35393e3` 只改变 CTBR 参考调节与观测打包前的接线，并增加测试；原生 SQP、CUDA ADMM/cuDSS、隐式反向、Actor、CTBR 模型及物理界仍与 M4 测试提交 `403fc2dd` 一致。候选与基线分开冻结，不把它重新命名为原版 A，也不改写旧导航 v2 的失败。[共同候选冻结记录](../results/turbompc-port/ctbr-reference-candidate-v1-freeze.json)

该候选用当前实测状态、当前原 SFC 和原路径，生成一条最长80步的规范离散 CTBR 制动轨迹。每步检查原输入、速度、真实加速度和同一 polytope 的全部面；有限近悬停终点只是这一条轨迹的证据，不是无限时域安全集。实际状态的见证与改变路径切向速度的虚拟参考见证分别保存；后者只用于固定9点速度网格的参考选择，不证明实际 Newton 跟踪、返回的完整 MPC 时序或未来未见障碍安全。观测在打包前适配，PPO 更新及重载继续使用冻结快照，不重复推进 Planner。

### 候选 PPO v4：一次实际更新通过

同一候选冻结记录下，原1环境、16步 rollout、1次迭代、2个 B8 minibatch、1 epoch 和两次重载协议已执行。独立 CPU 审计49项全部通过：全部34个 Actor 行 valid/solved/unbraked、最终/raw 状态0；两份实际 `8×7` 成本 VJP 逐值复算后16行均非零且有限，12个参数 hook 均有限、8个非零；两个优化器步、checkpoint 一致性以及 update/reload 无 Planner/env 重入通过。73个项目来源、144个候选冻结来源及实际库守卫通过。[PPO v4 独立验收](../results/turbompc-port/ctbr-train-smoke-v4-acceptance.json)

这只证明一次更新链，完成 episode 数为0；成功行记录未保存 governor 明细，不能从导航记录推断该训练中的参考适配次数。完整参数梯度、初始参数和前后动作数组未保存的限制与 v3 相同，参数差值和动作一致性仍按来源守卫下的进程内断言核验。验收 SHA256 为 `38e7d06a87c82bbb88ea1271d3ddc9a30727e577059ea972ccf2b135a3bd9ffd`，不能据此声称训练策略或导航通过。

### 导航 v3：390步已执行，首次新几何已超出全部原控制的可达集合

导航沿用原 seed0、stride10、目标10m、障碍距离1m、35s、1环境和固定成本，SQP60/ADMM32768、容差与物理界均未改变。共享 GPU 上自 UTC16:49:29.136023 开始，逐调用只读记录保留同一次实际观测、完整路径/nearest、参考调节明细和 solver 结果。392次观测中前390次求解已返回并执行动作：最终/raw/polish 状态均0，无 braking、无无效 SFC，实际命令/推力/状态违反计数均0；原 SQP 状态为389次0和1次1，后者通过既有有界收尾及原证书，原阶段没有被抹除。[导航 v3 完整派生验收](../results/turbompc-port/ctbr-real-low-v3-acceptance.json)

第一次适配发生在 t≈0.30s，将参考2m/s降到1.5m/s，全部前缀共5次观测适配。到 t≈3.89s，实测位置 x≈6.11825m，原当前几何仍可认证15步实际制动轨迹。下一次 t≈3.90s 的观测392把路径由206点改为231点、nearest 由6改为0，SFC 改为19+12个面，原时序为旧走廊6个 slot 后接新走廊9个 slot。此时实际位置为 `[6.13529301, -.005687929, 1.00066555]`，速度为 `[1.70149672, -.0433145314, .0448102988]`；实际制动见证在第4步离开所有单走廊，返回 reason5 并保持 nominal 参考、所有原面和时序不变。[前后同源几何记录](../results/turbompc-port/ctbr-real-low-v3-first-geometry-cpu-v1.json)

第一预测位置 `p1=p0+0.1v0` 对两块原走廊仍全部满足，最小法向距离分别为 `.341978109m` 和 `.051652206m`。但对这**首次失去实际见证的新几何**，复用独立有理数正比较系统与 Taylor 余项包络，遍历全部31个面及节点1…7，得到节点4的严格全控制下界：

| 原走廊与面 | 所有原允许控制下的 `n·p4` 下界 | 原面上界 | 必然原行违反下界 | 法向距离下界 |
| --- | ---: | ---: | ---: | ---: |
| 旧走廊 face14（原 slot3 / G row21） | 5.182164538954096 | 5.164368681086063 | .01779585786803274 | .023096991808053743m |
| 新走廊 face5 | 3.3814665907233885 | 3.3558011681258675 | .02566542259752116 | .029978762338189854m |

因此节点4不可能进入两块原走廊的并集；任意旧/新切换时序也不能从该保存状态修复问题。这个结论来自对全部原控制范围的保守外包络，未把一条启发式制动策略失败当作不可行证明，未替换 raw quaternion、放宽原平面或更改状态。[首次新几何可达性证书](../results/turbompc-port/ctbr-real-low-v3-first-geometry-reachability-cpu-v1.json)

Root 在证书核对后只向该次启动身份已验证的 PID1190009 发送 SIGINT。第391次求解尚未返回，在原生调用内受控中断；其 SQP/raw/polish 状态、残差和迭代数一律未知，不能声称 raw1 或60次预算耗尽。该次输入尚未执行下一动作。标准 wrapper 与 observer 的 finally 已完成，child130、outer1；原 `execution_provenance_passed=false` 原样保留。没有 episode 终局 records/metrics，故 goal/collision/timeout、完整 episode 数及其总计保持 null，结果是 **FAILED_INCOMPLETE、navigation_acceptance_passed=false**。[中断请求](../results/turbompc-port/ctbr-real-low-v3-interrupt-request.json) · [执行记录](../results/turbompc-port/ctbr-real-low-v3-execution.json) · [逐步记录汇总](../results/turbompc-port/ctbr-real-low-v3-trace/summary.json)

原73个运行来源、144个候选来源、328个 M4 来源和10个构建产物，以及实际加载库全部守卫通过；334个原始/诊断产物重哈希保持一致。派生验收 SHA256 为 `9b609108c71ca65920d91dc1838a7634cc1ae03c02c3d26ec3e57e55f9217772`，首个新几何快照为 `e6cfaccde65530a308dc0356cd21bb647f0f2f76399e159e7f7f6c342755fb9b`，全控制证书为 `fe8e503c269c36c2654071c14b05c70b628f70763624e2dea50dadbbad7da148`。旧几何中的有限制动见证没有提供未来感知更新的动态余量保证；该轮导航未通过。后续更保守参考策略的独立完整验收见下节，原失败不改写。

## 参考余量策略 V2：完整导航 v4 与 PPO v5 通过

冻结候选 `1a71d692c2eb2b3ca540146def685d7879e5852b` 在上一候选上加入明确的保守参考策略。它只读取当时实际状态、原 SFC 和原路径，不使用障碍事件的未来位置或触发时刻，不改变实际状态、模型、成本、物理界、原平面、求解预算或验收门槛。原生 SQP/ADMM/KKT、Actor 与 CTBR 模型仍与已测核心 `403fc2dd` 字节一致；新的参考前端单独冻结。[V2 共同冻结记录](../results/turbompc-port/ctbr-reserve-candidate-v2-freeze.json)

对每块原 polytope，按当前实际位置计算水平截面的中心内接圆半径：`r=min_i (-(n_i·p+b_i)/hypot(nx_i,ny_i))`。只把水平法向非零的面纳入半径，纯上下盖仍保留在全部三维平面检查中。**实际状态的制动见证保持上一版规则**；固定9点速度网格中的虚拟参考见证，额外要求全轨迹最大水平位移不超过同一块 polytope 的该半径。候选不能从另一块借用更大半径，也不能用虚拟速度替代实测速度作实际可行性证书。所有原输入、速度、加速度、完整三维面的逐步检查以及最长80步有限终点条件保持不变。

这是有意新增的参考选择策略，有限制动轨迹只支持当前几何下的一条局部见证，不认证返回的完整多走廊 NLP、实际 Newton 跟踪、未来几何或无限时域安全。19项 CPU 数学/失败样本测试和7项 env 接线回归通过；保存的旧不可行状态仍为实际见证失败并返回原参考，没有被重新标成可行。[V2 CPU 记录](../build/turbompc/ctbr-reserve-candidate/build/turbompc/ctbr-reserve-cpu-v1.json) · [独立设计审查](../build/turbompc/diagnostics/ctbr-reserve-independent-review-v1.json)

### 原协议完整导航：1076步到达目标

2026-11-03 UTC17:16:49.470874 至17:30:03.335600，标准 CLI 在共享 GPU 上完成原固定成本 low 协议：seed0、stride10、目标 `[10,0,1]m`、障碍距离1m、35s上限、1环境和1个完整 episode。预测 H15/dt0.1、控制周期0.01、物理步长1/300、m0.75/g9.81、SQP60/1e-7、ADMM32768/绝对1e-8、严格 raw 收敛、4轮精修/1e-11及最多2次线性校正均未改变。Actor 保持已验证的 `finish_iteration_limit=True`。child/outer 均以0退出，标准 wrapper 与逐调用记录完整落盘。[原 records](../results/turbompc-port/ctbr-real-low-v4/records.json) · [原 metrics](../results/turbompc-port/ctbr-real-low-v4/metrics.json) · [独立验收](../results/turbompc-port/ctbr-real-low-v4-acceptance.json)

| 原门槛或完整记录 | 实际结果 |
| --- | --- |
| 完整 episode / 控制步 | 1/1，1076步；没有筛除失败行或提前结束 |
| 目标 / 碰撞 / timeout | success=true、goal_crossed=true；collision=false、timeout=false |
| 原命令/推力/状态违反 | 各0次 |
| QP失败 / 前端失败 / 恢复 / 制动 | 各0次 |
| 接受且含 SFC 的步数 | 1076/1076；单步最多31个面 |
| 最小物理步净空 | `.2194174826145172m` |
| 返回轨迹原等式/不等式最大误差 | `2.55351295663786e-15` |
| 原 raw ADMM primal / dual 最大值 | `9.995746452773346e-9 / 9.898463252878995e-9` |
| 精修后 stationarity 最大值 | `9.370282327836321e-14` |
| 六项精修证书 | 全部1076行均不超过原 `1e-11` |

终点位置为 `[9.531530380249023,-.13249996304512024,1.0132378339767456]m`，距目标 `.4870269654597974m`，满足任务原有的0.5m目标球。沿用字段 `goal_crossed=true` 表示这一任务成功条件，**不是 x 已越过10m**。沿用字段 `maximum_qp_primal_residual` 表示返回 CTBR 轨迹的原等式/不等式误差，不能与 raw ADMM primal 混为一项。

5942条原 trace 包含1078次观测、1076次开始/完成求解、1076次结果记录、1076次实际动作和560份几何快照。全部返回行的 final/raw/polish 状态均为0；原 SQP 状态单独保留为1075次0和1次1。唯一达到 SQP60 上限的是 solve974/obs975，原 stationarity `1.6261975366305137e-7`，原 equality `6.108169525731455e-12`、原 inequality0；raw0且满足既有收尾资格，之后通过原六项证书。没有把原 SQP1 改写成原阶段成功。全部13个 polish 诊断字段保留，本轮最多1次 Newton，额外前向线性校正最多0次。[完整 trace 汇总](../results/turbompc-port/ctbr-real-low-v4-trace/summary.json)

参考策略从初始观测就将 nominal 2m/s 选为1.25m/s：同块半径 `.5498481475m`、选定虚拟轨迹最大水平位移 `.4880129130m`，实际制动见证通过。首次障碍激活后出现重大新路径/几何的 obs592，t≈5.90004s，实测 x≈5.96598m、vx≈1.09933m/s；实际见证仍通过，solve591 的 final/SQP/raw/polish 均0、stationarity约 `5.76e-14`。这些是新闭环实测结果，不是对旧不可行样本作了状态替换，也不构成任意未来障碍的保证。[对应几何快照](../results/turbompc-port/ctbr-real-low-v4-trace/geometry-00330.pt)

**reason6 的32次观测全部纳入验收**：obs678…709、solve677…708，t=`6.760059834…7.070066929s`。这些时刻实际状态制动见证均通过，但固定虚拟网格没有候选满足额外水平余量条件，因此按既有分支保持 nominal 参考。诊断没有被清零，也没有通过放松平面或物理界接受动作；对应32次原生求解、实际动作和原门槛全部通过。全部1078次观测的实际见证均通过，观测 reason 统计为1046次1和32次6；其中执行输入为1044次1和32次6。`actual_backup_matches_returned_schedule=false` 的保守返回值不等于已证明实际约束违反，也不能反向宣称多走廊时序获得了制动证书。

独立验收的31项原门槛/来源检查全部通过，578个原始及诊断产物重新哈希一致；73个运行来源、144个候选来源、328个 M4 来源和10个构建产物，以及实际加载库均匹配冻结记录。没有首次失败快照、异常/失败中止事件或未完成求解。该证据覆盖这一个固定 seed、固定成本低速 episode，不是多 seed 成功率或训练后策略评估。[执行守卫](../results/turbompc-port/ctbr-real-low-v4-execution.json) · [运行来源](../results/turbompc-port/ctbr-real-low-v4-provenance.json)

原计时包含首次编译、逐步记录和共享 GPU 条件：wall `774.1717577s`，Actor平均 `613.4606010ms`，step平均 `718.5051414ms`。本轮不按延迟判定 M5 通过；仿真控制周期0.01s不代表100Hz墙钟运行。线性 M4 的独立性能结论也不能直接外推到这里的非线性 CTBR 闭环。

### 同一候选的 PPO v5：一次真实更新通过

原1环境、16步 rollout、1迭代、2个 B8 minibatch、1 epoch及两次同快照重载协议完整通过，独立 CPU 审计49/49项为真。16个 rollout B1、2个 update B8和2个 reload B1共34行均 valid/solved/unbraked，最终和 raw 状态0；两份实际 `8×7` 成本 VJP 可逐值重算，16个更新行均非零且有限。12个参数梯度 hook 均有限、8个非零；参数更新、checkpoint 与两次 optimizer step、update/reload 不推进 Planner/env 的检查通过。共同候选冻结、73个来源及实际库守卫全部通过。[PPO v5 独立验收](../results/turbompc-port/ctbr-train-smoke-v5-acceptance.json)

本项完成 episode 数为0，只证明一次真实成本学习和 checkpoint 链路，不是训练收敛。成功行未保存 governor 明细，故 `applied_count=null`，不从导航反推训练中的适配次数。完整参数梯度、初始参数和前后动作数组未持久保存，参数差值和同观测动作一致性按来源守卫下的进程内断言核验，不能宣称独立重算这些差值。随机探索动作仍由实际环境检查，不能从优化均值诊断推断 episode 安全。

### 最终限定验收与版本边界

[最终组件索引 v2](../results/turbompc-port/native-port-completion-v2.json) 汇总七个独立验证域，全部通过，`qualified_migration_acceptance_passed=true`，来源/库/产物守卫全部通过。该索引复用各域已有完整证据，没有启动新的 GPU 测量。核心 M4 仍绑定 `403fc2dd`，新参考前端和本次导航/PPO 绑定 `1a71d69`；其余组件的继承按实际源文件字节或明确函数 AST 守卫判断，不把新版前端冒称为旧600步 Newton 验证时的前端。

共同 IR2 的基准资格保持显式：`unmodified_upstream_m4_passed=false`，未修改上游 A0 的失败没有被新成功覆盖。原导航 v1/v2/v3、原模型/精修失败和性能负结果也全部保留。当前通过范围为规定的原生迁移数值/性能、SFC与CTBR独立数值和闭环、一次原协议低速导航及一次 PPO 更新；没有扩展为森林/多 seed 导航、训练策略收敛、未见动态障碍安全或非线性墙钟实时能力。

| 本节最终证据 | SHA256 |
| --- | --- |
| `ctbr-reserve-candidate-v2-freeze.json` | `2d7dce0bc8861acf61bee79990b4a07209e948bbea39f409330ed771594254b6` |
| `ctbr-real-low-v4-acceptance.json` | `e56ee5fdc8687d37bec42ef42526d05e2a2dee2dbb3c66c1094ed2899139c526` |
| `ctbr-real-low-v4/records.json` | `5ea15f98e997a6fe4461cacde6ac95c7b8d4b326000e62e9d6610f4b499843f8` |
| `ctbr-real-low-v4/metrics.json` | `7c4169340d9a07deb61cf60c26aedf061c903b1ea5c875c564bf07dbbc166ee2` |
| `ctbr-real-low-v4-execution.json` | `96b049d522959493d952600d6bccdc7a0234df779a90290124374b810aacea5d` |
| `ctbr-train-smoke-v5-acceptance.json` | `008d2f44ec8e61e687cd6a00d34fe87941e22c82004c02aa9ebc434e30fb36dc` |
| `native-port-completion-v2.json` | `8dc396aeb05e8a363d613db58721581f6b8e357cdda46e05bac4a5dfe04c8dbb` |

## 证据索引

路径相对于工作树。运行结果在本地 `results/`，不随 Git 提交；源码、用例和作者 fixture 随项目版本管理。

| 文件 | SHA256 |
| --- | --- |
| `tests/data/ca_acmpc_forward_reference.npz` | `2b5691c389f79dc29da64e5415d5ab2ff0423f70f238a12925be93309cb58e90` |
| `src/isaaclab_ipc/turbompc/quadrotor.py`（v1 数值执行） | `eb2f2428a57225f99b94bff6bd299ce2f949c7abdc5b11921b978c5aea2a1832` |
| `src/isaaclab_ipc/turbompc/ctbr.py`（v1 数值执行） | `c084c5131b3ec265cfc6bda41bf49c72ee69e93cdfb5e115a706a5f5ca52273f` |
| `results/turbompc-port/ctbr-acceptance-cu130-v1.json` | `b5326841c5a6174164da378b86689389766b7e3b3c6ad4d1d41a0dc5fd9986bd` |
| `results/turbompc-port/path-constraints-cu130-v1.log` | `7b04c2b621f828aa980e40bbf0e6c8860e177d04bb2f9d1c5d7606d458248626` |
| `results/turbompc-port/ctbr-actor-cpu-v2.log` | `358cd4fa7fa0d4ba8c189a4159e859ff30f223b220e0ea9b2216f3ac0d8a8a4b` |
| `results/turbompc-port/ctbr-actor-cu130-v1.json` | `a3f0d950fee0885d50d72d415aaf3fbb0c672a4ac4ea0d0788c5ef93d410f91c` |
| `results/turbompc-port/ctbr-actor-cu130-v2.json` | `33593ffd5a128604e2c94068ff92a04693ae42a2a9d33a0cbf2e095dadc54162` |
| `results/turbompc-port/ctbr-actor-vjp-cu130-v1.json` | `e9c83f2556e9e3861c6919ccddd6db50f70bc64a2636725fd88b78c495788e2e` |

## 重跑接口检查

在 verified 工作树使用项目环境；GPU 检查与本任务其他 GPU 验证协调执行，共享 GPU 上既有训练无需停止。

```bash
env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WARP_CACHE_PATH="$PWD/build/warp-cache" \
  /home/tong/tongworkspace/free/isaaclab-IPC/.venv/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/test_turbompc_ctbr.py \
  tests/test_turbompc_ctbr_actor.py tests/test_turbompc_ctbr_polish.py \
  tests/test_turbompc_polish_integration.py -k 'not cuda'

bash scripts/turbompc/env.sh cu130 env PYTHONPATH="$PWD/src" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 WARP_CACHE_PATH="$PWD/build/warp-cache" \
  /home/tong/tongworkspace/free/isaaclab-IPC/.venv/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/test_turbompc_ctbr_actor.py -k cuda
```

作者源码只读位于原工作树 `build/ca-acmpc-reference/`。独立审计必须核对其 commit 与源文件 SHA，并为每次运行使用新的 JSON/NPZ 输出名称；CPU preflight、base-only、完整数值、模型闭环和 Newton 闭环的通过标志分别解释。Actor 的局部梯度与快照重放检查不代表 PPO 训练有效性、导航成功率或 100 Hz 墙钟实时能力。
