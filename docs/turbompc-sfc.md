# TurboMPC 的 IPC SFC 数值验收

## 当前结论

固定四类 IPC 线性 jerk/SFC 问题的数值与成本梯度审计已通过。最新回归对应 `403fc2ddb19e753ec959771cd1a4fa593328ce72` 的编译路径，结果为 [sfc-parity-compiled-cu130-v1.json](../results/turbompc-port/sfc-parity-compiled-cu130-v1.json)：`passed=true`、`sfc_numerical_gate_passed=true`、`sources_unchanged=true`，执行时间为2026-11-03 15:13:05–15:13:41 UTC。此前 `50b7298` 上的 eager `final-v1`、最初的 v3 和全部失败记录保留。

SFC 数学问题、原生求解、成本梯度及低速障碍、森林两个真实 Newton/MJWarp 闭环均已有通过证据。两次历史闭环各为一个固定种子及当时冻结的实现，不从单次成功推断总体成功率。当前核心本轮重跑编译数值与梯度协议，没有重跑这两个线性SFC真实场景；AC-MPC 非线性 CTBR 模型及其真实导航另有独立验证。

前置 M4 使用已明确披露的统一 IR2 修正配置。最新编译回归绑定正式 [final-v8](../results/turbompc-port/m4-ir2-accepted-final-v8.json)，其 `m4_gate_passed=true`，SHA256为 `9fd2a6e8d5e160b8ed6cc2eff3428e755bd9646fd4908d192c2d984e3eeb9e41`；四个 Batch 的400对前后向测量、数值、有限差分和来源门槛全部通过。历史各轮前置M4证据保留。未经修改的原版 A0 仍为 false，SFC 结果不改写它，也不覆盖中间正式 M4 轮次的失败。

## 代码与接口

实现冻结于 `53a3e64b8746bd0d1b4cdc2428b4187071027feb`。v3 运行时记录的 HEAD 是 `1cdb4a46065a387038f2c5fb52ac58dbd480f8c0`，当时 SFC 文件尚未提交；记录中的全部 11 个源码 SHA256 已逐项核对到上述冻结提交，均一致。

- 构造器：[ipc.py](../src/isaaclab_ipc/turbompc/ipc.py)。`IPCQPBuilder(torch.nn.Module, cfg=None)` 默认读取 `Config.load("avoid")`，`forward(obs, costs)` 返回 `(MPCInputs, guess)`，公开 `a/b` 缓冲区与 `dynamics()`。
- 独立审计：[check_sfc_parity.py](../scripts/turbompc/check_sfc_parity.py)。CPU OSQP 参考、两步长有限差分、原生前后向、来源检查和失败时落盘使用同一入口。
- CPU 测试：[test_turbompc_sfc.py](../tests/test_turbompc_sfc.py)。六项通过；Ruff 检查、格式检查及 AST 检查通过。

构造器只接收保存的状态、参考和走廊张量，不调用 Planner、Actor 网络或求解器。`sfc_valid` 的有效行筛选与失败制动由 Actor/执行层处理；对应集成验收单独记录。缓冲区与求解数据为 float64，成本输入可由网络产生并保留梯度。

## 保留的 IPC 数学契约

状态为 `[position, velocity, acceleration]`，控制为三轴 jerk。预测 15 步，每步 0.1 s；`U=(u0,...,u14)`，未来状态为 `X=M x0+C U`。本阶段保留原线性 jerk 模型，不引入四旋翼姿态动力学。

目标保持 `0.5 Uᵀ H U + gᵀ U`。OSQP-Eigen 0.7 实际丢弃原 IPC Hessian 的下三角连续性非对角项，当前实现保留这个已核实的发布行为：jerk 对角权重为 `R_u + [1,2,...,2,1] R_u_con`。没有补回被丢弃的交叉项，也没有改变目标函数的 1/2 系数。

七个成本参数沿用现有 Actor：前四项通过 `exp(log(2)*tanh(cost))` 调整位置、速度、加速度和 jerk 成本；后三项为 `0.2*tanh(cost)` 的线性位置偏置，使用基础位置权重。零成本恢复固定 IPC 目标。末端虚拟控制独立、无约束，只有单位正定二次项，其唯一最优值为零。

SFC 及状态约束保持以下规则：

- 平面采用 `[A,-b]`，表示 `A p <= b`；时间槽 k 约束 `x[k+1]`。初始状态不额外施加第一条预测 SFC。
- `pack_corridors` 已按原 `SetFSC` 的行为将每个平面块反序，展开后不再次反序。紧凑快照保存两个走廊、面 mask 和 15 个时间索引；`corridor_index=-1` 表示该步没有走廊。
- 不删平面、不去重、不归一化、不截断容量。几何层已处理安全半径，QP 层不再次收缩。
- 无效面为零系数和无限边界；无效填充值在计算前屏蔽，包括 NaN 占位。有效面出现非有限数据会报错。
- 初态速度和加速度裁剪、未来速度/加速度界、jerk 界及末端状态权重与当前 CPU `MPC.problem` 一致。
- 审计把分阶段数据精确凝聚回原行序：全部 jerk、全部 acceleration、全部 velocity、逐时间 SFC，然后比较 `H/g/A/l/u`。允许的操作只有已声明的变量消元和行索引映射。

CPU 测试还覆盖紧凑/稠密快照一致性、首个空走廊的时间索引、重复/非单位法向平面、初态裁剪，以及第 69 个有效面确实保留。

## 固定审计与参考方法

四个样例事先固定，没有依据结果拒绝、替换或筛选样本。它们在同一 B=4 的原生调用中执行：

| 样例 | SFC 设置 | OSQP 活动面数 / 行秩 |
| --- | --- | --- |
| `no_sfc` | 全时域无 SFC | 0 / 0 |
| `interior_sfc` | 第一槽为空，其余为宽走廊 | 0 / 0 |
| `terminal_active_sfc` | 末端 x 上界为 0.35 m | 1 / 1 |
| `redundant_active_sfc` | 保留同一活动面的两倍法向重复行 | 2 / 1 |

CPU 参考使用当前 IPC 凝聚矩阵和独立的 OSQP 1.0.5：`eps_abs=eps_rel=1e-10`、`max_iter=20000`、polishing 开启、warm start 关闭。该严格配置用于数学解比较；它不等同于原 CPU MPC 默认 `1e-3` 容差的迭代轨迹。参考与原生数据的全部有效约束逐项比较，包含无穷边界的符号。

梯度参考为独立 CPU OSQP 中央有限差分。每个样例检查全部七个成本参数，步长分别为 `1e-4`、`5e-5`，合计 112 个扰动求解，加四个基点求解。固定损失为全部 45 个 jerk 分量的加权和，权重由脚本中的固定正弦/余弦公式生成。两种步长都必须满足梯度门槛，且所有正负扰动的活动面集合必须与基点一致。最小非活动约束距离大于 4.09；两步长梯度的最大相对差约 `3.15e-9`。

原生梯度为同一标量损失对全部七个成本参数的梯度。该覆盖范围是每个样例一个固定 VJP；完整控制 Jacobian、几何梯度和非线性动力学梯度留待各自的独立审计。

## 两次保留的失败与最终停止条件

所有输入、外部验收阈值和 `SOLVED` 状态要求在三轮之间保持不变。

| 轮次 | ADMM 设置 | 失败事实 |
| --- | --- | --- |
| v1 | 4000 次；绝对/相对容差均 `1e-10` | 单活动面 raw QP 返回 `MAX_ITER`，虽然 SQP、解、物理残差和梯度已通过；总门槛失败。 |
| v2 | 当时生产配置 4096 次；绝对/相对容差均 `1e-8` | 同一样例 raw QP 提前返回 `SOLVED`，但 15 次 SQP 后仍未收敛，失败行梯度按既有语义为零；总门槛失败。 |
| v3 | 16384 次上限；绝对容差 `1e-7`、相对容差 0 | 四例 raw QP 和 SQP 全部 `SOLVED`；全部门槛通过。 |

v2 保存的证据明确显示：raw dual residual 为 `1.31776034e-5`，它低于 ADMM 相对阈值 `2.51605294e-5`，却高于 SQP 的绝对目标 `1e-6`。IPC 位置权重达到千量级，使相对 ADMM 阈值与外层绝对阈值不一致；每次额外 25 步的 QP 很快再次停止，外层预算耗尽。

最终生产 Actor 与审计统一采用 **绝对 ADMM `1e-7`、SQP `1e-6`**。内层绝对目标为外层的十分之一，不再随成本权重放大。16384 是上限，普通提前终止仍启用。其余参数保留投影硬约束、无 slack、`rho=0.1`、`sigma=1e-6`、`rho_f_factor=1000`、`alpha=1.6`、每 25 次检查/调整，以及最多 15 次无搜索 SQP。

这没有放宽原始约束 `1e-5`、控制误差 `1e-4`、梯度相对误差 `1e-3` 或原始 QP 必须 `SOLVED` 的门槛，也没有修改 M4 核心或既有测量。v1 未记录 raw 四元残差向量，因此不能事后补称取得了它的实际终止阈值；v2/v3 已记录 `[primal, dual, primal_limit, dual_limit]`。

## 初次 v3 实测结果

环境为 Python 3.12.14、PyTorch 2.11.0+cu130、CUDA 13.0、RTX 4090。审计采用 eager 执行，数值配置与生产 Actor 一致；本次不进行吞吐对照。

| 样例 | SQP 次数 | ADMM 次数 | 首条 jerk 最大绝对误差 | 全部 jerk 最大绝对误差 | 七成本梯度最大相对误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无 SFC | 1 | 175 | `8.65e-8` | `9.40e-8` | `5.65e-9` |
| 非活动 SFC | 1 | 175 | `9.06e-8` | `9.84e-8` | `5.71e-9` |
| 单活动面 | 1 | 8550 | `5.74e-8` | `3.34e-7` | `5.93e-9` |
| 重复活动面 | 1 | 1975 | `6.31e-8` | `3.09e-7` | `5.68e-9` |

本轮每个样例只有一次 SQP，因此表中最后一次 QP 迭代数也就是累计次数。两个活动面线性相关的样例仍保留两行；其成本梯度与独立有限差分一致。

| 汇总指标 | 最差实测值 | 验收条件 |
| --- | ---: | --- |
| 凝聚 `H/g/A/l/u` 数据差 | `5.6843e-14` | `atol=1e-10, rtol=1e-12` |
| 首条控制误差 | `9.0648e-8` | `<=1e-4` |
| 全部 45 个控制误差 | `3.3401e-7` | `<=1e-4` |
| 全部未来状态误差 | `3.3024e-8` | `<=1e-4` |
| 初态/动力学等式残差 | `3.1043e-11` | `<=1e-5` |
| 原始单位不等式违反 | `0` | `<=1e-5` |
| 独立完整系统 stationarity | `9.8688e-8` | `<=1e-5` |
| 独立凝聚系统 stationarity | `1.3710e-7` | `<=1e-5` |
| 七成本梯度相对误差 | `5.9325e-9` | 两步长分别 `<=1e-3` |

64 面容量增加 32 个无效面到 96 后，四例仍全部 `SOLVED`，全部控制的最大差为 `7.7384e-12`。源码前后哈希一致，加载的原生 `.so` 与成功构建记录匹配。

## 最终核心回归：final-v1

本轮对应提交 `50b7298b6fdecfd923d3566705ed68584f9c01f8`，`solver.py` SHA256 为 `ed21a70911fa200e6b57af91e69e6f67e99860af3037670735c37a0618540d29`。审计脚本仍为上述 v3 使用的同一文件，SHA256 未变。逐项核对后，四个样例的初态、参考、成本、平面数量、全部外部验收阈值、两种有限差分步长、OSQP 配置及 ADMM 配置与原 v3 完全一致；没有替换或筛选样本。

仍使用最多 15 次、绝对容差 `1e-6`、无搜索的 SQP，以及最多 16384 次、绝对容差 `1e-7`、相对容差 0 的 ADMM。新配置字段 `require_qp_convergence=False` 对应原协议的默认行为；独立审计本身继续要求四行的 SQP 和最终 raw QP 都为 `SOLVED`。该脚本没有 `--compile` 选项，本轮保持 **eager** 执行，没有将执行模式变更混入回归。

环境仍为 Python 3.12.14、PyTorch 2.11.0+cu130、CUDA 13.0、OSQP 1.0.5、RTX 4090。四个原生基点在同一 B=4 调用中执行，全部 112 个 CPU OSQP 正负扰动求解及活动面稳定性检查通过；这些是独立 OSQP 有限差分参考，不是 112 次原生扰动求解。

| 样例 | SQP / raw QP 状态 | SQP / 最后一次 QP 迭代数 | 首条 jerk 最大绝对误差 | 全部 jerk 最大绝对误差 | 七成本梯度最大相对误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| `no_sfc` | `SOLVED / SOLVED` | 1 / 175 | `8.6509e-8` | `9.4005e-8` | `5.6515e-9` |
| `interior_sfc` | `SOLVED / SOLVED` | 1 / 175 | `9.0649e-8` | `9.8385e-8` | `5.7103e-9` |
| `terminal_active_sfc` | `SOLVED / SOLVED` | 1 / 8550 | `5.7428e-8` | `3.3401e-7` | `5.9324e-9` |
| `redundant_active_sfc` | `SOLVED / SOLVED` | 1 / 1975 | `6.3112e-8` | `3.0874e-7` | `5.6798e-9` |

表中梯度误差取两个步长中的最大相对 L2 误差；每个步长均分别通过 `1e-3` 门槛。四行均只执行一次 SQP，最后一次 QP 迭代数与本轮累计次数相同。

| 汇总指标 | 最差实测值 | 保持的验收条件 |
| --- | ---: | --- |
| 凝聚 `H/g/A/l/u` 数据差 | `5.6843e-14` | `atol=1e-10, rtol=1e-12` |
| 首条控制误差 | `9.0649e-8` | `<=1e-4` |
| 全部 45 个控制误差 | `3.3401e-7` | `<=1e-4` |
| 全部未来状态误差 | `3.3025e-8` | `<=1e-4` |
| 初态/动力学等式残差 | `3.1043e-11` | `<=1e-5` |
| 原始单位不等式违反 | `0` | `<=1e-5` |
| 独立完整系统 stationarity | `9.8684e-8` | `<=1e-5` |
| 独立凝聚系统 stationarity | `1.3710e-7` | `<=1e-5` |
| 七成本梯度相对误差 | `5.9324e-9` | 两步长分别 `<=1e-3` |

七成本梯度的最大绝对误差为 `1.9088e-9`。raw QP 的最大 primal / dual residual 分别为 `1.3935e-10 / 9.8684e-8`，对应绝对停止阈值均为 `1e-7`。容量从 64 增至 96，新增 32 个带 NaN 占位的无效面后，四行状态仍全部通过，全部控制最大差为 `1.8824e-11`。

源码前后哈希一致，实际加载的原生 `.so` 与成功增量构建 manifest 对应，实际 cuDSS 路径与 SHA256 也已记录；具体库哈希与下列初次 v3 记录相同。该轮与独立 CTBR/真实 Newton 验证共享 GPU，**不以时延或吞吐作为 SFC 回归门槛**。本轮结论限于最终核心上的原 IPC 数学问题、真实约束与成本 VJP 回归；历史低速和森林的闭环范围仍以各自冻结实现和原始报告为准。

## 当前编译核心回归：compiled-cu130-v1

`403fc2d` 的 `_matvec` 优化只作用于编译路径，因此历史 eager 审计不能单独证明此次变更。新 producer 在 `build/turbompc/diagnostics/check_sfc_compiled_regression.py` 保存原脚本副本，只改三处：绝对项目根目录、`compile_kernels=True`、报告中的执行模式标签。将三处逆变换后，整个模块AST与原脚本完全一致；独立只读复核也确认字节等于这三次替换。原生产验收脚本未修改，准备证明保存在 `build/turbompc/sfc-compiled-producer-prep-v1.json`。

四个样例、112次独立OSQP有限差分、两步长、全部成本、平面/mask、预算、容差及总门槛均保持原协议。当前调用使用CUDA13环境、N15、float64，真实执行 `compile_kernels=True`。四行SQP/raw QP均为0，SQP均1次，最后一次QP迭代分别为175、175、8550、1975；活动重复面仍全部保留。

| 当前编译回归指标 | 最差实测 | 固定门槛 |
| --- | ---: | ---: |
| 首条控制误差 | `9.0648511e-8` | `1e-4` |
| 全部控制误差 | `3.3399769e-7` | `1e-4` |
| 全部未来状态误差 | `3.3022904e-8` | `1e-4` |
| 初态/动力学等式残差 | `3.1043293e-11` | `1e-5` |
| 原始单位不等式违反 | `0` | `1e-5` |
| 独立完整系统 stationarity | `9.8685216e-8` | `1e-5` |
| 独立凝聚系统 stationarity | `1.3709612e-7` | `1e-5` |
| 七成本梯度相对误差 | `5.9320661e-9` | 两步长各 `1e-3` |

容量64增加至96、无效面含NaN占位后，四行继续通过，全部控制最大差为 `1.9319657e-11`。实际加载的native/cuDSS与原CUDA13构建一致，全部源文件、HEAD、producer和前置M4哈希运行前后相同。该回归在共享GPU上完成，不增加性能门槛，不复用历史模型或闭环结果来代替新编译数值证据。

| 证据 | SHA256 |
| --- | --- |
| [编译回归报告](../results/turbompc-port/sfc-parity-compiled-cu130-v1.json) | `31f7a1e11513b369bb4b9213cd99b285c616a67f46168dfe21a5ef1bb5587e1e` |
| [原始数组](../results/turbompc-port/sfc-parity-compiled-cu130-v1.npz) | `1530e3186b69ae1d0b748325e4b9e7e732bf0cbda24304e7dd846708acd3e02d` |
| [执行日志](../results/turbompc-port/sfc-parity-compiled-cu130-v1.log) | `cb3a97f52195e09b2ee12c9cd2576856c800bc6eab136c44ff36bf5c05433a36` |

外围启动和源码守卫记录见 [sfc-parity-compiled-cu130-v1.execution.json](../results/turbompc-port/sfc-parity-compiled-cu130-v1.execution.json)。

## 历史证据与来源

路径均相对于本项目工作树。原始 JSON、NPZ 和失败版本脚本独占写入，未覆盖历史结果。

| 文件 | SHA256 |
| --- | --- |
| `results/turbompc-port/sfc-parity-cu130-final-v1.json` | `1daf8bbf87e26ca165f1a5b7d06c08e800a10a2ef46fa2bb35d4432a6d3f5200` |
| `results/turbompc-port/sfc-parity-cu130-final-v1.npz` | `d8c36468979fc2aa1313ef3df8b7de691a3aa4ace80878b193f140a245227f73` |
| `results/turbompc-port/sfc-parity-cu130-final-v1.log` | `0bb009b97e81657833423ec0e3595aca3d899e3b3a274c1add369bf99bffe712` |
| `results/turbompc-port/m4-ir2-accepted-final-v4.json` | `39f129f083a2d87178dbe7ab8547963bac836b923ed46a9a52e9da69907c9062` |
| `results/turbompc-port/sfc-parity-cu130-v3.json` | `3b8e6880cb79c90c410abddab266e739c649ab0367d6af02955c3036a0803c7e` |
| `results/turbompc-port/sfc-parity-cu130-v3.npz` | `9958378d1f214ae4809221964cd9bfa2b86b847b54d4c683be06131ea7241afd` |
| `results/turbompc-port/sfc-parity-cu130-v1.json` | `7d476c66f1f0da9116873b08457eafb2bc9c46f9dad90bb3e35b1a859c148f7d` |
| `results/turbompc-port/sfc-parity-cu130-v1.npz` | `04f68315066ea56668d6cf26af3a23502fda663f47b64cb3dc38f42c2787d1ba` |
| `results/turbompc-port/sfc-parity-cu130-v1-harness.py` | `bcee1d7d398987e320e1f573ca9f7875994593467a3eb29c55bd023659ef2cbf` |
| `results/turbompc-port/sfc-parity-cu130-v2.json` | `fba0ef2ccdfee514bc284fda3d4ac895d25dc6f82fd1afb400d87a67a50e5175` |
| `results/turbompc-port/sfc-parity-cu130-v2.npz` | `b62204a01e33d846dcd4a8c0b905a725740f0eb65436385aee1640c64a359bd8` |
| `results/turbompc-port/sfc-parity-cu130-v2-harness.py` | `42b165ce034b3be9f8806f29ea457d8819aff4820aadb25779a8f35c53fff9d7` |
| `results/turbompc-port/sfc-preflight-v1.json` | `bed85ef27f3c6943d05a8a4bcd9861a814c4ea125bf1de75d7ebe209b4dbab45` |
| `results/turbompc-port/m4-ir2-accepted-v1.json` | `1ab2df7b8438b7c80dee3a16b115f0d836f3a6054bf08dec41e9b59b6ba3e03f` |

正式审计所执行源码：

| 源码 | SHA256 |
| --- | --- |
| `src/isaaclab_ipc/turbompc/ipc.py` | `652abd6bfd03c53c534946254ef974e21b5816c94e39ddb2fddb5f99a31dafd7` |
| `scripts/turbompc/check_sfc_parity.py` | `451fe6fda817f9f88a128971f5cee056b7cc3da8dcd766bb9fa1b638c222b050` |
| `tests/test_turbompc_sfc.py` | `ddc1550f027838e5f10bd94b444d7a6163b50dc276f493fb1ddfadb7a5953bb3` |

加载来源：

- `build/turbompc/native/cu130/_turbompc_native.so`：`b90b553da01764a02ddeb6effcc71be62302aa576ddcc1a9223a4d59e5daea3e`。
- 私有 cu130 `nvidia/cu13/lib/libcudss.so.1`：`62ec19543bc4bc0614b214d2532e46af4c34f6db8a9290f15b57f78f7b7d30a9`。
- `build/turbompc/native/cu130/build-manifest.json`：`8a3fc6d75c38a158118b2dcc8ab9185e4d82d04571d6a9530709f18f38759643`；增量构建 ID 为 `20261103T065854.798634Z-764285-964c7058`。这是实际构建/加载关联记录，不是密码学构建证明。

日志保留为 `build/turbompc-sfc-preflight-v1.log` 和 `build/turbompc-sfc-parity-cu130-v1.log`、`v2.log`、`v3.log`。

## 重跑

在项目工作树执行，使用新的输出名称，避免覆盖既有证据：

```bash
env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD/src" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/tong/tongworkspace/free/isaaclab-IPC/.venv/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/test_turbompc_sfc.py

bash scripts/turbompc/env.sh cu130 env \
  PYTHONPATH="$PWD/src" WARP_CACHE_PATH="$PWD/build/warp-cache" \
  /home/tong/tongworkspace/free/isaaclab-IPC/.venv/bin/python -B \
  scripts/turbompc/check_sfc_parity.py \
  --output results/turbompc-port/sfc-parity-cu130-new.json \
  --m4-summary results/turbompc-port/m4-ir2-accepted-final-v4.json
```

增加 `--preflight` 可仅运行 CPU 矩阵、OSQP 和有限差分检查；该模式不会声明完整 SFC 数值门槛通过。运行时记录共享 GPU 情况；本次 final-v1 与其他数值/闭环验证并行，不用于性能比较。共享设备上的既有训练保持运行。

## 真实低速障碍闭环

固定实现 `53a3e64`、seed=0、stride=10、B=1，使用 `Isaac-IPC-Avoid-TurboMPC-Direct` 与零成本头。真实 Newton 电机力推进，控制 100 Hz、物理 300 Hz；20,000 rays / 10 Hz / 40 m 的 Mid-360 配置保持。输出位于 `results/turbompc-port/sfc-real-low-v1/`。

- 642 步到达目标，最终位置 `[9.524477,-0.148736,1.010899]` m；无碰撞、命令/推力/实际状态越界、QP 或前端失败、制动和恢复事件。
- `backend=turbompc-cuda`，642 次原生调用、642 个接受控制步；全部 642 步具有非空 SFC，最多 29 个有效平面。
- 最大接受解原始约束残差 `8.6331e-9`，最小物理步净间距 `0.09028` m。
- 642 个全部样本的 Actor P50/P95 为 `160.729 / 2510.896` ms，包含环境步骤的 P50/P95 为 `280.103 / 2615.221` ms；平均值分别 `761.346 / 997.287` ms。首步包含编译与 CUDA graph 初始化，数据未剔除首步或异常值。
- 该轮墙钟约 640.802 s（不含先前环境构建），仿真时间 6.42 s。**功能与约束闭环通过，不满足 100 Hz 墙钟实时控制。** 这些时延属于当时共享 GPU 负载，不能推断独占硬件时限；M4 的线性基准性能也不能代替 SFC 场景时延。

证据 SHA256：`records.json` 为 `f94f4b25acee9210065df5d7da10332b4a4ee9ff5fec8c487870b1d4ef53d5e0`；`metrics.json` 为 `bdbf6a21f6799f931d0c327e4b8332cc6a1a97638eb2ff137fab130fef798bcf`；`timing-ms.npy` 为 `86abdc44ce66c3bef25defd2e12b9c86f4dcf07261979ccf6e06224a1e876a77`。

## 真实森林闭环

固定实现 `c1889ae299e367c01d5ab6138ecb457cf39a4962`、seed=0、stride=6、B=1，使用 `Isaac-IPC-Forest-TurboMPC-Direct` 与零成本头。通过 Git archive 冻结 160 个源码文件，运行路径为 `build/turbompc/sfc-validation-c1889ae/`，允许 CTBR 在验证树独立开发而不改变正在执行的 SFC 源码。输出位于 `results/turbompc-port/sfc-real-forest-v1/`。

- 356 步到达 8 m 森林目标，最终位置 `[7.511107,0.010235,1.042960]` m；无碰撞、命令/推力/实际状态越界、QP 或前端失败、制动和恢复事件。
- 356 次原生调用、356 个接受步；全部 356 步有非空 SFC，最多 25 个有效平面。
- 最大接受解原始约束残差 `1.5756571e-7`，最小物理步净间距 `0.3939347` m。
- 全部 356 个样本的 Actor P50/P95 为 `574.354 / 5113.753` ms，完整控制步 P50/P95 为 `665.042 / 5271.135` ms，均值为 `1624.395 / 1751.262` ms。包含首步编译及所有尾部样本。
- 墙钟约 `623.902` s，仿真时间 3.56 s。该结果也不满足 100 Hz 墙钟实时控制。

独立来源旁车 `source-provenance.json` 核对运行前 manifest、运行后快照、Git 提交的 160/160 源文件以及几何扩展，全部一致；原始四个报告在旁车写入前后哈希不变。

| 文件 | SHA256 |
| --- | --- |
| `sfc-real-forest-v1/source-provenance.json` | `e002d4de73c6a0e9f7ac29116e342afd7dec1a24edf44316d64be50746fdc071` |
| `sfc-real-forest-v1/metrics.json` | `f260f94f89d0b84eab5ab044632f5a8a7625010f5a17641b8cd3b264284afd87` |
| `sfc-real-forest-v1/timing-ms.npy` | `ca015f14b60795498e6039dd170404cc80a3f779e479e995fd516d8d89933d4f` |
| 冻结快照 `source-manifest.json` | `aef8f74d8793302ee4b38de8c5916bc49699687f1c70e5423489b1ff4c7f7905` |

## 验证范围

已有中速真实场景的失败历史不因上述低速和森林成功而改变。这里证明固定场景的数值与真实闭环链条成立，没有进行多种子成功率或实时控制验收。

SFC 独立数值和低速真实闭环完成后，才开始 CTBR 的 Isaac Lab XYZW 到模型 WXYZ、世界系速度、机体系角速度、总推力单位和质量适配；非线性模型/Jacobian/梯度及闭环另行验收。
