# TurboMPC 原生移植进度

更新：2026-11-03（UTC）。计划：`docs/turbompc-torch-port-plan.md`；集中交付说明：[turbompc-port-delivery.md](turbompc-port-delivery.md)。

**原生Torch求解器已实现，提交 `403fc2d` 的 M4 `final-v8` 已正式通过；后续参考前端 `1a71d69` 的导航v4和PPO v5也已独立验收通过。** M4四批400对完整前向加反向、双步长有限差分、全部固定数值及性能门槛、328个源文件和10个构建产物来源守卫全部通过。A 明确为 `7050a4f + 三行 IR2`，原版 A0 仍失败。编译SFC四例/112次独立OSQP有限差分、CTBR模型v6的116个开环/有限差分样本和四例各60步、Newton600步、真实状态29次公共求解保留其实际403核心身份。最新导航在原seed0协议下完成1076步并达到原目标判据；PPO的一次更新/重载通过49项检查。历史导航v1/v2/v3失败不改写。
`final-v7` 的 B16 P50 `1.157486 > 1.1` 失败完整保留。剖析后增加编译matvec融合，先通过独立CPU与GPU诊断，再执行完整 final-v8；CUDA原生库、输入、预算、阈值和计时边界不变。
final-v8的206条负载记录利用率90%–100%、中位数99%；按用户授权在共享GPU上验收，不要求空闲。本轮通过不外推其他负载、独占性能或CTBR 10状态/4输入模型吞吐。
`final-v1/v2/v3` 失败、成本组装编译负收益及全部原始记录均保留。尤其 `final-v3` 在较低负载下的 B16 P50 失败尚未由相同条件的最终代码复测消除。
SFC 独立验证继续保留；CTBR 后续闭环和训练验收在专项说明中单列，不以 M4 通过替代。

## 固定输入

- 原始工作树：`free/.worktrees/isaaclab-IPC/turbompc-torch-native`，分支 `feat/turbompc-torch-native`。
- 固定验证工作树：`free/.worktrees/isaaclab-IPC/turbompc-torch-native-verified`，分支 `feat/turbompc-torch-native-verified`；共同基线 `3e1b459`。
- TurboMPC：`7050a4f367d08504ecb6f5820d9161e704df5816`。
- 历史 M0–M4 冻结被测代码：`a771284`；后续审计与文档修订不改动该次测量的求解器源码。
- QP 审计、修正后的正式 gate、独立 IR2 完整实验入口固定于 `710cc8c`；原生求解器和构建输入与 `a771284` 完全一致。
- SFC/CTBR 接入后的复测分别另存为 `candidate-ir2-official-linear-cu128-final-v1` 至 `final-v8`；各次源码与构建输入以对应报告的哈希守卫为准。
- 最新核心M4被测提交：`403fc2ddb19e753ec959771cd1a4fa593328ce72`；solver SHA256：`314e9d672b93161be2bfdfdfb855c5d68a7db07d6baac9bc731e78d16131581c`，adjoint SHA256：`6fa2ad79292c2a383a0b2355fd3617adff2b0a6aab8b195cab542d3df909beb0`，CTBR polisher SHA256：`21953024352cf57da5c830daacd677963958442350a38dbf9924025ceff05c71`。最新导航/PPO实际被测前端提交为 `1a71d692c2eb2b3ca540146def685d7879e5852b`，上述核心和原生库未改变；两种测试身份分开记录。
- 参考仓库只读；构建、虚拟环境和结果均位于本工作树。
- 已测前端的两个提交已经依次集成为 `a37cf21` 与 `282a3fa92c8b738e520ec331a081b61a84d5ce92`；运行代码与被测候选1a71逐字一致，后续仅更新交付文档。
- 原项目依赖锁与正在运行的训练不作修改。用户已授权共享 GPU 验收，不等待空闲。

原工作树后来出现另一个开发进程写入的 SFC、四旋翼和 Actor 草稿，并修改了 `solver.py` 和包入口。
为避免被测源码在运行中改变，历史 M4 首轮测量期间本验证树仅包含 M0–M4；SQP 和包入口通过 SHA256 恢复为先前实测版本，再保留本轮生命周期修复。
原工作树的并行修改没有被回退或纳入本分支。快照清单见 `turbompc-validation-snapshot.json`。
两树复用已安装的工具链、锁定环境和历史结果；本树的 native 扩展分别在自己的 cu128/cu130 目录构建。

## 当前阶段

| 阶段 | 状态 | 证据与边界 |
| --- | --- | --- |
| M0 | 完成 | CUDA 12/13 独立环境、原版安装检查、官方桥接前后向与 cuDSS smoke 通过 |
| M1 | 已实现，目标回归通过 | Torch Tensor / caching allocator / current stream 绑定；复用上游 ADMM/cuDSS；验证范围见下文 |
| M2 | 已实现，目标回归通过 | Torch QP 组装、自动微分线性化、merit 线搜索、真实残差和逐样本状态；未宣称完整上游 API 等价 |
| M3 | 已实现并验证 | 稀疏 KKT + 一阶 Torch autograd；成本、状态、约束和非线性有限差分通过 |
| M4 | 当前403fc2d的final-v8通过 | 四批400对、数值/FD/来源/性能全过；共同IR2基线及原版A0失败身份不变 |
| M5 | SFC与CTBR接入及既定数值/闭环验收通过 | 403核心：编译SFC、CTBR116个开环/FD与240步模型闭环、Newton600步/公共29；1a71前端：原协议1076步导航、PPO一次更新/重载；不外推多seed、CTBR森林、训练收敛或实时性 |

M3 的原生梯度数值检查与 M4 的原版一致性是两个检查。不能以有限差分通过替代用户规定的 A/B 梯度门槛。

## 环境和实现

`scripts/turbompc/env.sh cu128|cu130 COMMAND` 明确选择工具链、Python、原生扩展及编译缓存。
CUDA Toolkit 分别为 12.8.1/13.0.2；Torch 分别为 2.11.0+cu128/cu130；cuDSS 为匹配 CUDA 大版本的 1.0.0.35。
JAX/jaxlib 0.6.2 仅装于 cu128。详情见 `turbompc-cuda-environments.md`。

原生源代码位于 `src/isaaclab_ipc/turbompc/` 和 `native/turbompc/`；接口说明见 `turbompc-native.md`。
结构化多重打靶 QP 保留相邻成本块、等式、逐步不等式、乘子与完整 warm start。
支持 float64、批次隔离、线性及自动微分动力学和一阶隐式梯度；不声明完整上游 API 或二阶策略梯度。

已经修复同一 ADMM 工作区切换 CUDA stream 后产生 NaN 的问题：每条 stream 使用独立工作区。
新增回归保留原始 40×40 KKT 精度样本、A→B→A stream 调用和混合成功/失败批次。
在 C++ 绑定与 Python 缓存增加 `solve/close` 生命周期同步，避免无效句柄与 GIL 死锁。
新增两项带子进程超时的并发关闭回归。固定验证树的两套原生库分别重新构建，CUDA 12 与 CUDA 13 均 **14 项全部通过**。
测试覆盖解析 QP、warm start、current stream、KKT、成本/状态/约束梯度、非线性 Lagrangian Hessian、混合失败批次、编译与 eager 一致性和并发关闭。
cu130 环境不安装 JAX。测试日志：`build/turbompc-tests-cu128-verified.log`（36.13 s）、`build/turbompc-tests-cu130-verified.log`（36.59 s）。
构建日志：`build/turbompc-build-cu128-verified.log`、`build/turbompc-build-cu130-verified.log`。

## M1/M2 逐次 QP 对照

`scripts/turbompc/check_qp_parity.py` 对相同官方线性 B1/N15 输入执行有界 eager 审计。
它只临时观察实例上的组装/求解调用，结束后恢复；没有修改上游文件或数值算法。
结果：`results/turbompc-port/qp-parity-cu128-v1.json` 及同名 `.npz`，全部检查通过。

- 两端均 2 次 SQP、3 次 QP 组装；第三次组装仅用于末次收敛评估。
- 初始 QP 数据逐项完全相同，后两次矩阵最大绝对差异 `1.39e-17`；最终轨迹差异 `1.24e-15`。
- 逐次 ADMM 迭代序列均为 `[50, 25]`。上游公开字段是累计 75 次，原生公开字段是最后一次 25 次，统计口径不同。
- 将同一组捕获 QP 数组交给原始 A/B 求解器，两端均 `SOLVED=0`、50 次迭代，原变量差异 `1.21e-15`；独立等式违反量不超过 `3.88e-13`，边界违反量为零。
- 另将合法盒上界与初态等式设为相互矛盾，两端均 `PRIMAL_INFEASIBLE=2`、25 次迭代；没有通过设置非法的 lower>upper 来制造失败。
- 审计期间源文件哈希前后一致。该报告验证固定用例的组装、残差及状态，不将其扩展为所有问题的完整兼容性证明，也不作为性能测量。

## 历史共享 GPU 正式接口测试（未修改原版 A0）

入口：`scripts/turbompc/check_acceptance.py`。
该次 A 为未修改的原版 `MPCBackend + MPC.mean_only`；B 为原生 Torch，均使用 cu128。
固定官方线性动力学、N=15、float64、同一 256 行输入库的前缀、相同 SQP/ADMM 设置与显式种子。
每组 10 次预热、100 对 AB/BA 交替的完整前向加反向；同步墙钟计时；编译时间另列。
没有依据结果筛选或替换输入。本轮记录 171 次 GPU 负载采样，GPU 利用率 88%–100%（中位数 99%），全卡已用显存 20,358–21,476 MiB。
这些是共享设备的总负载，不能用作某个求解器的独立显存峰值。

固定代码最终结果：`results/turbompc-port/acceptance-linear-cu128-verified-v2.json` 及同名 `.npz`。
测量时间为 2026-11-03 07:03:22–07:06:35 UTC；JSON SHA256 为 `fc6ffc8d4d316c9265a53d49f255a04044d345ce44c1fa3d58f18bf81d10167d`。
上游源码、编译输入、实际加载库的 SHA256、运行前后原生 Python 源码及验收脚本检查全部通过。旧 `acceptance-linear-cu128-v1` 原样保留。

| Batch | A P50 ms | B P50 ms | P50 B/A | A P95 ms | B P95 ms | P95 B/A |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 53.450 | 37.900 | 0.709 | 84.739 | 61.084 | 0.721 |
| 16 | 60.722 | 59.575 | 0.981 | 86.436 | 78.611 | 0.909 |
| 64 | 61.729 | 62.346 | 1.010 | 83.481 | 75.847 | 0.909 |
| 256 | 79.490 | 74.685 | 0.940 | 106.088 | 91.449 | 0.862 |

四组延迟均低于 1.1×。以下误差取每组全部计时对的最大值，按每个样本计算梯度范数，不能用全批次平均掩盖个别失败：

| Batch | u0 最大绝对误差 | 成本梯度最大相对误差 | 初始状态梯度最大相对误差 | 原始约束最大违反量 |
| --- | --- | --- | --- | --- |
| 1 | 9.99e-16 | 3.09e-4 | 3.85e-4 | 7.49e-16 |
| 16 | 5.77e-15 | 4.94e-4 | 6.29e-4 | 4.34e-13 |
| 64 | 6.00e-15 | 9.33e-4 | **1.06e-3** | 4.34e-13 |
| 256 | 6.55e-15 | 9.09e-4 | **1.22e-3** | 4.53e-13 |

线性用例的成本梯度均满足 1e-3，但初始状态梯度在 B64 的 12/100 对和 B256 的 100/100 对超出该阈值。
原生全坐标、两步长中心有限差分的最大成本梯度相对误差为 1.29e-7，状态梯度为 1.06e-9。
A 的 B256 状态梯度与有限差分误差为 1.20e-3。原始 VJP 和官方回调在计时前后均为有限值。
这一用例仅验证固定严格内点，不涵盖所有稳定活跃集；活跃约束另有原生回归及下述完整官方 jerk 诊断。
共享 GPU 的绝对延迟取决于当时负载；此表是保留全部配对样本的实测结果，不是独占硬件吞吐保证。

### 验收工具审查后的门槛修正

冻结测量版曾将旧自定义 backend 报告的失败或缺失永久纳入总门槛。现已改为保留旧历史，仅用显式官方 jerk 复测判定已知问题。
复测必须匹配被测 A 的库哈希、官方源码、固定输入和求解设置；同时检查原始 NPZ 梯度、物理残差及稳定活跃集证据。
16 项 CPU gate 单元测试通过，IR2 候选不能被误认成原版 A。
原 `run_batch`、两端适配器、有限差分与计时函数未改，原始 GPU 结果没有重写。
独立重评记录 `results/turbompc-port/acceptance-linear-cu128-verified-v2-gate-reevaluation-v1.json` 明确标注没有重跑 GPU：
线性数值门槛仍失败、同一原版 A 的官方 jerk 复测也失败，因此修正 gate 后 M4 仍为 false。
实验输出覆盖风险也已修复；标准与规格两条审查轴的发现和处置见 `turbompc-review.md`。

## 原版 A0 的 KKT 精度问题与独立修正

早期 9 状态 jerk 诊断位于 `results/turbompc-port/sweep-linear-cu128-v1/`。
其桥接文件来自官方，但 backend 是自定义适配器，因此旧计时不充当正式官方接口验收。
旧案例的全批次成本梯度相对误差为 1.35e-3、2.20e-3、2.76e-3、约 2.85e-3。
原生与有限差分一致；原版未精化 KKT 的有限差分误差更大。该失败不会因换用更简单的线性案例而消失。

原版 KKT 未设置 `CUDSS_CONFIG_IR_N_STEPS`，cuDSS 1.0 默认值为 0。
原生 KKT 显式执行两次迭代精化；同时使用与原变量解耦的非活跃乘子单位行。
40×40 满秩、条件数约 33.87 的保留样本中，原版残差 3.38e-4，精化后的候选残差 8.61e-11。
这不是一个仅由严重病态条件数解释的失败；具体底层分解扰动原因尚未独立测定。

已经构建独立的原版 IR=2 候选，仅新增三行 cuDSS 配置，完整差异和命令见 `turbompc-kkt-reference-fix.md`。
它没有安装到正式 A，也没有覆盖原版库。精度诊断
`results/turbompc-port/kkt-reference-fix-cu128-diagnostic-v1.json` 已通过：

- 与保存的原生 B1 成本梯度相对差异：7.82e-9。
- 候选方向导数与中心有限差分相对误差：6.34e-7。
- 前向完整轨迹最大差异：2.05e-12。
- 正式 A 的 KKT 二进制哈希前后相同。

这一阶段候选只用于定位和审阅，未改变原版 A0 的基线定义；原版 A0/B 验收仍失败，性能通过不抵消数值失败。
本轮后续采用的共同 IR2 配置及独立汇总见下文，不覆盖该历史结论。

### 完整官方接口复现

`check_jerk_official.py` 使用未修改的 `MPCBackend + MPC.forward`，对同一个旧 B1 输入和完整控制序列损失复测。
全部 12 个成本和 9 个初始状态坐标使用 `1e-4`、`5e-5` 两组中心差分，另保留原 seed=45 的成本方向。
每次扰动均用官方前向结果的实际 KKT mask 验证活跃集；88 次扰动全部可行、收敛，活跃集合不变。

| 检查 | 原版 IR=0 | 独立 IR=2 候选 |
| --- | --- | --- |
| 成本梯度相对原生差异 | 1.66e-3 | 1.33e-8 |
| 成本梯度相对全坐标有限差分误差 | 1.66e-3 | 1.29e-6 |
| 初始状态梯度相对全坐标有限差分误差 | 1.01e-3 | 5.22e-7 |
| 活跃集稳定性、可行性与原始梯度有限性 | 通过 | 通过 |

结果位于 `results/turbompc-port/official-jerk-original-b1-v1.json` 和 `official-jerk-ir2-b1-v1.json`，各自保存原始 NPZ。
这排除了仅由早期自定义 backend 引入差异的解释。下述候选完整实验仍单独保存，没有取代原版 A。

### 历史 IR2 候选的完整四组实验

随后使用 `710cc8c` 的独立候选入口执行全部四组实验。每组固定输入的 SHA256、SQP/ADMM 设置、误差门槛、损失和计时方法均与原版正式实验一致。
A 的上游 Python、官方 backend/bridge 及其他 FFI 保持不变；唯一源码差异仍是 KKT 的三行 IR=2 配置。
每组 10 次预热、100 对 AB/BA 前向加反向，400 对数值检查全部通过。

结果：`results/turbompc-port/candidate-ir2-official-linear-cu128-v1.json` 及同名 `.npz`。
时间：2026-11-03 07:25:18–07:28:36 UTC；JSON SHA256 为 `29b56e0961c399b9c790be51f401bebb2f9e1b92cc49fda315266dad7aa4c490`。
GPU 负载共 177 次采样，利用率 94%–100%（中位数 100%）。源码、harness、构建记录和实际加载库的前后核对全部通过。

| Batch | IR2 候选 A P50 ms | B P50 ms | P50 B/A | IR2 候选 A P95 ms | B P95 ms | P95 B/A |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 65.628 | 43.651 | 0.665 | 86.986 | 60.421 | 0.695 |
| 16 | 69.197 | 60.730 | 0.878 | 99.638 | 74.270 | 0.745 |
| 64 | 72.028 | 63.660 | 0.884 | 100.867 | 82.352 | 0.816 |
| 256 | 91.200 | 79.428 | 0.871 | 117.125 | 96.823 | 0.827 |

全部计时对的最大误差：控制 `6.78e-15`、成本梯度相对误差 `5.81e-10`、初态梯度相对误差 `4.14e-10`、原始约束违反量 `4.53e-13`。
两端全部坐标、两个步长的有限差分也通过，最大成本梯度相对误差约 `1.29e-7`，最大初态梯度相对误差约 `1.08e-9`。
两轮共享负载不同，不能用原版和候选两张表中的绝对时间相减推断 IR2 的独立开销；各自配对的比值是该轮实测结果。

报告的 `candidate_suite_passed=true`，但 `m4_gate_passed=false`，资格固定为 `candidate-only-not-authorized-original-A`。
已安装原版 KKT 的 SHA256 前后仍为 `3adceabed469d2c2d074cc098acede55ebd9d6f1b4eca146096bfc528d112817`。
可供决定的具体修正是：A 使用固定 `7050a4f` 加上述三行 IR2，B 与当前门槛完全保持，原版失败记录继续保留。
这项修正不能被写成未修改原版 A0 通过。两类原始报告保持各自的含义。

### 历史共同 IR2 汇总与后续阶段启动

在完整结果披露后，用户再次指示继续完成 M4、SFC 和 AC-MPC。本轮后续将两端共同的 IR2 作为明确记录的数值配置，算法与所有验收阈值保持。
独立 `finalize_acceptance.py` 从原始数据复算四组 400 对的数值/性能、全部 FD、323 个源文件与 10 个产物；
另从官方 jerk 用例的 88 次控制输出扰动重建有限差分并核对实际活跃集，全部通过。
正式汇总为 `results/turbompc-port/m4-ir2-accepted-v1.json`，SHA256 `1ab2df7b8438b7c80dee3a16b115f0d836f3a6054bf08dec41e9b59b6ba3e03f`。

```text
baseline_role = upstream_7050a4f_plus_ir2
m4_gate_passed = true
original_baseline_passed = false
```

它采用已保存的真实 GPU 数据，没有改写 A0 失败，也没有把候选原始报告中的 false 覆写成 true。
该历史配置的 M4 门槛成立后，M5 开始线性 IPC/SFC 接入；此处的通过仅适用于上述保存的测量。
这份历史汇总不能代替后续代码复测；final-v1/v2/v3 的失败与最终代码 final-v4 的通过分别记录在下文。

## 最终复测 final-v1：数值通过，性能未过

SFC/CTBR 接入后的复测仍比较已披露的共同 IR2 配置。结果保存在
[`candidate-ir2-official-linear-cu128-final-v1.json`](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v1.json)
及同名 `.npz`；测量时间为 2026-11-03 09:37:19–09:39:35 UTC，
JSON SHA256 为 `7cc831041ee4dfbbe7f0f78372965ff35fa9a45e690d24ded54e476f1039b441`。

四组 case SHA256、求解合同、误差门槛、损失、float64/N15 和历史 IR2 实验逐项一致。
每组仍为 10 次预热、100 对 AB/BA 的同步完整前向加反向；没有删掉行、计时样本或离群点。
全部 400 对数值检查通过，两端全部坐标的双步长有限差分通过；源码、实际加载库与构建守卫均通过，原版 A0 库哈希前后不变。
计时样本的最大误差为：控制 `6.44e-15`，成本梯度相对误差 `5.65e-10`，初态梯度相对误差 `5.06e-10`，原始约束违反量 `4.53e-13`。

| Batch | IR2 A P50 ms | B P50 ms | P50 B/A | IR2 A P95 ms | B P95 ms | P95 B/A | 性能门槛 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.337 | 18.632 | **1.140** | 20.988 | 21.000 | 1.001 | 未通过 P50 |
| 16 | 22.260 | 29.241 | **1.314** | 24.306 | 31.533 | **1.297** | P50、P95 均未通过 |
| 64 | 25.319 | 32.873 | **1.298** | 27.164 | 37.880 | **1.394** | P50、P95 均未通过 |
| 256 | 29.056 | 34.940 | **1.202** | 31.315 | 37.283 | **1.191** | P50、P95 均未通过 |

性能要求仍是每个 Batch 的 P50、P95 均不高于 A 的 `1.1×`，因此四个 Batch 的综合性能门槛均未通过。
独立 IR2 驱动原始报告保留 `candidate_suite_passed=false`、`m4_gate_passed=false` 及其原资格字段；该轮未通过 M4。
本轮记录的 121 次全卡负载采样中，GPU 利用率为 0%–66%（中位数 0%）；历史 IR2 轮次为 94%–100%（中位数 100%）。
这些采样描述各轮共享环境，不要求 GPU 空闲，也不能据此挑选历史通过的一轮作为最终结论。原始历史结果与本次失败完整保留。

### 旧/现原生版本的调度诊断

[`m4-native-profile-cu128-v2/summary.json`](../results/turbompc-port/m4-native-profile-cu128-v2/summary.json)
和 [`analysis.json`](../results/turbompc-port/m4-native-profile-cu128-v2/analysis.json) 对同一个 B16 输入进行了 10 对旧/现原生代码的交替测量，
旧版为 `a771284`，两者全部数值检查一致，源码守卫通过。
未插桩的 P50 为旧版 `31.409 ms`、当前版 `31.572 ms`，现/旧比值 **1.00519**；P95 为 `33.129/33.283 ms`，比值 **1.00463**。

该对照表明旧、现版本已有相近的调度开销。另一次 5 步插桩记录中，完整前向加反向合计 `266.54 ms`，
GPU kernel/拷贝/memset 的忙碌时间并集为 `33.23 ms`（`12.47%`），且包含 6910 个 CUDA kernel。
这些证据把后续优化重点指向既有的外层调度、kernel 启动与同步；不能把本次性能失败简单归为新增 CTBR 路径造成的大幅回退。
插桩本身有额外开销，嵌套阶段时间不能相加；这 10 对诊断也不替代 M4 的四批次、每批100对 A/B 正式验收。
后续优化与完整复测继续保留相同数值门槛和固定输入，结果见下文。

## 最终复测 final-v2：200对完成，性能与编译完整性未过

首轮外层 SQP 融合冻结于 `8727b729e597ca3c741e61e562d8d63b6b3c7989`：将步长选择后的轨迹/ADMM/QP更新、
逐行接受与终止判定合并到 `_advance`，减少分散的张量调用和主机同步，保留原来的求解与接受语义。
随后仍按原四个 Batch、每组10次预热及100对完整前向加反向执行。输入 SHA256、求解合同、损失和数值/性能阈值与 final-v1 一致。

报告为 [`candidate-ir2-official-linear-cu128-final-v2.json`](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v2.json)
及同名 `.npz`，测量时间2026-11-03 10:05:09–10:07:27 UTC；
JSON SHA256 为 `f8e6583f32d39f96d19c5b28b66a5f63c2719197c4314772b29ffa8ee61bb93a`。

| Batch | 完成配对数 | IR2 A P50 ms | B P50 ms | P50 B/A | IR2 A P95 ms | B P95 ms | P95 B/A | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 100 | 19.487 | 17.157 | 0.88045 | 22.778 | 21.068 | 0.92491 | 本批次通过 |
| 16 | 100 | 25.104 | 26.544 | 1.05735 | 28.238 | 31.365 | **1.11072** | P95超过1.1 |
| 64 | 0 | — | — | — | — | — | — | 计时前编译变体达到上限 |
| 256 | 0 | — | — | — | — | — | — | 计时前编译变体达到上限 |

已完成的200对全部数值检查、B1/B16两端全部坐标的双步长有限差分均通过。计时对的最大误差为控制 `5.88e-15`、
成本梯度相对误差 `1.77e-10`、初态梯度相对误差 `9.64e-11`、原始约束违反量 `4.34e-13`。
B64/B256 均保留 `FailOnRecompileLimitHit: Hard failure due to fullgraph=True` 与 traceback；没有完成其正式预热、配对与FD，不能将前两组外推为四组数值通过。
源码/构建/实际加载库守卫通过，原版A0的KKT哈希不变；但 `at_least_100_pairs_per_batch`、`at_least_10_warmups_per_backend` 和总批次门槛均为false，
`candidate_suite_passed=false`、`m4_gate_passed=false`。报告中出现四个batch条目不代表四批次完成，全部失败和已测配对均保留。

### 编译守卫的CPU实证与第二轮修正

独立 [`m4-advance-guard-cpu-v1.json`](../results/turbompc-port/m4-advance-guard-cpu-v1.json) 使用冻结首轮源码和CPU eager graph backend，
保持 Torch 2.11.0 的 `recompile_limit=8`，只验证编译守卫，不测GPU性能。
未 detach 的探针按 batch、输入 `requires_grad` 和前次ADMM是否为空分别特化：B1生成4图，B16累计8图，B64首次调用触发相同上限异常。
同一调用序列仅对数值前向输入 detach 后，训练与有限差分输入共用图，四个batch合计8图且16次调用全部完成。
完整守卫日志保存在 [`m4-advance-guard-cpu-v2.log`](../results/turbompc-port/m4-advance-guard-cpu-v2.log)；首次入口缺少导入路径的失败日志也原样保留。

第二轮实现冻结于 `c40ac84cacd3589efd409e753ef30d1e07a0bbdc`，`solver.py` SHA256 为
`f9082d5672e998a5d8c1caf8529645dcba6efe183fa8ee8e4bd3b9881bcf0141`，包含两项修正：

- SQP数值前向统一使用detach后的输入，原始可微输入继续交给隐式autograd；没有改为对SQP迭代展开反传，也没有提高全局编译变体上限。
- 将反向的QP、Lagrangian曲率、活跃约束与上游梯度掩码准备融合为 `_adjoint_data`。仅精确的 `LinearDynamics` 类型跳过恒零动力学曲率；非线性模型和子类保留完整Hessian，路径约束仍先屏蔽非活跃乘子，避免修改共享成本存储。

该冻结实现的[三项GPU回归](../build/turbompc-sqp-fusion-correctness-cu128-v2.log)通过（41.21 s），
覆盖compiled/eager轨迹及成本梯度、曲面活跃约束的独立KKT/有限差分，以及线性活跃/非活跃约束的成本与状态梯度。
[两项CPU选择/状态回归](../build/turbompc-sqp-selection-fusion-v2.log)通过（21.12 s）。这些是针对性正确性检查，
不替代四批次各100对的正式M4；该版本的完整结果为以下 final-v3，最终代码另见 final-v4。

## 最终复测 final-v3：400对完成，B16 P50未过

使用上述 `c40ac84` / `f9082d56…` 核心，四批均完成100对、双步长有限差分及源码/构建检查，没有再次触发编译变体上限。
结果为 [`candidate-ir2-official-linear-cu128-final-v3.json`](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v3.json) 及同名 `.npz`，
测量时间为2026-11-03 10:23:27–10:25:47 UTC，JSON SHA256 为 `716e64eaec46f58e74bbd25ef8d4c1142caa845a937a821a949785954a200a98`。

| Batch | IR2 A P50 ms | B P50 ms | P50 B/A | IR2 A P95 ms | B P95 ms | P95 B/A | 性能门槛 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 19.458 | 16.446 | 0.84521 | 24.236 | 19.644 | 0.81052 | 通过 |
| 16 | 23.302 | 26.934 | **1.15585** | 28.595 | 30.503 | 1.06671 | P50未过 |
| 64 | 30.282 | 29.969 | 0.98966 | 33.498 | 33.547 | 1.00147 | 通过 |
| 256 | 40.551 | 37.925 | 0.93524 | 66.712 | 55.225 | 0.82781 | 通过 |

400对全部数值与FD通过，最大控制误差 `6.55e-15`、成本/初态梯度相对误差 `6.22e-10/3.70e-10`，原始物理违反量 `4.53e-13`。
唯一失败为 B16 P50，`candidate_suite_passed=false`；没有生成 `m4-ir2-accepted-final-v3.json`。
122条全卡负载记录的利用率为0%–99%、中位数19%，显存中位数10952 MiB。
该次较低负载失败继续有效；后续高负载下通过不能被写成在相同低负载条件下已经解决。

## 采用的优化与被否决的成本编译

最终代码采用三项有来源记录的优化，固定输入、损失、SQP/ADMM容差及正式计时边界均未改变：

1. `8727b72` 将外层 SQP 的线搜索、轨迹/ADMM选择、QP重组、误差与逐行状态判定融合为 `_advance`。
2. `c40ac84` 的数值前向统一使用 detach 后输入，原可微输入仍传入隐式反传；融合 `_adjoint_data`，精确 `LinearDynamics` 跳过恒零曲率。四批缓存检查保持 `recompile_limit=8`，没有提高全局上限。
3. `50b7298` 对精确 `LinearDynamics` 直接使用 Jacobian=M 和等式右端常量 `-offset`（无offset时为0），省去两次相消矩阵运算；用同形状、同dtype/device的零张量减offset保留旧提升与广播语义。非线性模型及子类保留原线性化路径，浮点舍入差异不通过放宽验收掩盖。

| 有界B16诊断 | 新/旧P50 | 新/旧P95 | 数值/状态/迭代 | 原始记录 |
| --- | --- | --- | --- | --- |
| 外层状态融合 | 0.802765 | 0.799923 | 全部一致 | [state-fusion](../results/turbompc-port/m4-state-fusion-b16-cu128-v1/summary.json) |
| 反向准备融合与detach | 0.956805 | 0.992094 | 全部一致，四批缓存通过 | [adjoint-fusion](../results/turbompc-port/m4-adjoint-fusion-b16-cu128-v1/summary.json) |
| 解析线性组装 | 0.956840 | 0.752052 | 全部一致，四批缓存通过 | [affine-assembly](../results/turbompc-port/m4-affine-assembly-cu128-v1/summary.json) |

这些是分别执行的同进程10次预热、10对诊断，不能把跨轮比值相乘当作端到端保证。解析组装新侧在6/10对更快，
其较大的P95变化主要对应旧侧两次约35.8 ms，不能从10对推断稳定尾延迟幅度。
最终解析组装通过线性成本/状态梯度和曲面活跃约束KKT/FD两项GPU回归；offset类型兼容的6项CPU回归通过。

另一次尝试仅将 `Native` 适配器的成本/seed组装原样提取并编译，全部工作仍在完整前向加反向计时内，
但 [`m4-cost-assembly-fusion-cu128-v1/summary.json`](../results/turbompc-port/m4-cost-assembly-fusion-cu128-v1/summary.json) 的结果是负收益：
旧 eager 侧P50/P95为 `45.648/55.502 ms`，compiled侧为 `61.559/68.311 ms`，比值 `1.348556/1.230798`，10对中9对更慢。
数值/状态/四批缓存和来源检查通过，负载采样中位100%；该优化已回退，不属于最终代码。
当前 `check_acceptance.py` 完整字节恢复为 `bf8b2bf68e7f810ee5fc619f6096842015261894ef2574358348bcf85c77db23`。

原负收益报告、NPZ及producer完整保留。原实验及其记录的快照位置只保存了旧适配器完整源码和新适配器哈希，未同时保存新harness完整字节。
事后从Git `c40ac84` 与精确Native变更重建，确认完整文件SHA256等于运行前后记录的
`ef26365e984cf224b56da20b7bc918f25a4fb875fa717bd11dcc23d393cf3021` 后，才另存到
[`m4-cost-assembly-fusion-cu128-v1-source-recovery`](../results/turbompc-port/m4-cost-assembly-fusion-cu128-v1-source-recovery/recovery.json)。
该归档包含完整harness、差异和来源说明，明确标记 **post-run exact-hash recovery**，不能称为运行时捕获的源码快照，也没有重跑或改写旧结果。

## 最终复测 final-v4：记录的共享条件下通过

被测提交 `50b7298b6fdecfd923d3566705ed68584f9c01f8`，solver SHA256
`ed21a70911fa200e6b57af91e69e6f67e99860af3037670735c37a0618540d29`。
两端使用同一CUDA12环境、float64、N=15、相同官方4状态/2控制线性用例与256行输入库前缀。
每批各端10次预热、100对AB/BA同步墙钟完整前向加反向，全部400对连续保留；成本和状态全部坐标采用两个步长FD。
没有改动输入、损失、求解设置、`1e-4/1e-3/1e-5/1.1×`门槛或计时边界；M4的 `require_qp_convergence` 保持false。

A明确为上游 `7050a4f` 加以下三行KKT配置，B同样使用IR2；上游Python、官方backend/bridge及其他FFI不变：

```cpp
int refinement_steps = 2;
CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS,
                           &refinement_steps, sizeof(refinement_steps)));
```

### 完整性能表

测量时间：2026-11-03 10:52:19–10:55:21 UTC。

| Batch | IR2 A P50 ms | B P50 ms | P50 B/A | IR2 A P95 ms | B P95 ms | P95 B/A | 门槛 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 62.0052 | 37.6646 | 0.607442 | 83.9947 | 56.8036 | 0.676277 | 通过 |
| 16 | 76.1391 | 57.5634 | 0.756029 | 103.0078 | 73.1675 | 0.710310 | 通过 |
| 64 | 64.9256 | 53.6776 | 0.826756 | 92.7434 | 72.1948 | 0.778436 | 通过 |
| 256 | 72.5556 | 63.3456 | 0.873063 | 102.2556 | 80.8914 | 0.791070 | 通过 |

### 完整数值表

以下是每批全部100对、逐样本误差的最大值，未使用批次平均掩盖失败。

| Batch | u0最大绝对误差 | 成本梯度最大相对误差 | 初态梯度最大相对误差 | 原始物理约束最大违反量 |
| --- | --- | --- | --- | --- |
| 1 | 8.88e-16 | 1.29e-13 | 3.24e-12 | 6.11e-16 |
| 16 | 5.88e-15 | 1.56e-10 | 9.90e-11 | 4.34e-13 |
| 64 | 6.22e-15 | 6.26e-10 | 4.34e-10 | 4.35e-13 |
| 256 | 6.66e-15 | 5.46e-10 | 3.56e-10 | 4.53e-13 |

全部400对状态有效，原始VJP和回调有限，两端全坐标双步长FD均通过，最大成本/初态FD相对误差约 `1.2863e-7/1.0932e-9`。
正式汇总另从已保存的官方jerk稳定用例重建88个扰动，实际活跃集不变，成本/初态FD最大相对误差 `1.284e-6/5.219e-7`，物理违反量 `1.654e-11`。
这项jerk证据复核没有重跑GPU，原始输入和三行IR2补丁均固定。

### 作用范围与可复现证据

本轮162条全卡负载采样的利用率为92%–100%、中位数99%，已用显存中位数20288 MiB；各批计时窗口的利用率中位数为99%–100%。
这些是共享设备总负载，不是B的独立显存峰值。**本次通过仅适用于记录的共享条件；最终核心尚未在与final-v3相同的较低负载条件下完成验收。**
不能将final-v4的高负载通过写成已消除final-v3的低负载失败，不能从不同轮次的绝对耗时推算某项优化的独立加速，也不声明100 Hz墙钟实时控制达标。

327个源文件和10个构建产物运行前后哈希一致，实际加载路径、CUDA12库、单一原生扩展及候选KKT选择均通过。
候选KKT哈希为 `1c194207f9f8930c9072e440b19646d4adfeaa238185c53865da5008d1af30dc`；原版KKT仍为
`3adceabed469d2c2d074cc098acede55ebd9d6f1b4eca146096bfc528d112817`，没有覆盖安装或将A0失败改写为通过。

| 证据 | SHA256 |
| --- | --- |
| [final-v4候选报告](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v4.json) | `7febfbd2b86e929ccd479f1b5e3bec4c67b1a30b3956a04ec53156b597bb6c1c` |
| [final-v4原始数组](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v4.npz) | `d06503af86f7e8ad4c5d26c06367b9dc54ec0e881e8b57c062a86a6f8d220756` |
| [正式最终汇总](../results/turbompc-port/m4-ir2-accepted-final-v4.json) | `39f129f083a2d87178dbe7ab8547963bac836b923ed46a9a52e9da69907c9062` |

CPU预检为 `m4-final-preflight-v5.json/.npz/.log`，测量与汇总日志为 `m4-final-cu128-v4.log`、`m4-finalize-cu128-v4.log`。
候选报告保留 `qualification=candidate-only-not-authorized-original-A`、`candidate_suite_passed=true`、`m4_gate_passed=false`。
独立CPU finalizer复算原始配对、数值、FD与来源后才生成采用共同IR2配置的汇总：

```text
baseline_role = upstream_7050a4f_plus_ir2
selection_note = continued implementation with explicitly disclosed equal IR2 config
m4_gate_passed = true
original_baseline_passed = false
numerical_and_performance_passed = true
same_original_inputs_settings_and_thresholds = true
```

复现与环境选择命令见 [CUDA环境](turbompc-cuda-environments.md)。上述范围内M4门槛成立；后续CTBR验证仍独立进行。

## M5 已核对的接口条件

SFC 继续使用已有 `[A,-b]` 半空间和 PathCover/RISP 安全半径，不重复收缩。
第 i 段走廊约束 x[i+1]，完整保留有效平面和掩码；成本时序、9 状态 jerk、预测 dt=0.1 s 与 100 Hz 控制频率分别处理。
先与现有 IPC 相同 QP 比较，再进行真实执行配置的闭环验证。

AC-MPC 状态为 `[p, q_WXYZ, v_world]`，控制为 `[总推力_N, omega_body]`。
Isaac Lab 3.0 的 XYZW 必须重排；当前 rate controller 接收比推力，因此总推力需除以实际质量。
作者离散模型采用位置/速度 Euler 和四元数指数更新，不应擅自替换成 RK4；零角速度处也需验证一阶及二阶导数。
作者默认质量、项目实际质量和重力参数分别核对，禁止直接抄默认值。
SFC 实现冻结于 `53a3e64`。四个固定样例的矩阵/控制/梯度与严格 OSQP 对照通过，完整结果和两轮失败定位见 `turbompc-sfc.md`。
最终生产配置为 ADMM 绝对 `1e-7`、相对 0、迭代上限 16384；外层 SQP `1e-6`，避免 IPC 千量级成本放大内层相对停止阈值。
CPU 组装六项、原生 Actor 四项均通过；切换到最终停止条件后，原生前向/可行性和网络梯度/观测重放两项针对性复核通过（6.78 s）。
新增独立 `Isaac-IPC-Avoid-TurboMPC-Direct` 与 `Isaac-IPC-Forest-TurboMPC-Direct` 任务，原入口保持原行为。
低速真实 Newton/MJWarp 闭环已完成：642 步到达，无碰撞、命令/实际状态越界和求解失败；接受解最大约束残差 `8.63e-9`，最小物理净间距 `0.09028 m`。
该共享负载下 Actor P50/P95 `160.73/2510.90 ms`，并未达到 100 Hz 墙钟实时控制。SFC 数值/闭环通过后开始 CTBR 实施。
森林验证使用 `c1889ae` 冻结快照完成：356 步到达，碰撞/约束越界/求解失败/恢复均为0；最大接受解原始约束残差 `1.58e-7`，最小物理净间距 `0.39393 m`。Actor P50/P95 为 `574.35/5113.75 ms`。全部160个源码及几何库前后核对通过；完整证据见 `turbompc-sfc.md`。
首个后端固定为上游默认 ADMM-cuDSS 与稀疏 KKT；不额外重写数值迭代。

最终核心的SFC数值回归 [`sfc-parity-cu130-final-v1.json`](../results/turbompc-port/sfc-parity-cu130-final-v1.json) 已通过；详细结果由 `turbompc-sfc.md` 单列。

CTBR 模型、七成本接口、127 维策略观察和真实非线性加速度约束已实现；116个开环/FD样本与Actor接口通过。
历史完整模型 v4 在40/240次求解、各例实际执行9步后停止；末次墙面行 raw QP 成功，但原KKT修正的 stationarity 块线性残差 `4.2719453554e-11 > 1e-11` 被拒绝。Newton v4 悬停200步通过、阶跃实际执行9步后同样被拒绝。两类实际失败均已通过保存原数据重放定位，并在 `29563a5` 加入同一原KKT上最多2次残差校正；没有删除有效SFC面、改变物理模型或放宽证书。

完整 [`ctbr-acceptance-cu130-v5.json`](../results/turbompc-port/ctbr-acceptance-cu130-v5.json) 通过原116个开环/FD及四例各60步模型闭环，全部356次 outer/raw/polish均成功。最大成本梯度相对误差 `3.59454e-7`、模型物理违反 `2.77556e-17`、独立动力学残差 `4.57548e-14`。原六证书全部≤`1e-11`；282个Newton步仅两步各追加一次线性校正，最终KKT残差最大 `2.07308e-13`。

独立Newton验证 [`ctbr-newton-cu130-v5/report.json`](../results/turbompc-port/ctbr-newton-cu130-v5/report.json) 完成悬停200步和位置阶跃400步，全部通过。阶跃终端位置误差 `0.000375824654 m`、速度 `0.000762822215 m/s`；最大预测约束违反量 `1.30904e-12`，实际状态越界为0，全部600×6项精修证书通过。每一步使用真实RateController、Thruster和Newton电机力，没有回写预测状态；仿真100 Hz步长不表示100 Hz墙钟实时。

实际Newton第9步的额外反向检查发现：原七成本FD本已通过 `1e-3`，但反向原KKT残差较大。`b45fd01` 增加仅CTBR启用的原方程认证：按固定上游范数缩放RHS、最多4次同矩阵缺陷校正，并同时要求归一化与实际残差除以该范数≤`1e-11`。失败显式报错；默认M4/线性SFC不启用。41项精修/伴随oracle、8项接口集成和13项Actor CPU检查通过；最终CUDA反向、导航与PPO证据另行验收。

历史失败、数值修正、精确源码与每轮边界见 [CTBR 专项说明](turbompc-ctbr.md)。SFC独立证据继续保留；M4、模型闭环、物理闭环及训练分别判定。

## 可选修正接口接线后的 M4 final-v5

本轮于2026-11-03 12:08:51–12:11:11 UTC执行，随后由CPU finalizer复核，正式 `m4_gate_passed=true`。默认求解路径仍为 `polisher=None`，没有启用 CTBR 专项数值修正；原版 A0 的 `original_baseline_passed=false` 保持。四批各100对、每端10次预热、AB/BA各50对、双步长FD、原输入与全部阈值均未改变。

| Batch | A P50 ms | B P50 ms | P50 B/A | A P95 ms | B P95 ms | P95 B/A |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 22.0601 | 16.8797 | 0.765168 | 33.8630 | 26.0426 | 0.769058 |
| 16 | 29.0460 | 29.4944 | 1.015435 | 38.2816 | 36.5387 | 0.954472 |
| 64 | 33.3029 | 31.6269 | 0.949672 | 47.4446 | 41.2459 | 0.869349 |
| 256 | 39.2788 | 35.6801 | 0.908382 | 52.6993 | 47.2610 | 0.896806 |

全部400对数值与状态检查通过。计时样本最大控制误差 `6.43929e-15`，成本梯度相对误差 `5.09471e-10`，初态梯度相对误差 `4.45110e-10`，原物理约束违反 `4.52945e-13`。双步长FD的成本/初态梯度最大相对误差分别 `1.28621e-7 / 1.05771e-9`，扰动物理违反 `4.54414e-13`。CPU汇总另验证88个既有官方jerk扰动及其真实活动集合，不重新运行该GPU实验。

328个源文件与10个构建产物前后守卫通过，原生C++和二进制没有修改或重建。123条全卡负载记录的利用率为53%–100%、中位89%；已用显存为14877–19588MiB、中位15097MiB。四批配对计时窗口的利用率中位数为96.5%、96.5%、97%、97%。该轮通过不能代替与旧final-v3相同较低负载条件的复测，也不能将不同共享负载下的绝对时间差归因于单项优化。

| 证据 | SHA256 |
| --- | --- |
| [final-v5候选报告](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v5.json) | `7da70ac21581ecee6da685f0d1468f4ac34bce8b7b618499e2ae0cacb151eb1a` |
| [final-v5原始数组](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v5.npz) | `d848db85c71b09a3e2cbb75820817aa5b18c630184e62058a502658cd639ea1c` |
| [final-v5正式汇总](../results/turbompc-port/m4-ir2-accepted-final-v5.json) | `9b8579a137216569139cd29388d3fba2af1b9b3ce92dbc883ec3f1a9efb32526` |

候选报告仍保留 `candidate-only` 与自身 `m4_gate_passed=false`；独立正式汇总才声明已采用共同 IR2 配置的 M4 通过。上述精确源码和执行范围独立于后续 CTBR 专项修正实验。

## 最终反向认证接线后的 M4 final-v6

提交 `b45fd01f124f647712e0463553a39c3e270dc9b0` 已包含CTBR有界原KKT反向校正及显式optional dispatch。默认 `polisher=None` 的可执行分支保持原数学行为，仍对该最终代码完整复测；没有用源码等价检查代替性能测量。

GPU测量于2026-11-03 `13:24:19.451504–13:26:42.980385 UTC` 执行，CPU finalizer于13:29:37生成正式汇总。仍是官方4状态/2控制线性用例，N=15、float64、四批各100对完整前向+反向、每端10次预热、原AB/BA顺序和全部原门槛。CTBR精修及认证反向没有在M4计时用例启用，M4不代表10状态/4控制非线性CTBR吞吐。

| Batch | A P50 ms | B P50 ms | P50 B/A | A P95 ms | B P95 ms | P95 B/A |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 24.4517 | 19.0631 | 0.779623 | 30.9843 | 21.9921 | 0.709783 |
| 16 | 27.0314 | 25.4089 | 0.939978 | 40.8224 | 34.5530 | 0.846423 |
| 64 | 34.4608 | 31.5986 | 0.916944 | 50.6964 | 40.6282 | 0.801402 |
| 256 | 41.2562 | 37.8834 | 0.918249 | 61.8312 | 54.4570 | 0.880737 |

全部400对求解状态、数值和延迟门槛通过。计时对最大控制误差 `6.43929e-15`、成本梯度相对误差 `4.88560e-10`、初态梯度相对误差 `4.17124e-10`、原始物理违反 `4.52923e-13`；双步长有限差分通过。CPU汇总还复核既有官方jerk的88个稳定活动集扰动，物理违反最大 `1.6538e-11`，没有重新执行该GPU实验。

125条全卡负载记录的利用率18%–99%、中位89%；已用显存14476–15976MiB、中位15043MiB。328个源文件与10个构建产物前后守卫通过，实际加载库及原版KKT保持检查通过。没有重编原生库，没有改变固定输入、成本、求解参数、精度、同步计时边界或样本保留规则。本次通过按用户授权的共享GPU条件成立，不要求另行等待空闲，也不外推其他负载或100Hz墙钟实时能力。

| 证据 | SHA256 |
| --- | --- |
| [final-v6候选报告](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v6.json) | `c175635775de1db504f85917feaf6c28a5d7fca7e692be2129fe108f74274550` |
| [final-v6原始数组](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v6.npz) | `6742bee01f6fcb2144914495c106864ca175864b295cde02c6ea5d4b3a8891bd` |
| [final-v6正式汇总](../results/turbompc-port/m4-ir2-accepted-final-v6.json) | `2ecb230460447fffbe3912b4a4fd943cee65bfef86f7b66378fb6f258abe04b3` |

正式 `m4_gate_passed=true`、`numerical_and_performance_passed=true`、`same_original_inputs_settings_and_thresholds=true`，基线角色明确为 `upstream_7050a4f_plus_ir2`；`original_baseline_passed=false`仍保留。第一次CPU finalizer因没有显式选择候选PYTHONPATH而拒绝，未生成正式JSON；修正测试进程环境后，只复核原数组即完成汇总，未重复GPU测量。历史原版A0、final-v1/v2/v3失败以及final-v4/v5通过证据均未覆盖。

反向接线范围证明位于 [scope报告](../build/turbompc-certified-adjoint-scope-v1.json)：对比 `29563a5` 的110个运行源文件，前向计算AST保持一致；仅4个声明生产文件发生变化。它支持保留v5完整240步模型与600步Newton前向证据，但不代替新backward、实际导航和PPO验证。

## 迭代上限收尾接线后的 M4 final-v7：失败保留

提交 `50858a03b59f71f5082d177beeafe7d4052d3267` 增加仅由CTBR Actor显式开启的迭代上限收尾资格，原始SQP状态、误差和QP结果继续独立保存。候选56项CPU检查与verified Actor/训练审计42项检查通过；同一新核心于2026-11-03 `14:17:33–14:20:21 UTC` 执行原M4协议。M4仍为 `polisher=None`，没有启用CTBR收尾。

| Batch | A P50 ms | B P50 ms | P50 B/A | A P95 ms | B P95 ms | P95 B/A |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 42.5567 | 21.7710 | 0.511577 | 65.5560 | 37.8173 | 0.576871 |
| 16 | 37.5061 | 43.4127 | **1.157486** | 73.4034 | 56.8544 | 0.774547 |
| 64 | 34.7548 | 36.2007 | 1.041602 | 76.4273 | 52.4688 | 0.686520 |
| 256 | 57.9840 | 52.3709 | 0.903196 | 94.8501 | 79.3064 | 0.836124 |

全部400对数值、状态、双步长FD及来源检查通过；328个源文件和10个产物前后相同。唯一失败是B16 P50，比允许值多 `2.156077 ms`，因此整体 `m4_gate_passed=false`。149条负载样本的利用率为16%–100%、中位97%；共享负载或bootstrap区间都不能改变这一判定，也不能单凭它们确定根因。GPU实验只执行一次，没有筛样或原样重试。

[拒绝汇总](../results/turbompc-port/m4-ir2-rejected-final-v7.json) 从原候选JSON重新计算全部计时，并独立保存失败原因；它不是accepted finalizer报告。原数值数组继续保留于 [NPZ](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v7.npz)，原计时、原始检查和负载在 [候选JSON](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v7.json)。

| 证据 | SHA256 |
| --- | --- |
| final-v7候选JSON | `863e5be8797b9831932026caee4108238ccf3517c81362d529c4f71cd9ffdb4c` |
| final-v7原始NPZ | `2939a49cbaa2eec1bf3eae22d5f2f775010f352744241275628c2dd4a2362458` |
| final-v7拒绝汇总 | `95d3aa349de7a695ee4270c96a55d8e87f12d1c0d9d9a8a6818e5b8f41a4ce1b` |

该次失败后未立即重跑：先对原生路径做有界分阶段诊断，只有实质优化经过核对后才执行下述 final-v8。旧final-v6通过没有被用来替代这一失败。

## 编译矩阵向量积融合后的 M4 final-v8

在 `50858a0` 的分阶段剖析中，每次完整前后向的四次SQP更新含186个kernel、28次复制。候选仅在编译的合法float64路径将 `_matvec` 改成乘积归约，允许Inductor与残差/线搜索融合；eager及其他输入仍调用原matmul。归约次序可能改变浮点舍入，故必须重新进行数值验证，不能宣称逐位等价。

独立39项CPU检查通过，覆盖高精度收缩参考、广播/非连续/空维、非法形状、实际CPU编译、vmap的一阶与二阶导数以及既有失败状态。B16的30对原生旧/新实现诊断保留全部样本，首控制误差 `8.882e-16`、成本梯度相对差 `1.474e-10`，原始状态/迭代数一致；完整前后向P50/P95比为 `0.611599/0.947209`。剖析中的SQP更新kernel降为110、复制为0。它是内部诊断，不是用户要求的JAX桥接对照，也不把共享负载下的时间差完全归因于某一优化。

另一个ADMM准备融合候选的30对诊断，P50比为 `1.06378`、P95比为 `0.96476`，未改善完整P50，因此未采用。两项诊断都保留原数据，未修改原生库、算法、求解预算或门槛。

生产提交 `403fc2d` 的CUDA12.8与CUDA13.0核心回归各12项全部通过，分别44.38s和51.61s；两套原生库哈希前后不变。之后于 `2026-11-03 15:03:23.551770–15:07:18.299336 UTC` 执行完整M4：原官方4状态/2输入线性模型、float64、N15、四批各100对、各端10次预热、AB/BA各50对、同步墙钟完整前后向。全部样本、双步长有限差分和来源检查纳入总门槛。

| Batch | A P50 ms | B P50 ms | P50 B/A | A P95 ms | B P95 ms | P95 B/A |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.4296 | 22.6144 | 0.604184 | 67.5885 | 41.6325 | 0.615970 |
| 16 | 54.7341 | 46.1155 | 0.842536 | 93.8562 | 62.2049 | 0.662768 |
| 64 | 54.3316 | 47.9596 | 0.882720 | 97.3391 | 65.2386 | 0.670220 |
| 256 | 69.4670 | 62.1124 | 0.894128 | 123.4148 | 82.6032 | 0.669314 |

| 全部400对中的最差指标 | 实测 | 固定门槛 |
| --- | ---: | ---: |
| 首控制量绝对差 | `6.55032e-15` | `1e-4` |
| 成本梯度逐行相对差 | `5.97646e-10` | `1e-3` |
| 初态梯度逐行相对差 | `3.55820e-10` | `1e-3` |
| 原始单位约束违反 | `4.52922e-13` | `1e-5` |
| P50 / P95 B/A | 全部低于1.1 | 各批分别 `<=1.1` |

CPU finalizer独立检查原始数组及官方jerk稳定FD证据后，正式给出 `m4_gate_passed=true`、`baseline_role=upstream_7050a4f_plus_ir2`、`original_baseline_passed=false`。328个源文件及10个构建产物全部未改变。206条GPU负载采样范围90%–100%、中位数99%；没有等待空闲、删样本或以同代码重复测量选取更有利负载。该线性M4通过不等于CTBR吞吐通过，更不构成100Hz墙钟实时控制保证。

| 证据 | SHA256 |
| --- | --- |
| [正式汇总](../results/turbompc-port/m4-ir2-accepted-final-v8.json) | `9fd2a6e8d5e160b8ed6cc2eff3428e755bd9646fd4908d192c2d984e3eeb9e41` |
| [原始候选报告](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v8.json) | `7b70ef7621462bd757c56bc0592e6898ad3dacf25dfdfbfc96e78b670ec0efc0` |
| [原始数组](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v8.npz) | `e81b82f0d66d9e4cf195f7472f6002c2e4d4b58274a0507fbac65ea0b39d50dd` |

两套核心回归见 [matvec-core-results-v1.json](../results/turbompc-port/matvec-core-results-v1.json)，完整执行收尾见 [m4-final-v8.execution-completion.json](../results/turbompc-port/m4-final-v8.execution-completion.json)。当前编译SFC回归及CTBR后续验证分别见 [SFC专项](turbompc-sfc.md) 和 [CTBR专项](turbompc-ctbr.md)。

## 历史403核心复验与导航v2阻塞

当时冻结版本的[总体索引v1](../results/turbompc-port/native-port-completion-v1.json)已经完成，SHA256为 `82e9a4d2d6860a29ab54634743b1919bf2141aa23f34a138f96c9302a8fd8081`。七个验证域中六个通过、实际导航失败，`qualified_migration_acceptance_passed=false`；所有HEAD、源文件、构建产物和索引前后哈希守卫通过。已终结的失败报告不等于完成episode，索引保留了这一区别。后续新参考前端的成功结果另列在下文，不能回写这一历史索引。

`403fc2d` 的[模型v6报告](../results/turbompc-port/ctbr-acceptance-cu130-v6.json)和[执行收尾](../results/turbompc-port/ctbr-acceptance-cu130-v6.execution.json)现已完成：原116个开环/有限差分样本及四例各60步模型闭环全部通过，输入、预算、阈值、被测源码和实际加载库均保持原协议。与作者动力学前向差异最大 `4.98962e-13`；四例双步长成本梯度相对差最大 `3.59409e-7`。模型位置阶跃的最终位置误差为 `0.00652466 m`，固定墙面案例为 `0.00389987 m`。这些是独立NumPy离散模型闭环，不是障碍导航。

当前Newton悬停200步/阶跃400步、公共真实状态29样本及一次PPO更新也分别完成；PPO的两批8×7成本VJP均有限且非零，checkpoint重载动作一致。一次更新尚不构成训练收敛或导航成功率。完整原始记录与数值边界见[CTBR专项](turbompc-ctbr.md)。

[导航v2失败验收](../results/turbompc-port/ctbr-real-low-v2-acceptance.json)已封存为 `FAILED_INCOMPLETE`：最后完整日志为第300步，首个失败快照保留；随后只对本次进程受控SIGINT，未取得终局records，不推断goal、collision或timeout。原SQP和原始QP均耗尽既定预算，polisher按资格规则没有处理原始QP失败候选。

这次失败不同于v1的可修正KKT精度问题。[全控制可达性报告](../results/turbompc-port/ctbr-real-low-v2-corridor-reachability-cpu-v1.json)对保存的原二进制输入所定义的数学离散模型给出必要界：预测node4在两个原走廊中都不可能满足全部平面，至少一个面的归一化违反下界分别为 `0.0373412 m` 和 `0.0427631 m`。因此，增加求解迭代或仅改变两走廊切换时刻均不能解决该保存问题；报告没有把硬件浮点运算声称为严格区间认证。

当时下一项实现位于独立 `build/turbompc/ctbr-reference-candidate` 工作树：在冻结Actor观测之前，利用当前CTBR状态和已经观测的原走廊调节参考推进，保留原几何、名义stride、物理界、H15/dt0.1和验收门槛。原始/调整后参考与调度明确区分。后续新规划问题的成功也不能描述为这个原失败QP已被求解。

### CTBR参考调节候选v1

候选已提交为 `3f9cd34`，分支 `feat/turbompc-ctbr-reference`，仍独立于固定验证分支。实现只改CTBR参考生成与父环境的恒等适配接点，求解器、模型、Actor、原物理界和场景参数保持原版本。在同一次观测中读取真实状态与规划器完整路径；先构造原允许控制下、单个当前原走廊内的有限制动轨迹，再从已记录的有限候选网格选择参考速度，并沿完整原路径重分配时间。实际状态与替换切向速度的虚拟状态分别检查，完整原平面与重复面均保留；局部制动见证不构成返回MPC时间分配、Newton物理或未来地图的递归可行性保证。

15项[数学回归](../build/turbompc/ctbr-reference-candidate/tests/test_ctbr_reference.py)、7项[观测接口回归](../build/turbompc/ctbr-reference-candidate/tests/test_ctbr_reference_env.py)及13项既有Actor CPU回归通过。数学记录为[ctbr-reference-cpu-v1.json](../build/turbompc/ctbr-reference-candidate/build/turbompc/ctbr-reference-cpu-v1.json)，未初始化CUDA。开阔案例的2/4m/s名义参考保持原样；受限案例按已检查网格选择较低速度；旧导航v2失败状态仍为实际制动未认证，原几何有效标记和输入保持不变。另修复了非零float32环境原点的观测精度比较，真实float64 CTBR状态不被回写或裁剪。

候选与原M4组件的[冻结清单](../results/turbompc-port/ctbr-reference-candidate-v1-freeze.json)已生成，SHA256为 `f61ef8e3bea9ac72dd20755eef79ddaf6dabc7f59895f57046b5288544761c2b`。原seed0导航v3于 `2026-11-03 16:49:29.136023 UTC` 在共享GPU启动。记录器分别保存求解输入和物理步后的观测，原始/调整后参考、几何变化与实际状态均有同次调用快照。

导航v3已[封存为FAILED_INCOMPLETE](../results/turbompc-port/ctbr-real-low-v3-acceptance.json)，不是完成episode。390次返回且执行的求解均final/raw/polish成功、未制动、几何有效且物理违反计数为零；其中原SQP为389次状态0和1次状态1，后一例由既定finisher认证收尾。首次新障碍几何在仿真3.90秒到达，状态位置x约6.13529m；[独立全控制必要界](../results/turbompc-port/ctbr-real-low-v3-first-geometry-reachability-cpu-v1.json)证明，该状态的node4在两个原走廊中分别至少违反0.0230970m/0.0299788m法向距离。新几何出现时才降低参考已经太迟，仅调整切换时刻也不可行。

第391次求解未返回时，依据已保存的独立不可行证书，于17:07:08 UTC只对本次子进程受控SIGINT；其状态、迭代和残差保持unknown/null，未将中断伪报为耗尽原预算。child130/outer1、无终局records、完整来源守卫和334件产物均保留。该结果检验了候选v1的因果限制，不能归结为数值容差不足。

同一冻结候选的[PPO v4独立验收](../results/turbompc-port/ctbr-train-smoke-v4-acceptance.json)49项通过：两批8×7原始成本VJP的16行均有限、非零，update/reload阶段不调用Planner或环境，checkpoint和optimizer一致。范围仍为一次更新、0个完整episode；成功行未持久化governor诊断，不能推断其实际触发次数。

### 候选v2：当前水平净空内的制动余量，原协议导航通过

新的隔离工作树为 `build/turbompc/ctbr-reserve-candidate`，从v1的`3f9cd34`出发。只在虚拟参考速度的选择中增加一条保守规则：构造的有限制动轨迹必须留在同一原polytope内，并将水平位移限制在该polytope当前高度截面的内接圆盘半径内。半径由已观测原平面的水平法向与当前余量计算；纯上下盖不产生水平半径限制。实际状态的原制动检查、MPC全体几何/物理约束、名义stride、目标、触发距离和时限不变。

这是一项明确的新参考策略，会因可见侧壁净空降低狭窄走廊内的推进速度；它不是原约束的必要速度上限，也不保证未来未知障碍、Newton跟踪或无限时域安全。候选提交为 `1a71d692c2eb2b3ca540146def685d7879e5852b`；19项数学CPU检查与7项观测接口CPU检查通过。开阔场景2/4m/s保持原名义参考；实际状态检查、所有原平面和物理界未变。同一块polytope的水平半径与完整轨迹绑定，不能从不同polytope借用余量。冻结清单为 [ctbr-reserve-candidate-v2-freeze.json](../results/turbompc-port/ctbr-reserve-candidate-v2-freeze.json)，SHA256 `2d7dce0bc8861acf61bee79990b4a07209e948bbea39f409330ed771594254b6`。

#### 原协议导航v4：完整终局与31项独立验收通过

本轮在共享GPU于 `2026-11-03 17:16:49.470874–17:30:03.335600 UTC` 执行，保留原seed0、1环境、固定成本、stride10、goal10m、障碍距离1m、35s、全部感知/物理/求解参数。标准CLI输出完整1个episode，1076步均有实际原生求解、结果接受和SFC约束；原返回状态/raw ADMM/polish全部0。31项独立CPU审计核对原终局、5942个完整事件、560个几何快照及578项产物，全部通过。[导航v4验收](../results/turbompc-port/ctbr-real-low-v4-acceptance.json)，SHA256 `e56ee5fdc8687d37bec42ef42526d05e2a2dee2dbb3c66c1094ed2899139c526`。

| 原协议指标 | 导航v4结果 |
| --- | ---: |
| 完整episode/原目标判据成功 | 1 / 1 |
| 碰撞、超时、物理/推力/状态违反 | 全部0 |
| 求解失败、前端失败、恢复事件、制动步 | 全部0 |
| 接受求解步、SFC约束步 | 各1076 |
| 最小物理子步间距 | 0.2194174826m |
| 返回轨迹原等式/不等式最大误差 | 2.5535129566e-15 |
| 返回stationarity最大值 | 9.3702823278e-14 |
| raw ADMM primal/dual最大值 | 9.9957464528e-9 / 9.8984632529e-9 |

终点真实位置为 `[9.53153038,-.132499963,1.013237834]m`，到配置目标 `[10,0,1]` 的距离为 `.487026965m`，满足原任务 `<.5m` 到达判据；不是声称x已越过10m。原SQP为1075次状态0和1次状态1；后者是solve974/obs975的60轮迭代上限、stationarity `1.626197537e-7`，经已经固定的finisher通过原证书。原阶段信息保留，没有修改raw状态或原预算。

参考诊断完整保留32次reason6（obs678–709，仿真t6.760060–7.070067s；solve677–708）：有限虚拟速度网格没有候选通过额外水平余量条件，因而保留名义参考、`selected_backup_certified=false`。这些行的实际状态制动见证仍已认证，原MPC在完整原约束下求解并通过。额外余量只是参考筛选规则，不是每步成立的硬不变量；未计算的时序匹配标记也不能解读为物理违规。没有剔除这些行或放宽原验收。

计时原样记录：墙钟774.172s，actor均值613.461ms、step均值718.505ms，包含首次编译与只读记录器开销；仿真控制周期为.01s。该数据不是CTBR的热态吞吐验收，也不支持100Hz墙钟实时结论。[原records](../results/turbompc-port/ctbr-real-low-v4/records.json) · [原metrics](../results/turbompc-port/ctbr-real-low-v4/metrics.json)。

#### 同候选PPO v5：49项检查通过

[PPO v5独立验收](../results/turbompc-port/ctbr-train-smoke-v5-acceptance.json)的SHA256为 `008d2f44ec8e61e687cd6a00d34fe87941e22c82004c02aa9ebc434e30fb36dc`。原16步rollout、2个B8 minibatch、1 epoch、1迭代与两次重载协议未变：34个Actor行全部valid/solved/unbraked，outer/raw状态0；两份完整8×7成本VJP逐值复算，16行均有限且非零；12个MLP参数hook有限、8个非零。更新与重载阶段Planner、env.step、reset及恢复调用全部0，12个optimizer state都为step2，checkpoint逐tensor一致。

最大参数改变量为 `2.0013286849e-4`；确定性动作在同一克隆观测上按 `atol=1e-6, rtol=0` 重载一致。完整成本VJP可独立复算，动作数组与初始参数未保存，后两项按来源固定的进程内断言核对；不声称独立重算了未持久化数组。训练范围是一轮更新、0个完整episode，不能用于训练收敛或导航成功率结论。训练成功行未持久化详细参考诊断，触发次数保持unknown，不从导航轨迹代推。

### 最终组件索引v2

[总体索引v2](../results/turbompc-port/native-port-completion-v2.json)已单次CPU汇总完成，SHA256 `8dc396aeb05e8a363d613db58721581f6b8e357cdda46e05bac4a5dfe04c8dbb`，`qualified_migration_acceptance_passed=true`。M4、编译SFC、CTBR模型、Newton、真实状态public29、导航v4和PPO v5七域原门槛及六项来源守卫全真；1187个文件哈希、24组源码/动态库边界、104个403到候选逐字一致的组件文件均核对，`provenance_issues=[]`。

索引明确区分核心实际测试提交403、前端实际测试提交1a71和汇总时文档HEAD497。Newton使用原IPC环境直接控制，仅相关质量检查函数与候选保持AST一致，不能称旧600步是在新参考前端执行。`unmodified_upstream_m4_passed=false`继续保留；旧false索引v1和三轮导航失败未覆盖。CTBR吞吐、通用导航成功率及训练收敛均未评估。

两个已测前端提交顺序合入verified分支后，[独立源码集成核验](../results/turbompc-port/native-port-integrated-source-v1.json)12项全真：144项冻结来源、155项Git运行/测试文件及全部fixture一致；CUDA12基准产物、实际Nav/PPO加载的CUDA13 native/cuDSS/geometry/runtime与manifest，以及候选库别名全部一致；无额外未追踪运行文件。集成代码提交为 `282a3fa92c8b738e520ec331a081b61a84d5ce92`，报告SHA256 `366fdf9428b6e408133172576037dc045ce4ccf32fcd2f2768ab193209ab1cc4`。与被测候选的分支差异仅文档；本次核验未调用Torch、仿真或GPU，不伪称在新Git提交重跑了旧测试。
