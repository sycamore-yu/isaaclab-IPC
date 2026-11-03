# TurboMPC 原生移植进度

更新：2026-11-03（HKT）。计划：`docs/turbompc-torch-port-plan.md`。

## 固定输入

- 原始工作树：`free/.worktrees/isaaclab-IPC/turbompc-torch-native`，分支 `feat/turbompc-torch-native`。
- 固定验证工作树：`free/.worktrees/isaaclab-IPC/turbompc-torch-native-verified`，分支 `feat/turbompc-torch-native-verified`；共同基线 `3e1b459`。
- TurboMPC：`7050a4f367d08504ecb6f5820d9161e704df5816`。
- 参考仓库只读；构建、虚拟环境和结果均位于本工作树。
- 原项目依赖锁与正在运行的训练不作修改。用户已授权共享 GPU 验收，不等待空闲。

原工作树后来出现另一个开发进程写入的 SFC、四旋翼和 Actor 草稿，并修改了 `solver.py` 和包入口。
为避免被测源码在运行中改变，本验证树仅包含 M0–M4；SQP 和包入口通过 SHA256 恢复为先前实测版本，再保留本轮生命周期修复。
原工作树的并行修改没有被回退或纳入本分支。快照清单见 `turbompc-validation-snapshot.json`。
两树复用已安装的工具链、锁定环境和历史结果；本树的 native 扩展分别在自己的 cu128/cu130 目录构建。

## 当前阶段

| 阶段 | 状态 | 证据与边界 |
| --- | --- | --- |
| M0 | 完成 | CUDA 12/13 独立环境、原版安装检查、官方桥接前后向与 cuDSS smoke 通过 |
| M1 | 已实现并验证 | Torch Tensor / caching allocator / current stream 绑定；复用上游 ADMM/cuDSS |
| M2 | 已实现并验证 | Torch QP 组装、自动微分线性化、merit 线搜索、真实残差和逐样本状态 |
| M3 | 已实现并验证 | 稀疏 KKT + 一阶 Torch autograd；成本、状态、约束和非线性有限差分通过 |
| M4 | 已执行，未通过 | 共享 GPU 延迟通过；原版稀疏 KKT 的梯度精度仍有未解决失败 |
| M5 | 未开始 | SFC/AC-MPC 的接口与单位已只读核对；按用户门槛暂停接入 |

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
CUDA 12 全部 12 项目标测试通过；CUDA 13 原有 9 项和新增 3 项分别通过。
日志：`build/turbompc-tests-cu128-v2.log`、`build/turbompc-tests-cu130.log`、`build/turbompc-regressions-cu130-v2.log`。
随后已在 C++ 绑定与 Python 缓存增加 `solve/close` 生命周期同步，避免无效句柄与 GIL 死锁。
新增两项带子进程超时的并发关闭回归。两套 CUDA 在原树冻结前均通过 14 项目标测试；本固定验证树的最终复核记录另行追加。

## 共享 GPU 正式接口测试

入口：`scripts/turbompc/check_acceptance.py`。
本次 A 为未修改的原版 `MPCBackend + MPC.mean_only`；B 为原生 Torch，均使用 cu128。
固定官方线性动力学、N=15、float64、同一 256 行输入库的前缀、相同 SQP/ADMM 设置与显式种子。
每组 10 次预热、100 对 AB/BA 交替的完整前向加反向；同步墙钟计时；编译时间另列。
没有依据结果筛选或替换输入。记录 304 次 GPU 负载采样。

结果：`results/turbompc-port/acceptance-linear-cu128-v1.json` 及同名 `.npz`。

| Batch | A P50 ms | B P50 ms | P50 B/A | A P95 ms | B P95 ms | P95 B/A |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 84.390 | 58.922 | 0.698 | 112.012 | 81.595 | 0.728 |
| 16 | 93.653 | 77.853 | 0.831 | 130.833 | 105.862 | 0.809 |
| 64 | 105.979 | 85.919 | 0.811 | 147.080 | 116.975 | 0.795 |
| 256 | 112.018 | 96.696 | 0.863 | 146.244 | 121.549 | 0.831 |

四组延迟均低于 1.1×。以下误差取每组全部计时对的最大值，按每个样本计算梯度范数，不能用全批次平均掩盖个别失败：

| Batch | u0 最大绝对误差 | 成本梯度最大相对误差 | 初始状态梯度最大相对误差 | 原始约束最大违反量 |
| --- | --- | --- | --- | --- |
| 1 | 7.77e-16 | 3.24e-4 | 4.29e-4 | 8.60e-16 |
| 16 | 5.77e-15 | 4.94e-4 | 6.16e-4 | 4.34e-13 |
| 64 | 6.11e-15 | 9.37e-4 | **1.07e-3** | 4.35e-13 |
| 256 | 6.44e-15 | 9.13e-4 | **1.22e-3** | 4.53e-13 |

线性用例的成本梯度均满足 1e-3，但初始状态梯度在 B64 的 9/100 对和 B256 的 100/100 对超出该阈值。
原生全坐标、两步长中心有限差分的最大成本梯度相对误差为 1.29e-7，状态梯度为 1.07e-9。
A 的 B256 状态梯度与有限差分误差为 1.20e-3。原始 VJP 和官方回调在计时前后均为有限值。
这一用例仅验证固定严格内点，不涵盖所有稳定活跃集；活跃约束另有原生回归与旧 jerk 诊断。

## 未解决的 KKT 精度问题

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

候选只用于定位和审阅。未经明确改变基线定义，不能用它把原版 A/B 失败标记为通过。
M4 当前仍未通过；性能通过不抵消数值失败。

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
这排除了仅由早期自定义 backend 引入差异的解释。候选没有性能验收资格，也没有取代原版 A。

## M5 已核对的接口条件

SFC 继续使用已有 `[A,-b]` 半空间和 PathCover/RISP 安全半径，不重复收缩。
第 i 段走廊约束 x[i+1]，完整保留有效平面和掩码；成本时序、9 状态 jerk、预测 dt=0.1 s 与 100 Hz 控制频率分别处理。
先与现有 IPC 相同 QP 比较，再进行真实执行配置的闭环验证。

AC-MPC 状态为 `[p, q_WXYZ, v_world]`，控制为 `[总推力_N, omega_body]`。
Isaac Lab 3.0 的 XYZW 必须重排；当前 rate controller 接收比推力，因此总推力需除以实际质量。
作者离散模型采用位置/速度 Euler 和四元数指数更新，不应擅自替换成 RK4；零角速度处也需验证一阶及二阶导数。
作者默认质量、项目实际质量和重力参数分别核对，禁止直接抄默认值。
这些是只读接口核对，尚未写入 SFC 或 AC-MPC 接入代码，也未声称完成闭环验证。
首个后端固定为上游默认 ADMM-cuDSS 与稀疏 KKT；不额外重写数值迭代。
