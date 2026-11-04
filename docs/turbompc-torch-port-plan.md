# TurboMPC PyTorch 原生移植执行计划

## 范围与基线

- 目标：在 Isaac Lab 3.0 中运行无 JAX 运行时依赖的 PyTorch 原生 TurboMPC。复用 TurboMPC 的 CUDA ADMM / cuDSS、稀疏 KKT 求解算法及原生实现。
- 起始开发分支：`feat/turbompc-torch-native`，基点为 `isaaclab-IPC` 提交 `3e1b459`。
- 固定验证分支：`feat/turbompc-torch-native-verified`，工作树为同级 `turbompc-torch-native-verified`。原工作树出现另一个进程的 SFC/AC-MPC/Actor 并行修改后，单独保存 M0–M4 求解器检查点；原工作树及其修改保留。第一阶段不修改正式任务入口；M5 使用独立任务 ID，保留原 MPC 默认行为。
- 上游参考：`ToyotaResearchInstitute/turbompc` 提交 `7050a4f367d08504ecb6f5820d9161e704df5816`，MIT 许可证。参考副本：`/home/tong/tongworkspace/reference_repos/TurboMPC`，保持只读。
- **第一阶段只比较 PyTorch 原生版与原版 JAX + 官方 PyTorch 桥接。** 不提前加入 IPC SFC、CA-AC-MPC 动力学或新的强化学习算法。
- 标准执行模型、理想/真实配置、场景和恢复策略遵守 `CONTEXT.md`、`docs/adr/`、`docs/execution-profiles.md`。

## 已核对的工程边界

- 原版 TurboMPC：高层 SQP、QP 数据组装、非线性约束和导数在 JAX；CUDA 原生库分别包含 ADMM/cuDSS、融合 ADMM/PCG 与稀疏 KKT 求解。
- `admm_cudss_ffi.cc` 和 `cudss_sparse_kkt_ffi.cc` 依赖 XLA FFI 的 `XLA_FFI_CallFrame` 与 `FfiDeviceAllocator`。不能直接从 PyTorch 加载原版 `.so` 后宣称完成原生移植。
- 建议新增独立 C++/CUDA 绑定：复用 `.cu` 求解核心，将输入改为 PyTorch CUDA Tensor 的设备指针，将工作区分配改为 Torch/CUDA 管理，并使用调用方 CUDA Stream；不修改参考仓库。
- 高层 SQP 与约束接口使用 Torch；前向调用复用的 CUDA ADMM；反向调用 CUDA KKT，接入 `torch.autograd.Function` 或 `torch.library.register_autograd`。
- 工具链：`cu128` 使用 CUDA Toolkit 12.8.1、Torch 2.11.0+cu128、JAX 0.6.2；`cu130` 使用 CUDA Toolkit 13.0.2、Torch 2.11.0+cu130，不安装 JAX。两者使用匹配 CUDA 大版本的 cuDSS 1.0.0.35，构建产物和缓存独立。准确版本及重建命令见 `turbompc-cuda-environments.md`。不得修改 Isaac Lab 现有 `uv.lock` 来解决基线环境冲突。

## M0：构建与基准环境

1. 锁定上游参考提交、许可证、原版 solver 参数与 CUDA FFI ABI。
2. 在独立环境运行原版 `turbompc-check`、官方 PyTorch/JAX 桥接前向和反向，不影响 IPC 原虚拟环境。
3. 在独立构建环境确认 CUDA 编译器、PyTorch C++/CUDA ABI、cuDSS 与 cuSPARSE 版本，执行最小原生编译、设备指针和当前 CUDA Stream 测试。
4. 固定相同 GPU、输入、数据类型、收敛容差、Warm Start、目标和约束，保存可重放 OCP 输入。

完成条件：两端可重复执行；GPU 后端确实启用；记录准确依赖与 commit。无法编译时不得进入数值/性能比较。

## M1：移植原生 QP 求解

1. 添加独立 PyTorch CUDA 扩展与必要的上游 MIT 版权声明，复用 `admm_cudss.cu`、块三对角求解与工作区逻辑。
2. 以结构化时域 QP 数据为入口，保留成本、初始状态等式、离散动力学等式、逐步不等式和 ADMM Warm Start。
3. 返回解、乘子、残差、求解状态、实际迭代数及工作区句柄。支持 float64、单批与批量 GPU 输入。
4. 与原版 JAX 的**同一个 QP 子问题**比较前向解、残差、失败状态和 CPU/GPU 同步。

完成条件：一致的停止准则和求解状态；满足双方设定的原始单位残差阈值；不通过删除约束或增设松弛变量伪造可行性。

## M2：PyTorch SQP

1. 移植高层 SQP：状态/控制轨迹、动力学线性化、代价二次近似、约束线性化、局部 QP 组装、步长选择及收敛判断。
2. 先使用 TurboMPC 官方线性/简单非线性测试问题，不接 IPC 任务逻辑；逐项对比每次 SQP 的 QP 数据和最终轨迹。
3. 返回逐批次求解状态和原始单位残差，保留失败信息；支持显式初始轨迹与 ADMM warm start。

完成条件：前向控制、轨迹、残差与原版数值一致；没有 JAX 运行时依赖。

## M3：CUDA KKT 与 Torch Autograd

1. 复用稀疏 KKT CUDA 求解实现，以隐式微分返回成本矩阵、线性成本、初始状态、约束矩阵与边界的梯度。
2. 非线性动力学使用实际离散模型的自动微分，并在反向 KKT 中包含完整拉格朗日 Hessian。
3. 在稳定可微点执行中心有限差分；验证活跃约束、非活跃约束、混合成功/失败批次与非默认 CUDA stream。失败样本不返回伪造的成功状态。
4. 单独记录数值内核相对上游的改动；不得静默修改用于正式比较的原版 A。

完成条件：原生成本及状态梯度通过数值检查；原版 VJP 的一致性和完整性能进入 M4 统一验收。

## M4：唯一性能对照与通过门槛

仅对比以下两组：A = 原版 TurboMPC JAX + 官方 Torch/JAX 桥接；B = PyTorch SQP + 复用原生 CUDA 内核 + Torch KKT。

- 同一问题：先用上游 OCP，`N=15`，固定输入、权重、约束、ADMM/SQP 参数、float64、Warm Start；原版桥接按原样保留复制和同步。
- 批量：`B=1,16,64,256`；分别报告冷启动与热启动。先预热并排除 JIT/编译时间，再单独报告冷启动。
- 计时：正式门槛使用同步的完整前向加反向墙钟延迟，报告 P50/P95、吞吐、失败率、残差和迭代数；各阶段耗时只用于定位。原版 A 必须调用官方 `MPCBackend` 与官方 Torch/JAX 桥接。
- 共享 GPU：用户已明确授权，不等待空闲。使用固定输入、A/B 与 B/A 交替配对，记录 GPU 负载，保留全部有效测量与失败记录。共享负载本身不构成拒绝验收的理由。
- 数值门槛：可行样本最终最大约束违反量不超过 `1e-5`；`u_0` 最大绝对误差不超过 `1e-4`；稳定可微点成本梯度相对误差不超过 `1e-3`。同时检查原始 VJP 与中心有限差分，不能利用桥接层的非有限值替换掩盖错误。
- 性能门槛：在目标批量下，PyTorch 原生版完整前向+反向的 P50/P95 不高于原版桥接的 `1.1×`；不得通过降低精度、减少约束、放宽容差或静默接受失败解实现。
- 任一数值或性能门槛失败，先定位 SQP 组装、数值精度、Kernel Launch、稀疏分解、工作区和流同步；**暂停 SFC/动力学扩展**。
- 正式入口为 `scripts/turbompc/check_acceptance.py`。旧的 `run_turbompc_comparison.py` 保留为顺序诊断，其自定义桥接后端与旧结果不替代正式验收。

## M5：门槛通过后扩展任务

### M5a：IPC SFC

1. 将原版 IPC 的 9 状态、3 维 Jerk、15 步和原预测间隔原样接入独立 TurboMPC Torch 后端。
2. 使用固定容量、有效掩码保留每步完整 SFC 平面，沿用现有 PathCover/RISP、地图安全半径和约束时序，不删除有效平面。
3. 使用投影硬约束并关闭 SFC Slack，保留速度、加速度和 Jerk 限制；求解成功后仍逐条验证预测安全和动力学残差。
4. 与 IPC 原 OSQP/CUDA ADMM 对照相同 QP、真实 Newton 环境的完成率、碰撞率、失败率及墙钟耗时。

### M5b：SFC 验证通过后接入 AC-MPC 非线性动力学

1. 先核对 CA-AC-MPC 作者代码中状态、输入、积分与单位定义，再实现相同非线性模型；不将 IPC Jerk 当作电机推力命令。
2. 使用 SQP 重复线性化非线性动力学。保留原 SFC 的位置硬约束，补充动力学约束和控制限幅。
3. 先固定成本对照非线性模型与线性基线，再开启可微成本 Actor。分别报告模型变化和学习带来的收益。
4. 限时 SQP 失败时保留现有真实配置的制动重规划处理，禁止执行未验证的轨迹。
5. Isaac Lab 3.0 的 XYZW 四元数进入 WXYZ 模型前必须重排；统一世界系线速度、机体系角速度、总推力 N、控制器比推力 m/s² 与实际质量参数。先复现作者的离散积分，再进行独立数值和真实闭环验证。

## 交付

- 独立原生 Torch 控制器与可加载 CUDA 扩展，不依赖 JAX 运行时。
- 原版 JAX 桥接基准脚本、完全相同输入快照、数值与有限差分对照测试。
- JSON/CSV 性能与残差记录、确定性的复现命令和依赖锁定。
- M4 通过后再交付独立 SFC 与 AC-MPC 动力学实验；仅通过验收后讨论合并。

## 当前进度

- M0–M3 的环境、原生接口、Torch SQP 与隐式微分已经实现。CUDA 12 和 CUDA 13 原生目标测试通过。
- M4 的官方接口和共享 GPU 配对实验已完成。未修改 A0 的稀疏 KKT 精度问题和失败记录保留；采用两端共同 IR2 配置的正式汇总通过，不能表述为未修改 A0 通过。后续核心修改分别复测并独立保存，最新版本、精确源码和原始数据以 [进度记录](turbompc-torch-progress.md) 为准。
- M5 的 IPC/SFC 原生接口、同 QP 组装、成本梯度及低速和森林真实闭环已经验证。CTBR 非线性动力学、完整 KKT 成本梯度、模型闭环和 Newton 物理闭环已实现并通过相应检查。后续参考前端按当前已观测走廊的水平余量约束虚拟制动轨迹，候选 `1a71d69` 的原协议固定成本低速导航v4完整执行1076步并达到原目标判据，PPO v5一次更新/重载通过49项检查。核心测试保留原 `403fc2d` 身份，新前端由独立导航/PPO报告绑定，不用旧组件测试替代前端验证。具体结果、历史失败与范围见 [SFC 验证](turbompc-sfc.md)、[CTBR 验证](turbompc-ctbr.md) 和 [进度记录](turbompc-torch-progress.md)。这些验证不覆盖CTBR森林、多seed成功率、训练收敛或100Hz墙钟实时性。
