# TurboMPC 原生 Torch 移植交付

日期：2026-11-03 UTC。工作树：`/home/tong/tongworkspace/free/.worktrees/isaaclab-IPC/turbompc-torch-native-verified`；独立分支：`feat/turbompc-torch-native-verified`。

两个已测参考前端提交已顺序集成为 `a37cf21` 和 `282a3fa92c8b738e520ec331a081b61a84d5ce92`。集成后的144项冻结源码、155项追踪运行/测试文件（含二进制fixture）、CUDA12基准产物、CUDA13实际加载库与manifest均核对一致；与被测候选的Git差异只有文档。源码映射检查12项全真，未重新执行GPU测试。[集成一致性报告](../results/turbompc-port/native-port-integrated-source-v1.json)

原生求解器、CUDA12/13独立环境、共同IR2配置的M4、IPC SFC和AC-MPC CTBR接入已经完成。最新固定成本导航按原seed0协议完整执行1076步并到达目标；同一参考前端的一次PPO更新与checkpoint回放也通过。总体验收的限定结论为 `qualified_migration_acceptance_passed=true`，**未经修改上游A0的M4仍为false**。[总体索引v2](../results/turbompc-port/native-port-completion-v2.json)

## 实现和实际被测来源

| 层次 | 当前实现 | 验证身份 |
| --- | --- | --- |
| CUDA内核 | 复用上游ADMM/cuDSS与稀疏KKT；Torch缓存分配器和调用方stream替换XLA FFI；按设备/stream隔离工作区 | CUDA12和13各自构建、回归 |
| Torch高层 | QP组装、SQP、自动微分线性化、merit线搜索、真实约束与逐行状态、一阶隐式梯度 | 核心 `403fc2d`，M4 final-v8 |
| IPC SFC | 完整原平面/掩码、硬约束、9状态3jerk模型；独立Actor与任务入口 | 编译数值回归403；物理low/forest保持各历史报告提交 |
| CTBR模型 | 10状态4控制、作者canonical离散映射、完整Lagrangian Hessian、可选有界原KKT精修和认证反向 | 模型v6、Newton v6、真实状态public29：403 |
| CTBR参考前端 | 只用当前已观测SFC、实测状态及原路径选择参考速度；在观测冻结前适配 | 导航v4、PPO v5：`1a71d69` |

原生运行路径无JAX依赖。一般求解器支持float64和一阶隐式梯度；CTBR精修路径限定为成本参数D/E/q梯度。当前不提供完整上游API、float32、二阶策略梯度或整个求解器的CUDA Graph捕获。[原生接口](turbompc-native.md)

旧组件报告没有被改名为新前端测试。总体索引分别复核1187个文件、24组源码/动态库边界及104个逐字不变的组件文件；Newton的质量检查函数另核对AST。原始A0、M4失败轮次、导航v1/v2/v3及失败前缀均保留。

## CUDA12与CUDA13分别使用

| 项目 | cu128 | cu130 |
| --- | --- | --- |
| CUDA Toolkit | 12.8.1 | 13.0.2 |
| NVCC | 12.8.93 | 13.0.88 |
| PyTorch | 2.11.0+cu128 | 2.11.0+cu130 |
| cuDSS | cu12 1.0.0.35 | cu13 1.0.0.35 |
| JAX/jaxlib | 0.6.2，仅用于对照 | 不安装 |

在工作树根目录执行，环境选择只作用于该命令及子进程：

```bash
bash scripts/turbompc/env.sh cu128 nvcc --version
bash scripts/turbompc/env.sh cu130 nvcc --version

bash scripts/turbompc/env.sh cu128 python YOUR_TEST.py
bash scripts/turbompc/env.sh cu130 python YOUR_TEST.py
```

Python依赖、原生扩展及编译缓存按profile隔离。A/B正式框架对照同用CUDA12，CUDA13单独验证兼容性与Isaac Lab接入；不把跨CUDA版本差异归因于框架。Isaac任务使用cu130配置和项目既有Isaac Python，具体启动与重建命令见[双环境说明](turbompc-cuda-environments.md)。本次没有修改全局驱动、共享项目依赖锁或其他训练进程。

## M4：共同IR2配置通过

A仍使用官方 `MPCBackend + MPC.mean_only` 桥接，仅在独立副本新增三行cuDSS `IR_N_STEPS=2` 配置；B使用相同IR2。上游A0原库和失败记录完整保留。A0在官方jerk案例的成本梯度相对差约 `1.66e-3`，因此不能声称未修改原版通过。该修正没有放宽门槛或修改原问题。[基线精度说明](turbompc-kkt-reference-fix.md)

正式final-v8固定float64、N15、官方4状态/2输入线性模型、B=1/16/64/256，每批100对完整前向加反向、各端10次预热、50对AB及50对BA、同步墙钟计时，包含全部数值与双步长有限差分门槛。206条全卡负载记录为90%–100%、中位99%，按用户授权共享GPU执行。[M4报告](../results/turbompc-port/m4-ir2-accepted-final-v8.json)

| Batch | A P50 ms | B P50 ms | P50 B/A | A P95 ms | B P95 ms | P95 B/A |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.4296 | 22.6144 | 0.604184 | 67.5885 | 41.6325 | 0.615970 |
| 16 | 54.7341 | 46.1155 | 0.842536 | 93.8562 | 62.2049 | 0.662768 |
| 64 | 54.3316 | 47.9596 | 0.882720 | 97.3391 | 65.2386 | 0.670220 |
| 256 | 69.4670 | 62.1124 | 0.894128 | 123.4148 | 82.6032 | 0.669314 |

| 全部计时对的最差指标 | 实测 | 门槛 |
| --- | ---: | ---: |
| 首控制绝对差 | 6.55032e-15 | 1e-4 |
| 成本梯度相对差 | 5.97646e-10 | 1e-3 |
| 初态梯度相对差 | 3.55820e-10 | 1e-3 |
| 原始单位约束违反 | 4.52922e-13 | 1e-5 |
| 各批P50/P95 B/A | 均通过 | 1.1 |

旧final-v7的B16 P50=1.157486×失败保留。先通过剖析把编译SQP的矩阵向量积改为可融合的乘积归约，经数值与导数检查后才执行final-v8。未原样重复计时选取更有利负载；成本组装和ADMM准备融合的负收益候选未采用。共享负载下的结果不外推为其他负载的保证，也不代表CTBR吞吐。

## SFC和CTBR验证

| 验证域 | 结果与范围 |
| --- | --- |
| 编译SFC | 原四例与112次独立OSQP有限差分通过；首控制差9.06485e-8、成本梯度差5.93207e-9；原平面和约束保留 |
| SFC物理闭环 | 历史low 642步、forest 356步完成，失败/违规计数0；保持原报告提交身份，不能替代新CTBR森林验证 |
| CTBR模型v6 | 116个开环/FD样本和4×60步模型闭环通过；作者前向最大差4.98962e-13、梯度FD最大相对差3.59409e-7 |
| Newton v6 | 200步悬停、400步位置阶跃通过；阶跃最终位置误差0.000375825m；真实电机/刚体执行 |
| 保存真实状态public29 | 基点及双步长成本FD共29次通过；保留原SQP阶段，最终原证书及伴随残差通过 |
| 原协议导航v4 | 1个完整seed0固定成本低速episode，1076步、31项独立验收通过 |
| PPO v5 | 16步rollout、2个B8 minibatch、1 epoch、1次更新、checkpoint回放，49项独立验收通过 |

证据：[SFC专项](turbompc-sfc.md)、[模型v6](../results/turbompc-port/ctbr-acceptance-cu130-v6.json)、[Newton v6](../results/turbompc-port/ctbr-newton-cu130-v6/report.json)、[导航v4](../results/turbompc-port/ctbr-real-low-v4-acceptance.json)、[PPO v5](../results/turbompc-port/ctbr-train-smoke-v5-acceptance.json)。

CTBR入口显式将Isaac状态的XYZW重排为WXYZ，位置与SFC采用同一环境原点、线速度为世界系、角速度为机体系。优化器输入是总推力N，环境动作使用F/m；物理资产总质量0.7499999404kg与模型/控制器0.75kg核对。预测步长0.1s、控制周期0.01s和物理子步0.003333s分别记录。[CTBR契约](turbompc-ctbr.md)

### 最新导航终局

保留原seed0、stride10、目标10m、障碍触发距离1m、35s时限及传感器/物理/求解配置。完整1076步的碰撞、超时、物理/推力/状态违反、求解失败、前端失败、恢复及制动计数全部0；最小物理子步间距 `0.2194174826m`，原返回轨迹最大约束误差 `2.5535129566e-15`。[原records](../results/turbompc-port/ctbr-real-low-v4/records.json)

终点 `[9.53153038,-.132499963,1.013237834]m` 到目标 `[10,0,1]` 的距离为 `.487026965m`，满足原 `<.5m` 目标判据。1076次final/raw/polish均成功；原SQP为1075次状态0及1次状态1，后一行经已经固定的有界KKT精修通过原证书，原状态未抹除。

额外水平余量筛选在obs678–709共32次没有认证虚拟候选，代码保留名义参考，随后原MPC仍按全部硬约束求解并通过。它不是已认证的零速返回，也不是每时刻成立的安全不变量；这些行未被剔除。真实状态制动见证与虚拟参考见证分开保存。

原metrics的墙钟774.172s、actor均值613.461ms和step均值718.505ms包含首次编译、共享负载及记录器开销。这不是热态吞吐测量，更不支持100Hz墙钟实时性结论。[原metrics](../results/turbompc-port/ctbr-real-low-v4/metrics.json)

### 最新PPO更新

全部34个Actor行valid/solved/unbraked，outer/raw状态0；两份完整8×7成本VJP独立复算，16行梯度均有限且非零。12个MLP参数hook均有限，其中8个非零。update和reload阶段Planner、环境推进、reset和恢复调用均0；12个optimizer state均在step2，checkpoint逐tensor一致。

最大记录参数改变量为 `2.0013286849e-4`，同一克隆观测的确定性动作回放在 `atol=1e-6, rtol=0` 下相等。完整初始参数和前后动作数组未保存，后两项是来源固定的进程内断言，不能声称独立重算了缺失数组。该测试仅一轮更新、0个完整episode，不能用来证明训练收敛或策略成功率。

## 遇到的问题与实施调整

1. **原XLA接口不能直接成为Torch算子。** 保留CUDA算法，替换分配器、Tensor指针及stream调用，补齐工作区隔离与关闭同步。
2. **上游未精化KKT的梯度精度不足。** 保留A0失败，单独构建双方共同IR2配置。CTBR另使用原KKT残差驱动的有界前后向校正，不增加原变量正则、不放宽原约束。
3. **共享GPU下B16的完整延迟超门槛。** 先诊断并融合编译矩阵向量积，验证后再做新的完整正式实验；失败轮次不覆盖。
4. **CTBR沿用IPC参考推进会在新障碍出现时进入不可恢复状态。** 对导航v2/v3保存输入构造全允许控制的必要外包络，证明仅改走廊切换时刻也无法修复。随后在独立前端候选中，依据当前可见走廊的水平余量筛选虚拟制动轨迹，提前选择较低参考速度；原状态、几何面、物理界、成本权重、预算及场景不变。参考与走廊时序发生明确记录的变化，因此新问题的成功不表示原失败QP已被求解。候选v1失败后才加入V2余量规则，并执行新版本完整导航；没有使用隐藏的未来障碍位置。

参考规则是一种保守的局部启发式，不是未来地图、无限时域或Newton跟踪安全证明。它默认用于CTBR任务，父级IPC hook保持identity；CTBR森林没有在本轮做闭环验收。

## 交付边界与证据

已完成的范围是上述固定协议与实际样本。未修改A0的M4、多seed/中高速通用导航成功率、CTBR森林、非线性CTBR吞吐、100Hz墙钟实时性和训练收敛均没有通过声明。已有中速IPC/AC-MPC历史失败不被本次低速成功改写。

| 核心证据 | SHA256 |
| --- | --- |
| [总体索引v2](../results/turbompc-port/native-port-completion-v2.json) | `8dc396aeb05e8a363d613db58721581f6b8e357cdda46e05bac4a5dfe04c8dbb` |
| [集成一致性](../results/turbompc-port/native-port-integrated-source-v1.json) | `366fdf9428b6e408133172576037dc045ce4ccf32fcd2f2768ab193209ab1cc4` |
| [M4 final-v8](../results/turbompc-port/m4-ir2-accepted-final-v8.json) | `9fd2a6e8d5e160b8ed6cc2eff3428e755bd9646fd4908d192c2d984e3eeb9e41` |
| [导航v4](../results/turbompc-port/ctbr-real-low-v4-acceptance.json) | `e56ee5fdc8687d37bec42ef42526d05e2a2dee2dbb3c66c1094ed2899139c526` |
| [PPO v5](../results/turbompc-port/ctbr-train-smoke-v5-acceptance.json) | `008d2f44ec8e61e687cd6a00d34fe87941e22c82004c02aa9ebc434e30fb36dc` |

完整诊断、未采用优化和逐阶段历史见[移植进度](turbompc-torch-progress.md)。Git保存源码、测试、fixture和文档；原始结果与构建环境留在项目的results/build目录，复制Git仓库时需另行保留这些证据。原main、原开发树源码和参考仓库没有被本次代码集成改动；没有推送远端。
