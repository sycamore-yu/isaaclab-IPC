# TurboMPC 原生 PyTorch 求解器

新 AC-MPC 主线遵循 [ADR-0008](adr/0008-native-acmpc-pathcover.md)。
它采用 28N 成本、默认 N=5，并复用本文的原生求解器。
当前 Actor 不启用 CTBRPolisher，也不使用恢复控制。
下文关于旧 CTBR Actor 默认修正与恢复的描述属于历史实现。
当前配置、IR2 构建和实测数据见 [原生 AC-MPC 验证](acmpc-native-validation.md)。

## 当前 CTBR 默认路径

CTBR 默认使用 `cudss_inverse_chunk`，通用 ADMM 的公开默认仍保留 vector 路径。
cuDSS 沿用原 QP 分解，并求解单位矩阵右端项。
固定矩阵期间使用 float64 矩阵向量积，按原检查、rho 更新和预算边界合并迭代。
每次 QP 或 rho 变化都会更新相应结果。
反向 KKT 继续使用 IR2；该路径不降低原收敛预算或约束精度。

动力学、雅可比、QP/SQP 张量、伴随准备与 Actor 后处理使用固定形状 `torch.compile`。
热启动在 GPU 上移位并积分；原生 ADMM 和主机终止判断仍在编译区域之外。
当前路径没有启用完整求解器 CUDA Graph 捕获。

当前 worktree 的 `build/turbompc/native/cu130` 链接到 `native-v35/cu130`。
也可通过 `TURBOMPC_NATIVE_BUILD_DIR` 显式选择已构建目录。
真实 Python/CUDA 工具链路径和独立候选构建命令见 [README](../README.md#依赖与构建)。
共享 `.venv` 不需要重新安装依赖。

V35 完整 P0 的 17 个阶段及 26 项集成合同通过。
N=5、B=256 的原生前向 p95 为 18.980340 ms，前后向总计 p95 为 28.583600 ms。
首次编译单列；这些数字不包含 PathCover 或物理执行。
完整数据位于 `results/p0-v35-validation/`，后续复用必须保持记录中的源码与原生库身份。

本目录实现结构化多重打靶 MPC。原生求解路径只使用 PyTorch、CUDA 和 cuDSS；
JAX 仅用于独立对照。

## 来源和当前实现

固定上游提交为 `7050a4f367d08504ecb6f5820d9161e704df5816`。
`native/turbompc/upstream/provenance.json` 保存原始文件摘要，MIT 许可证随源码保留。
CUDA ADMM 迭代直接复用上游实现。Torch 侧负责 QP 组装、SQP、线搜索和伴随系统。
XLA FFI 分配器替换为 PyTorch CUDA 缓存分配器；算子使用调用者的 CUDA Stream。

支持 float64、批量非线性动力学、二次成本、相邻控制成本、线性不等式及固定非线性阶段约束。
反向采用完整拉格朗日 Hessian，而不是仅对前向 Gauss–Newton 近似求导。
默认求解路径对成本矩阵、线性项、初始状态、线性约束矩阵和边界提供一阶隐式梯度。可选 CTBR 修正路径只接受成本参数梯度，见下文。
物理参数当前由动力学对象固定，不声明支持其学习。

当前只支持 projection、无松弛变量的约束模式。其他模式显式报错。
未实现完整 TurboMPC API、float32、二阶策略梯度或完整求解器 CUDA Graph 捕获。
ADMM 按设备和 CUDA stream 分别保留工作区，可顺序跨 stream 调用；
输入和输出在不同 stream 之间的依赖仍由调用者显式建立。外部 stream 的所有者应
保持底层 stream 有效，直到相应求解器关闭。高层 `TorchMPC` 实例按顺序使用。

`compile_kernels=True` 将每轮 SQP 的线搜索、候选选择、真实残差和状态更新合并到一个纯张量编译区域；CUDA ADMM 与主机终止判断仍在区域外。反向的 QP、活动集和完整 Hessian 准备也一并编译。精确线性模型直接只读使用成本 Hessian，非线性动力学和活动路径约束仍加入完整曲率。所有原始状态、预算、失败优先级和失败行零梯度规则保留。

数值前向内部使用输入的 detached 视图，以避免同一 batch 在训练和有限差分调用中因 `requires_grad` 不同而生成重复图；隐式 Autograd 保存的仍是原可学习输入。没有提高 PyTorch 的全局编译变体上限。源码冻结的四批同进程检查覆盖可微调用、无梯度调用、再可微调用，数值与来源记录见进度页；这项检查不替代完整 M4。

对于类型恰好为 `LinearDynamics` 的仿射模型 `f(z)=Mz+b`，QP 直接使用固定 `M` 和动力学平移项 `-b`，省去每次重算 `Jz-f(z)` 的矩阵乘法及消减。子类和一般非线性模型仍调用各自的自动线性化，避免跳过重写的动力学。偏置的形状、广播和 dtype 语义由独立物理残差回归覆盖。这三项优化均未改变 CUDA ADMM 算法、M4 输入、容差或计时边界；最新正式 M4 为 [final-v8](../results/turbompc-port/m4-ir2-accepted-final-v8.json)。

提交 `403fc2d` 进一步把编译路径中的合法 float64 矩阵向量积写成乘积归约，使 Inductor 可以与线搜索及残差运算融合。普通 eager 保留原 matmul；其他 dtype、设备、布局或非法收缩维也回原算子，不依靠逐元素广播接受错误形状。浮点归约顺序与 FMA 舍入可能变化，不能宣称逐位等价；39项 CPU 数学/导数/状态回归及30对 B16 原生版本内部诊断已通过。该诊断的完整前后向P50/P95比值为 `0.611599/0.947209`，并观察到每次完整调用的SQP更新kernel数 `186→110`。随后正式四批M4 final-v8全部通过，CUDA12/13各12项核心回归及编译SFC数值回归也通过。未采用的ADMM准备融合候选及其P50负收益保留为独立实验。

CTBR核心在 `403fc2d` 完成116个开环/有限差分样本、四例各60步模型闭环、600步Newton闭环和真实状态公共29样本。随后参考前端候选 `1a71d69` 保留这些核心源码与原生库，实际执行原协议导航v4和PPO v5：导航完整1076步、原目标判据通过、碰撞/超时/求解失败/恢复/制动均为0；PPO的一次更新与checkpoint回放通过49项独立检查。新前端只按当前可见SFC与状态保守选择参考速度，保留原几何、物理界、预算与容差，详情见 [CTBR验证](turbompc-ctbr.md)。旧核心测试不冒称在新前端版本重跑；导航v1/v2/v3失败继续保留。单条低速固定成本导航和一次PPO更新不代表多seed成功率、CTBR森林验证、训练收敛或100Hz墙钟实时性。

## 构建与运行

统一环境和锁定依赖见 `turbompc-cuda-environments.md`。
原生内核必须使用与 PyTorch 相同主版本的 CUDA 工具链。

```bash
# CUDA 12.8 一致性对照环境；不改动主项目 .venv。
bash scripts/turbompc/env.sh cu128 python scripts/build_turbompc.py

bash scripts/turbompc/env.sh cu128 env PYTHONPATH="$PWD/src" \
  python scripts/benchmark_turbompc.py --backend torch --compile \
    --batch 16 --repeats 30 --warmup 5 --finite-difference \
    --output results/new-native-b16.json
```

原生构建也已验证 Torch 2.11.0 / CUDA 13.0。使用私有 CUDA 13 工具链时，
NVCC、CRT 和 NVVM 必须同为 13.0.88，避免 pip 解析到不兼容的 13.4 组件。
构建产物位置由 `TURBOMPC_NATIVE_BUILD_DIR` 选择；默认是
`build/turbompc/native/cu128` 或 `build/turbompc/native/cu130`，由当前 Torch CUDA 版本决定。
没有运行时自动下载、隐式编译或 CPU 回退。

## 数学接口

`MPCInputs` 的时间维度是 `N+1`，包含终端控制节点，与上游一致。
每步令 `z=[x,u]`，目标为：

```
sum_k (0.5 * z[k].T D[k] z[k] + q[k].T z[k])
  + sum_k z[k+1].T E[k] z[k]
```

约束是 `x[0]=initial_state`、`f(x[k],u[k])-x[k+1]=0`，以及
`lower[k] <= G[k] z[k] <= upper[k]`。`E[k]` 是 Hessian 的下非对角块。
失败状态不会被隐藏。失败样本的成本梯度为零，混合批次仍可训练有效样本。
活跃集切换处不保证经典导数存在；有限差分验收只用于局部规则解。

### 固定非线性阶段约束

`PathConstraints` 位于 `isaaclab_ipc.turbompc.constraints`，通过关键字 `TorchMPC(..., constraints=path)` 接入。它表示 `lower <= h(z[k]) <= upper`，支持阶段掩码；CTBR 将它用于真实世界系加速度约束。函数及上下界是固定物理定义，不接受需要学习的边界或函数参数。

SQP 将该函数及上下界一起线性化，merit 线搜索和最终物理残差继续计算真实 `h(z)`。反向 KKT 加入活动阶段约束的完整拉格朗日 Hessian；非活动乘子在曲率计算前归零。线性 `G/lower/upper` 的梯度仍只返回原始线性行，不能与内部追加的非线性行混淆。

`MPCResult.errors` 保持三列 `[stationarity, equality, physical_inequality]`。启用阶段约束时，另返回 `normal_cone_error`，检查 `h - clip(h+y, lower, upper)`；因此“可行且一阶梯度小、但乘子不属于正确法锥”的伪解不能通过。未启用时该字段为 `None`，线性 M4 路径的成功条件保持。

外层 `status` 与 `result.admm.status` 分别描述最终求解结果和最后一个 QP。默认 `SQPConfig.require_qp_convergence=False` 保留原版行为：内层上限耗尽时，外层可能已经满足自己的物理 KKT 容差。CTBR 显式设置为 `True`，SQP 只有二者同时成功才提前结束；否则将已有 ADMM warm state 传给下一轮，使用剩余 SQP 预算。预算耗尽先记录 SQP 失败；CTBR Actor 显式启用下述有界收尾，最终状态由原问题证书决定。Actor 再次检查最终状态和原始 QP 均为0，否则进入恢复流程。M4 固定使用默认 False，并在汇总时拒绝混入 True 配置。

对应回归见 `tests/test_turbompc_constraints.py`：物理残差、阶段 mask、活动曲率、法锥检查、线性参数梯度，以及未启用约束的原有线性/非线性梯度回归。CTBR 模型及单位契约见 `turbompc-ctbr.md`。

### 可选 CTBR 原问题数值修正

`TorchMPC(..., polisher=CTBRPolisher())` 默认仅处理 SQP 与原始 QP 均成功的候选，用同一个原问题的完整 KKT 方程修正轨迹和乘子。默认 `polisher=None`；M4 和线性 SFC 路径不启用它。CTBR Actor 显式启用，最多4次 Newton，原问题证书容差为 `1e-11`。这项修正针对多重射击误差传播造成的下一步固定位置不可行，不改变 SFC 面、物理上下界、动力学、目标或 ADMM 算法。

提交 `50858a0` 增加显式 `CTBRPolisher(finish_iteration_limit=True)` 选项，CTBR Actor 默认启用。它还允许“原 SQP 状态为1（预算耗尽）、原始 QP 状态为0、轨迹及原残差有限”的候选进入相同收尾；其他失败状态不能进入。仍须通过全部原证书和外层原容差，才将最终状态标为0；失败返回状态4并按既有规则对成本返回零梯度。SQP 上限60、Newton上限4、线性校正上限2、所有容差均未改变，也不根据误差接近阈值直接接受候选。独立构造的 `CTBRPolisher()` 默认保持 `finish_iteration_limit=False`。

每个可处理行独立使用 B=1 的原生 cuDSS 工作区。它检查 stationarity、等式、全部物理不等式、法锥、从真实初态完整展开后的原约束违反和 shooting gap。首个未来位置满足精确关系 `p1=p0+dt*v0`；只有这一有代数证书的固定位置行从 KKT 活动等式基中排除，原不等式始终保留检查，对应乘子贡献转移至初态和动力学等式。一般相关约束不会被任意删掉或加入原变量正则项。

每次 Newton 方向还要通过原 KKT 三块残差检查。初解未达到 `1e-11` 时，对同一个矩阵使用 `b-Kδ` 作为 RHS，最多追加2次线性校正，并累计全部原变量及乘子方向；每次都重新检查原 RHS。预算耗尽即拒绝，不放宽阈值。这一修复针对 v4 保留下来的墙面模型第10次求解和 Newton 真实状态第10次求解，其单点重放及完整 v5 闭环均已通过。

`MPCResult.admm` 保留未经改写的原始结果；启用修正时，`sqp_status/sqp_errors` 保存收尾前的原 SQP 状态及三项残差，`iterations` 仍是实际 SQP 次数。未配置 polisher 时新增两个字段为 `None`。`MPCResult.polish` 的13个字段另存修正轨迹、乘子、实际活动基、次数、原 KKT 线性残差及六项物理证书。其中 `linear_residual_history` 保存初解与每次校正的三块残差，`linear_refinement_steps` 保存追加次数。修正状态为0表示成功，1表示未尝试，2表示失败；失败原因单列。失败不会将原始 QP 状态改成成功，外层使用状态4表示修正拒绝，Actor 按既有规则制动并对成本参数返回零梯度。

隐式反向使用修正后的轨迹、动力学乘子、约束乘子和经过认证的活动集合，重算完整 Lagrangian Hessian。该路径只支持 `D/E/q` 的一阶成本梯度；初态、`G` 和边界必须固定，输入验证发生在 numerical detach 之前。数学依据是固定物理约束下可能出现非唯一的冗余面乘子，不能据此提供唯一的物理参数导数。

可选 polisher 可以通过 `solve_adjoint` 进一步认证反向 KKT 解。CTBR 的实现固定每行上游向量无穷范数 `s`（零向量取1），先解归一化 RHS，再还原原尺度；若残差超标，则对同一个完整 KKT 矩阵最多追加4次原缺陷校正。每行必须同时满足归一化方程残差及实际原方程残差除以 `s` 不超过 `1e-11`，才返回 `v/w/wg`。归一化保证上游尺度的一致性，精度提升来自残差校正。已通过行保持不变；非有限解、原生异常或预算耗尽均抛出明确异常，不能把成功前向的未认证梯度静默置零。

`CTBRPolisher.last_adjoint_diagnostics` 只表示最后一次反向调用的 detached 残差、尺度和校正次数，不附着于历史 `MPCResult`，也不是训练全程日志。默认 `polisher=None` 的 M4 与线性 SFC 路径继续直接调用原伴随求解器。

迭代上限收尾的15项接口集成与41项精修/伴随 CPU 检查通过，覆盖 eager/compiled、原状态和预算保留、认证失败零梯度、混合批次及未开启选项的历史行为；Actor和训练审计另有42项 CPU 检查通过。v5 已完成116个开环/FD样本、四例各60步模型闭环，以及 Newton 悬停200步和阶跃400步。认证反向的新29个真实状态样本及原四例116个开环/FD样本也均通过。实际导航与 PPO 更新按独立记录验收，不能仅据局部测试称全部集成完成。各轮实际成败及原始数据见 [CTBR 专项说明](turbompc-ctbr.md)。

原生结果中的 ADMM 迭代数记录最后一次 QP 求解；官方桥接汇总的是整个 SQP 的累计数。
两字段不能直接当作同一统计口径比较。逐次审计工具会记录完整序列并分别计算累计值。

## 显式数值差异

1. 稀疏 KKT 增加 2 次 cuDSS 迭代精化。未修改目标、约束或原变量正则项。
   原始 cuDSS 配置在条件数约 33.87 的 40 维系统上产生 `3.38e-4` 残差；
   两次精化后残差约 `1e-10`，回归阈值为 `1e-9`。复现数据和上游差异见
   `tests/data/README.md` 与 `native/turbompc/upstream/PATCHES.md`。
2. 非活跃不等式的乘子行使用单位对角块，避免原版稀疏组装中的空方程。
   这些行与原变量解耦，不向原变量 Hessian 添加正则项。
3. 每次接受 SQP 步后检查真实非线性动力学和一阶残差，包括最后预算步。
   不沿用上游最后一步可能跳过该检查的行为。不以未检查解宣称收敛。

未经精化的原版 JAX-cuDSS 梯度并非所有样本的可靠真值。
未经修改的原版 A0 仍保留独立构建与失败报告。原版 JAX 稠密 KKT 及有限差分仅用于误差诊断，
不能代替用户规定的 A/B 梯度门槛；后续共同 IR=2 配置的 A 明确标注为原版加三行修正。
线性 B=1 的原生/原版稠密梯度最大差异为 `2.76e-8`，初始状态梯度为 `8.16e-8`。
不能通过增大误差阈值使原生实现追随上游数值误差。

## 对照方法

正式入口是 `scripts/turbompc/check_acceptance.py`，只比较原版官方
`MPCBackend + MPC.mean_only` 与原生 Torch，使用相同 CUDA 12 环境。
固定 B=1/16/64/256、N=15、float64，每组至少 100 对 AB/BA 交替计时。
它检查全部门槛，任一失败返回非零退出码。已知 jerk 精度问题用显式指定的完整官方接口复测报告核验，历史自定义 backend 结果单独保留。

`benchmark_turbompc.py`、`run_turbompc_comparison.py` 和
`benchmark_turbompc_paired.py` 是早期诊断工具。它们复用了官方桥接文件，但使用自定义
backend，不能视为完整官方接口的正式验收。JAX 单独计时与稠密 KKT 仅作定位。

所有测试保留原始时间样本、输入、轨迹、梯度、求解状态和独立物理残差。
首次编译单独计时。前向＋反向包含同一个控制轨迹损失。
JAX/桥接工作区在解释器退出前显式释放，避免 CUDA 库销毁顺序导致退出崩溃。
这只修改测试进程生命周期，不修改官方桥接文件或求解算法。

同进程 A/B 的共享显存不能解释为独立后端峰值。用户已授权在当前共享 GPU 上验收，
脚本记录外部负载；通过结论只适用于所记录的共享测试条件。
共同 IR2 配置的 M4 已通过，未修改原版 A0 的失败仍独立保留。SFC 数值接入见 `turbompc-sfc.md`，真实闭环与后续 AC-MPC 状态以 `turbompc-torch-progress.md` 为准。
