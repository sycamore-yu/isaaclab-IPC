# TurboMPC 原生 PyTorch 求解器

本目录实现结构化多重打靶 MPC。原生求解路径只使用 PyTorch、CUDA 和 cuDSS；
JAX 仅用于独立对照。

## 来源和当前实现

固定上游提交为 `7050a4f367d08504ecb6f5820d9161e704df5816`。
`native/turbompc/upstream/provenance.json` 保存原始文件摘要，MIT 许可证随源码保留。
CUDA ADMM 迭代直接复用上游实现。Torch 侧负责 QP 组装、SQP、线搜索和伴随系统。
XLA FFI 分配器替换为 PyTorch CUDA 缓存分配器；算子使用调用者的 CUDA Stream。

支持 float64、批量非线性动力学、二次成本、相邻控制成本和线性不等式。
反向采用完整拉格朗日 Hessian，而不是仅对前向 Gauss–Newton 近似求导。
成本矩阵、线性项、初始状态、线性约束矩阵和边界具有一阶隐式梯度。
物理参数当前由动力学对象固定，不声明支持其学习。

当前只支持 projection、无松弛变量的约束模式。其他模式显式报错。
未实现完整 TurboMPC API、float32、二阶策略梯度或完整求解器 CUDA Graph 捕获。
ADMM 按设备和 CUDA stream 分别保留工作区，可顺序跨 stream 调用；
输入和输出在不同 stream 之间的依赖仍由调用者显式建立。外部 stream 的所有者应
保持底层 stream 有效，直到相应求解器关闭。高层 `TorchMPC` 实例按顺序使用。

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
正式 A 仍使用未修改的原版 CUDA 实现。原版 JAX 稠密 KKT 及有限差分仅用于误差诊断，
不能代替用户规定的 A/B 梯度门槛；独立 IR=2 修正候选也不能冒充原版 A。
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
