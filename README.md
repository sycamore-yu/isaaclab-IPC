# Isaac Lab IPC

本分支建立原生 **AC-MPC + PathCover** 导航主线。Isaac Lab / Newton 负责物理、LiDAR 和 PPO 训练；MARSIM / Flightmare 负责正式评测。架构决策见 [ADR-0008](docs/adr/0008-native-acmpc-pathcover.md)。

| 配置 | 前端 | 成本 | 输出 |
| --- | --- | --- | --- |
| A0 | 目标观测，无几何前端 | 逐节点学习成本 | TurboMPC → 4D CTBR |
| A1 | 完整原生 PathCover | 固定二次成本 | 同一 TurboMPC → 4D CTBR |
| A2 | 与 A1 相同 | 逐节点学习成本 | 同一 TurboMPC → 4D CTBR |

PathCover 原生前端依次执行 NanoVoxMap、全图 DMP、RISP 和 GCOPTER/MINCO。它输出连续 PVA/yaw、轨迹段时间及有序 SFC。Actor 读取冻结的观测，用逐节点成本构造非线性 MPC；PyTorch 原生 TurboMPC 复用 CUDA/cuDSS 求解与反向传播。

原生 DMP 直接搜索完整可用网格。正式 Dense 失败快照表明，旧 JPS 引导区域内没有合法连接，完整网格中存在合法路径。上游 JPS 源码与完整链路对照继续保留。`search_radius` 仅用于该上游对照，生产 DMP 不裁剪引导区域。

搜索网格直接读取膨胀后的整数占据位图。细格与粗格仅触边时，不将相邻粗格标成障碍。两种分辨率需具有可恢复的有理比例，分子和分母上限为 1,000,000；常用的 0.1/0.25、0.1/0.3 及同分辨率均支持。诊断点云仍保留原 float 输出，正常规划不经该转换。

- **训练**：Isaac Lab 3 / Newton (MJWarp)，真实电机力、接触和 LiDAR。使用标准 RSL-RL PPO；Actor 学习成本，Critic 估计长期价值。
- **正式闭环评测**：MARSIM + Flightmare，控制频率 100 Hz 仿真时间，MPC 以 10 Hz 仿真时间重规划。也保留可选墙钟模式，用于计算延迟测试。
- **外部对照**：原版 [SUPER](https://github.com/hku-mars/SUPER) 独立运行；其代码不属于上述两条路径。
- **预测模型**：10D CTBR，状态为位置、WXYZ 四元数和世界速度。优化控制为总推力 N 与机体角速度，执行时将总推力除以质量。
- **成本节点**：默认 N=5，含当前节点，共 4 段 0.1 s 转移。Actor 输入 69 维特征，输出 140 个成本参数。旧 7 参数检查点不适用于该契约。

Newton 与 Flightmare 的力矩分配均保留可行请求的总推力。若旋翼目标力越界，执行器先最大化横滚与俯仰力矩的共同缩放系数，再选择最近的可行偏航力矩。两端保留各自的电机动态模型；CTBR 指令不等于瞬时实际机体响应。

## 依赖与构建

Linux、Python 3.12、CUDA 13、Isaac Lab 3、Newton、Warp 和原生 TurboMPC CUDA 扩展。原生前端还需要 Eigen3、Boost 头文件和 Qhull。使用项目锁定的 `uv.lock`；参考源码保持只读。

当前工作树已经准备运行环境、资产和原生库。MPC 默认目录 `build/turbompc/native/cu130` 指向已验收的 `build/turbompc/native-v35/cu130`。日常训练和评测直接使用下文入口。

首次准备新环境时使用锁文件，并生成传感器与森林资产：

```bash
uv sync
.venv/bin/python scripts/prepare_mid360.py --output build/sensors
.venv/bin/python scripts/build_forest.py \
  --pcd /home/tong/tongworkspace/reference_repos/IPC/hku-mars/mars_uav_sim/uav_simulator/map_generator/resource/rc_forest.pcd \
  --output build/forest
```

当前机器的 CUDA 工具链和原生构建解释器位于主仓库。以下命令将候选库写入独立目录：

```bash
CUDA_HOME=/home/tong/tongworkspace/free/isaaclab-IPC/build/turbompc/toolchains/cuda-13.0.2 \
TURBOMPC_NATIVE_BUILD_DIR="$PWD/build/turbompc/native-local/cu130" \
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
  /home/tong/tongworkspace/free/isaaclab-IPC/build/turbompc/envs/cu130/bin/python \
  scripts/build_turbompc.py

.venv/bin/python scripts/build_pathcover_native.py \
  --cuda-home /home/tong/tongworkspace/free/isaaclab-IPC/build/turbompc/toolchains/cuda-13.0.2 \
  --cuda-arch 89 --jobs 2 \
  --build-dir build/pathcover_native/local/objects \
  --output-dir build/pathcover_native/local
```

运行 MPC 时用 `TURBOMPC_NATIVE_BUILD_DIR` 选择已验收的目录。前端的 `--output-dir` 只生成候选库；省略该选项会写入当前工作树的 `src/isaaclab_ipc`。RTX 4090 使用架构 89，其他 GPU 需指定其计算能力。原生构建和数值检查见 [TurboMPC 原生实现](docs/turbompc-native.md)。

A1 和 A2 使用相同的动力学、输入界、速度界与 SFC 硬约束。安全距离只在前端处理一次。历史 IPC 的 jerk 模型、源文件和验证数据继续保留，其入口仍为 `scripts/evaluate_native.py` 和 MARSIM `--policy ipc`。

CTBR 的 Newton 与 MARSIM 工厂使用零额外参考余量。机体间隙由原始占据盒与机体椭球处理，MPC 使用原始 SFC。独立 PathCover 的默认参考余量仍为 0.15 m；它只作用于惩罚中的半空间副本，并按走廊及交集的内切半径裁减。日志记录实际配置。

前端返回数值有限的优化参考，并记录软约束残差。`optimizer_quadrature_feasible` 只表示原求积节点质量判据，不证明完整曲线可行。完整参考质量、MPC 求解状态和实际执行约束分别验收。MPC 对预测节点施加输入界、速度界和 SFC 硬约束；未收敛调用仍记为求解失败。

CTBR 两端的 GCOPTER 迭代上限为 2000，原停止阈值保持不变。真实 Dense 失败输入在原 1000 步上限处仍持续下降，并在第 1019 步满足原阈值。独立 PathCover 默认上限仍为 1000。该配置的正式任务与延迟以验证记录为准。

## 统一入口

### Isaac Lab / Newton

```bash
export PYTHONPATH="$PWD/src"
export TURBOMPC_NATIVE_BUILD_DIR="$PWD/build/turbompc/native-v35/cu130"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2

# A0：Forest 导航任务，无几何前端
.venv/bin/python scripts/acmpc.py train --task Isaac-ACMPC-PathCover-Forest-Direct \
  --num-envs 16 --iterations 32 --audit-training --output results/my-acmpc \
  env.frontend=none env.max_planes=0 agent.seed=0 \
  agent.num_steps_per_env=16 agent.algorithm.num_mini_batches=2 \
  agent.algorithm.num_learning_epochs=1 \
  agent.actor.qp_linear_solver=cudss_inverse_chunk

# A1：PathCover + 固定成本 CTBR
.venv/bin/python scripts/acmpc.py evaluate --task Isaac-ACMPC-PathCover-Forest-Direct \
  --fixed-cost --num-envs 1 --output results/my-pathcover-fixed

# A2：PathCover + 学习成本 CTBR
.venv/bin/python scripts/acmpc.py train --task Isaac-ACMPC-PathCover-Forest-Direct \
  --num-envs 16 --iterations 32 --audit-training --output results/my-pathcover-acmpc \
  agent.seed=0 agent.num_steps_per_env=16 \
  agent.algorithm.num_mini_batches=2 agent.algorithm.num_learning_epochs=1 \
  agent.actor.qp_linear_solver=cudss_inverse_chunk

# IPC 的 Newton 独立回归；使用同一原生 TurboMPC
.venv/bin/python scripts/evaluate_native.py --num-envs 1 --stride 10 \
  --distance 1 --output results/my-native-check
```

上述 A0/A2 使用相同的 Forest 与训练预算，各采集 8192 次交互。`Isaac-ACMPC-CTBR-Direct` 用于空场景数值与性能基线。输出目录必须是新目录；继续训练时添加 `--checkpoint <路径>`，环境从新回合开始。学习成本评估将 `--fixed-cost` 换为 `--checkpoint <路径>`。修改节点数使用 `env.ctbr_model.cost_nodes`，步长使用 `env.ctbr_model.dt`。

Newton 的策略周期为 0.1 s，电机反馈和物理周期为 1/300 s。训练与 MARSIM 使用相同的 CTBR 观测和检查点语义。Avoid 任务保留突发移动障碍压力测试，其结果独立记录。

### MARSIM

```bash
# 当前 worktree 已构建的独立动态执行器
export IPC_MARSIM_RUNTIME="$PWD/build/marsim-rp-priority-v20"
source scripts/marsim_ros2_env.sh
export DISPLAY=:11 ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export TORCHINDUCTOR_CACHE_DIR="$PWD/build/torchinductor"

# A1：原生 PathCover + 固定成本 CTBR
.venv/bin/python scripts/run_marsim_ros2.py --mode dynamic --time-mode sim \
  --benchmark high_speed --policy fixed-ctbr --frontend pathcover \
  --reference-speed 3 --capture-reference-plans --timeout 180 --domain 232 \
  --render-prefix 'vglrun -d egl0' --output results/my-marsim-fixed

# 学习成本 AC-MPC；检查点必须使用当前原生 Actor 格式
.venv/bin/python scripts/run_marsim_ros2.py --mode dynamic --time-mode sim \
  --benchmark high_speed --policy acmpc --checkpoint results/my-pathcover-acmpc/checkpoint.pt \
  --frontend pathcover --reference-speed 3 --capture-reference-plans \
  --timeout 180 --domain 233 --render-prefix 'vglrun -d egl0' \
  --output results/my-marsim-acmpc
```

`dense` 场景修改 `--benchmark dense` 与输出目录。A0 使用其无前端训练 checkpoint，并加 `--frontend none`。每次运行使用新输出目录和空闲 domain。`:11` 与 `egl0` 是当前机器的显示配置。原 SUPER 的 `ideal/dynamic` 双执行模式继续保留。完整任务、实际失败和时钟证据见 [MARSIM 评测](docs/marsim-ros2.md)。

原生 PathCover 的正式基线采用官方 3 m/s 名义参考速度，与 Newton 训练一致。
使用 `--reference-speed` 可显式调整，最终值仍受场景和 CTBR 速度界限制。
`nominal_velocity` 控制速度成本，`max_velocity` 用于独立的物理参考质量检查。
默认物理参考上限为 High-speed 10 m/s、Dense 8 m/s、Newton 10 m/s。
这些值取自场景与 CTBR 速度界。优化器固定实测初始位置、速度与推力轴；初始总推力参与优化，测量加速度只提供初值。
日志分别记录名义速度超额和物理约束违反。
控制周期与原始任务点保持独立配置。

当前 CTBR 参考与 MPC 的物理推力范围均为 `[0, 22.0725] N`，质量为 `0.75 kg`。参考成本使用 `2.094972067 N` 的正名义下界与物理上界。参考倾角界为 `π/2`，速度惩罚权重为 `100`；参考软倾角界不是 MPC 的硬倾角约束。独立 PathCover 默认与历史 `6 m/s` 配置分别记录。

### 完整参考的离线检查

启用 `--capture-reference-plans` 并结束运行后，使用可提交的通用检查入口：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src .venv/bin/python scripts/analyze_marsim_references.py \
  --run-dir results/my-marsim-fixed \
  --output-dir results/my-marsim-fixed-analysis
```

该工具按时间戳和计划编号精确匹配全部已返回参考，并检查保存的多项式、yaw、原 SFC 和初末边界。失败尝试与窗口外计划分别保留。它使用 CPU，不重新运行前端或生成 yaw。

结果写入 `summary.json`、`summary.md` 和逐计划 `references.json`。退出码 0 表示报告生成完成；质量结论读取 `quality.reference_quality_passed`，并与任务结果和全部规划尝试的延迟分别判断。零返回计划的质量为 `null`；A0 无 PathCover，不适用该参考检查。

报告保存输入、分析代码、运行提交、原生库、执行器及 checkpoint 来源。可选提供保存的 `--identity`、`--identity-start`、`--identity-end` 三份记录核对发布身份。多项式根和角速度局部精化不构成区间证明；100 Hz 实际 SFC 采样也不能替代连续执行证明。

身份核对支持 schema 2 和 schema 3。schema 3 的配置清单引用、配置变更与 P0 复用来源按捕获记录保留。工具不再次核验配置清单文件，也不要求前端构建提交等于运行提交。

可选的 `--frontend-manifest` 只接受含 `so_sha256` 的原生库发布清单。只改变运行配置时，应提供实际复用的原生库发布清单。配置清单不能替代原生库发布清单。

## 验证

当前分支的配置、实测值与未通过项见 [原生 AC-MPC 验证](docs/acmpc-native-validation.md)。历史文档的通过结论不代替新主线验收。

原生 TurboMPC 的 SFC、隐式梯度、线性／CTBR 模型与 CUDA 后端共享同一套数值测试。

```bash
PYTHONPATH="$PWD/src" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
uv run ruff check src scripts tests
```

源码与文档由 Git 保存；`results/`、`recordings/`、`build/` 和 `.venv/` 不提交。评测中的碰撞、超时、QP 失败及物理约束违反均保留在原始输出目录。

相关文档：[训练与奖励](docs/acmpc.md)、[物理执行协议](docs/execution-profiles.md)、[TurboMPC 原生实现](docs/turbompc-native.md)、[CTBR 模型](docs/turbompc-ctbr.md)、[来源与许可](docs/provenance.md)。
