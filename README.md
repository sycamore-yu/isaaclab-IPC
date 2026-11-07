# Isaac Lab IPC

本分支建立原生 **AC-MPC + PathCover** 导航主线。Isaac Lab / Newton 负责物理、LiDAR 和 PPO 训练；MARSIM / Flightmare 负责正式评测。架构决策见 [ADR-0008](docs/adr/0008-native-acmpc-pathcover.md)。

| 配置 | 前端 | 成本 | 输出 |
| --- | --- | --- | --- |
| A0 | 目标观测，无几何前端 | 逐节点学习成本 | TurboMPC → 4D CTBR |
| A1 | 完整原生 PathCover | 固定二次成本 | 同一 TurboMPC → 4D CTBR |
| A2 | 与 A1 相同 | 逐节点学习成本 | 同一 TurboMPC → 4D CTBR |

PathCover 原生前端依次执行 NanoVoxMap、JPS/DMP、RISP 和 GCOPTER/MINCO。它输出连续 PVA/yaw、轨迹段时间及有序 SFC。Actor 读取冻结的观测，用逐节点成本构造非线性 MPC；PyTorch 原生 TurboMPC 复用 CUDA/cuDSS 求解与反向传播。

- **训练**：Isaac Lab 3 / Newton (MJWarp)，真实电机力、接触和 LiDAR。使用标准 RSL-RL PPO；Actor 学习成本，Critic 估计长期价值。
- **正式闭环评测**：MARSIM + Flightmare，控制频率 100 Hz 仿真时间，MPC 以 10 Hz 仿真时间重规划。也保留可选墙钟模式，用于计算延迟测试。
- **外部对照**：原版 [SUPER](https://github.com/hku-mars/SUPER) 独立运行；其代码不属于上述两条路径。
- **预测模型**：10D CTBR，状态为位置、WXYZ 四元数和世界速度。优化控制为总推力 N 与机体角速度，执行时将总推力除以质量。
- **成本节点**：默认 N=5，含当前节点，共 4 段 0.1 s 转移。Actor 输入 69 维特征，输出 140 个成本参数。旧 7 参数检查点不适用于该契约。

## 依赖与构建

Linux、Python 3.12、CUDA 13、Isaac Lab 3、Newton、Warp 和原生 TurboMPC CUDA 扩展。原生前端还需要 Eigen3、Boost 头文件和 Qhull。使用项目锁定的 `uv.lock`；参考源码保持只读。

```bash
uv sync
uv run python scripts/prepare_mid360.py
uv run python scripts/build_pathcover_native.py
# 原生 CUDA 扩展的版本、编译方法与数值检查见 docs/turbompc-native.md。
```

若 CUDA 工具链不在默认构建目录，给前端构建命令添加 `--cuda-home <toolkit目录>`。RTX 4090 使用默认 `--cuda-arch 89`，其他 GPU 应指定其计算能力。独立 worktree 可以共享现有运行环境，但应使用当前 worktree 的 `src` 和扩展构建目录。

A1 和 A2 使用相同的动力学、输入界、速度界与 SFC 硬约束。安全距离只在前端处理一次。历史 IPC 的 jerk 模型、源文件和验证数据继续保留，其入口仍为 `scripts/evaluate_native.py` 和 MARSIM `--policy ipc`。

## 统一入口

### Isaac Lab / Newton

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# A0：独立 AC-MPC 基线
.venv/bin/python scripts/acmpc.py train --task Isaac-ACMPC-CTBR-Direct \
  --num-envs 4 --iterations 32 --output results/my-acmpc

# A1：PathCover + 固定成本 CTBR
.venv/bin/python scripts/acmpc.py evaluate --task Isaac-ACMPC-PathCover-Forest-Direct \
  --fixed-cost --num-envs 1 --output results/my-pathcover-fixed

# A2：PathCover + 学习成本 CTBR
.venv/bin/python scripts/acmpc.py train --task Isaac-ACMPC-PathCover-Forest-Direct \
  --num-envs 4 --iterations 32 --output results/my-pathcover-acmpc

# IPC 的 Newton 独立回归；使用同一原生 TurboMPC
.venv/bin/python scripts/evaluate_native.py --num-envs 1 --stride 10 \
  --distance 1 --output results/my-native-check
```

森林任务需先生成 `build/forest`。A0 与 A2 比较训练结果时，应使用同一 Forest 任务；A0 添加 `env.frontend=none env.max_planes=0`。`Isaac-ACMPC-CTBR-Direct` 用于独立数值与性能基线。

Newton 的策略周期为 0.1 s，电机反馈和物理周期为 1/300 s。训练与 MARSIM 使用相同的 CTBR 观测和检查点语义。Avoid 任务保留突发移动障碍压力测试，其结果独立记录。

### MARSIM

```bash
source scripts/marsim_ros2_env.sh
export DISPLAY=:11

# A1：原生 PathCover + 固定成本 CTBR
.venv/bin/python scripts/run_marsim_ros2.py --mode dynamic --time-mode sim \
  --benchmark high_speed --policy fixed-ctbr --timeout 120 --domain 130 \
  --render-prefix 'vglrun -d egl0' --output results/my-marsim-fixed

# 学习成本 AC-MPC；检查点必须使用当前原生 Actor 格式
.venv/bin/python scripts/run_marsim_ros2.py --mode dynamic --time-mode sim \
  --benchmark high_speed --policy acmpc --checkpoint results/my-pathcover-acmpc/checkpoint.pt \
  --timeout 120 --domain 131 --render-prefix 'vglrun -d egl0' \
  --output results/my-marsim-acmpc
```

`dense` 场景修改 `--benchmark dense` 与输出目录。A0 使用 `--frontend none`。原 SUPER 的 `ideal/dynamic` 双执行模式继续保留。完整任务、实际失败和时钟证据见 [MARSIM 评测](docs/marsim-ros2.md)。

原生 PathCover 默认名义参考速度为 6 m/s，与 Newton 训练一致。
使用 `--reference-speed` 可显式调整，最终值仍受场景和 CTBR 速度界限制。
优化器保留真实初始 PVA。若初速已超出名义值，速度验收界限为
`max(名义速度, 初速) + 0.001 × 名义速度`；日志保留相对名义值的实际超额。
控制周期与原始任务点保持独立配置。

## 验证

当前分支的配置、实测值与未通过项见 [原生 AC-MPC 验证](docs/acmpc-native-validation.md)。历史文档的通过结论不代替新主线验收。

原生 TurboMPC 的 SFC、隐式梯度、线性／CTBR 模型与 CUDA 后端共享同一套数值测试。

```bash
PYTHONPATH="$PWD/src" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
uv run ruff check src scripts tests
```

源码与文档由 Git 保存；`results/`、`recordings/`、`build/` 和 `.venv/` 不提交。评测中的碰撞、超时、QP 失败及物理约束违反均保留在原始输出目录。

相关文档：[训练与奖励](docs/acmpc.md)、[物理执行协议](docs/execution-profiles.md)、[TurboMPC 原生实现](docs/turbompc-native.md)、[CTBR 模型](docs/turbompc-ctbr.md)、[来源与许可](docs/provenance.md)。
