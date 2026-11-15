# AC-MPC 路径进度优化

## 当前实现

`progress` 模式保留 PathCover 和 4D CTBR 执行接口。优化状态为
`[p_world, q_wxyz, v_world, s]`，优化输入为 `[F_N, omega_body, nu]`。
虚拟状态满足 `s_next = s + 0.1 * nu`。默认保留 N=5，即 0.4 s 预测窗口。

MINCO 的曲线按每段归一化参数采样，转为近似弧长。控制器不再追踪原时间表、
原速度或原加速度。每次规划在当前进度附近建立局部模型：

```
r(s) = r(s_bar) + tangent(s_bar) * (s - s_bar)
```

该局部模型在一次求解期间固定。位置与进度、速度与虚拟速度的交叉项进入
同一个二次成本和 KKT 系统。物理 CTBR 动力学、CUDA QP、隐式梯度后端均复用。
140D 成本头保持不变；策略输入增加各节点几何信息，从 69D 变为 104D。

## 安全空间与进度

PathCover 每次重规划返回最多 3 个有序 SFC。各节点根据几何进度预览选择
原始多面体，并将进度限制在对应弧长区间。精确交界同时使用相邻走廊。
旧固定时间截面的前后 ±0.5 m 约束在 `progress` 模式中不使用。

每条新局部路径从实测位置开始，因此局部 s0=0；不会将上一条曲线的虚拟
进度累加到新曲线。低速时用至少 1 m/s 的几何预览选择线性化点，以避免
整段预测重复使用起点附近的退化切线。该预览不改变实测速度、虚拟速度初值
或进度上下界。

物理执行、到达半径、终止条件和实际 PCD 碰撞判定保持原配置。ROS 的 SFC
执行残差使用当前执行区间对应的 MPC 多面体，不再使用 MINCO 时钟选走廊。

## 成本与本轮参数

原 14D 可学习残差包含位置/进度误差、姿态正则、速度/虚拟速度一致性和
CTBR 控制正则。额外固定成本为沿切线滞后误差、虚拟速度正则、相邻虚拟
速度差和末端进度奖励。

| 参数 | 本轮训练及评测 |
| --- | ---: |
| 进度奖励 | 12 |
| 滞后误差权重 | 50 |
| 虚拟速度权重 | 0.1 |
| 虚拟速度变化权重 | 0.2 |
| 虚拟速度上限 | 6 m/s |
| 节点数 / 步长 | 5 / 0.1 s |

进度奖励 12 来自训练地图的固定成本诊断。默认参数 6 保留为初始设置；
正式命令显式指定 12，评测从同一检查点恢复参数。神经成本与物理速度上限
均未借此修改。

## 已完成的机制检查

测试覆盖：MINCO 时间缩放不改变空间采样；新增 s 动力学的导数；位置/进度
和速度/虚拟速度的成本耦合；原始 SFC 与弧长区间；走廊交界；一次原生
KKT 成本梯度与有限差分；起步几何预览。原 CTBR 回归测试保持通过。

Newton Forest 诊断保存在 `results/acmpc-progress-20261115/`：

| 配置 | 到达时间 | 碰撞 / QP 失败 |
| --- | ---: | ---: |
| 固定时间历史基线 | 3.50 s | 0 / 0 |
| 原起步采样，进度奖励 6 | 4.00 s | 0 / 0 |
| 原起步采样，进度奖励 12 | 5.10 s | 0 / 0 |
| 修正几何预览，进度奖励 6 | 3.60 s | 0 / 0 |
| 修正几何预览，进度奖励 12 | 2.80 s | 0 / 0 |

原起步采样在静止时给所有节点同一条近乎竖直的切线，导致先爬升后错过目标。
诊断状态记录见 `reward12-trace.npz`；修正只改变几何线性化预览。

## 运行

```bash
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TURBOMPC_NATIVE_BUILD_DIR="$PWD/build/turbompc/native-v35/cu130"
export LD_LIBRARY_PATH="$PWD/build/turbompc/envs/cu130/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

.venv/bin/python scripts/acmpc.py train \
  --task Isaac-ACMPC-PathCover-Forest-Direct --num-envs 16 --iterations 32 \
  --output results/my-progress-training \
  env.sfc_constraint_mode=progress env.progress_config.reward=12.0 \
  agent.seed=0 agent.num_steps_per_env=16 \
  agent.algorithm.num_mini_batches=2 agent.algorithm.num_learning_epochs=1 \
  agent.actor.compile_kernels=False
```

MARSIM 配对测试给固定成本和学习成本传入同一检查点。两者只改变 `--policy`，
其余参数、原生求解器和编译模式保持一致：

```bash
export IPC_MARSIM_RUNTIME="$PWD/build/marsim_ros2_relocated"
source scripts/marsim_ros2_env.sh
DISPLAY=:11 .venv/bin/python scripts/run_marsim_ros2.py \
  --mode dynamic --time-mode sim --benchmark dense \
  --policy fixed-ctbr --checkpoint results/my-progress-training/checkpoint.pt \
  --frontend pathcover --sfc-constraint-mode progress --domain 164 \
  --render-prefix 'vglrun -d egl0' --output results/my-progress-dense-fixed \
  --timeout 180
```

学习成本使用 `--policy acmpc`；High-speed 使用 `--benchmark high_speed`。
分别使用新的输出目录和 ROS domain。`polytope`、`section` 保持固定时间语义。
