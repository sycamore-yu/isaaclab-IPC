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
140D 成本头保持不变；策略输入增加各节点几何信息，从 69D 变为 109D。

## 安全空间与进度

PathCover 每次重规划返回最多 3 个有序 SFC。各节点根据几何进度预览选择
原始多面体，并将进度限制在对应弧长区间。精确交界同时使用相邻走廊。
旧固定时间截面的前后 ±0.5 m 约束在 `progress` 模式中不使用。

每条新局部路径从实测位置开始，因此局部 s0=0；不会将上一条曲线的虚拟
进度累加到新曲线。低速时用至少 1 m/s 的几何预览选择线性化点。对整条
局部曲线计算曲率速度上限，并沿弧长反向传播制动速度曲线。到任务终点时，
按剩余实际距离约束虚拟进度速度。MINCO 只决定空间路径形状及其切线、
曲率，不向控制器施加固定的时间速度参考。

物理执行、到达半径、终止条件和实际 PCD 碰撞判定保持原配置。ROS 的 SFC
执行残差使用当前执行区间对应的 MPC 多面体，不再使用 MINCO 时钟选走廊。

## 成本与本轮参数

原 14D 可学习残差包含位置/进度误差、姿态正则、速度/虚拟速度一致性和
CTBR 控制正则。额外固定成本为轮廓误差、速度与虚拟进度的一致性、
沿切线滞后误差、虚拟速度正则、相邻虚拟速度差和末端进度奖励。

| 参数 | 本轮训练及评测 |
| --- | ---: |
| 进度奖励 | 12 |
| 滞后误差权重 | 50 |
| 虚拟速度权重 | 0.1 |
| 虚拟速度变化权重 | 0.2 |
| 虚拟速度上限 | 4 m/s |
| 最大侧向参考加速度 | 3.5 m/s² |
| 进度制动加速度 | 4 m/s² |
| 轮廓误差权重 | 25 |
| 速度一致性权重 | 10 |
| 节点数 / 步长 | 5 / 0.1 s |

进度奖励 12 来自训练地图的固定成本诊断。上述配置用于后续完整闭环，
神经成本维度、物理输入界和四旋翼动力学未修改。原方案的奖励 6、
速度上限 6 m/s 已保留在 Git 历史中。

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
  env.sfc_constraint_mode=progress \
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

## 后续修复与结果

初版在 Dense 的急弯处出现 QP 未收敛和 PathCover 初始运动种子不可行。
原控制器曾执行未收敛的极端控制解。现在仅在 MPC 成功时更新热启动；
失败时保持上一次有效 CTBR 控制，不增加第二套控制器。
同时根据实际转弯几何生成加速度/姿态参考，强化实际速度与进度耦合，
对全局部路径反向传播弯道制动速度曲线。
新增虚拟进度导数直接复用 CTBR 物理雅可比与 Hessian。

固定成本完整 MARSIM 结果，seed 0、PathCover、动态 Flightmare、
100 Hz 仿真控制、10 Hz 进度 MPC：

| 场景 | 固定进度历史基线 | 自由进度 | 实际净空 | QP 失败 | 规划 p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 95.54 s | **84.19 s** | 0.218 m | 0 | 551.21 ms |
| High-speed | 70.99 s | **55.56 s** | 0.631 m | 0 | 235.93 ms |

以上自由进度任务均完成 2/2 航点，碰撞及实际 SFC 违反均为 0。
相比固定进度，飞行时间分别缩短约 11.9% 与 21.7%。MPC 计算延迟
明显上升，尚不符合 100 ms 的墙钟规划目标。
记录分别在 `results/acmpc-progress-20261115-v2/marsim-{dense,high-speed}-fixed-v5/`。
初版及中间失败尝试保留在 `results/acmpc-progress-20261115/` 和
`results/acmpc-progress-20261115-v2/`，不覆盖历史记录。
