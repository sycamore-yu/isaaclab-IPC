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

## 配对训练和闭环结果（2026-11-15，最终 V5）

训练前半段在第 19 次更新附近遇到前端失败：失败行仍使用 7D
`progress_reference` 占位数据，而当前观测为 8D。修复该接口后，从
`train-v5-8192/model_16.pt` 恢复第 17–31 次更新，累计完成 32 次
PPO 更新、8192 次环境交互。最终检查点：

`results/acmpc-progress-20261115-v2/train-v5-resume17-31/checkpoint.pt`

SHA256：
`ffdabf4236453f7b56c32a8beab205372dde74d323462a124cde7c3b1a9dd974`

最终检查点动作重载通过。5 个 Newton Forest 配对种子均成功到达，
无碰撞、QP 未收敛或前端失败。固定成本平均飞行时间 **2.68 s**，
学习成本 **2.72 s**，学习未提高短程任务速度。

4 场正式 MARSIM 使用上述**同一个检查点**，物理动力学、PathCover
几何、自由进度和求解器设置一致。仅切换固定成本/学习成本：

| 指标 | Dense 固定 | Dense 学习 | High-speed 固定 | High-speed 学习 |
| --- | ---: | ---: | ---: | ---: |
| 完成航点 | 2/2 | 2/2 | 2/2 | 2/2 |
| 飞行时间（s） | 87.55 | **81.96** | 55.59 | 55.57 |
| 飞行距离（m） | 229.23 | 228.06 | 199.48 | 199.50 |
| 最小障碍净空（m） | 0.166 | 0.228 | 0.616 | 0.643 |
| MPC 未收敛（次） | 2 | 2 | 0 | 0 |
| 实际 SFC 最大正违反（mm） | 9.27 | 6.53 | 0 | 0 |
| 规划耗时 p95（ms） | 756.73 | 829.63 | 401.13 | 391.51 |

四场均无物理碰撞、无超时，且实际达到全部官方航点。
Dense 两组各有 **1 个控制时刻**出现 SFC 正违反：
固定成本为 9.70 s，学习成本为 37.20 s。该项与物理障碍碰撞
分开记录，不能写作零 SFC 违反。

相同自由进度控制器下，学习成本令 Dense 到达时间缩短 5.59 s
（约 6.4%）。High-speed 的 0.02 s 差异不构成明显效率优势。
两场学习策略均略微提高最小障碍净空。历史固定时间基线
Dense 95.54 s、High-speed 70.99 s，用于算法进展参考；
它们不是本次重新运行的第五、第六个配对实验。

本轮使用非编译推理（`compile_kernels=False`）。规划 p95 仍超过
100 ms，实时性和 Dense 的少量数值约束违反是后续性能改进目标，
**不影响上述已观测的到达时间结果**。

原始数据与完整机器可读汇总：
`results/acmpc-progress-20261115-v2/marsim-{dense,high-speed}-{fixed,learned}-paired-v5/`
和 `results/acmpc-progress-20261115-v2/paired-v5-summary.json`。
Rerun 录制位于 `recordings/acmpc-progress-{dense,high-speed}-{fixed,learned}-paired-v5.rrd`。

## SUPER 同场景对照、虚拟速度与求解预算（2026-11-15）

使用仓库已有的 ROS 2 SUPER 接口，重新编译原生
ROG-Map / CIRI / MINCO / FSM。SUPER 和 AC-MPC 共享 MARSIM
Flightmare 刚体、动力学参数、点云地图和两个官方航点。
SUPER 通过位置/PVA 轨迹跟踪器执行，AC-MPC 直接执行 CTBR；
两者的**规划与控制接口不同**，但动力学执行器相同。

| 算法与控制 | Dense | High-speed |
| --- | --- | --- |
| SUPER 原生 ROS 2 | 4.15 s 物理碰撞，0/2 到达 | 21.13 s，2/2 |
| AC-MPC 旧版固定进度，固定成本 | 95.54 s，2/2 | 70.99 s，2/2 |
| AC-MPC 自由进度，学习成本，原 4 m/s 上限 | 81.96 s，2/2 | 55.57 s，2/2 |
| AC-MPC 自由进度，学习成本，场景原限速 + 8/1024 预算 | **83.23 s，2/2** | **47.29 s，2/2** |

原生 SUPER High-speed 飞行距离 205.03 m，实测最大速度
25.57 m/s，最小障碍净空 0.421 m。新版 AC-MPC High-speed
飞行距离 201.30 m，实测最高速度 6.55 m/s，最小障碍净空
0.651 m。原生 SUPER High-speed 飞行时间约为新版 AC-MPC
的 44.7%。目前真实速度差距远大于单纯的规划计算耗时差距。

SUPER Dense 在相同 Flightmare 和 0.20 m 碰撞半径下复测两次，
均在 4.15 s 发生碰撞。另一次把独立碰撞评测半径临时缩至
0.195 m，仍在 4.15 s 碰撞。原半径下，首次碰撞时的 SUPER
**参考位置**仍有 0.394 m 原始 PCD 净空，但实际机体与参考
位置偏差 0.409 m，实际净空为 −0.0024 m。问题主要是
轨迹跟踪误差，不是参考位置与障碍的几何重叠；不能将
4.15 s 作为 SUPER 的成功到达时间。

为了与 SUPER 的场景配置对应，MARSIM 自由进度虚拟速度上限
由 4 m/s 调整为 **Dense 8 m/s / High-speed 25 m/s**。
通过模型加载后替换进度配置完成，不修改学习网络权重，
也不修改 Newton 训练配置。原生 SFC、Flightmare 的
0.20 m 碰撞半径与高度边界保持不变。

该上限仅约束虚拟进度 `nu`。实际 CTBR 预测模型仍限制
速度各轴不超过 10 m/s、机体系角速度为 (3,3,1.5) rad/s；
曲线参考仍使用侧向加速度 3.5 m/s²、制动加速度 4 m/s²。
这些限制，以及 0.4 s 的 MPC 预测窗口，比 25 m/s
虚拟速度上限更直接地影响 High-speed 的实际飞行速度。

冻结同一份 Dense / High-speed SQP 输入测试表明：
把 SQP/QP 迭代预算从 25/4096 缩到 8/1024，
首个 CTBR 动作的最大分量变化分别不超过约 0.0005。
三份输入均出现未严格收敛的状态码，但其原始不等式
残差量级为 10⁻⁷ m。MARSIM 推理现在允许
SQP 状态为迭代上限、原始残差不超过 0.001、
预测 SFC 正违反不超过 0.01 m，且观测有效的解执行。
原始状态码、严格收敛状态和提前接受次数均保留记录；
物理碰撞与原始 PCD 检验不变。

原地图边界下的重新运行：

| 指标 | Dense 学习成本 | High-speed 学习成本 |
| --- | ---: | ---: |
| 航点 / 碰撞 | 2/2、0 | 2/2、0 |
| SQP 严格收敛 / 提前接受 | 712 / 95 | 434 / 12 |
| 被拒绝的 MPC 解 | 0 | 0 |
| 实际 SFC 最大正违反 | 0 | 0 |
| 完整规划 p95 | 468.90 ms | 308.48 ms |

原始数据：
`results/acmpc-super-speed-match-20261115/`，
对应成功目录
`dense-learned-fast8-1024-native-ceiling/`、
`high-speed-learned-fast8-1024-native-ceiling/`、
`super-dynamic-high_speed/` 以及 SUPER Dense 失败记录。
时间使用仿真时钟，计算耗时使用墙钟；当前离线重放
不把 100 ms 规划 p95 当作到达率的独立拒绝条件。
实际硬件部署时，规划耗时仍会影响控制滞后。

### High-speed 几何侧向加速度消融

为判断 25 m/s 虚拟速度仍无法达到 SUPER 速度的原因，
另使用相同的 Actor 权重，把**单一**非学习参数
`progress_config.lateral_acceleration` 从 3.5 提至
10 m/s²。其余成本、参考和动力学参数不变；新检查点位于
`results/acmpc-super-speed-match-20261115/checkpoint-lateral10-ablation.pt`，
对应源检查点、参数差分和 SHA256 记录在同目录 JSON。
所有 Actor 张量权重均与原检查点完全相同。

| High-speed | 侧向加速度 3.5 | 侧向加速度 10 |
| --- | ---: | ---: |
| 航点到达 | 2/2 | 2/2 |
| 飞行时间 | 47.29 s | **42.76 s** |
| 路径长度 | 201.30 m | 207.29 m |
| 平均距离/时间 | 4.26 m/s | **4.85 m/s** |
| 最大实际速度 | 6.55 m/s | 6.35 m/s |
| 最小 PCD 净空 | 0.651 m | 0.672 m |
| SQP 严格收敛 / 提前接受 | 434 / 12 | 392 / 9 |
| 被拒绝的 MPC / 碰撞 / SFC 正违反 | 0 / 0 / 0 | 0 / 0 / 0 |
| 规划 p95 | 308.48 ms | **234.81 ms** |

改进来自更高的**持续平均速度**，而不是更高的最高瞬时速度。
这说明局部 MINCO 曲率与基于曲率的速度限制是当前主要
飞行时间瓶颈之一。10 m/s² 为成功的单场候选，
不自动替换所有场景的训练配置。
对应原始结果在
`results/acmpc-super-speed-match-20261115/high-speed-learned-lateral10-final/`；
Rerun 位于
`recordings/acmpc-progress-high-speed-lateral10-20261115.rrd`。

### 参考 Isaac-PathCover 的预见范围与速度消融

`Isaac-PathCover` 沿用 SUPER 的按米数规划视野：
Dense 7 m，High-speed 30 m，并保留旧轨迹前缀。
本仓库的 DMP 每次搜索到官方航点，但 PathCover 仅取
前 3 个安全走廊生成局部 MINCO 轨迹。局部轨迹长度
不固定；MPC 另有 0.4 s 的短预测窗口，每 0.1 s 重规划。

为直接统计真实局部预见距离，进度采样现在把已计算的
局部弧长传给 `local_path_length_m` 指标，不再重复积分。
本轮 High-speed 最佳配置的局部轨迹长度中位数为
**11.73 m**（p95 16.34 m）；Dense 名义速度 8 m/s
实验的中位数为 **4.21 m**（p95 6.32 m）。

为改善速度，保持 Actor、原 SFC、CTBR 刚体预测及
Flightmare 电机参数不变，分别修改一个几何规划参数：

| High-speed 配置 | 结果 | 飞行时间 | 路径长度 | 规划 p95 |
| --- | --- | ---: | ---: | ---: |
| 侧向加速度 3.5；MINCO 名义速度 3；3 SFC | 2/2 | 47.29 s | 201.30 m | 308.48 ms |
| 侧向加速度 10；名义速度 3；3 SFC | 2/2 | 42.76 s | 207.29 m | **234.81 ms** |
| **侧向加速度 10；名义速度 10；3 SFC** | **2/2** | **41.90 s** | **199.43 m** | 275.28 ms |
| 侧向加速度 20；名义速度 3；3 SFC | 2/2 | 46.18 s | 223.19 m | 442.75 ms |
| 侧向加速度 10；名义速度 3；6 SFC | 1/2，前端中断 | 37.46 s | 176.50 m | 407.37 ms |

完成的 4 场 High-speed 测试均为 0 碰撞、0 次被拒绝的
MPC 解、0 次实际 SFC 正违反。扩大到 6 SFC 增加了
计算开销，并在返航时触及原地图的有效高度限制，
因而已撤销该代码改动。

Dense 将 MINCO 名义速度从 3 提到 8 m/s 后，飞行
时间由 **83.23 s** 变成 **83.90 s**，同时出现
2 次被拒绝的 MPC 解。故只推荐 High-speed 使用
名义参考速度 10 m/s；Dense 继续使用 3 m/s。

最优 High-speed 直接用已有接口复现：

```bash
export IPC_MARSIM_RUNTIME="$PWD/build/marsim_ros2_relocated"
export TURBOMPC_NATIVE_BUILD_DIR="$PWD/build/turbompc/native-v35/cu130"
source scripts/marsim_ros2_env.sh
.venv/bin/python scripts/run_marsim_ros2.py \
  --mode dynamic --time-mode sim --benchmark high_speed \
  --policy acmpc --frontend pathcover --sfc-constraint-mode progress \
  --checkpoint "$PWD/results/acmpc-super-speed-match-20261115/checkpoint-lateral10-ablation.pt" \
  --reference-speed 10 --domain 178 --render-prefix 'vglrun -d egl0' \
  --output results/high-speed-reference10-repeat --timeout 120
```

**约束区分：** CTBR MPC 对预测速度施加每轴 ±10 m/s，
对角速度施加 (3,3,1.5) rad/s，并限制总推力加速度
0～29.43 m/s²；这些是预测优化的约束。
Flightmare 无独立平移速度截断，实际角速度限制为
10 rad/s，旋翼推力限制对应最大总推力加速度
29.43 m/s²，电机滞后时间常数为 0.01 s。
在此前 42.76 s 的成功飞行中，角速度 X/Y 触及
99% MPC 限幅的比例只有 0.50% / 0.75%，
最大推力加速度为 16.33 m/s²，真实速度峰值
为 6.35 m/s。因此当前加速重点应为曲线形状、
局部预见与进度成本，而非修改仿真动力学或
继续提高基本未触发的控制上限。

完整证据：
`results/acmpc-super-speed-match-20261115/speed-optimization-summary.json`。
最佳 High-speed Rerun：
`recordings/acmpc-progress-high-speed-ref10-20261115.rrd`。
该结果来自单次完整配对条件下的消融，不代表
多随机种子下的全局最优。
