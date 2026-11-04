# MARSIM ROS 2 评测

**当前验收状态：接口和构建已完成。SUPER high_speed 在 ideal、dynamic 两种模式下均完成完整往返。密集场景和 AC-MPC 的 100 Hz 策略输出要求尚未通过，不能认定全部验收完成。**

MARSIM 负责正式评测，提供 `ideal` 和 `dynamic` 两种执行模式。Isaac Lab/Newton 继续使用原训练入口。ROS 节点复用项目的 Planner、PathCover/RISP、MPCActor 和 QP，不启动 Isaac Lab 仿真进程。本文记录当前实现、复现方法和实际成功／失败结果。

## 1. 工作树与参考版本

本次开发分支为 `feat/marsim-ros2-acmpc-20261104`。独立工作树：

```text
/home/tong/tongworkspace/free/.worktrees/isaaclab-IPC/marsim-ros2-acmpc-20261104
```

原项目为 `/home/tong/tongworkspace/free/isaaclab-IPC`。原工作树改动及训练进程保留。

| 参考 | 路径 | 固定版本 |
|---|---|---|
| SUPER | `/home/tong/tongworkspace/reference_repos/SUPER` | `2ad3419c127a617c6d7df6925e81a14175a9c096` |
| Flightmare | `/home/tong/tongworkspace/reference_repos/flightmare` | `d4218aedac18cbe9364a0a0df10ab992c4b65e4f` |
| IPC | `/home/tong/tongworkspace/reference_repos/IPC/hku-mars` | 只读参考；实际策略复用本项目已有实现 |

构建脚本检查前两个仓库的版本，并通过 `git archive` 生成私有副本。参考仓库不接收补丁。来源写入 `build/marsim_ros2/sources.json`。

## 2. 环境与隔离构建

当前环境使用 ROS 2 Jazzy 和 Python 3.12。策略与监控使用项目 `.venv/bin/python`；ROS C++ 包通过系统 Python 构建。ROS 官方要求虚拟环境解释器与 ROS 二进制的 Python ABI 匹配，不能只靠追加 `PYTHONPATH` 解决版本差异。[Python 环境规范](https://github.com/ros2/ros2_documentation/blob/jazzy/source/How-To-Guides/Using-Python-Packages.rst)

构建需要 C++17、CMake、colcon、ament、Eigen、yaml-cpp、PCL、OpenCV、OpenMP、GLFW/OpenGL，以及 ROS 消息、TF 和 `pcl_conversions`。SUPER 任务包还依赖 `mavros_msgs`、`geographic_msgs` 和 `unique_identifier_msgs`。

本工作树已将缺少的依赖准备到以下前缀：

```text
build/marsim_ros2/deps/root/usr
build/marsim_ros2/deps/root/opt/ros/jazzy
```

构建脚本接受已有依赖前缀，不负责下载或安装系统软件。迁移机器时，先提供上述依赖，再构建。当前下载缓存位于 `build/marsim_ros2/deps/downloads`。

首次构建先加载系统 ROS。此时评测 overlay 可能尚不存在。

```bash
cd /home/tong/tongworkspace/free/.worktrees/isaaclab-IPC/marsim-ros2-acmpc-20261104
source /opt/ros/jazzy/setup.bash

python3 scripts/build_marsim_ros2.py \
  --super-root /home/tong/tongworkspace/reference_repos/SUPER \
  --flightmare-root /home/tong/tongworkspace/reference_repos/flightmare \
  --deps-prefix build/marsim_ros2/deps/root/usr \
  --deps-prefix build/marsim_ros2/deps/root/opt/ros/jazzy \
  --jobs 2

source scripts/marsim_ros2_env.sh
```

构建和安装都限制在 `build/marsim_ros2/`。脚本支持 `--prepare-only`、`--upstream-only` 和 `--packages`，供分阶段构建使用。通常先完整构建，再用 `--packages isaaclab_marsim` 增量编译桥接节点。

之后每个评测终端都执行：

```bash
source scripts/marsim_ros2_env.sh
```

这个脚本加载 Jazzy、私有依赖及评测 overlay。它设置项目源码路径和 `IPC_MARSIM_PYTHON`，并将 OpenMP/OpenBLAS 线程数设为 1。

### 本地点云传输

环境脚本为以下变量提供默认值；已有非空值保持不变：

| 变量 | 默认值 |
|---|---|
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_AUTOMATIC_DISCOVERY_RANGE` | `SYSTEM_DEFAULT` |
| `FASTRTPS_DEFAULT_PROFILES_FILE` | `$IPC_MARSIM_ROOT/ros2/isaaclab_marsim/config/fastdds_local.xml` |

该 Fast DDS profile 显式启用 32 MiB 共享内存（SHM）段，并将 SHM 消息上限设为 8 MiB。UDP 传输及发现地址限制为 `127.0.0.1`，保留单机评测范围。

本配置使用 `SYSTEM_DEFAULT`，由 XML 管理发现与传输。本机实测中，`LOCALHOST` 会旁路自定义 SHM 配置，导致约 2 MiB 点云丢帧。若旧终端已有该值，应在启动本次评测前改为：

```bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT
```

配置修复后的独立传输探针以 10 Hz 发送 60 帧、每帧 2 MiB，接收端全部收到。这只验证消息传输，不是飞行或导航验收证据。

### NVIDIA LiDAR 渲染

MARSIM 使用原 OpenGL LiDAR 渲染器。当前机器通过已有 X 显示 `:11` 与 VirtualGL 的 `egl0` 使用 NVIDIA GPU：

```bash
export DISPLAY=:11
```

统一 Launch 的渲染参数为 `render_prefix:="vglrun -d egl0"`。该值是当前机器配置；迁移机器需确认实际显示和 GPU 设备。仅有 ROS 话题存在，不能证明地图已加载或点云正确。

## 3. 一个 Launch，两个执行器

统一入口：

```bash
export ROS_DOMAIN_ID=124
export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT

ros2 launch isaaclab_marsim benchmark.launch.py \
  execution:=dynamic \
  benchmark:=high_speed \
  policy:=super \
  render_prefix:="vglrun -d egl0" \
  output:="$PWD/results/example-dynamic-super" \
  timeout:=120
```

使用未被其他实验占用的 ROS domain。原始话题名保持不变，因此同一 domain 只运行一个评测图。

| 参数 | 取值 | 行为 |
|---|---|---|
| `execution` | `ideal` | 启动原 `perfect_drone_node`，执行 PositionCommand |
| `execution` | `dynamic` | 启动 `flightmare_node` 和独立 `marsim_lidar_node` |
| `benchmark` | `high_speed`、`dense` | 加载对应原地图、LiDAR 和规划配置 |
| `policy` | `super` | 原 SUPER 规划器 |
| `policy` | `ipc` | 固定成本 IPC；内部 ROS 参数名为 `fixed` |
| `policy` | `acmpc` | 从检查点加载学习成本 MPCActor |
| `checkpoint` | 检查点绝对路径 | `acmpc` 必填 |
| `executor_parameters` | 可选 ROS 参数 YAML | 仅 dynamic 执行器；用于记录明确的参数对照 |
| `record` | `true`、`false` | 是否随 Launch 启动监控，默认开启 |
| `mission` | `true`、`false` | 是否启动原航点任务 |
| `rviz` | `true`、`false` | 默认关闭 |

Launch 按 `execution` 分支只创建一个执行器。监控完成，或执行器、渲染器、策略进程退出时，Launch 关闭本次运行。

`dynamic + super` 接收原 PositionCommand，由 C++ 位置跟踪器转换为 CTBR，再调用 Flightmare。该路径保留原 SUPER 的规划接口。

`dynamic + ipc/acmpc` 由 Python 节点直接发布 CTBR。`ideal + ipc/acmpc` 发布同一策略求得的 PositionCommand，由原 PerfectDrone 执行。两种模式都使用实际收到的速度构造观测。

ideal 发布端按 0.01 s 控制步长推进选定的 jerk，得到本步位置、速度和加速度；MPC 仍使用训练时的 0.1 s 预测步长。原 PerfectDrone 没有独立加速度话题，策略使用最近已发布的 PositionCommand.acceleration。dynamic 使用实际 AccelStamped。两种步长和加速度来源都写入策略 metadata。

### 推荐的自动记录入口

`run_marsim_ros2.py` 创建独立进程组，记录启动命令，并在结束时停止本次创建的进程。它默认使用 domain 124，也可以通过 `--domain` 指定。

停止 Launch 时，运行器先只向父 Launch 进程发送 SIGINT，由 Launch 关闭子节点，避免重复发送 SIGINT。等待 10 s 仍未退出时，才向本次进程组发送 SIGTERM。监控进程单独停止。

```bash
DISPLAY=:11 .venv/bin/python scripts/run_marsim_ros2.py \
  --mode dynamic --benchmark dense --policy ipc \
  --domain 124 --render-prefix "vglrun -d egl0" \
  --output results/example-dynamic-ipc-dense --timeout 120
```

每次使用新的输出目录。运行器拒绝复用已存在的目录；单独监控允许目录存在，但拒绝覆盖其结果文件。

### 执行器参数对照

默认跟踪增益仍为 `position_gain=4.0`、`velocity_gain=3.0`、`attitude_gain=6.0`。只有显式提供参数文件时才覆盖执行器参数。统一 Launch 使用 `executor_parameters:=/absolute/path/parameters.yaml`；运行器对应 `--executor-parameters /absolute/path/parameters.yaml`。运行器拒绝在 ideal 模式使用该选项，并将文件哈希写入 `run.json` 的 `executor_parameters_sha256`。

参数文件采用标准 ROS 2 YAML，例如本次已记录的增益对照文件 `results/marsim-ros2-20261104/tracker-gains-8-5-12.yaml`：

```yaml
flightmare_node:
  ros__parameters:
    position_gain: 8.0
    velocity_gain: 5.0
    attitude_gain: 12.0
```

该对照在 dense 场景更早碰撞，没有改为默认值。参数文件只用于可追溯的对照，不改变官方地图、航点、阈值或任务成功判据。复现时保留参数文件及其哈希。

## 4. 官方入口与完整任务

原任务入口仍可使用：

```bash
DISPLAY=:11 vglrun -d egl0 ros2 launch mission_planner benchmark_high_speed.launch.py
DISPLAY=:11 vglrun -d egl0 ros2 launch mission_planner benchmark_dense.launch.py
```

这两条命令分别启动，不能放在同一 ROS domain 同时运行。它们使用原 PerfectDrone。上游高速文件名实际为 `banchmark_high_speed.launch.py`；构建副本保留该文件，并添加正确拼写的别名。

记录官方入口的完整任务：

```bash
DISPLAY=:11 .venv/bin/python scripts/run_marsim_ros2.py \
  --mode ideal --benchmark high_speed --policy super --official \
  --domain 124 --render-prefix "vglrun -d egl0" \
  --output results/example-official-high-speed --timeout 120
```

dense 运行将 `--benchmark` 改为 `dense`，同时更换输出目录。

两项任务都使用原 `benchmark.txt`：

```text
0 50 1.5 1
0 -50 1.5 1
```

初始位置为 `[0, -50, 1.5]`。完整任务为名义距离 200 m 的往返。监控要求按顺序进入两个航点的 1 m 半径内，判定条件为严格小于 1 m。只完成去程或短距离动作，不算完整任务成功。

| 地图 | 原 PCD | 有效点数 | LiDAR 距离／频率 |
|---|---|---:|---|
| high_speed | `random_map_24_6635.pcd` | 186905 | 70 m／10 Hz |
| dense | `random_map_2_26609.pcd` | 412021 | 15 m／10 Hz |

配置来自 `src/isaaclab_ipc/config/super_benchmarks.yaml`。完整协议另见 [SUPER benchmark 说明](super-benchmarks.md)。

## 5. CTBR、坐标系与时钟

CTBR 表示总比推力与三轴机体角速度。命令使用 `geometry_msgs/TwistStamped`：

| 字段 | 含义 | 单位 |
|---|---|---|
| `twist.linear.z` | 所有旋翼总推力除以质量 | m/s² |
| `twist.angular.x/y/z` | 机体系角速度 | rad/s |

悬停总比推力约为 9.81 m/s²。它不是总推力 N，也不是单旋翼推力。默认动态参数为质量 0.75 kg、惯量对角线 `[0.012, 0.012, 0.012]` kg·m²、臂长 0.2 m、电机时间常数 0.01 s、角速度控制增益 45 s⁻¹。单旋翼最大推力为 5.518125 N，因此总比推力上限为 29.43 m/s²。

Newton 推力常数的转速单位为 rev/s。桥接按 `8.175e-5 / (2π)²` 转为 Flightmare 使用的 N/(rad/s)²。动态执行保留 Flightmare 原分配矩阵、电机指数响应和 RK4 积分。原始分配可行时，旋翼力逐元素保持；分配超限时使用第 9 节说明的饱和处理修正。

| Topic | 类型 | 坐标约定 |
|---|---|---|
| `/planning/pos_cmd` | `mars_quadrotor_msgs/PositionCommand` | 世界系位置、速度、加速度、jerk，yaw 为 rad |
| `/planning/click_goal` | `geometry_msgs/PoseStamped` | 原任务目标，世界系位置 |
| `/marsim/ctbr` | `geometry_msgs/TwistStamped` | 总比推力及机体系角速度 |
| `/marsim/odometry` | `nav_msgs/Odometry` | pose：world；child：base_link；twist：机体系 |
| `/lidar_slam/odom` | `nav_msgs/Odometry` | 兼容接口；速度为世界系 |
| `/marsim/acceleration` | `geometry_msgs/AccelStamped` | 积分后的世界系加速度 |
| `/cloud_registered` | `sensor_msgs/PointCloud2` | 世界系点云 |
| `/marsim/lidar_pose` | `geometry_msgs/PoseStamped` | 仅 dynamic；渲染器实际使用的位姿 |
| `/marsim/policy_stats` | `std_msgs/String` | 策略统计 JSON |

ROS 标准要求 Odometry 的 pose 属于 `header.frame_id`，twist 属于 `child_frame_id`。[Odometry 定义](https://github.com/ros2/common_interfaces/blob/jazzy/nav_msgs/msg/Odometry.msg)

原 PerfectDrone 的 `child_frame_id` 为空，速度仍是世界系。动态兼容话题明确使用 world/world。标准动态话题使用 world/base_link，Python 节点以 `R @ v_body` 转成世界速度。

ROS 消息和本文 NPZ 中的四元数顺序为 XYZW。Flightmare/Eigen 四元数构造时显式使用 `(w,x,y,z)` 参数。Isaac Lab 3.x 已将原 2.x 的 WXYZ 约定改为 XYZW；本项目使用 3.0，不能沿用旧版本的排序假设。[Isaac Lab 3.0 迁移指南](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/migrating.html)；[3.0 数学 API](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab/isaaclab.utils.html)

传感订阅使用 best-effort、volatile QoS，兼容原 SUPER/MARSIM 发布端。Python 策略的里程计、加速度和点云订阅使用 `KEEP_LAST(1)`，只保留最新样本，避免规划或 QP 耗时导致旧状态积压。Python 策略命令、策略统计和动态里程计采用可靠发布；原 SUPER PositionCommand 保持 best-effort。消息中的 frame 与单位仍需遵守上述契约。

任务目标事件单独使用 `RELIABLE + TRANSIENT_LOCAL + KEEP_LAST(1)`。构建副本中的 mission 目标发布器、SUPER 的 `/planning/click_goal` 订阅器和 Python 策略目标订阅器使用这一组合。这样后启动的策略仍能收到当前目标。SUPER 的普通 `/goal_pose` 点击入口保留原 QoS，点云和里程计订阅也不改为持久订阅。

当前节点不发布 `/clock`，也未启用 `use_sim_time`。消息使用 ROS 节点时钟；监控用单调时钟限制总运行时间。Flightmare 每个控制 tick 积分固定的 `1/control_frequency`。回调落后时，不应把仿真步长等同于实际墙钟频率。[ROS 2 时钟设计](https://design.ros2.org/articles/clock_and_time.html)

## 6. AC-MPC 检查点与观测

已完成严格权重加载检查的检查点：

```text
/home/tong/tongworkspace/free/isaaclab-IPC/results/mid360-retrain-matrix-20261102/train/checkpoint.pt
```

同目录 `config.json` 记录训练任务 `Isaac-IPC-Avoid-ACMPC-Direct`、场景 `avoid`、`reference_strides=[10,20]`。ROS 单机使用第一个 stride，即 10。该检查点不是 SUPER 两项地图的任务成功证明。

本次 dynamic AC-MPC 与固定成本 IPC 对照记录的 `checkpoint_sha256` 均为 `55ef3968672c22d5ad74d730de6c57c00d0f0bee8a224da31e9c2541f423bd77`。IPC 使用同一检查点恢复的模型配置，但不启用学习成本。

当前适配器只接受 `isaaclab_ipc.rl.actor:MPCActor` 的 124 维观测检查点。127 维 CTBR 模型检查点会被拒绝，避免改变训练时的预测模型。

| 顺序 | 特征 | 维数 |
|---:|---|---:|
| 1 | 世界速度／10 | 3 |
| 2 | 世界加速度／20 | 3 |
| 3 | body-to-world 旋转矩阵，按行展开 | 9 |
| 4 | 目标位置减当前位置，再除以 10 | 3 |
| 5 | 15 个参考位置减当前位置，再除以 10 | 45 |
| 6 | 15 个参考速度／10 | 45 |
| 7 | 15 个通道裕量，截断到 [-10,10] 后除以 10 | 15 |
| 8 | braking 标志 | 1 |

MPC 状态、参考轨迹和约束保持 SI 单位。缩放只作用于策略特征。检查点的 MPC 参数、成本、推力限制及参考 stride 保留；评测场景的前端工作空间来自对应 benchmark 配置。

```bash
IPC_ACMPC_CHECKPOINT=/home/tong/tongworkspace/free/isaaclab-IPC/results/mid360-retrain-matrix-20261102/train/checkpoint.pt

DISPLAY=:11 .venv/bin/python scripts/run_marsim_ros2.py \
  --mode dynamic --benchmark high_speed --policy acmpc \
  --checkpoint "$IPC_ACMPC_CHECKPOINT" \
  --domain 124 --render-prefix "vglrun -d egl0" \
  --output results/example-dynamic-acmpc --timeout 120
```

Actor 和 QP 默认在 CPU 上运行；现有增量地图前端仍使用 CUDA。控制 timer 请求 100 Hz，但实际频率取决于点云处理、规划、QP 和调度。热态单次 QP 耗时不能替代完整控制频率测量。当前记录没有拆分各前端阶段耗时，不能将控制回调中的全部额外耗时归因于 GPU。

## 7. 输出、判定与频率

运行器写入：

| 文件 | 内容 |
|---|---|
| `run.json` | Launch／监控命令、ROS domain、Git HEAD、`source_sha256`、`middleware_environment` 和可选 `executor_parameters_sha256` |
| `launch.log`、`monitor.log` | 本次进程日志 |
| `metrics.json` | success/collision/timeout、任务进度、速度、净空、频率、QP 失败和耗时 |
| `trajectory.npz` | 实际轨迹、速度、四元数、加速度、CTBR、PositionCommand、LiDAR 及渲染位姿 |
| `policy_stats.jsonl` | 监控收到的完整策略统计 |
| `policy/policy.jsonl` | 策略节点直接写出的每次控制统计 |
| `policy/metadata.json` | 检查点路径及 `checkpoint_sha256`、实际模型配置、观测顺序、推力限制和求解器 |

`source_sha256` 保存本次运行纳入记录的 Python、C++、shell、XML 等源码文件哈希，包含 Fast DDS profile。`middleware_environment` 保存 RMW、发现范围、`FASTDDS_BUILTIN_TRANSPORTS` 和 profile 路径，便于复现传输环境。原始参考版本另见 `build/marsim_ros2/sources.json`。

`trajectory.npz` 的 `position_commands` 数组保存收到的 `/planning/pos_cmd` 原始指令，形状为 `(N, 16)`。列顺序为消息时间、接收 elapsed、世界系位置 3 维、速度 3 维、加速度 3 维、jerk 3 维、yaw 和 yaw_dot。单位为 SI；完整列名写入 `metrics.json` 的 `array_columns.position_commands`，消息数为 `position_command_messages`。这些记录用于离线分析跟踪误差，不改变任务判据。

未使用相应接口时，消息数量为 0，相应指标为 null。例如原 SUPER 不提供本项目的 QP 统计。null 不表示 QP 失败数为 0。

监控复用完整原 PCD 解析，去除 NaN 后建立 CPU KDTree。碰撞模型为半径 0.2 m 的球体，检查地图点与地面。它还检查相邻里程计位置之间的线段，减少高速越过障碍的漏检。该指标是 PCD 几何碰撞判定，不是 Flightmare 的刚体接触反馈。

持续频率定义为：

```text
rate_hz = (收到的消息数 - 1) / (最后时间戳 - 最初时间戳)
```

`odometry_timing`、`ctbr_timing` 同时保存 count、duration、间隔中位数、P95 和最大值。`ctbr_reception_timing` 使用接收墙钟时间。间隔中位数的倒数不能作为持续频率。

早期 `official-ideal-high-speed` 记录包含 2154 个里程计样本，跨度 118.6463 s，持续频率为 18.1464 Hz。其间隔中位数为 10.0119 ms，P95 却为 216.5738 ms。旧 JSON 的 `odometry_rate_hz` 使用了中位数倒数；代码已经修正，旧原始结果保留，复算应使用 NPZ 时间戳。

监控和运行器的退出码：

| 退出码 | 含义 |
|---:|---|
| 0 | 完成全部官方航点 |
| 2 | PCD／地面碰撞 |
| 3 | 超时、没有里程计，或监控被中断 |
| 4 | 运行器发现 Launch 提前结束 |

静态检查或短动作检查不能代替完整往返任务。报告必须同时给出结果、运行配置和实际测得的频率。

## 8. 基础运动与 LiDAR 验证

基础检查在原 high_speed 起点附近执行，参考轨迹沿 +Y 移动 3 m。5 个阶段各持续 2 s：悬停、加速至 0.75 m/s、匀速转向 60°、制动、再次悬停。参考位置、速度、加速度及 jerk 在阶段边界连续。

使用同一未占用 domain，在 3 个终端分别运行。每个终端先加载环境，并设置相同的 domain：

```bash
source scripts/marsim_ros2_env.sh
export ROS_DOMAIN_ID=125
export ROS_AUTOMATIC_DISCOVERY_RANGE=SYSTEM_DEFAULT
IPC_MARSIM_CONFIG="$IPC_MARSIM_ROOT/build/marsim_ros2/install/perfect_drone_sim/share/perfect_drone_sim/config/high_speed.yaml"
```

终端 1：

```bash
ros2 run isaaclab_marsim flightmare_node --ros-args \
  -p config_path:="$IPC_MARSIM_CONFIG" -p command_source:=position_command
```

终端 2：

```bash
DISPLAY=:11 vglrun -d egl0 ros2 run isaaclab_marsim marsim_lidar_node --ros-args \
  -p config_path:="$IPC_MARSIM_CONFIG" -p odom_topic:=/marsim/odometry
```

终端 3：

```bash
.venv/bin/python scripts/check_marsim_dynamics.py \
  --output results/example-dynamic-motion --timeout 45
```

脚本只发布 PositionCommand。它从实际里程计计算各阶段误差，并记录实际加速度与生成的 CTBR。输出为 `metrics.json` 和 `trajectory.npz`。全部检查通过时退出 0，否则退出 1。

LiDAR 检查同时要求：渲染位姿匹配同时间戳里程计；点云与渲染位姿时间戳匹配；渲染位置随飞行变化；点云载荷存在变化。检查还要求实际位置与已发布目标存在可测差异，用于排除位置直接赋值。

原 PerfectDrone 命令契约由 `scripts/check_marsim_ideal.py` 检查。先在隔离 domain 启动原节点，再运行：

```bash
.venv/bin/python scripts/check_marsim_ideal.py \
  --output results/example-ideal-contract.json --timeout 45
```

这里的 `--output` 是 JSON 文件；动态检查的 `--output` 是目录。

## 9. 私有构建副本中的兼容修复

SUPER 副本切换到上游自带的 ROS 2 CMake 和 package.xml。额外补齐消息依赖、GLFW include 导出、`pcl_conversions`、`std::isnan` 及 `<cstdint>`，并移除覆盖依赖搜索路径的 IDE 提示。

任务节点保留首个待发布目标，直到发现订阅者才发送。原配置的 `publish_dt=99999` 使首次目标丢失后很难再次触发。构建副本还将任务目标发布端和 SUPER 的 `/planning/click_goal` 订阅端改为可靠、持久、深度 1 的 QoS；Python 策略端同步匹配。该补丁只作用于任务目标事件，普通 `/goal_pose` 点击入口和传感器 QoS 保持原约定。航点、阈值和任务顺序不变。**PerfectDrone 的 C++ 执行源码没有修改。**

Flightmare 副本的补丁记录在 `build/marsim_ros2/flightmare/isaaclab_marsim.patch`：

1. CTBR 限幅使用总推力除以质量，修正比推力与单旋翼力混用。
2. 归一化积分得到的 `next_state` 四元数。
3. 提供惯量对角线设置接口，匹配现有训练物理参数。
4. 加载转速与推力映射后重新计算旋翼最大推力。
5. 提供角速度增益设置接口。
6. 提供电机初始推力设置接口，以悬停转速初始化。
7. 修正饱和时的旋翼力分配；可行的原始分配逐元素保持。

原实现先用逆分配矩阵计算旋翼力，再独立截断各旋翼。较大的 yaw 请求会消耗推力裕量，部分组合会将 roll/pitch 力矩抵消为 0。私有副本仅在原分配超限时采用新处理：先保留总推力，再以同一比例缩放 roll/pitch 分量，最后将剩余裕量分配给 yaw。该处理改变了饱和行为，不能表述为完全保留上游分配算法；原矩阵、电机响应和 RK4 仍保留。

桥接位置跟踪器同时修正反馈比力的导数：`F_dot = jerk_d + kp * (v_d - v) + kd * (a_d - a_actual)`。姿态角速度前馈使用完整导数，其中 `a_actual` 为积分得到的世界系实际加速度。

数值检查 `ros2/isaaclab_marsim/tests/check_flightmare_allocation.cpp` 链接与执行器相同的 Flightmare 动力学库。`results/marsim-ros2-20261104/native-allocation.json` 记录 3773 组边界检查通过；其中 173 组原始可行分配的最大逐元素误差为 0 N，总推力最大误差为 1.90735e-6 N。这些结果验证分配实现，不替代 ROS 飞行、LiDAR 或完整任务验收。

完成正常构建后，可单独启用并运行这项 CTest：

```bash
source scripts/marsim_ros2_env.sh
cd build/marsim_ros2
cmake -S ../../ros2/isaaclab_marsim -B build/isaaclab_marsim -DBUILD_TESTING=ON
cmake --build build/isaaclab_marsim --target check_flightmare_allocation -j2
ctest --test-dir build/isaaclab_marsim -R '^flightmare_allocation$' --output-on-failure
```

桥接节点直接编译所需 Flightmare 动力学源码，不引入 Unity 运行链路。MARSIM 渲染节点的头文件配置以 `perfect_drone_sim` 为 PCD 根目录；原渲染库保留自己的 shader 和扫描模式根目录。这是桥接构建路径修复，未重写 LiDAR 算法。

## 10. 实测结果与未通过项

以下结果来自本次独立功能工作树的实际源码运行。运行时包含未提交改动，不能表述为测试了干净的 `7d6da6b` 提交。`run.json` 的 `git_revision` 只记录当时 HEAD；实际文件版本还应核对 `source_sha256`、参数文件哈希和构建副本来源。

结果目录统一位于 `results/marsim-ros2-20261104/`。各行数字来自相应目录的 `metrics.json`。墙钟耗时使用 `elapsed_wall_s`，包含启动等待；轨迹采样跨度另见 `trajectory_duration_s`。最小净空使用包括地面的 `minimum_clearance_m`。所有完整任务都要求按原协议完成 2 个航点，未放宽阈值。

汇总文件 `summary.json` 收录最终 7 项任务、命令契约、基本动作、分配数值检查和最终源码 SHA256，并明确记录 `tested_uncommitted_working_tree=true`、`all_acceptance_criteria_passed=false`。下表另列出一次跟踪增益对照。

### 命令契约、基本动作与实际 LiDAR

| 检查 | 已确认结果 | 证据 |
|---|---|---|
| 原 PerfectDrone 命令直通 | 位置误差 0 m，速度误差 0 m/s，旋转矩阵误差约 1.67e-16 | `ideal-contract.json` |
| 动态基本动作 | `passed=true`；悬停、加速、转向、制动、终点位置、高度和实际状态差异检查全部通过 | `dynamic-motion-v6/metrics.json` |
| 动态 LiDAR | 101 条渲染位姿、101 帧点云与 101 条对应里程计时间戳匹配；位置误差 0 m，姿态误差最大约 2.70e-8 rad；点云载荷有 101 个不同摘要 | 同上 |

基本动作测试的 CTBR 持续频率为 100.0007 Hz。实际状态与全部已发布位置目标的最大最小距离为 0.009213 m。它验证了真实积分状态和原 MARSIM 渲染路径，不等于完成官方往返任务。

### 完整任务结果

| 结果目录 | 结果／航点 | 墙钟耗时 s | 轨迹 m | 最小净空 m | CTBR Hz |
|---|---|---:|---:|---:|---:|
| `official-ideal-high-speed-v3` | success，2/2 | 18.2430 | 202.7019 | 0.583463 | — |
| `dynamic-super-high-speed-v3` | success，2/2 | 24.5617 | 206.0797 | 0.363057 | 100.0004 |
| `official-ideal-dense-v2` | timeout，1/2 | 120.0241 | 107.6283 | 0.167549 | — |
| `dynamic-super-dense-v2` | collision，0/2 | 9.9700 | 8.3667 | -0.001872 | 99.9999 |
| `dynamic-super-dense-gains` | collision，0/2；增益 8/5/12 对照 | 6.3906 | 4.9780 | -0.004556 | 99.9970 |
| `dynamic-ipc-high-speed-v2` | collision，0/2 | 64.3900 | 94.4911 | -0.017258 | 19.0883 |
| `dynamic-acmpc-high-speed-v4` | collision，0/2 | 52.6044 | 78.8439 | -0.011345 | 17.0972 |
| `ideal-acmpc-high-speed` | timeout，0/2 | 120.0416 | 19.0829 | 0.671987 | — |

ideal 模式执行 PositionCommand，不发布 CTBR，因此该列为“—”。两个官方 ideal 运行的实际里程计持续频率分别为 46.0840 Hz（high_speed）和 54.3379 Hz（dense）。不要用原 10 ms timer 设置替代这些实测频率。

高速场景的两个 SUPER 运行完成完整往返，但日志仍包含原规划器轨迹优化和重规划警告，不能称为完全无告警。ideal dense 日志包含 `PlanFromRest`／`GenerateExpTrajectory` 失败重试；超时时仅完成去程。dynamic dense 日志包含角速度、推力、位置约束违例和 `GeneratePolytopeFromLine` 失败。增大跟踪增益的单次对照更早碰撞，因此保留默认 4/3/6。密集场景验收尚未通过，现有证据也不足以将原因归结为单个增益。

### 固定成本与学习成本策略

dynamic IPC 与 AC-MPC 使用同一检查点恢复的 MPC 配置。IPC 使用固定成本和 OSQP；AC-MPC 加载实际 MPCActor 权重并使用可微 QP。它们不是相同求解器的纯成本消融，也不能假定所有运行环境完全相同。实际中间件环境以各 `run.json` 为准；例如已记录的增益对照使用 `SUBNET`，而环境脚本的默认值是 `SYSTEM_DEFAULT`。

| 运行 | 控制统计数 | QP 失败 | 控制耗时均值／P95 ms | QP 耗时均值／P95 ms | 点云／渲染位姿匹配 |
|---|---:|---:|---:|---:|---:|
| `dynamic-ipc-high-speed-v2` | 1076 | 1 | 51.1513／113.4667 | 8.7495／10.9431 | 632/632 |
| `dynamic-acmpc-high-speed-v4` | 793 | 0 | 57.2936／110.6856 | 14.7024／16.1887 | 515/515 |
| `ideal-acmpc-high-speed` | 2657 | 0 | 41.3492／78.5582 | 12.3422／12.8819 | 原 PerfectDrone 不发布渲染位姿 |

两种 Python 策略均实际运行到碰撞终止，并保存了完整记录。AC-MPC 的 QP 失败数为 0，但任务仍碰撞；QP 求解成功不能代替导航成功。动态执行器里程计约为 100 Hz，而两种策略的 CTBR 持续频率只有 19.0883 Hz 和 17.0972 Hz。100 Hz 策略输出要求尚未达到。

ideal AC-MPC 运行 120.0416 s 后超时，未到达第一个航点。它发布了 2657 条 PositionCommand，收到 1189 帧点云。按 NPZ 中命令时间戳计算，PositionCommand 持续频率为 23.3587 Hz；里程计为 66.7395 Hz。该模式不发布 CTBR。虽然没有报告碰撞或 QP 失败，但完整任务未完成，控制耗时也超过 10 ms 预算。该结果同样不能记为验收通过。

### 修复范围与证据边界

最终运行使用了前述 PCD 根目录、Fast DDS 大点云传输及目标 QoS 修复。Flightmare 的饱和分配行为也有明确修改：原可行分配不变，超限时保留总推力、同比例处理 roll/pitch，最后处理 yaw。这些结果不能归为完全未修改的 Flightmare 上游行为。

分配数值检查覆盖 3773 组组合，CTest 为 1/1 通过；证据为 `native-allocation.json` 和 `native-allocation-ctest*.log`。它们补充数值正确性证据，不替代飞行验收。此前空点云、传输异常、碰撞和超时目录均保留。当前已确认的成功仅覆盖上述具体任务，不能推断全部验收通过。

### 原工作树、参考仓库与实验保留

`preservation-check.json` 记录原 main 工作树仍保留已有未提交改动。SUPER、Flightmare 和 IPC 参考仓库的 tracked／staged diff 均为空，检查点 SHA256 与运行记录一致。这不表示原工作树或参考目录没有既有未跟踪文件。

运行结束后的检查显示，3 个原有 `isaaclab-navrl` 训练进程仍在 GPU 进程列表，本次评测进程列表为空。较早记录的 PID 820562 未出现在该快照中；本次没有停止它，其退出原因未核实。因此只能报告检查时仍运行的进程，不能声称所有历史训练 PID 始终存活。
