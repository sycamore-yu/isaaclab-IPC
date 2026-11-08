# MARSIM ROS 2 评测

新 AC-MPC 分支按 [ADR-0008](adr/0008-native-acmpc-pathcover.md) 使用 `--policy fixed-ctbr` 或 `--policy acmpc`，直接执行 4D CTBR。`--frontend none` 选择 A0；默认 `pathcover` 选择完整原生前端。以下历史记录中的 124D jerk 检查点和二重轨迹验收不属于新 CTBR 路径。

当前原生 CTBR 主线的实际结果见 [原生 AC-MPC 验证](acmpc-native-validation.md)。
训练和正式评测命令见 [README](../README.md#marsim)。历史 V15 固定成本的 High-speed 成功、Dense 失败及其延迟记录保留，不代表当前版本。

V40 六场比较已完成。A1/A2 在 Dense 和 High-speed 均到达 2/2 航点。
其 planning p95 分别为 A1 88.790715 / 43.485337 ms、A2 91.701845 / 48.218813 ms。
3203 次 MPC 与 3203 份完整参考通过，实际 100 Hz 采样 SFC 正违反为 0。
A0 两场均碰撞，作为无前端负基线保留。逐场时间、速度、慢调用和输入核对见 [V40 正式验收](acmpc-native-validation.md#v40-正式验收)。

当前 Flightmare 饱和分配保留可行请求的总推力，先最大化横滚与俯仰力矩的共同缩放系数，再选择最近的可行偏航力矩。原可行分配直接保留。电机延迟与 RK4 动态执行继续运行。第 8–11 节中的旧分配及结果属于历史版本，不替代使用同一执行器的 A0/A1/A2 比较。

当前工作树为 `/home/tong/tongworkspace/free/.worktrees/isaaclab-IPC/acmpc-native-pathcover-v1`，分支为 `feat/acmpc-native-pathcover-v1`。
当前运行目录为 `build/marsim-shader-v39`，通过 `IPC_MARSIM_RUNTIME` 选择。它复用 V20 动态执行程序，并使用独立构建的修复版 LiDAR 渲染库。CTBR 的当前 3 走廊配置及阶段验收见原生验证记录。
正式配置显式使用 `--mode dynamic --time-mode sim --reference-speed 3 --timeout 180`。
`High-speed` 是场景名称，不表示名义速度为 10 m/s。

`--capture-reference-plans` 保存全部已返回参考，用于离线完整曲线检查。
该选项默认关闭，不增加在线重复积分或参考验收。
任务结果、全部成功与失败规划尝试的延迟、参考质量、预测节点约束及 100 Hz 采样实际约束分别统计。
任务 success 或退出码 0 本身不代表规划 p95 和完整参考质量通过。
`scripts/analyze_marsim_references.py --run-dir <运行目录> --output-dir <新目录>` 提供可复现的 CPU 离线参考分析，命令和统计口径见 README。它读取保存的曲线，不重新运行优化或生成 yaw。

以下历史 SUPER 结果包括 ideal high_speed、ideal dense 和 dynamic high_speed 的完整往返。历史 dynamic dense 曾碰撞，Python 策略也曾超时；这些结果及原始证据仍保留在第 11 节。

MARSIM 负责正式评测，提供 `ideal` 和 `dynamic` 两种执行模式。Isaac Lab/Newton 继续使用原训练入口。ROS 节点复用项目的 Planner、PathCover/RISP、MPCActor 和 QP，不启动 Isaac Lab 仿真进程。本文记录当前实现、复现方法和实际成功／失败结果。

## 1. 历史工作树与参考版本

以下基础设施最初在 `feat/marsim-ros2-acmpc-20261104` 开发。历史独立工作树：

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

默认构建和安装目录为 `build/marsim_ros2/`；使用 `--workspace <独立目录>` 可隔离候选。脚本支持 `--prepare-only`、`--upstream-only` 和 `--packages`，供分阶段构建使用。通常先完整构建，再用 `--packages isaaclab_marsim` 增量编译桥接节点。

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

私有 SUPER 准备脚本修正 `360camera.vs` 的量程外顶点：明确放到裁剪体外，并写入点大小。原分支的全零齐次坐标与未定义点大小曾使全部深度像素接近零，随后被盲区过滤为空点云。修复保留有效点的算式和原缓冲区顺序。隔离 60 帧全部有效，并与旧版有效帧的 XYZ 逐位一致；证据见原生验证记录的 V39 渲染器章节。

渲染器在运行时从编译期 `ROOT_DIR` 读取 shader。复现必须同时冻结运行库和该目录的资源。运行器在 `run.json` 中记录实际依赖解析得到的渲染库路径及 SHA256；同一个 LiDAR 可执行文件不等于同一个渲染器。

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

ideal 发布端按实际发布间隔推进上一条已发命令及其 jerk，得到本步位置、速度和加速度；正常发布间隔约为 0.01 s。MPC 仍使用训练时的 0.1 s 预测步长。原 PerfectDrone 没有独立加速度话题，策略用收到的精确位置、速度匹配已发 PositionCommand，再取该命令的加速度，避免拼接不同执行时刻的状态。历史按里程计确认清理，并限制为完整预测时域内的命令数量。dynamic 使用实际 AccelStamped。

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

Actor 和 QP 默认在 CPU 上运行；现有增量地图前端仍使用 CUDA。规划、观测构造和 QP 在一个独立进程中运行。ROS 主进程读取最新状态，以 100 Hz 执行已有 jerk 序列并生成 CTBR。一次只提交一个规划请求，完成后再提交最新状态和点云。这样不会积压历史请求，也不会让持有 Python GIL 的几何前端阻塞控制回调。

控制频率与求解频率分别测量。`solve` 事件记录一次真实求解，`control` 事件记录一次实际发布。复用同一条序列时，不重复累计 QP 失败或求解耗时。前端、观测、QP、总规划、控制回调耗时分别保存；同时记录状态年龄、解的源状态年龄和执行序列年龄。

新解从接受时刻开始执行，不能把求解期间的时间当作该序列已经执行的时间。正常动态控制在接受解时，以最新实测 PVA 建立执行参考；各 0.1 s 结点使用原 `mpc.a/b`，区间内按相同 jerk 动力学和真实经过时间求参考状态。控制回调将参考加速度、当前段 jerk 和最新姿态交给原 `flatness_torch`。源状态超过完整 1.5 s 预测时域时仍进入原恢复逻辑。

接受前还会复查同一条 jerk 序列。复查使用未截断的实际执行 PVA、原 `MPC.problem` 约束、原 `M/C` 以及原求解器的原始可行性容差。它涵盖 jerk、加速度、速度和安全飞行走廊约束。不可行的延迟计划进入已有制动和重规划流程，不改变预测模型、成本或边界。ideal 先将上一条已发布 PVA 推进到当前控制时间戳，再用同一 PVA 复查和发布，避免检查与执行状态不一致。

`reanchor_rejected` 表示延迟计划在实际执行状态下不可行，累计数为 `reanchor_rejections`；它与 QP 求解失败分开统计。每个 `solve` 事件保存完整小型 jerk 序列、源状态、未截断源观测、接受状态、紧凑走廊与索引、容差、约束类别和残差。`control` 事件不重复这些数组。完整复查耗时另存为 `acceptance_ms`。

这是多速率执行适配，不等同于每 10 ms 重解 MPCActor。不能在每个 10 ms 回调中反复计算“最新实测加速度 + 0.1 × 缓存 jerk”；该做法会重复施加前视量。恒定竖直 jerk 为 1 m/s³ 时，旧做法在理想推力响应下于 0.1 s 内累计到 1.0 m/s²，正确推进为 0.1 m/s²。控制日志保存参考加速度、实测加速度和 jerk，支持复算。

## 7. 输出、判定与频率

运行器写入：

| 文件 | 内容 |
|---|---|
| `run.json` | Launch／监控命令、ROS domain、Git HEAD、源码及执行文件 SHA256、中间件环境和可选执行器参数 SHA256 |
| `launch.log`、`monitor.log` | 本次进程日志 |
| `metrics.json` | success/collision/timeout、任务进度、速度、净空、频率、QP 失败、延迟计划拒绝和各阶段耗时 |
| `trajectory.npz` | 实际轨迹、速度、四元数、加速度、CTBR、PositionCommand、LiDAR 及渲染位姿 |
| `policy_stats.jsonl` | 监控收到的完整策略统计 |
| `policy/policy.jsonl` | 策略节点直接写出的控制及求解事件 |
| `policy/metadata.json` | 检查点路径及 `checkpoint_sha256`、实际模型配置、观测顺序、推力限制和求解器 |
| `super_time_consuming.csv` | 本轮 SUPER 实际产生的规划阶段耗时；运行结束后归档 |

`source_sha256` 保存本次运行纳入记录的 Python、C++、shell、XML 等源码文件哈希，包含 Fast DDS profile。`executable_sha256` 保存已安装的 Flightmare、MARSIM LiDAR、PerfectDrone 和 SUPER 可执行文件哈希，可区分源码已修改但尚未重编译的情况。`middleware_environment` 保存 RMW、发现范围、`FASTDDS_BUILTIN_TRANSPORTS` 和 profile 路径。原始参考版本另见 `build/marsim_ros2/sources.json`。

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

任务节点保留首个待发布目标，直到发现订阅者才发送。原配置的 `publish_dt=99999` 使首次目标丢失后很难再次触发。构建副本还将任务目标发布端和 SUPER 的 `/planning/click_goal` 订阅端改为可靠、持久、深度 1 的 QoS；Python 策略端同步匹配。该补丁只作用于任务目标事件，普通 `/goal_pose` 点击入口和传感器 QoS 保持原约定。航点、阈值和任务顺序不变。**PerfectDrone 执行类及其定时回调保持原源码。**

私有 ROS 2 main 另修复一个已复现的 SIGINT 关闭竞态：全局地图的 1 ms 定时器可能在 context 关闭后调用 `count_subscribers`，抛出 `std::runtime_error`。main 仅在该节点 context 已关闭且错误文本匹配这条订阅者计数异常时正常收尾。运行中异常和其他关闭异常仍抛出。该补丁不改变飞行期间的执行行为，也不修改原参考仓库。

后续实飞定位了 SUPER ROS 2 的起步时间问题。原代码把静止起步轨迹的时间原点设在求解开始时，首次发布会跳过求解期间的轨迹段。它还将滚动重规划的 100 ms 衔接期限用于静止起步，导致大量有效解被丢弃。私有副本改为在提交静止轨迹时设置时间原点，并同步平移 Exp、Backup 的绝对起点。多项式、相对衔接时刻和原滚动重规划期限保持不变。

该修正要求起步前和提交时的实际速度均不超过 0.05 m/s，里程计年龄不超过 0.1 s，求解期间位移不超过地图分辨率的 1/10。ROGMap ROS 2 原回调未填充速度，现从原世界系里程计补齐，并用短互斥锁读取一致的状态快照。静止条件不满足时仍保持原任务状态并重试。

同时删除一次输出未被使用的 Backup 重复优化。SUPER 多线程执行器新增 `executor_threads` 参数，默认 4；原来的自动线程数在本机创建了 40 个工作线程。构建脚本创建上游二进制轨迹日志需要的空目录，避免 `git archive` 未包含空目录而使退出保存报错。这些是私有副本中的运行修正，不能称为完全未修改的原 SUPER 规划过程。

Flightmare 副本的补丁记录在 `build/marsim_ros2/flightmare/isaaclab_marsim.patch`：

1. CTBR 限幅使用总推力除以质量，修正比推力与单旋翼力混用。
2. 归一化积分得到的 `next_state` 四元数。
3. 提供惯量对角线设置接口，匹配现有训练物理参数。
4. 加载转速与推力映射后重新计算旋翼最大推力。
5. 提供角速度增益设置接口。
6. 提供电机初始推力设置接口，以悬停转速初始化。
7. 修正饱和时的旋翼力分配；可行的原始分配逐元素保持。

原实现先用逆分配矩阵计算旋翼力，再独立截断各旋翼。较大的 yaw 请求会消耗推力裕量，部分组合会将 roll/pitch 力矩抵消为 0。私有副本仅在原分配超限时采用新处理：优先保留可实现的 roll/pitch 力矩，再选择最接近请求值的可行总推力，最后分配 yaw。roll/pitch 同比例缩放，保留力矩方向。饱和时实际总推力可能偏离请求值；原分配矩阵、电机响应和 RK4 仍保留。

桥接位置跟踪器同时修正反馈比力的导数：`F_dot = jerk_d + kp * (v_d - v) + kd * (a_d - a_actual)`。姿态角速度前馈使用完整导数，其中 `a_actual` 为积分得到的世界系实际加速度。

最终跟踪器复用 SUPER 的 `FlatnessMap`。姿态反馈参考 [PX4 reduced-attitude 构造](https://px4.github.io/Firmware-Doxygen/db/de4/_attitude_control_8cpp_source.html)，优先对齐机体推力轴，再处理 yaw。yaw 权重为 0.4，未补偿 yaw 增益，因此也降低 yaw 带宽。控制器将负竖直合力及其导数投影到 0；完整角速度前馈的范数上限为原生角速度限值的一半，即 5 rad/s。姿态反馈优先，前馈在剩余绝对速率预算内统一缩放。输出满足各轴 ±10 rad/s、总比推力 [0,29.43] m/s²。

上述限制是新增的执行控制规则，不是对 PX4 整个控制器的移植，也不保证实际倾角始终小于 90°。生产节点和 `tests/replay_position_commands.cpp` 共用 `position_command_tracking.hpp`。PVAJ 外推、水平与竖直解耦、signed 前馈预算等未通过反例检查，均未加入生产路径。

数值检查 `ros2/isaaclab_marsim/tests/check_flightmare_allocation.cpp` 链接与执行器相同的 Flightmare 动力学库。最新 `results/marsim-ros2-followup-20261104/native-allocation-rp-priority.json` 覆盖 3773 组边界输入。其中 173 组原始可行分配的逐元素误差为 0 N；2912 组需要调整总推力，最大调整为 11.0363 N。可实现 roll/pitch 目标的最大力矩误差为 1.55e-7 N·m。数值检查不替代实际飞行。

完成正常构建后，可单独启用并运行这项 CTest：

```bash
source scripts/marsim_ros2_env.sh
cd build/marsim_ros2
cmake -S ../../ros2/isaaclab_marsim -B build/isaaclab_marsim -DBUILD_TESTING=ON
cmake --build build/isaaclab_marsim --target check_flightmare_allocation -j2
ctest --test-dir build/isaaclab_marsim -R '^flightmare_allocation$' --output-on-failure
```

同一 `BUILD_TESTING=ON` 配置还提供 `replay_position_commands` 目标。输入是无标题 CSV：时间、P/V/A/jerk 各 3 维、yaw、yaw_dot，共 15 列。时间以首个 CTBR tick 为 0。运行方式为 `replay_position_commands input.csv output.csv [kp kd attitude_gain]`，默认增益与节点一致，为 4/3/6。它复用生产控制函数及原生动力学，但不会重新规划或渲染 LiDAR，因此只用于分析已捕获指令，不能代替 ROS 任务验收。

桥接节点直接编译所需 Flightmare 动力学源码，不引入 Unity 运行链路。MARSIM 渲染节点的头文件配置以 `perfect_drone_sim` 为 PCD 根目录；原渲染库保留自己的 shader 和扫描模式根目录。这是桥接构建路径修复，未重写 LiDAR 算法。

## 10. 第一阶段结果（历史记录）

本节保留首次实现的成功和失败证据。后续修复及当前验收结论见第 11 节；本节的控制频率、分配优先级和失败结论只对应当时版本。

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

## 11. 后续修复与当前实飞结果

本节结果位于 `results/marsim-ros2-followup-20261104/`。仍使用原 PCD、原 2 个航点、严格小于 1 m 的航点判据、0.2 m 碰撞半径和 120 s 截止时间。没有通过降低速度、放宽净空或延长超时取得通过结果。

该目录的 `summary.json` 汇总最新 7 项完整任务、动作检查、约束复算、源码及二进制哈希、回归检查和保留检查。它同时索引两阶段共 50 份实际运行指标，并明确记录 `all_acceptance_criteria_passed=false`。运行时的未提交源码由各 `run.json` 哈希标识，交付提交另行记录，不能用后来生成的提交号替换原运行的 HEAD。

### SUPER 官方任务

| 结果目录 | 执行模式 | 结果／航点 | 墙钟 s | 轨迹 m | 最小净空 m |
|---|---|---|---:|---:|---:|
| `official-ideal-dense-rest-clock` | ideal | success，2/2 | 84.959 | 218.716 | 0.175609 |
| `official-ideal-dense-rest-clock-repeat` | ideal | success，2/2 | 88.586 | 219.585 | 0.142427 |
| `official-ideal-high-speed-rest-clock` | ideal | timeout，1/2 | 120.017 | 174.807 | 0.347029 |
| `official-ideal-high-speed-rest-clock-repeat` | ideal | success，2/2 | 24.844 | 204.736 | 0.395609 |
| `official-ideal-high-speed-shutdown-fixed` | ideal | success，2/2 | 20.138 | 203.383 | 0.424550 |
| `dynamic-super-high-speed-bounded-default` | dynamic | success，2/2 | 16.800 | 202.466 | 1.049630 |
| `dynamic-super-dense-bounded-default` | dynamic | collision，0/2 | 5.849 | 4.814 | -0.029024 |
| `dynamic-super-dense-bounded` | dynamic，增益 16/8/12 | collision，0/2 | 6.816 | 7.350 | -0.004892 |

最终默认增益保持 4/3/6。16/8/12 只作为已记录的诊断对照，没有设为默认。最新官方 high_speed、dense 成功运行均正常退出，二进制规划日志保存成功。规划器仍会报告优化失败和重规划超时警告，这些警告会影响任务完成率。

`official-ideal-high-speed-shutdown-fixed` 使用已重编译的 PerfectDrone main。4 个官方入口节点全部正常退出，日志没有 ERROR；这验证了关闭补丁后的完整任务路径。此前 ideal v5 的关闭异常及失败日志仍保留，详见 `perfect-drone-shutdown-fix.json`。

ideal dense 的两次成功支持起步时间修正有效，但不是成功率统计。高速首次复测在途中静止约 98 s，反复报告 Exp 优化失败，最终超时；相同配置重复运行成功。该失败记录保留，不能据一次成功宣称所有重复运行都无回归或必然成功。

### 最终动态动作和 LiDAR

`dynamic-motion-bounded-default/metrics.json` 中全部检查为 true。该检查使用最终默认位置跟踪器，完成悬停、加速、转向、制动及终点保持。实测 CTBR 为 99.9996 Hz。

- 101 帧点云均匹配对应渲染位姿。
- 101 个渲染位姿均匹配实际里程计时间戳。
- 位置差为 0 m，最大姿态差为 2.73e-8 rad。
- 点云有 99 个不同载荷摘要。
- 实际位置与全部已发位置目标之间存在最大 0.00715 m 的最小距离，排除了位置直接赋值。

原 PerfectDrone 执行类仍未修改。第 9 节的 main 关闭修复只处理已关闭的 context。此前 `ideal-contract.json` 的位置、速度直通及姿态验证继续适用。

### Python 策略实飞

以下 v3、v4 运行保留了执行前可行性复查加入前的诊断证据。

| 结果目录 | 结果 | 轨迹 m | 命令频率 Hz | 真实求解更新 Hz | QP 失败 |
|---|---|---:|---:|---:|---:|
| `dynamic-acmpc-async-v3` | collision | 59.504 | 99.9742 | 15.6767 | 0 |
| `dynamic-ipc-async-v3` | collision | 86.498 | 99.9760 | 15.3670 | 2 |
| `ideal-acmpc-async-v3` | timeout | 109.771 | 99.3541，PositionCommand | 5.6206 | 127 |
| `ideal-acmpc-async-v4` | collision | 48.056 | 99.7002，PositionCommand | 14.0820 | 0 |

两种动态策略使用同一检查点配置。AC-MPC 控制回调均值／P95／最大值为 1.158／1.623／7.547 ms，3926 个回调均未超过 10 ms。固定 IPC 为 1.191／1.665／19.187 ms，5438 个回调中 1 个超过 10 ms。持续频率约为 100 Hz，不表示每次调度都满足硬实时期限。

动态 AC-MPC 实际运行到 44.348 s 后撞地。最后位置高度为 0.19316 m，对 0.2 m 半径的机体构成碰撞。末 1 s 有 81/100 次命中检查点保留的 12.6129 m/s² 推力上限。一个末段参考需要 21.2294 m/s²，虽然仍满足 MPC 的分量加速度边界，却超过推力能力。末段参考更新也降至约 5.65 Hz，解年龄均值达到 342 ms。证据同时涉及物理可执行性和计算延迟，不能把失败完全归因于检查点。

3449 个正常动态控制样本的推力及横向角速度重算误差不超过 1.78e-15；同计划区间内的参考积分误差不超过 3.24e-15。此前重复施加 0.1 s 前视量的缺陷未重现。独立里程计与加速度订阅在 3926 个回调中有 4 个出现约 10 ms 的一帧错位，加速度差最大为 0.01914 m/s²；它们发生在正常飞行段，不能解释末段失稳。

ideal v3 还暴露出独立的观测拼接缺陷：它会将某条已执行命令的位置、速度与较新命令的加速度拼接。最新代码已按精确 P/V 匹配同一命令的加速度。该修正不改变动态路径，也不能消除长时间求解期间复用旧 jerk 的稳定性限制。

修正后的 ideal v4 已完成真实复测，在 34.410 s 后报告碰撞。共发出 2956 条 PositionCommand；控制回调均值／P95／最大值为 0.673／1.136／5.891 ms，没有 QP 失败或节点异常。该结果验证最新适配代码实际运行并记录失败，不表示检查点完成了原往返任务。

v4 还暴露出延迟计划的状态偏移：最后一条计划的求解源加速度为 `[9.023,-15.176,-4.275]` m/s²，接受时的执行加速度已变为 `[21.420,-18.830,-11.401]` m/s²。因此不能直接把同一解平移到新 PVA 并声称仍满足原约束。第 6 节的接受前复查修复了这一契约缺口；它不等于重新求解 QP。

最新 v5 使用该复查，实际完成以下 3 次完整时限运行。它们都在原 120 s 截止时间内未完成第一个航点。

| 结果目录 | 结果 | 轨迹 m | 命令频率 Hz | 真实求解更新 Hz | QP 失败 | 延迟计划拒绝 |
|---|---|---:|---:|---:|---:|---:|
| `ideal-acmpc-async-v5` | timeout，0/2 | 52.937 | 98.7135，PositionCommand | 9.9530 | 0 | 115 |
| `dynamic-acmpc-async-v5` | timeout，0/2 | 11.850 | 99.5210，CTBR | 5.3049 | 117 | 28 |
| `dynamic-ipc-async-v5` | timeout，0/2 | 12.126 | 99.8940，CTBR | 32.2400 | 2976 | 31 |

三次运行的最小净空分别为 0.668347、1.278089 和 1.277897 m。动态两次没有节点异常；ideal 在超时停止后出现 PerfectDrone 关闭错误，详见对应 `launch.log`。第 9 节记录了随后完成的关闭修复，其后官方 high_speed 完整运行和关闭均通过。不能把保持无碰撞或 QP 成功当作完成任务。

动态 AC-MPC 与固定 IPC 的命令间隔 P95 分别为 11.211 和 11.788 ms，最大间隔分别为 375.130 和 24.376 ms。平均输出接近 100 Hz，但不满足每次调度均在 10 ms 内完成的硬实时保证。`control_ms` 在最终控制统计序列化和发布前采样，因此也不是完整回调上界。实际命令时间戳的间隔才覆盖该调度结果。

动态 AC-MPC 的 QP 耗时均值／P95 为 137.356／653.100 ms；固定 IPC 为 5.768／8.898 ms。后者的更新频率较高包含大量失败求解，不表示有效轨迹更新更快。v5 日志保留每次求解、拒绝、恢复和执行事件，支持后续核对停滞原因。

三组运行共执行 329 次接受前检查。独立复算与全部日志判定一致；174 次拒绝均存在真实走廊约束违反，没有误拒绝证据。动态 AC-MPC 最终停在高度 4.79416 m，固定 IPC 停在 4.91673 m，均超过当前走廊顶面 4.75 m。末段速度和加速度已接近 0，未触发源状态截断。

原模型步长为 0.1 s，竖直 jerk 下界为 -50 m/s³，因此静止状态在第 1 个预测结点最多下降约 0.008333 m。两组末状态仍分别违反顶面约 0.03582 和 0.15840 m。独立线性可行性求解也确认原约束不可行。现有制动规则基于速度减速，不会主动将位置移回走廊；未发现“已静止且 QP 成功，却无法退出制动”的状态机样本。这解释了持续失败，但不等于完成导航验收。证据位于各运行的 `analysis-v5.json`。

抬升主要发生在制动区间，两组分别在这些区间上升约 3.015 和 3.100 m。全部竖直 jerk 符合原普通制动公式，没有检测到 safe-climb 指令。这是阶段关联，当前证据没有隔离抬升的单一原因。

### dense 尚未解决的执行问题

两次最终 dense 碰撞都发生在高于地面的 PCD 障碍处。默认增益运行的参考位置曲线最小净空为 0.23677 m，最终实际跟踪误差为 0.79523 m。16/8/12 对照的末端参考净空仅为 0.03580 m，实际误差为 0.07251 m，发生擦碰。

当前分配矩阵和每旋翼 5.518125 N 上限给出的全局力矩范数上界为 1.560761 N·m。惯量为 0.012 I，因此世界系角速度平均变化率的宽松上界为 130.063 rad/s²。两个实际捕获的参考分别要求 337.646 和 1428.300 rad/s²，已有正常相邻命令区间超过该上界。计算还忽略了电机延迟和同时满足指定总推力的要求，因此不能靠当前控制器精确执行这两条完整姿态参考。

这只证明具体参考区间不可精确执行，不证明整个 dense 任务无法完成。剩余问题是规划参考、旋翼能力和停车点跟踪余量尚未匹配。当前没有通过修改官方速度、地图、任务判据、Newton 训练代码或 MPC 模型来掩盖失败。

对应证据包括两次 dense 目录中的 `reference-feasibility-analysis.json`、`tracking-analysis.json`、`holding-causality-analysis.json`，以及 `bounded-replays/` 和 `signed-budget/`。负结果和中间失败均保留。

### 构建和保留检查

后续已重编译 ROGMap、SUPER 规划器和 MARSIM 桥接包。新建的干净 SUPER 副本与实际构建副本比较，553 个源码文件完全一致，证据为 `super-fresh-prepare.json`。这验证了修复可从固定参考版本重建，不依赖手工修改后的旧构建目录。

此后仅新增 PerfectDrone main 的关闭补丁。该 main 已另行验证从固定参考源码生成的一致性和重复 prepare 的幂等性，并完成增量编译和上述官方完整任务复测。原 PerfectDrone 执行类的哈希仍与参考一致。

最新 Python 回归检查为 41 项通过，没有跳过项。它涵盖 ROS 策略、评测统计、执行语义、MPC 和 SUPER 任务配置；Newton 任务工厂使用原有地图资产完成配置检查。Ruff 与 `git diff --check` 通过。原生分配 CTest 为 1/1 通过，覆盖 3773 组输入。上述检查补充软件和数值证据，不替代本节记录的真实任务运行。

最终 `preservation-final.json` 记录：原目录仍在 main，HEAD 为 `7d6da6b`；已有工作树改动列表与前一阶段一致。3 个参考仓库的 tracked/staged diff 均为空。RL、Planner、执行模型和 MPC 源码相对任务基线没有差异。检查时原 `isaaclab-navrl` 训练进程 PID 1647733 仍在 GPU 上运行，本次评测进程列表为空。

## 12. SUPER MARSIM 真实运行 Rerun 回放

`scripts/export_marsim_super_rerun.py` 使用**此前已完成的真实 ROS 2 评测记录**，不重新运行仿真。它把实际里程计位置、四元数、速度、逐段净空、发布的 `PositionCommand`、实际到达航点和任务结局写入 RRD。静态地图使用对应评测的**完整原始 PCD 有效点**，并核对点数。RRD 自包含地图和飞行数据。

现有四份录制在工作树 `recordings/` 下，不提交 Git：

| 文件 | 场景 | 执行模式 | 真实任务结果 |
|---|---|---|---|
| `super_marsim_high_speed_ideal.rrd` | high_speed | ideal | success，2/2 |
| `super_marsim_dense_ideal.rrd` | dense | ideal | success，2/2 |
| `super_marsim_high_speed_dynamic.rrd` | high_speed | dynamic | success，2/2 |
| `super_marsim_dense_dynamic_collision.rrd` | dense | dynamic | collision，0/2 |

使用 Rerun 桌面端直接打开文件：

```bash
rerun recordings/super_marsim_high_speed_ideal.rrd recordings/super_marsim_dense_ideal.rrd \
  recordings/super_marsim_high_speed_dynamic.rrd recordings/super_marsim_dense_dynamic_collision.rrd
```

重新从指定评测生成独立文件，例如：

```bash
PYTHONPATH=src .venv/bin/python scripts/export_marsim_super_rerun.py \
  --run results/marsim-ros2-followup-20261104/official-ideal-dense-rest-clock-repeat \
  --output recordings/my_super_dense_ideal.rrd

rerun rrd verify recordings/my_super_dense_ideal.rrd
```

Rerun 中蓝色为逐步累计的真实飞行轨迹，灰色为完整真实轨迹，洋红色为**已发布**的 `PositionCommand` 样本，绿色为实际到达航点。速度和净空使用评测保存的原始数值。录制还保留了航点切换、末端碰撞或成功状态。

评测 NPZ **没有保存原始 LiDAR 点云帧和 SUPER 优化器候选轨迹**。因此 RRD 不提供这两类数据，不使用 PCD 地图伪装实时 LiDAR，也不把已发布的 PositionCommand 样本冒充完整的规划曲线。四份录制均通过 Rerun 的 `rrd verify` 解码检查。

## 13. 同步仿真时间（100 Hz）执行模式

新增 `--time-mode sim` / `time_mode:=sim`，**仅用于 Flightmare dynamic**。默认的 `wall` 模式及上游 SUPER PerfectDrone ideal 路径均保持原行为。原版 SUPER 的 15 Hz 重规划是配置的墙钟规划频率，不等于 IPC 的 MPC；本模式将其改为 15 Hz *仿真时间*重规划，同时按仿真步发布 100 Hz PositionCommand。

独立 `ros_sim_clock.py` 是步进协调者。每个 tick 在 `/clock` 发布确切的 `t=n×0.01 s` 后等待 `/marsim/sim_policy_done`。策略处理本次传感器与规划，发布当前时间戳的 PositionCommand 或 CTBR；之后才向 Flightmare 发送 `/marsim/sim_step` 并等待 `/marsim/sim_step_done`。原生 Flightmare 只积分 **1 个 0.01 s 步**（RK4 子步保留）。每 10 步，MARSIM 按实际里程计执行一次原生 LiDAR 渲染并确认 `/marsim/sim_scan_done`。下一个仿真步必须等待这些确认；任何 MPC 求解、OpenGL 渲染或 ROS 执行耗时都只消耗墙钟时间，**不插入额外物理积分**。

AC-MPC 使用同一个训练检查点和 jerk/QP／走廊约束，按每 10 个仿真步求解 1 次（10 Hz *仿真时间*），其间每个 0.01 s 步执行已接受的序列。控制发布 **100 Hz 仿真时间**。真实求解器仍在独立进程运行，但模拟时钟等待计算结果，避免因为机器速度变慢使轨迹被动老化。记录 `policy_*_timing`（仿真时间频率）与 `policy_*_wall_timing`（墙钟频率），另记 `real_time_factor=trajectory_duration_s/elapsed_wall_s`。

使用同一套完整往返任务与原始 PCD，仿真时间截止 `--timeout 120` 秒：

```bash
source scripts/marsim_ros2_env.sh
export DISPLAY=:11
.venv/bin/python scripts/run_marsim_ros2.py \
  --mode dynamic --time-mode sim --benchmark high_speed --policy super \
  --output results/my-super-high-speed-simtime --timeout 120 --domain 147 \
  --render-prefix 'vglrun -d egl0'
```

对于 AC-MPC，把 `--policy super` 改为 `--policy acmpc`，并提供此前正式评测使用的相同 `--checkpoint`。dense 场景只替换 `--benchmark dense` 和独立结果目录及 ROS domain。结果记录在 `results/marsim-simtime-20261104` 下；不把仿真时间频率冒充硬件实时控制能力。

### 完整 120 s 仿真时间评测（2026-11-04）

四组均已完成真实 ROS 2 图、Flightmare 原生动力学、MARSIM LiDAR 和原始 PCD 的监控；不存在用离线轨迹代替闭环运行的情况。全部保持相同 100 Hz 控制和 10 Hz LiDAR 仿真时间频率。SUPER 保留原先 15 Hz 轨迹重规划目标；AC-MPC 固定每 0.1 s 仿真时间求解 1 次。

| 结果目录 | 策略／场景 | 结果／航点 | 仿真时间 s | 墙钟耗时 s | 实时因子 | 最小净空 m |
|---|---|---|---:|---:|---:|---:|
| `full-super-high_speed` | SUPER high_speed | success，2/2 | 16.49 | 28.18 | 0.585 | 0.8031 |
| `full-super-dense` | SUPER dense | collision，0/2 | 4.15 | 10.03 | 0.414 | −0.0125 |
| `full-acmpc-high_speed` | AC-MPC high_speed | timeout，1/2 | 120.00 | 238.92 | 0.502 | 0.6715 |
| `full-acmpc-dense` | AC-MPC dense | timeout，0/2 | 120.00 | 1048.03 | 0.115 | 0.8584 |

在四个原始 `trajectory.npz` 中，所有相邻里程计时间戳均相差 0.01 s（绝对误差小于 `1e-14 s`）；LiDAR 每 0.1 s 采样，全部点云时间戳匹配实际渲染位姿。每组 `run.json` 的 114 个源码和 4 个二进制哈希均与此次构建一致。对应证据：`summary.json` 和 `timing_validation.json`。

AC-MPC 两场景都有 11701 次控制发布和 1171 次 QP 求解，仿真时间测得 100/10 Hz。墙钟下高速度场景的控制／求解更新为 **51.03/5.10 Hz**，dense 为 **11.27/1.13 Hz**。这两个低频是实际计算速度，并未改变仿真控制步长。

AC-MPC high_speed 在仿真时间 63.39 s 到达第 1 航点；返航到 120 s 时位于约 `[0.154, -43.831, 1.654] m`，尚未进入最后 1 m 到达半径。最大实飞速度为 **1.954 m/s**，QP 失败及重锚约束拒绝均为 **0 次**。它未完成 120 s 任务，不应计为成功。

AC-MPC dense 仅飞行 **1.142 m**，最终停在约 `[-0.430, -49.532, 2.442] m`。1171 次求解中 **1161 次达到迭代上限而未被判为成功**，QP 达到 **4096 次迭代**，原始残差约 `2e-5`。这不是 LiDAR 帧率或仿真时钟故障。对仿真时间 5.0、8.0、30.2 s 的失败样本，使用原始 MPC 的 jerk、速度、加速度和 SFC 约束构造线性可行性问题，HiGHS 均找到可行解。详见 `acmpc_dense_linear_feasibility.json`；**不能将此失败解释为走廊约束必然不可行**，后续应优先分析求解器收敛和目标成本。所用检查点任务为 `Isaac-IPC-Avoid-ACMPC-Direct`，并不是针对 SUPER 两场景训练的策略。

SUPER dense 即使使用同步仿真时间仍在高于地面的 PCD 障碍处碰撞。与上一轮墙钟 dynamic dense 失败相符，时间切换没有修复其旋翼动力学的轨迹跟踪能力限制。

从已完成的 SUPER 两场景生成了新 Rerun 回放：`recordings/super_marsim_high_speed_simtime.rrd` 和 `recordings/super_marsim_dense_simtime_collision.rrd`，均通过 `rerun rrd verify`。结果目录以及录制文件不提交 Git；只提交实现与本文。

独立进行了原 `--time-mode wall` 的 SUPER dynamic high_speed 6 s 回归探针：里程计实测 **99.9996 Hz 墙钟**，收到 49 帧 LiDAR、227 条 PositionCommand 和 526 条 CTBR；程序按预设 6 s 墙钟截止，无异常退出。证据在 `wall-regression-super-high-6s/`。此短时探针用于确认墙钟接口保留，不计入完整往返任务成功率。

## 历史两条 IPC 路径（统一 TurboMPC）

本节保留 2026-11-04 至 2026-11-05 的 jerk 版本及其结果。
当前 10D CTBR、28N 成本入口见 [README](../README.md) 和 [ADR-0008](adr/0008-native-acmpc-pathcover.md)。
当时维护固定成本 IPC 和学习成本 AC-MPC 两条算法路径。
二者使用相同的 PathCover/RISP、原始 SFC、jerk 预测模型、约束和 `TorchMPC` 原生 CUDA 求解器。
该版本通过 `--policy ipc` 或 `--policy acmpc --checkpoint ...` 选择成本。
其学习策略使用 124D 输入和 7D 成本参数。

该历史版本的启动入口为 `scripts/run_marsim_ros2.py`。
`sim` 控制周期为 0.01 s，MPC 以 0.1 s 仿真时间重规划。
`wall` 用于实时计算延迟实验，原有 SUPER 仅作外部比较。

该版本的检查点通过 `rl/actor.py:MPCActor` 进入 TurboMPC。
当时已移除独立的 PyTorch ADMM 生产实现。
两条策略都要求 CUDA 原生扩展，使用 `scripts/marsim_ros2_env.sh` 设置默认构建路径。
以下运行记录保留原始求解器版本与负结果，不重命名历史数据。

### high_speed 首次 IPC 求解：SFC 时间分配

2026-11-05 使用固定成本 IPC、原始 MARSIM LiDAR、Flightmare dynamic 和 100 Hz 仿真时间重新获取首个真实规划问题。原时序在预测第 4 步强制进入第二个 PathCover 走廊，而初始机体几乎静止。保留全部 jerk、速度、加速度与 SFC 约束时，HiGHS 判断原 QP 不可行，OSQP 返回 `primal infeasible`（72 ms）。原生 TurboMPC 求解 14.41 s 后达到 16384 次 ADMM 迭代上限；前端耗时 328.82 ms。

现在以当前位置和沿路径方向的实际速度、加速度硬界估算可达到上一走廊终点的最早预测时刻，再决定第二走廊的生效步数。该规则只改变 SFC 时间分配，保留原始走廊平面、状态硬界及预测模型；达到该必要可达条件不等于保证任意后续 MPC 问题都可行。

修正后的真实首个 QP 将走廊分配从前 3／后 12 步改成前 6／后 9 步。HiGHS 验证新的原问题可行；TurboMPC 在 2.05 s 内成功求解，ADMM 迭代 7600 次。6 s MARSIM 探针取得 0 次 QP 失败、5.84 m 飞行距离、5.99 m/s 最高速度、0.667 m 最小净空，控制／MPC 仿真频率为 100／10 Hz。原始实测输入和两次监控证据分别保存在 `results/ipc-two-pipelines-20261105/first-fixed-high-speed/` 和 `fixed-high-speed-timed-sfc-6s/`；首次问题的独立 OSQP/HiGHS 检查结果保存在 `first-fixed-qporacle.npz`。

同一修正完成了 **120 s 仿真时间上限内的完整 high_speed 正式任务**：去程航点在 26.61 s 到达，返程在 42.06 s 到达，最终结果为 **success、2/2 航点、无碰撞**。总航程 200.955 m，最高实际速度 11.539 m/s，最小完整 PCD 净空 0.3687 m。累计 1 次 QP 求解失败、3 次计划重锚拒绝，均由原有控制和恢复逻辑处理。控制频率 100 Hz 仿真时间、MPC 频率 10 Hz 仿真时间；421 帧 LiDAR 时间戳匹配渲染位姿、4207 次物理状态采样的时间间隔均为 0.01 s。

性能仍是下一阶段工作：完整任务墙钟耗时 460.53 s，实时因子 0.0913；QP 求解延迟 P50/P95 为 0.712/2.078 s，最长一次 41.868 s，发生于仿真时间 22.69 s（内部 ADMM 达到 16384 次迭代上限）。因此，**仿真时间闭环成功与墙钟实时性能是两项不同结论**。正式原始记录在 `results/ipc-two-pipelines-20261105/fixed-high-speed-timed-sfc-full/`。本次运行的 109 个源码及 4 个可执行文件哈希均与被测代码一致。

### 重构后的运行检查

学习成本 AC-MPC 使用迁移后的检查点，在 high_speed 动态模式运行 6 s 仿真时间：601 次里程计采样（100 Hz）、301 次有效控制回调（100 Hz）、31 次 MPC 更新（10 Hz），QP 失败 0 次。结果：`results/ipc-two-pipelines-20261104/probe-acmpc-eager/`。6 s 是接口探针的截止时间，不属于完整任务超时评测。

固定成本 IPC 的原生求解器已通过真实 PathCover/SFC 输入的约束、数值一致性和 Newton 3 步物理回归，均未发生 QP 失败。high_speed MARSIM 动态探针在首个求解时达到 60 s 墙钟等待上限，尚未通过该场景的闭环验收；证据为 `results/ipc-two-pipelines-20261104/probe-ipc-eager/`。不把它与成功的 AC-MPC 探针混为同一结果。
