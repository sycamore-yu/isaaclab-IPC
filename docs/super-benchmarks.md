# SUPER High-speed / Dense 场景接入

新增两组独立 Isaac Lab 任务：SUPER 地图和场景参数，IPC 线性可微 MPC，Newton 真实执行。
原有 Avoid / Forest 任务默认值不变。使用固定 IPC 成本录制，不把未在这些地图训练的旧策略称为已适配策略。

## 上游实际使用的任务

来源：[hku-mars/SUPER](https://github.com/hku-mars/SUPER/tree/2ad3419c127a617c6d7df6925e81a14175a9c096)，
固定 revision `2ad3419c127a617c6d7df6925e81a14175a9c096`。
以 ROS1 `mission_planner/launch/benchmark_high_speed.launch` 和 `benchmark_dense.launch` 为入口，
两者实际均加载 `mission_planner/data/benchmark.txt`，不是按文件名猜测 Dense 的任务文件。

两组起点均为 **(0, -50, 1.5) m**，初始 yaw 为 **1.5741 rad**。
任务依次发布 **(0, 50, 1.5)**、**(0, -50, 1.5)**：是往返任务，不是只飞第一段 100 m。
任务文件每个 waypoint 的切换距离为 **1 m**，优先于 `waypoint.yaml` 的全局 `switch_dis=3`。
程序启动延迟为 3 s。

| 参数 | High-speed | Dense |
| --- | --- | --- |
| PCD | `random_map_24_6635.pcd` | `random_map_2_26609.pcd` |
| `max_vel` (m/s) | 25 | 8 |
| `max_acc` (m/s²) | 20 | 15 |
| `max_jerk` (m/s³) | 120 | 120 |
| `max_omg` (rad/s) | 5 | 2.5 |
| 比推力下限 / 上限 (m/s²) | 6 / 25 | 6 / 17 |
| 地图分辨率 (m) | 0.3 | 0.05 |
| 膨胀分辨率 × 步数 | 0.3 × 2 | 0.1 × 3 |
| `map_size` (m) | [15, 110, 6] | [15, 110, 6] |
| 固定地图原点 (m) | [0, 0, 1.5] | [0, 0, 1.5] |
| 虚拟地板 / 天花板 (m) | -0.5 / 5 | -0.1 / 3.5 |
| 机器人半径 (m) | 0.2 | 0.2 |
| 规划 horizon (m) | 30 | 7 |
| 雷达类型 | GENERAL_360 | GENERAL_360 |
| 雷达距离 (m) | 70 | 15 |
| 雷达频率 (Hz) | 10 | 10 |
| 角分辨率 (deg) | 0.3 | 0.4 |
| 垂直 FOV (deg) | 178 | 178 |
| 盲区 (m) | 0.1 | 0.1 |
| `downsample_res` (m) | 0.1 | 0.1 |

完整原始配置、launch 选择和解析后的任务保存在
`results/super-benchmark-20261103/source-contract.json`；未改写只读参考仓库。

## 已准备的地图

```bash
env -u PYTHONPATH .venv/bin/python scripts/build_forest.py \
  --pcd /path/to/SUPER/mars_uav_sim/perfect_drone_sim/pcd/random_map_24_6635.pcd \
  --output build/super/high-speed --drop-nan

env -u PYTHONPATH .venv/bin/python scripts/build_forest.py \
  --pcd /path/to/SUPER/mars_uav_sim/perfect_drone_sim/pcd/random_map_2_26609.pcd \
  --output build/super/dense --drop-nan
```

新增 ASCII float32 PCD 支持，原二进制读取入口和默认非有限值拒绝行为不变。
原地图分别含 110 / 2,734 行无坐标 NaN，转换时显式用 `--drop-nan` 移除，
保留全部 186,905 / 412,021 个有限点。无限值仍拒绝。
原 PCD 保持只读，独立碰撞输入保留全部有限点；射线网格为 0.1 m 体素表面，
分别为 720,492 / 1,299,012 个三角形。
新任务使用官方 MultiMeshRayCaster 查询该网格。MARSIM 的点云投影/覆盖渲染与体素网格射线查询是不同的几何观测实现。

## 运行与录制

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/acmpc.py evaluate \
  --task Isaac-IPC-SUPER-HighSpeed-ACMPC-Direct --fixed-cost --num-envs 1 \
  --output results/my-super-high-speed \
  --record recordings/my-super-high-speed.rrd

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/acmpc.py evaluate \
  --task Isaac-IPC-SUPER-Dense-ACMPC-Direct --fixed-cost --num-envs 1 \
  --output results/my-super-dense \
  --record recordings/my-super-dense.rrd
```

输出目录和 RRD 必须不存在。每次运行包含 3 s 启动悬停和两个连续航点，
中途不重置机体、速度或传感地图。启动期间参考位置和所有预测步的 SFC 保持当前区域，
3 s 后发布首个导航目标。评估期限为 180 s，这是本项目的测试预算；原 waypoint mission 没有超时。

RRD 显示整张地图、起点/往返目标、当前航点、实际机体轨迹、SFC、参考/预测轨迹、
速度、间距、制动和最终结果。碰撞或超时也会结束并保留录制。

## 参数如何进入 IPC/Newton

- 地图分辨率、固定 `map_size` 和原点用于原生 A* 的实际占据网格。
  虚拟地板/天花板同时约束 A* 与 SFC；膨胀尺寸采用上游的分辨率乘步数。
  A* 使用欧氏启发式和 26 邻域。Dense 网格包含 79.2 M 个占据单元，
  搜索节点改为按访问分配，避免为所有单元创建对象；旧默认场景的保存路径逐点一致。
- 固定地图累积已观测点，不预装独立碰撞 PCD 到规划器。
  局部路径目标范围取上游 `planning_horizon` 的 30 / 7 m。
- GENERAL_360 不是完整角度网格每行都扫描：MARSIM 每帧随机偏移、取 128 个舍入后的图像行。
  新模式保留此规律及图像行翻转，High-speed / Dense 每帧分别为 153,600 / 115,200 条射线。
  官方 MultiMesh 负责位姿和网格查询，只在 10 Hz 更新其方向缓冲区；不继承新传感器类。
  Python RNG 提供可复现种子，随机数序列与上游 C++ 引擎不同。
- 速度、加速度和 jerk 数值写入 IPC 各轴硬界；参考路径采样间距按 25 / 8 m/s 名义推进速度计算。
  MPC 仍为 15 步、每步 0.1 s，控制仍为 100 Hz。名义参考速度不是实测飞行速度。
- 平坦性命令采用源配置的比推力范围与角速度模上限；机器人半径为 0.2 m，
  同时用于 SFC 与完整 PCD 的物理子步碰撞统计。Newton 仍使用现有 IPC 机体和电机。

SUPER 的速度/加速度罚项针对向量模；IPC 仍为逐轴硬界，二者不是同一约束集合。
`records.json` 因此另记实际速度/加速度/角速度和规划 jerk 的向量模越界步数、
实际峰值速度及已到达航点。原 SUPER 的 MINCO 优化、备份轨迹、15 Hz FSM、
平坦性模型质量/气动参数和理想轨迹执行不替换当前 IPC/Newton 算法。

## 本轮运行证据

`results/super-benchmark-20261103/` 保存完整源配置、有效运行配置核对、测试与运行记录。
正式 RRD：`recordings/super_high_speed_ipc_newton.rrd`、`recordings/super_dense_ipc_newton.rrd`。
原先两次 RRD 为历史尝试，不作为已完成任务的成功率。High-speed 曾在局部终点被占据时出现不可恢复异常；Dense 旧尝试尚无完整 episode 统计。新结果必须使用独立文件名，并检查实际终止记录。
