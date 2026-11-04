---
status: accepted
---

# MARSIM 负责正式评测，Isaac Lab/Newton 负责训练

用户于 2026-11-04 指定正式评测使用 MARSIM，并保留两种执行模式。

- `ideal` 启动原 SUPER `perfect_drone_sim`，执行 `PositionCommand` 的位置、速度、加速度和 yaw。
- `dynamic` 启动 Flightmare `flightlib::Quadrotor`，接收总比推力（m/s²）与机体角速度（rad/s）。
- 动态模式使用角速度控制、旋翼分配、电机响应和 RK4 积分。LiDAR 使用积分后的实际位姿。
- ROS 2 Launch 在一次运行中只启动一种执行器。策略可选择原 SUPER、固定成本 IPC 或学习成本 AC-MPC。
- AC-MPC 复用训练时的 Planner、PathCover/RISP、观测构造、MPCActor 和可微 QP，不修改预测模型。
- 原参考仓库保持只读。ROS 2 构建和必要的上游编译适配位于独立构建副本。

这里的 MARSIM `ideal/dynamic` 是评测执行器选择。
既有 `ExecutionProfile.ideal/real` 仍定义 IPC 历史复现与 Newton 训练行为，不能将两者混为一谈。
SUPER 理想执行不要求将策略读取的真实速度置零。

验收必须运行真实 ROS 节点和实际 LiDAR。
分别记录成功、碰撞、超时、轨迹、速度、完整 PCD 净空、QP 失败和墙钟耗时。
100 Hz 发布频率与 MPC 求解耗时分别测量，不能用设定的定时器周期代替测量。

构建、运行与验收证据见 [MARSIM ROS 2 评测](../marsim-ros2.md)。
