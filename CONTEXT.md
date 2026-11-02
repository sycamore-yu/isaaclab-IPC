# IPC 导航术语

本项目讨论四旋翼的 Integrated Planning and Control。

## Language

**IPC（Integrated Planning and Control）**：
将局部路径规划和预测控制结合的四旋翼导航方法。
来源为 hku-mars/IPC。
_Avoid_：Incremental Potential Contact。

**安全飞行走廊（SFC）**：
由凸多面体表示的可飞行区域。IPC 将走廊边界作为预测位置的硬约束。
_Avoid_：避障奖励、碰撞软惩罚。

**jerk**：
加速度对时间的导数，单位为 m/s³。
IPC 的预测控制输入为三轴 jerk。
_Avoid_：电机推力、角速度命令。

**机体角速度命令（body-rate command）**：
机体系下的目标角速度，单位为 rad/s。
该命令与总推力命令共同控制四旋翼。
_Avoid_：姿态角、jerk。

**突现障碍**：
飞行过程中突然进入传感范围或原本被遮挡的障碍物。
_Avoid_：所有动态障碍物。

**AC-MPC（Actor-Critic Model Predictive Control）**：
将可微 MPC 放入 actor 的强化学习方法。
神经网络学习成本参数，MPC 决定控制动作。
_Avoid_：残差控制、MPC 专家模仿、AC4MPC。

**神经成本映射（neural cost map）**：
从观测到 MPC 二次成本参数的映射。
_Avoid_：直接动作策略、奖励函数。

**IPC 基线**：
按固定发布代码定义的非学习导航方法。
_Avoid_：学习成本后的 IPC、AC-MPC 扩展。
