---
status: accepted
---

# RL 学习 MPC 成本，控制动作由 MPC 生成

用户于 2026-11-02 指定 RL 采用类似 AC-MPC 的角色。
策略由神经成本映射与可微 MPC 组成。
PPO 更新需要把梯度穿过 MPC 传回成本网络。
Newton/MJWarp 负责环境交互，不需要反传物理轨迹。

用户允许将复杂 RL 留到下一阶段。
作者求解器不支持 IPC 的状态硬约束，因此先验收非学习 IPC。
随后开发可微硬约束 QP 与自定义 actor，复用标准训练入口。
普通 PPO 策略或成本参数作为环境动作，均不替代该交付。

IPC 基线保留固定成本，AC-MPC 扩展单独验收。
首版扩展只学习成本；A* 和 SFC 构造不纳入反传。
具体成本参数化和训练预算在第二阶段确定。

原始来源：[论文 III-F](https://arxiv.org/html/2306.09852v8#S3.SS6)、[作者代码](https://github.com/uzh-rpg/acmpc_public/tree/c59e53aec11c1fffa8b69d99b0ee7879ba7ccb28)。
接入依据见 [设计调查](../design.md#ac-mpc-接入方式)。
