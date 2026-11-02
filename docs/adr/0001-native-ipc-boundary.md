---
status: accepted
---

# 保留 IPC 数学问题，替换运行平台和计算实现

用户于 2026-11-02 选择以发布代码为复刻基准。
固定 revision 为 `0f959ca7995e71a1fcc4942bd4a86869ab6ee975`。
保留 IPC 的 A*、SFC 硬约束和 jerk-MPC。
Newton/MJWarp 负责刚体物理，Warp 承载批量计算。
cuRobo 只复用符合 IPC 数学语义的组件。
用 cuRobo 默认轨迹优化器替换 MPC，会改变约束和求解方法。

目标包采用 Isaac Lab 标准外部包和任务注册入口。
物理状态由 Isaac Lab 管理，不另建刚体积分器。
论文差异单独记录，不合并为另一套默认参数。
数值疑点先用原求解器核对，不静默修正。

依据见 [设计调查](../design.md)。
