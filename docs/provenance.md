# 源码来源与算法差异

## IPC 源码基准

四旋翼 IPC 位于参考目录的 `hku-mars` 子目录。
上层同名仓库是 Incremental Potential Contact，不能作为本项目基准。

固定算法 revision：`0f959ca7995e71a1fcc4942bd4a86869ab6ee975`。
本地 Docker 复现工程 HEAD：`c831816`。
其 ROS 入口和证据收集脚本来自 `repro/`。
`repro/dynipc/` 属于其他算法，未作为本次基线。

`build_native.py` 从固定 Git revision 提取 A*，只移除 ROS I/O 并释放析构资源。
源码差异位于 `build/native/port.patch`。
默认走廊使用 vendored PathCover/RISP，版本、许可与适配见 [走廊说明](pathcover.md)。
走廊输出、参考时序和 MPC 之间仍使用原来的半空间接口。

IPC 根目录无许可证，`package.xml` 的 license 为 `TODO`。
当前只验证本地可编辑安装，不发布带算法源码或二进制的 wheel。
正式分发前需要澄清上游授权。

## 物理资产

资产取自邻近 `isaaclab-DyIPC/src/isaaclab_dynipc/assets/`。
复制时所在提交为 `fe9dabc9a2781efc20e72a324137b43cc1497df7`。
该工作树有未提交修改，尤其是 `rate_control.py`。
因此本项目保留的是当时的文件快照，不是该提交的干净副本。
复制文件保留 BSD-3-Clause 来源声明。
本项目只改自身命名空间和碰撞球半径，未修改来源工作树。

无人机使用官方 `Thruster` 电机模型。
Newton articulation 适配保留物理状态、四电机和力矩分配。
电机逆动态补偿来自上述快照，属于执行层扩展。
质量 0.75 kg、转动惯量 0.012 kg·m²、碰撞半径 0.25 m。
这些是重建值，不是经过辨识的原版飞行器参数。
控制频率 100 Hz，物理频率 300 Hz。
真实配置只有 reset 写入位姿；飞行通过电机力和力矩推进。
理想配置在独立理想执行器中直接采用规划位置。

## 已明确的兼容性差异

- OSQP-Eigen 0.7.0 只接收 Hessian 上三角。
  原 IPC 写在下三角的连续性非对角项因此被丢弃。
  当前目标矩阵保留这一实际行为。
- CPU 基线使用 OSQP 1.0.5；Docker 使用 OSQP 0.6.2。
  默认停止容差与自适应 rho 时序会使控制量不同。
  严格容差对照验证同一 QP 的数学解，不代表默认迭代轨迹一致。
- 理想配置首次 QP 失败时抛出明确错误。
  后续失败使用上一条 jerk 序列，按原时序推进并截断索引。
  真实配置使用用户批准的制动重规划，见 ADR-0003。
- A* 端点迁移最多搜索 20 m，找不到则报错。
  原代码在没有自由端点时会无限循环。
- 原避障配置的 `perfect_simu` 会把观测速度置零。
  理想诊断脚本保留该行为，并直接采用预测位置。
  Newton 路径使用真实速度和官方 PVA 净加速度。
- 原森林接收 PVAJ 并使用 MARSIM 的 SO3 控制器。
  Newton 路径执行 IPC 的平坦变换和机体角速度命令。
- 森林地图相同，感知表面重建不同。
  当前射线命中体素表面，不复刻 MARSIM 的点云平面插值。
  完整原始 PCD 独立承担碰撞判定。

## Mid-360 数据来源

方向表来自 AME-2 `leggedrobotics/ame2_minimal`，固定 revision
`34b6fe9eae6ff2c599ca6ecffa24c76e9bce8148`。
`scripts/prepare_mid360.py` 下载方向表与 GPL-3.0 许可证到被 Git 忽略的 `build/sensors/`。
数据与来源清单保留该许可，不按本项目的 BSD 代码声明重新授权。
未复制 AME-2 的传感器类或 `_update_buffers_impl`；射线查询由官方 MultiMesh 承担。
实现与验收见 [官方组件迁移](official-components.md)。

## GPU 范围

Newton/MJWarp、力矩计算、碰撞检查和射线查询在 GPU。
QP 使用 Torch 批量矩阵和 CUDA graph。
A*、PathCover/RISP、点云历史聚合、QP 组装仍在 CPU。
CPU 几何扩展内部有静态随机数状态，当前不并发调用。
端到端 GPU 前端和消除逐环境 Python 循环尚未完成。

cuRobo 未接入运行路径。
默认 cuRobo 轨迹求解器不能替代 IPC 的状态与 SFC 硬约束 QP。
本阶段复用 Warp 和 Isaac Lab 的射线接口，不增加无实际调用的依赖。
