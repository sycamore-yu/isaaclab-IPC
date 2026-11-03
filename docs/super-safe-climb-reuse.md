# SUPER：低高度 A* 恢复与 LOONG 风格轨迹复用

本次仅将经过测试的 P0 安全爬升和 P1 轨迹复用合并到 IPC。
沿用原 Isaac Lab 3.0、Newton/MJWarp、GENERAL_360、Warp 增量建图、
原生 A*、PathCover/RISP，以及 15 × 0.1 s jerk-MPC。
SFC 交集中心路径点仍留在实验分支，不进入本次正式代码。

## P0：低高度起点死锁恢复

当 A* 起点落入膨胀占据单元、水平端点修复失败时，
寻找同一水平位置上方的自由终点。
使用**已观测点云**检查当前位置到目标位置的机体扫掠线段，
并为体素误差留出余量。
恢复时维持原有水平 Brake，并通过受 jerk 上限约束的高度控制爬升。
地图更新或机体横向漂移时重新检查恢复路径；发现障碍则停止爬升。
达到可规划高度且速度、加速度降低后，返回正常 A* 与 SFC 流程。

此前 High-speed 在仿真 28.33 s、位置约
(-0.960, -8.860, 0.834) m 出现持续前端失效。
将相同位置用作 Newton 初始状态的 5 s 短测显示：
安全爬升触发 1 次，在 4.61 s 左右重新获得有效 SFC，
末端高度 1.314 m，未碰撞。
该短测仍有 168 次其他前端失败和 106 次 QP 失败，
故该结果仅验证低高度死锁可恢复。

## P1：安全预测轨迹复用

正常 MPC 成功求解时，保存前 3 个预测位置及其 SFC 快照。
仅在目标未变、当前位置和复用段通过**最新膨胀占据图**
连续路径检查后，将安全前段连接到下一条 A* 路径。
重新制动、目标变更或新增障碍侵入时不使用旧轨迹。
此机制不修改 MPC 动力学、代价、碰撞半径和 SFC 硬约束。

## 配置与验证

仅 SUPER 任务默认启用 P0。
由于相同种子的短程对照显示 High-speed 有收益而 Dense 出现退化，
目前仅对 High-speed 默认启用轨迹复用，Dense 默认关闭。

| High-speed，10 s，种子 0 | P0 only | P0 + 轨迹复用 |
|---|---:|---:|
| QP 失败 | 282 | 217 |
| Brake 步数 | 629 | 525 |
| 前进距离 | 2.004 m | 6.182 m |
| 峰值速度 | 0.84 m/s | 5.42 m/s |
| 前端失败 / 碰撞 | 0 / 0 | 0 / 0 |

Dense 同种子 4 s 对照：未启用轨迹复用时 QP 失败为 0，
前进 1.495 m；启用时 QP 失败为 1，前进 0.534 m。
以上仅为短闭环结果，不代表完整 SUPER 100 m 往返成功率。

原始证据保存在独立实验工作树：
`/home/tong/tongworkspace/free/.worktrees/isaaclab-IPC/super-loong-safe-recovery-20261103/results/super-loong-20261103/`。
该处包含 `records.json`、`metrics.json` 及对应 RRD；
历史运行及中断的 Dense 正式评估未被覆盖。

验证命令：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check src scripts tests
```
