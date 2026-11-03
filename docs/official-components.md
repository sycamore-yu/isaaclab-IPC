# 默认 Mid-360，状态、事件和记录交给官方组件

本页保留首次官方组件迁移的结果。后续默认传感器已采用 ADR-0006 的共享 BVH 与融合射线实现，
仍由官方 MultiMesh 管理目标、位姿和生命周期；当前设计与验证见 [规模优化](scale-optimization.md)。

先准备方向表，再使用原有评估或训练命令，无须选择传感器后端：

```bash
uv run python scripts/prepare_mid360.py
# 只需下载一次，已有 build/sensors 时不要重复执行。
env -u PYTHONPATH uv run python scripts/evaluate_native.py \
  --stride 10 --distance 1 --output results/my-mid360
```

用户已接受低速最小间距下降的取舍，Mid-360 成为真实配置的默认输入。
旧 MiniCF 方向公式、自建射线查询与真实配置合成点云分支已移除。
原版 `ideal` 对照仍保持合成点云和位置直接赋值。

## 方向窗口不接管射线查询

`make_cfg()` 将传感器声明在 `cfg.scene`，支持标准配置解析与覆盖。
`mid360.py` 配置官方 `MultiMeshRayCasterCfg`，不继承传感器类、不重写 `_update_buffers_impl()`。
每 0.1 s 随机选取 AME-2 方向表的连续 20,000 行，写入所选环境的 `ray_directions.torch`。
射线起点、机体旋转、多环境网格、移动障碍变换与命中结果均由官方组件管理。
同时更新的环境共享该扫描窗口；部分 reset 只更改对应环境。
`cfg.seed` 和 `env.reset(seed=...)` 控制窗口随机序列。

- 扫描模式：Mid-360 随机方向窗口，`ray_decimation=1`。
- 频率 10 Hz，最大距离 40 m；小于 0.3 m 的命中保留原过滤规则。
- 安装位于机体原点、轴对齐，XYZW 四元数为 `(0,0,0,1)`。
- 森林为静态网格；墙与突现障碍启用官方变换跟踪。
- 保持 Newton 物理复制；用官方 `usd_replicate` 实例化各环境的森林 USD 查询目标。
  否则源环境有网格、其他环境没有静态目标，会导致不同环境测量不一致。
- 方向表为 800,000×3 float32，加载时验证、归一化并缓存；不复制到每个环境。

数据固定为 [AME-2 revision 34b6fe9](https://github.com/leggedrobotics/ame2_minimal/tree/34b6fe9eae6ff2c599ca6ecffa24c76e9bce8148)，
下载时连同 GPL-3.0 许可证和来源清单保存于 `build/sensors/`。
缺少方向表会直接报错，不退回旧扫描器。

## 官方组件接管了哪些职责

| 原职责 | 当前实现 | 保留的含义 |
| --- | --- | --- |
| `body_state()` 读取后端数组 | `ArticulationData.root_link_pose_w` / `root_com_vel_w` | 机体位姿、质心线速度和世界角速度 |
| 电机力自行算加速度 | `PvaCfg`，将 `lin_acc_b` 旋转到世界系 | 世界系净加速度，单位 m/s² |
| 物理接触观测 | `ContactSensorCfg` → `extras["physical_contact"]` | 接触力模长大于 1e-6 N |
| reset、突现障碍时序、扫描随机窗口 | `EventManager` 的 reset / ipc_control terms | 保留原障碍激活时序、重置电机状态 |
| 轨迹、MPC 输入、Rerun | `RecorderManager` + terms | 输入对应本步动作，输出为执行后的状态 |

PVA 是无噪声仿真状态传感器，不是加入偏置和噪声的 IMU。
自由飞行与旧加速度公式相差最多 9.55e-7 m/s²；有接触时，PVA 还包含接触响应。
这时两者含义不同，采用净加速度更符合 IPC 状态输入。
部分 reset 后加速度归零，未重置环境的状态与加速度不变。

**森林几何碰撞仍检查完整 PCD**，保持 0.25 m 安全球和每物理子步锁存。
森林树木没有刚体碰撞体，ContactSensor 不能替代这个成功率判据。
地面、墙和突现障碍的刚体接触由 Newton/MJWarp 与 ContactSensor 管理。
为森林添加实体接触会改变动力学，未混入本次组件替换。

## 记录入口保持不变

`scripts/acmpc.py evaluate --record path.rrd` 现通过 RecorderManager 的 Rerun term 记录。
直接使用环境时，可在创建环境前设置：

```python
cfg.recorder_dir = "results/my-dataset"  # flight.hdf5
cfg.rerun_path = "recordings/my-forest.rrd"  # 仅支持单环境森林 MPC 任务
```

两项均可省略或单独启用。已有同名文件时报错，不覆盖结果。
HDF5 保存动作、MPC 状态/参考/SFC 快照，以及执行后的位姿、速度、加速度、接触和终止状态。
reset 自动分割 episode，close 导出剩余片段。RRD-only 不生成 HDF5。
`RecorderManager` 统一关闭；仅对官方 close 重入增加幂等保护，避免析构时再次访问已关闭的场景。

## 同场景试验结果

固定成本 IPC、OSQP；每格 **1 个 episode**，不是成功率矩阵。
下表 CPU 配对采用相同 Newton/MJWarp 后端、场景、种子和目标。
耗时是仿真时间，不是墙钟运行速度。

| 场景 | 旧输入 | Mid-360 默认输入 | 物理子步最小间距，旧 → 新 |
| --- | --- | --- | --- |
| 低速近障碍 | 到达，6.34 s | 到达，6.39 s | 0.1497 → 0.0834 m |
| 森林 8 m | 到达，3.54 s | 到达，3.56 s | 0.3923 → 0.3820 m |

两组均无碰撞、命令越界、QP 失败或制动恢复。
只替换状态/PVA/事件组件、保持旧扫描时，低速仍为 6.34 s，间距 0.149665 m。
扫描变化使轨迹和净空变化；用户已明确接受该取舍并批准切换。

实际 GPU 路径也各运行 1 次：低速到达 6.40 s、间距 **0.0687 m**；
森林 8 m 到达 3.54 s、间距 **0.3928 m**。
均无碰撞、命令越界、QP 失败或恢复。
未据共享 GPU 上的墙钟时间主张加速。

## 验证证据与未改变的边界

证据目录：`results/official-components/`。

- `probe_final.log`：官方状态/PVA/事件的双环境飞行轨迹与迁移前快照配对；位姿/速度误差阈值 1e-7。
- `probe_reset_pva.log`：有横向加速度的环境部分 reset，归零且不改变另一环境。
- `probe_multimesh_materialized.log` / `probe_default_multimesh_gpu.log`：CPU 和 GPU 双环境各 16,017 个初始命中，局部坐标一致；20,000 rays、40 m、换窗口、部分 reset 通过。
- `probe_contact_recorder.log`：接触/非接触环境区分、自由落体加速度与 HDF5 部分 reset 分段通过。
- `probe_mpc_recording.log` / `probe_default_recording.log`：动作、15×32 SFC 和状态记录逐步对齐；RRD 解码与实际视图检查通过。
- `low-default-final/`：移除旧扫描分支后的默认 CLI 完整到达，与 Mid-360 候选结果一致。
- `final-targeted-tests.log`：28 项针对性测试通过；lint / format / diff 检查通过。
- `low-*` / `forest8-*`：上述导航结果与轨迹；失败尝试日志也保留。

继续使用官方 `InteractiveScene`、`DirectRLEnv` 的奖励/终止生命周期、任务注册、
`Thruster` 电机模型和 RSL-RL PPO。Direct 任务不为形式统一再套一层 Observation/Reward Manager。
IPC A*、PathCover、硬约束 QP、控制分配和完整 PCD 安全检查是算法职责，保留自有实现。
当前 Newton 版本的官方通用 Multirotor 路径仍不能直接替代本地 articulation / 力回调适配。

组件迁移当轮未重训策略，也未重跑中高速、32 m 或完整 10 次成功率矩阵。
后续重训与新矩阵见 [奖励与重训](reward.md)。
历史训练与录制仍对应旧输入；不把短程通过推断为全部场景无退化。
