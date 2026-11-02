---
status: accepted
---

# Mid-360 默认输入，官方组件负责仿真生命周期

用户批准机体原点、轴对齐、40 m 的 Mid-360 随机方向窗口，
并在审阅低速最小间距下降后明确接受取舍、设为默认。
采用官方 `MultiMeshRayCasterCfg`，仅更新方向缓冲区，不继承自定义传感器类。

位姿/速度迁移至 ArticulationData，加速度迁移至 PVA，
物理接触使用 ContactSensor，事件与记录分别由 EventManager、RecorderManager 管理。
保留完整 PCD 的几何碰撞判据及原 `ideal` 执行语义。
删除旧扫描运行分支；历史实验和录制不改写。

用法、来源、验证和剩余范围统一见 [官方组件迁移](../official-components.md)。
