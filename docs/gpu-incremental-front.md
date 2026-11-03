# SUPER 增量建图与 Warp 膨胀

## 改动范围

仅 `frontend.gpu_incremental=True` 的 SUPER High-speed/Dense 任务启用。其他 Avoid/Forest 配置不变。原始完整 PCD 仍仅用于传感模拟和独立碰撞检测，规划器只累积已经观测到的点云。

1. `incremental_voxels.py` 按每帧新点更新体素质心。保留原 `voxel_centroids` 对旧质心加权为 1 的迭代语义，以及 float32 输入/质心截断。只有新增或变更的质心进入 GPU 膨胀。
2. `warp_inflation.py` 对新增、替换质心的障碍膨胀足迹执行 Warp 原子计数。删除旧质心足迹后再增加新足迹；避免历史障碍永久膨胀。GPU 持有 `int32` 计数，占据图压缩为 32 位位图。
3. `Grid.load_bits` 将压缩位图复制进原生 A*，按位直接查询，不展开为完整 CPU 字节栅格。每帧只传递约 9.9 MB Dense 地图位图，之后依旧使用上游固定版 A* 和 Floyd 简化。
4. PathCover/RISP 保持 CPU，依旧读取相同顺序与数值的质心云；不改变 SFC 半空间、规划成本及安全半径。
5. 局部 A* 无自由端点时返回可记录的前端错误，使用既有 Brake/Replan；不再使评估进程异常退出。

Dense 分辨率 0.05 m、网格范围 `[15,110,6]`，约 79.2 M cells。单环境 CUDA `int32` 计数约 316.8 MB；压缩占据图约 9.9 MB。多环境共享静态物理网格不意味着能共享彼此的已观测占据图。该方案目前针对单环境 benchmark；扩到大规模并行时应使用稀疏/局部子图并重新评估显存。

## 对照证据

`results/super-benchmark-20261103/warp-incremental-probe-v2.json` 用 SUPER Dense PCD 点云检查 18.3 万初始点和两轮各 1.2 万新增点。

| 输入 | 旧 CPU 体素+膨胀 | 增量体素+Warp+位图传输 | 同输入一致性 |
|---|---:|---:|---|
| 首轮 183014 个点 | 967.5 ms | 282.2 ms | 质心与全部占据单元一致 |
| 次轮新增 12000 点 | 1031.6 ms | 30.2 ms | 质心与全部占据单元一致 |
| 第三轮新增 12000 点 | 1116.2 ms | 34.0 ms | 质心与全部占据单元一致 |

这些是共享 RTX 4090 上的同输入单次探针，不含传感器、A*、SFC、QP、Newton 或 Rerun，不能直接推断完整训练加速或 100 Hz 实时性。

测试入口：

```bash
env -u PYTHONPATH .venv/bin/python scripts/build_native.py \
  --upstream /home/tong/tongworkspace/reference_repos/IPC/hku-mars
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q tests/test_incremental_map.py tests/test_super_benchmark.py
```

正式场景保留 SUPER 的 3 s 启动等待、100 m 往返、10 Hz GENERAL_360 和 100 Hz IPC。需要用独立 RRD/结果目录完成每张地图真实 Newton 评估；不能把短闭环超时当作完整任务成功。
