# Isaac Lab IPC

本项目维护两条无人机导航路径：**固定成本 IPC** 和 **学习成本 AC-MPC**。两者共用 IPC 前端、PathCover/RISP 安全飞行走廊和原生 TurboMPC 求解器；区别只有 MPC 成本是否由策略网络学习。

| 路径 | 成本 | 求解器 | 输出 |
| --- | --- | --- | --- |
| IPC | 原 IPC 固定二次成本 | TurboMPC CUDA | 15 步 jerk → 4D CTBR |
| AC-MPC | 124 维观测 → 7 个学习成本参数 | 同一个 TurboMPC CUDA | 15 步 jerk → 4D CTBR |

```text
Mid-360 点云 + 状态 + 目标
            ↓
A* → PathCover/RISP → SFC + 参考轨迹
            ↓
       固定成本 / 学习成本
            ↓
          TurboMPC
            ↓
    jerk → 总比推力、机体角速度
            ↓
   Newton 训练 / MARSIM 闭环评测
```

- **训练**：Isaac Lab 3 / Newton (MJWarp)，真实电机力、接触和 LiDAR。AC-MPC 使用标准 RSL-RL PPO，只训练成本网络。
- **正式闭环评测**：MARSIM + Flightmare，控制频率 100 Hz 仿真时间，MPC 以 10 Hz 仿真时间重规划。也保留可选墙钟模式，用于计算延迟测试。
- **外部对照**：原版 [SUPER](https://github.com/hku-mars/SUPER) 独立运行；其代码不属于上述两条路径。
- **非线性模型研究**：CTBR 属于 AC-MPC 的可选动力学模型，使用独立 127 维观测和检查点；不另建算法 pipeline。

## 依赖与构建

Linux、Python 3.12、CUDA 13、Isaac Lab 3、Newton、Warp 和原生 TurboMPC CUDA 扩展。使用项目锁定的 `uv.lock`；参考源码保持只读。

```bash
uv sync
uv run python scripts/prepare_mid360.py
uv run python scripts/build_native.py --upstream /home/tong/tongworkspace/reference_repos/IPC/hku-mars
# 原生 CUDA 扩展的版本、编译方法与数值检查见 docs/turbompc-native.md。
```

两条路径使用相同的 MPC 约束：9D 状态 `(p,v,a)`、3D jerk、预测步长 0.1 s、15 步预测、固定速度／加速度／jerk 界和原始 SFC 半空间。

## 统一入口

### Isaac Lab / Newton

```bash
# 学习成本 AC-MPC
.venv/bin/python scripts/acmpc.py train --task Isaac-IPC-Avoid-ACMPC-Direct \
  --num-envs 4 --iterations 32 --output results/my-acmpc

# 相同环境、相同 TurboMPC：固定成本 IPC
.venv/bin/python scripts/acmpc.py evaluate --task Isaac-IPC-Avoid-ACMPC-Direct \
  --fixed-cost --num-envs 1 --output results/my-ipc

# 已训练的 AC-MPC
.venv/bin/python scripts/acmpc.py evaluate --task Isaac-IPC-Avoid-ACMPC-Direct \
  --checkpoint results/my-acmpc/checkpoint.pt --num-envs 1 --output results/my-acmpc-eval

# IPC 的 Newton 独立回归；使用同一原生 TurboMPC
.venv/bin/python scripts/evaluate_native.py --num-envs 1 --stride 10 \
  --distance 1 --output results/my-native-check
```

森林场景对应 `Isaac-IPC-Forest-ACMPC-Direct`，需先生成 `build/forest`。CTBR 模型可通过 `Isaac-IPC-Avoid-ACMPC-CTBR-Direct` 或森林变体运行，检查点不得与 jerk 模型互换。

### MARSIM

```bash
source scripts/marsim_ros2_env.sh
export DISPLAY=:11

# 固定成本 IPC
.venv/bin/python scripts/run_marsim_ros2.py --mode dynamic --time-mode sim \
  --benchmark high_speed --policy ipc --timeout 120 --domain 130 \
  --render-prefix 'vglrun -d egl0' --output results/my-marsim-ipc

# 学习成本 AC-MPC；检查点必须使用当前原生 Actor 格式
.venv/bin/python scripts/run_marsim_ros2.py --mode dynamic --time-mode sim \
  --benchmark high_speed --policy acmpc --checkpoint results/my-acmpc/checkpoint.pt \
  --timeout 120 --domain 131 --render-prefix 'vglrun -d egl0' \
  --output results/my-marsim-acmpc
```

`dense` 场景只修改 `--benchmark dense` 与输出目录。详细评测协议、真实失败记录、时钟及 LiDAR 验收见 [MARSIM 评测](docs/marsim-ros2.md)。

## 验证

原生 TurboMPC 的 SFC、隐式梯度、线性／CTBR 模型与 CUDA 后端共享同一套数值测试。

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv run ruff check src scripts tests
```

源码与文档由 Git 保存；`results/`、`recordings/`、`build/` 和 `.venv/` 不提交。评测中的碰撞、超时、QP 失败及物理约束违反均保留在原始输出目录。

相关文档：[训练与奖励](docs/acmpc.md)、[物理执行协议](docs/execution-profiles.md)、[TurboMPC 原生实现](docs/turbompc-native.md)、[CTBR 模型](docs/turbompc-ctbr.md)、[来源与许可](docs/provenance.md)。
