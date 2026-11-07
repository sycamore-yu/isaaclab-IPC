# 历史 IPC 走廊：PathCover/RISP

本文记录旧 IPC 的单多面体替换。新 AC-MPC 主线使用完整原生 PathCover，
包括 NanoVoxMap、JPS/DMP、RISP 和 GCOPTER/MINCO。
当前接口与验收见 [ADR-0008](adr/0008-native-acmpc-pathcover.md) 和
[原生 AC-MPC 验证](acmpc-native-validation.md)。下文保留原始范围与结果。

用户在审阅现有 7 次测试和剩余验证缺口后，明确指定采用该默认设计。
EMVP 已从运行代码和构建输入移除；历史来源、失败记录和录制保留。

## 只替换单多面体生成

来源为 [PathCover](https://github.com/kunalnk123690/PathCover)，固定 revision 为
`8723a05d1993bf0199a453b47dc6ee167d493c23`。
代码和 BSD-3-Clause 许可位于 `third_party/pathcover/`。

IPC 保留 A*、点云历史、参考采样和走廊时序。
走廊函数复用 PathCover 的 RISP 单多面体内核。
PathCover 的整条路径覆盖调度不接入 IPC；同时接入会改变已限定的替换范围。
MPC 成本、状态和输入限制、Newton 物理及恢复策略保持不变。
学习策略使用原检查点，不重新训练。

PathCover 输出 `A p ≤ b`，绑定返回 `[A, -b]`。
IPC 继续按 `n·p+d ≤ 0` 读取每个平面。

## 保留 0.3 m 安全距离

PathCover 原工程使用外部膨胀地图，当前 IPC 输入则是未膨胀点云。
对非空点云，适配将 `b` 改为 `b-r‖Aᵢ‖`，其中 `r=cfg.params["sfc_dis"]`。
当前配置 `r=0.3 m`；不使用 PathCover 演示工程的其他地图参数。
空点云保持原 IPC 工作空间边界。
下边界为原版 `z≥-0.5`，对应 `Ap≤b` 的 `b=+0.5`。
2026-11-03 修复了此前误写 `-0.5` 的符号错误；此前受影响结果保持为历史记录，诊断见 [奖励与重训](reward.md)。

每次生成后检查种子与线段终点是否仍在多面体内。
种子失去安全距离时，前端报错并交给既有恢复策略。
长走廊不能包含终点时，返回空候选，IPC 保留原来的短走廊调度。
不会删除障碍点或放宽安全距离来取得可行走廊。

RISP 的 `alpha` 保持作者默认值 0.01。
全局 `rand()` 改为每次调用独立的 `mt19937`，种子为 0。
该调整固定几何输出，避免多个环境共享随机数状态。

## 使用标准构建和评估入口

默认 `_geometry.corridor` 直接使用 PathCover/RISP，不保留运行时后端开关。
需要 Eigen3、G++ 和 `libqhull-dev`；本机已具备这些依赖。

```bash
env -u PYTHONPATH uv run python scripts/build_native.py \
  --upstream /home/tong/tongworkspace/reference_repos/IPC/hku-mars
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  uv run python scripts/acmpc.py evaluate \
  --checkpoint results/acmpc-development/resume/checkpoint.pt \
  --task Isaac-IPC-Forest-ACMPC-Direct --num-envs 1 \
  --output results/my-pathcover-forest env.goal_x=32 env.episode_length_s=90
```

固定成本测试将 `--checkpoint` 及其路径换为 `--fixed-cost`。
8 m 森林测试将 `env.goal_x=32` 改为 `env.goal_x=8`。
当前入口不再调用旧生成器作并行计时。
下述同输入计时由切换前的候选测试取得，原始数据保留在本地结果目录。

## 已完成的 7 个固定场景 episode

每格仅 1 次确定性运行，种子为 0。没有重新训练。
基线来自此前相同配置的 CUDA QP 与 Newton 测试。

| 场景与成本 | 原走廊 | PathCover 候选 | 候选最小间距 |
| --- | --- | --- | ---: |
| low-fixed | 到达，6.54 s | 到达，6.35 s | 0.1513 m |
| low-learned | 到达，6.60 s | 到达，6.35 s | 0.1564 m |
| medium-fixed | 碰撞，2.34 s | 碰撞，2.43 s | -0.0112 m |
| medium-learned | 碰撞，2.35 s | 碰撞，2.43 s | -0.0139 m |
| high-fixed | 碰撞，1.44 s | 碰撞，1.96 s | -0.0196 m |
| high-learned | 碰撞，1.44 s | 碰撞，1.96 s | -0.0186 m |
| forest32-fixed | 到达，17.83 s | 到达，13.71 s | 0.0942 m |

固定成本 32 m 森林的恢复次数从 5 次降到 0 次。
其最小间距从 0.1387 m 降到 0.0942 m，仍未碰撞。
该次路线更快，但走廊生成本身未变快。
同输入的 7872 次调用，原生成器平均 1.747 ms，候选平均 1.814 ms。
候选中有 5228 次返回空的长走廊候选，由 IPC 继续原来的候选搜索。

平面符号、端点包含、0.3 m 净空和确定性已检查。
原先产生 NaN 的输入被拒绝为无效长走廊，不再输出 NaN。

## 剩余测试被显存占用阻塞

8 m 森林固定成本在环境初始化时首先报 `Warp CUDA error 2: out of memory`。
当时其他 NavRL 任务占用约 22.2 GiB 显存，总量约 24 GiB。
该次没有完成 episode，后续非法地址错误不能计为候选碰撞。

剩余测试为 8 m 森林固定/学习成本，以及 32 m 森林学习成本。
显存可用后继续；已有 7 次结果和失败日志保持原样，不重复运行。
这些验证仍待完成，不阻止用户已批准的默认设计切换。
历史对照结果与录制仍作为来源证据保留。

证据根目录为 `results/pathcover-evaluation/`。
`summary.json` 保存汇总，`forest8-fixed.log` 保存初始化失败。

## 迁移与恢复

独立候选构建和评估脚本已撤下，所有入口统一使用标准原生模块。
旧生成头文件及迁移前二进制存档在 `results/pathcover-default-migration/previous/`。
当前构建不再读取这些存档。
恢复旧走廊时，必须恢复对应源码版本并重新构建原生模块。
来源许可位于 `third_party/pathcover/LICENSE`。
