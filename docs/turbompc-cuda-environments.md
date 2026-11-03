# TurboMPC CUDA 12 / CUDA 13 测试环境

## 配置

环境用于 TurboMPC 原生移植；最终冻结验证分支为 `feat/turbompc-torch-native-verified`。
驱动由服务器提供。每个 profile 独立选择编译器、Python 依赖和扩展缓存。

最新 M4 已在 `final-v8` 记录的共享 GPU 条件下通过：被测提交 `403fc2ddb19e753ec959771cd1a4fa593328ce72`，solver SHA256 `314e9d672b93161be2bfdfdfb855c5d68a7db07d6baac9bc731e78d16131581c`。A 为上游 `7050a4f + 三行 IR2`，B 同样 IR2，使用默认 `polisher=None`；未经修改的原版 A0 仍失败。
本轮四批各 100 对完整前向加反向、双步长 FD 和全部固定门槛通过；328 个源文件与 10 个产物的来源守卫通过。正式结论见 [final-v8 汇总](../results/turbompc-port/m4-ir2-accepted-final-v8.json)，SHA256 `9fd2a6e8d5e160b8ed6cc2eff3428e755bd9646fd4908d192c2d984e3eeb9e41`；原始测量见 [final-v8 候选报告](../results/turbompc-port/candidate-ir2-official-linear-cu128-final-v8.json)。完整性能、数值表及来源见 [移植进度](turbompc-torch-progress.md)。
验收按用户授权使用共享 GPU，不要求空闲。该结果不外推为其他负载或独占条件；final-v1/v2/v3、final-v7 的失败以及被否决的成本组装、ADMM 准备融合实验均保留。

| 项目 | `cu128` | `cu130` |
| --- | --- | --- |
| CUDA Toolkit 发布版本 | 12.8.1 | 13.0.2 |
| NVCC | 12.8.93 | 13.0.88 |
| PyTorch | 2.11.0+cu128 | 2.11.0+cu130 |
| cuDSS wheel | nvidia-cudss-cu12 1.0.0.35 | nvidia-cudss-cu13 1.0.0.35 |
| JAX / jaxlib | 0.6.2 / 0.6.2 | 不安装 |
| GPU 编译目标 | sm_89（RTX 4090） | sm_89（RTX 4090） |
| 用途 | 原版 JAX 桥接及 Torch 同版本对照 | Torch 与 Isaac Lab CUDA 13 兼容验证 |

PyTorch 官方同时提供 2.11.0 的 cu128 和 cu130 wheel。
上游 TurboMPC 固定提交 `7050a4f367d08504ecb6f5820d9161e704df5816` 要求 JAX 0.6.2 与 cuDSS >=1.0。
原版 JAX 基线保留 CUDA 12。先比较 A12/B12，再报告 B13 的兼容性和性能。
仅比较 A12/B13，不能单独归因于框架迁移。

## 运行命令

从工作树根目录执行。`env.sh` 只影响本次命令及其子进程。

```bash
bash scripts/turbompc/env.sh cu128 nvcc --version
bash scripts/turbompc/env.sh cu130 nvcc --version

bash scripts/turbompc/env.sh cu128 python -c \
  'import torch; print(torch.__version__, torch.version.cuda)'
bash scripts/turbompc/env.sh cu130 python -c \
  'import torch; print(torch.__version__, torch.version.cuda)'

# 所有后续测试都使用相同入口选择环境。
bash scripts/turbompc/env.sh cu128 python YOUR_TEST.py
bash scripts/turbompc/env.sh cu130 python YOUR_TEST.py

# 需要交互操作时，启动对应环境的子 shell。
bash scripts/turbompc/env.sh cu128 bash --noprofile --norc
```

## 重建

`requirements-cu128.txt`、`requirements-cu130.txt` 固定实际验证的完整 Python 依赖。
对应 `.in` 文件保存直接依赖，供主动升级时重新解析。
`setup.sh` 使用 `uv --no-config`，避免继承 Isaac Lab 项目的依赖覆盖配置。
PyTorch 通过 `--torch-backend` 选择官方 wheel，其余包通过 PyPI 解析。

```bash
bash scripts/turbompc/setup.sh cu128
bash scripts/turbompc/setup.sh cu130
```

若当前 PATH 中没有 uv，可显式指定已安装的命令：

```bash
TURBOMPC_UV=/home/tong/.local/bin/uv bash scripts/turbompc/setup.sh cu128
```

项目 `.venv/bin/python` 提供 Python 3.12 解释器；也可通过 `TURBOMPC_PYTHON` 指定。
脚本不会重写项目 `pyproject.toml` 或 `uv.lock`。
重新同步 profile 后，需要重新执行下述原版 TurboMPC 构建命令。

## 路径与隔离

以下路径均相对于工作树根目录：

| 内容 | 路径 |
| --- | --- |
| CUDA 12 Toolkit | `build/turbompc/toolchains/cuda-12.8.1` |
| CUDA 13 Toolkit | `build/turbompc/toolchains/cuda-13.0.2` |
| Python 环境 | `build/turbompc/envs/cu128`、`build/turbompc/envs/cu130` |
| Torch 扩展缓存 | `build/turbompc/torch-extensions/<profile>` |
| 原生 TurboMPC 扩展 | `build/turbompc/native/<profile>` |
| TorchInductor / Triton 缓存 | `build/turbompc/inductor/<profile>`、`build/turbompc/triton/<profile>` |
| 原版 TurboMPC 构建副本 | `build/turbompc/upstream` |
| NVIDIA 下载与校验清单 | `build/turbompc/downloads`、各 Toolkit 的 `manifests/` |

Toolkit 使用 NVIDIA 官方 redist 组件，并逐包核对 SHA256。
包含 NVCC、CUDART、CCCL、cuSPARSE、nvJitLink 和 NVRTC。
cuBLAS 由每个 Python 环境的匹配官方 wheel 提供。
CUDA 13 另外安装拆分后的 CRT 和 NVVM。
这些组件满足当前 CUDA、Torch 扩展和 TurboMPC 构建需求。

`env.sh` 设置 `CUDA_HOME`、`CUDACXX`、`CUDAToolkit_ROOT`、`CUDSS_DIR` 与缓存路径。
它清除外部 `PYTHONPATH`、`PYTHONHOME` 和 `LD_LIBRARY_PATH`，防止加载另一套环境的包或 CUDA 库。
自定义扩展通过 RPATH 定位本环境 wheel 的动态库。
JAX 关闭显存预分配，以便在共享 GPU 上执行小规模检查。
编译默认使用 2 个 CPU 作业。

cuDSS 1.0 的 NVRTC 是加载时依赖。
两套环境分别锁定匹配 Toolkit 的 cuBLAS、NVRTC 和 builtins。
CUDA 12 与 CUDA 13 的 cuDSS 库不能互换。

## 验证命令

结果路径必须尚不存在，失败记录也会保留。

```bash
bash scripts/turbompc/env.sh cu128 python scripts/turbompc/check_toolchain.py \
  --output results/turbompc-cu128-smoke.json
bash scripts/turbompc/env.sh cu130 python scripts/turbompc/check_toolchain.py \
  --output results/turbompc-cu130-smoke.json
```

检查包括编译与加载、float64 运算、Torch 设备指针、非默认 CUDA Stream、cuDSS 稀疏求解。
报告记录实际动态库路径。稀疏求解的解误差与最大绝对残差须不超过 `1e-12`。

原版 TurboMPC 在固定提交的独立副本中构建。参考仓库只提供源码读取。

```bash
TURBOMPC_UV=/home/tong/.local/bin/uv \
bash scripts/turbompc/env.sh cu128 bash scripts/turbompc/build_baseline.sh \
  /home/tong/tongworkspace/reference_repos/TurboMPC
bash scripts/turbompc/env.sh cu128 turbompc-check
bash scripts/turbompc/env.sh cu128 python scripts/turbompc/check_baseline.py \
  --output results/turbompc-jax-bridge-smoke.json
```

上述检查属于 M0 环境验证，不能替代完整MPC前向、梯度及延迟验收。
历史共同 IR2 的 M4 门槛成立后开始 SFC/CTBR 接入；当前含可选 CTBR 前后向认证接口及编译模式矩阵向量乘法表达优化的代码另经 final-v8 完整复测，作用范围见上文与进度记录。

### 原生求解器与正式 A/B 验收

两个 profile 分别构建原生库，避免跨 CUDA 大版本加载：

```bash
bash scripts/turbompc/env.sh cu128 python scripts/build_turbompc.py
bash scripts/turbompc/env.sh cu130 python scripts/build_turbompc.py

bash scripts/turbompc/env.sh cu128 env PYTHONPATH="$PWD/src" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_turbompc_native.py tests/test_turbompc_solver.py \
  tests/test_turbompc_regressions.py tests/test_turbompc_lifecycle.py
```

将测试命令中的 `cu128` 改为 `cu130` 即可使用无 JAX 的原生环境。
`env.sh` 选择的扩展、Inductor 和 Triton 缓存均位于对应 profile 下。
如需覆盖路径，应在 `env.sh` 之后通过命令级 `env` 明确设置。

以下命令保留未经修改原版A0的独立回归。该A0已有KKT梯度失败记录，不能将环境smoke或性能通过写成A0的M4通过：

```bash
bash scripts/turbompc/env.sh cu128 python scripts/turbompc/check_acceptance.py \
  --upstream-repo /home/tong/tongworkspace/reference_repos/TurboMPC \
  --batches 1 16 64 256 --pairs 100 --warmup 10 --compile \
  --official-regression-report results/turbompc-port/official-jerk-original-b1-v1.json \
  --output results/turbompc-port/acceptance-linear-cu128-new.json
```

输出路径须使用新名称。脚本保留 JSON、输入及输出快照、原始配对耗时、负载记录和失败原因。
用户已授权共享测试，不需要等待 GPU 空闲；全部数值、梯度与性能门槛仍然适用。
当前结果与已知限制见 `turbompc-torch-progress.md`。
`--official-regression-report` 必须是相同原版库、相同固定输入的完整官方 jerk 复测；脚本检查报告及 NPZ 原始数组。
它独立判定已知精度问题，旧自定义 backend 的历史报告不再永久否决新的官方结果。
该报告的生成命令及 IR2 候选的独立使用方法见 `turbompc-kkt-reference-fix.md`。

### 当前采用的共同IR2配置：M4完整复现

使用已构建的独立IR2源码/FFI副本，通过命令级 `PYTHONPATH` 显式选择；不得覆盖已安装原版A0。
三行变更仅将A的KKT `CUDSS_CONFIG_IR_N_STEPS` 设置为2。B原本也是IR2，官方Python、backend/bridge与其他A FFI保持上游版本。
所有结果路径须为新名称，以下入口均独占保存，不覆盖历史报告。

```bash
# CPU来源预检；不执行GPU求解。
bash scripts/turbompc/env.sh cu128 env CUDA_VISIBLE_DEVICES="" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/build/turbompc/kkt-reference-fix/cu128/source" \
  python scripts/turbompc/check_candidate_acceptance.py --preflight-only \
  --batches 1 16 64 256 --pairs 100 --warmup 10 --compile \
  --output results/turbompc-port/m4-ir2-preflight-rerun.json

# 完整GPU实验：4批、各100对、双步长FD、全部固定门槛与负载记录。
bash scripts/turbompc/env.sh cu128 env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/build/turbompc/kkt-reference-fix/cu128/source" \
  python -u scripts/turbompc/check_candidate_acceptance.py \
  --batches 1 16 64 256 --pairs 100 --warmup 10 --compile \
  --output results/turbompc-port/candidate-ir2-official-linear-cu128-rerun.json

# 实验全过后，在源文件和产物仍冻结时执行CPU汇总；不重跑GPU。
bash scripts/turbompc/env.sh cu128 env CUDA_VISIBLE_DEVICES="" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/build/turbompc/kkt-reference-fix/cu128/source" \
  python scripts/turbompc/finalize_acceptance.py \
  --candidate results/turbompc-port/candidate-ir2-official-linear-cu128-rerun.json \
  --jerk results/turbompc-port/official-jerk-ir2-b1-v1.json \
  --original-jerk results/turbompc-port/official-jerk-original-b1-v1.json \
  --original-gate results/turbompc-port/acceptance-linear-cu128-verified-v2-gate-reevaluation-v1.json \
  --output results/turbompc-port/m4-ir2-accepted-rerun.json
```

候选driver固定保留 `qualification=candidate-only-not-authorized-original-A` 与 `m4_gate_passed=false`，通过字段是 `candidate_suite_passed`。
finalizer复算全部原始数据、88个稳定jerk扰动、固定设置与来源后，独立记录本轮已披露的采用配置：
`baseline_role=upstream_7050a4f_plus_ir2`、`original_baseline_passed=false`，只有全部检查真实通过才有 `m4_gate_passed=true`。
它不会把旧A0或候选报告改写成另一种角色。保留原报告/NPZ、预检和负载日志是复现的一部分。

历史 final-v4 绑定提交 `50b7298b6fdecfd923d3566705ed68584f9c01f8`，solver SHA256
`ed21a70911fa200e6b57af91e69e6f67e99860af3037670735c37a0618540d29`；327个源文件与10个产物前后相同，原版KKT哈希不变。
该阶段采用外层 SQP 融合、数值前向 detach/反向准备融合、精确线性模型解析组装；成本组装编译的负收益候选已回退。
其完整源码仅以 [事后精确哈希恢复归档](../results/turbompc-port/m4-cost-assembly-fusion-cu128-v1-source-recovery/recovery.json) 保留，不作为生产代码或新的验收数据。
final-v5、final-v6 的通过记录保留；final-v7 的 B16 P50 失败也保留。当前 final-v8 在提交 `403fc2d` 上重跑原协议，通过全部 P50/P95 的 `1.1×` 门槛；新变化只在符合形状、类型和设备条件的编译路径中，将小矩阵向量乘法写成逐项乘积与求和，eager 路径保留原表达。ADMM 准备融合候选未采用。该轮共享负载结果不能替代独占或另一固定负载条件的实验。
CUDA 13 的原生兼容验证与 CUDA 12 的 A/B 性能结论分别记录，不以 A12/B13 替代同版本框架对照。

## 官方依据

- [CUDA 官方组件归档与校验](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#tarball-and-zip-archive-deliverables)
- [CUDA 12.8.1 清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json)
- [CUDA 13.0.2 清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json)
- [PyTorch 2.11.0 安装版本](https://pytorch.org/get-started/previous-versions/#v2110)
- [cuDSS 1.0 安装与动态库要求](https://docs.nvidia.com/cuda/cudss/1.0.0/getting_started.html)
- [JAX 安装与 CUDA 动态库路径](https://docs.jax.dev/en/latest/installation.html)

## 本机验证记录

验证日期：2026-11-03（HKT）。GPU 为 RTX 4090，驱动 595.84，Python 3.12.14。

两套环境均已实际安装、编译并运行。`cu128` 的 68 个锁定依赖、`cu130` 的 43 个锁定依赖均与安装版本一致，`uv pip check` 通过。
原版 TurboMPC 额外安装到 `cu128` 后，69 个包的依赖检查也通过。`cu130` 未安装 JAX。

| 检查 | `cu128` | `cu130` |
| --- | --- | --- |
| Torch CUDA 扩展编译与加载 | 通过 | 通过 |
| 编译 / 加载的 CUDA runtime | 12.8 / 12.8 | 13.0 / 13.0 |
| float64 CUDA 运算最大绝对误差 | 0 | 0 |
| cuDSS 稀疏求解最大绝对残差 | 8.881784197001252e-16 | 8.881784197001252e-16 |
| cuDSS 解最大绝对误差 | 4.440892098500626e-16 | 4.440892098500626e-16 |
| Torch 设备指针一致 | 通过 | 通过 |
| Torch 非默认 current stream 一致 | 通过 | 通过 |
| CUDA 用户态动态库来自所选环境 | 通过 | 通过 |

结果文件：

- `results/turbompc-cu128-smoke.json`
- `results/turbompc-cu130-smoke.json`

原版 `7050a4f` 在独立副本中编译成功。`turbompc-check` 的 cuSolver、fused PCG、fused cuDSS、PCG、cuDSS、cuDSS KKT 和 Eigen 检查全部通过。
官方 JAX/PyTorch 桥接在 `float64`、预测长度 15、Batch 1 的线性问题上完成前向及反向：

| 项目 | 实测结果 |
| --- | --- |
| 前向 / 反向后端 | `admm_fused_cudss` / `direct_cudss_ffi` |
| 求解状态 | `SOLVED`，1 次 SQP，50 次 ADMM |
| 原始单位的等式最大绝对残差 | 3.6080253368320214e-13 |
| 边界最大违反量 | 0 |
| 原始 VJP | 全部有限 |
| 返回 Torch 的成本参数梯度 | 有限且非零 |

桥接结果记录在 `results/turbompc-jax-bridge-smoke.json`；原版安装检查日志位于 `build/turbompc-installation-check.log`。

以上表格是 M0 环境验证记录。后续已完成原生求解器实现，历史固定验证树的 CUDA 12/13 各 14 项目标测试已通过。最新提交 `403fc2d` 在两个 profile 分别重跑 `test_turbompc_native.py`、`test_turbompc_solver.py`、`test_turbompc_regressions.py` 的 12 项核心回归，结果如下；本轮没有重跑 lifecycle 测试，也没有重建原生库。

| 最新核心回归 | 结果 | 实际耗时 | 日志 |
| --- | --- | --- | --- |
| `cu128` | 12 项通过 | 44.38 秒 | [CUDA 12 日志](../results/turbompc-port/matvec-core-cu128-v1.log) |
| `cu130` | 12 项通过 | 51.61 秒 | [CUDA 13 日志](../results/turbompc-port/matvec-core-cu130-v1.log) |

[核心回归来源报告](../results/turbompc-port/matvec-core-results-v1.json) 记录源文件及扩展在测试前后完全相同。两套已构建原生库的 SHA256 均保持不变：

| 原生库 | SHA256 |
| --- | --- |
| `build/turbompc/native/cu128/_turbompc_native.so` | `44d85435440597af217e1ce89a586998115629b4817bc7f7222a2634ca26cf45` |
| `build/turbompc/native/cu130/_turbompc_native.so` | `b90b553da01764a02ddeb6effcc71be62302aa576ddcc1a9223a4d59e5daea3e` |

未经修改的原版 A0/B 正式实验虽通过性能门槛，KKT 梯度仍失败，该结论及原库保留。最新代码在已披露共同 IR2 配置下通过 [final-v8 正式汇总](../results/turbompc-port/m4-ir2-accepted-final-v8.json)，其 `baseline_role=upstream_7050a4f_plus_ir2`、`original_baseline_passed=false`。完整实测表、原始结果、历史失败与作用范围见 [移植进度](turbompc-torch-progress.md) 与 [KKT 参考修正说明](turbompc-kkt-reference-fix.md)。
