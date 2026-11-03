# TurboMPC CUDA 12 / CUDA 13 测试环境

## 配置

环境用于 TurboMPC 原生移植；最终冻结验证分支为 `feat/turbompc-torch-native-verified`。
驱动由服务器提供。每个 profile 独立选择编译器、Python 依赖和扩展缓存。

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

上述检查属于 M0 环境验证。完整 MPC 前向、梯度及延迟验收仍按迁移方案执行。
M4 的数值、梯度及性能门槛全部通过后，才接入 SFC 和 AC-MPC 非线性动力学。

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

正式 M4 使用相同 CUDA 12 环境中的 A/B，以共享 GPU 上的同步配对延迟作为验收：

```bash
bash scripts/turbompc/env.sh cu128 python scripts/turbompc/check_acceptance.py \
  --upstream-repo /home/tong/tongworkspace/reference_repos/TurboMPC \
  --batches 1 16 64 256 --pairs 100 --warmup 10 \
  --official-regression-report results/turbompc-port/official-jerk-original-b1-v1.json \
  --output results/turbompc-port/acceptance-linear-cu128-new.json
```

输出路径须使用新名称。脚本保留 JSON、输入及输出快照、原始配对耗时、负载记录和失败原因。
用户已授权共享测试，不需要等待 GPU 空闲；全部数值、梯度与性能门槛仍然适用。
当前结果与已知限制见 `turbompc-torch-progress.md`。
`--official-regression-report` 必须是相同原版库、相同固定输入的完整官方 jerk 复测；脚本检查报告及 NPZ 原始数组。
它独立判定已知精度问题，旧自定义 backend 的历史报告不再永久否决新的官方结果。
该报告的生成命令及 IR2 候选的独立使用方法见 `turbompc-kkt-reference-fix.md`。

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

以上表格是 M0 环境验证记录。后续已完成原生求解器实现，固定验证树的 CUDA 12/13 各 14 项目标测试全部通过。
Batch 1/16/64/256 的共享 GPU 正式 A/B 验收也已执行，P50/P95 均满足 1.1×，但原版 KKT 梯度精度仍使 M4 失败。
完整实测表、原始结果路径、构建来源和修正候选见 `turbompc-torch-progress.md` 与 `turbompc-kkt-reference-fix.md`。
