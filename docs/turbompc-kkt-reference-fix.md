# JAX KKT IR=2：修正配置与验证记录

2026-11-03，本轮继续实施明确采用共同 IR2 数值配置：A 为上游 `7050a4f`
加下述三行 KKT 修正，B 使用原生实现已有的 IR2。独立汇总
`results/turbompc-port/m4-ir2-accepted-v1.json` 已核验精度、性能、固定输入和来源，
记录这一修正配置的 `m4_gate_passed=true`，允许随后开展 SFC 与 AC-MPC 验证。

未修改原版 A0 的失败保留为 `original_baseline_passed=false`。候选实验入口及其旧报告
仍保留 `candidate-only-not-authorized-original-A` 和 `m4_gate_passed=false`；本轮采用状态
由新的独立汇总记录。已安装原版包、原版库和环境默认值保持原样。

## 精确差异

源码来自 TurboMPC `7050a4f367d08504ecb6f5820d9161e704df5816` 的 `git archive`。
唯一源码修改位于 `turbompc/solvers/backward/csrc/cudss_sparse_kkt.cu`：

```diff
         CUDSS_CHECK(cudssConfigCreate(&config));
+        int refinement_steps = 2;
+        CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS,
+                                   &refinement_steps, sizeof(refinement_steps)));
         CUDSS_CHECK(cudssDataCreate(handle, &data));
```

这与 Torch 原生版本的 KKT 数值改动一致。ADMM、SQP、FFI allocator、CSR 组装、
求解参数和官方 PyTorch/JAX 桥接保持上游代码。cuDSS 默认 IR 为 0；候选将其设为 2，
保留默认 `IR_TOL=0`，因此固定执行两次改进。

## 独立构建

在工作树根目录运行：

```bash
bash scripts/turbompc/env.sh cu128 python scripts/turbompc/build_kkt_reference_fix.py \
  /home/tong/tongworkspace/reference_repos/TurboMPC
```

脚本仅构建 `cudss_sparse_kkt_ffi` CMake target，显式关闭 GPU post-build check，
使用已安装 cu128、JAX/jaxlib 0.6.2、cuDSS 1.0 与 sm_89。它不运行 pip install 或 CMake install。
构建前后核对已安装原版 KKT 库的 SHA256；构建日志按时间保留。

全部候选内容位于 `build/turbompc/kkt-reference-fix/cu128/`：

- `source/`：固定提交的独立源码副本，仅上述源文件有改动。
- `build/libcudss_sparse_kkt_ffi.so`：独立构建产物。
- `source/turbompc/solvers/_ffi/libcudss_sparse_kkt_ffi.so`：诊断进程的明确加载位置。
- `kkt-ir2.patch`、`manifest.json`、带时间的 manifest 和日志：精确差异、SHA256 与完整编译命令。

为保持可选 CPU 后端的可用性，脚本还复制已安装 A 的 Eigen 扩展到候选源码目录；
这些二进制未重新编译，内容不变，其 SHA256 也记录在 manifest 中。

## 显式选择候选

上游 `ffi_loader.py` 的函数支持调用方传入一个环境变量名，但
`backward_kkt_cudss_ffi.py` 调用 `load_ffi_lib(_LIB_NAME)` 时没有传入该参数。
因此该提交没有可用于 KKT 库的官方环境变量开关。

这里使用上游已有的搜索顺序：当前源码包内的 `_ffi/` 优先，其次是已安装包。
候选 `_ffi/` 中仅放 KKT 库，其他 CUDA FFI 继续从已安装 A 中查找。
必须在 `env.sh` 之后为单个诊断命令设置 `PYTHONPATH`，因为 `env.sh` 会清除外部继承值。

以下是供后续单独执行的 GPU 精度诊断；构建脚本不会执行它：

```bash
bash scripts/turbompc/env.sh cu128 env \
  PYTHONPATH="$PWD/build/turbompc/kkt-reference-fix/cu128/source" python - <<'PY'
from pathlib import Path
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
from turbompc.solvers.backward import backward_kkt_cudss_ffi as backend

expected_library = Path(
    'build/turbompc/kkt-reference-fix/cu128/source/turbompc/solvers/_ffi/libcudss_sparse_kkt_ffi.so'
).resolve()
assert Path(backend._LIB._name).resolve() == expected_library
with np.load('tests/data/turbompc_kkt_accuracy_reproducer.npz', allow_pickle=False) as fixture:
    row, col, values, rhs = (fixture[key] for key in ('row', 'col', 'values', 'rhs'))
matrix = np.zeros((40, 40), dtype=np.float64)
np.add.at(matrix, (np.repeat(np.arange(40), np.diff(row)), col), values)
pattern = SimpleNamespace(rowPtr=row, colIdx=col, N_kkt=40, nnz=values.size)
workspace = backend.CudssSparseKktWorkspace(pattern, np.float64)
try:
    answer = np.asarray(backend._call_ffi(jnp.asarray(values), jnp.asarray(rhs), workspace))
    residual = np.max(np.abs(matrix @ answer - rhs))
    error = np.max(np.abs(answer - np.linalg.solve(matrix, rhs)))
    print({'role': 'IR=2 candidate diagnostic, not M4', 'library': str(expected_library),
           'max_abs_residual': float(residual), 'max_abs_solution_error': float(error)})
    assert residual <= 1e-9 and error <= 1e-9
finally:
    workspace.close()
PY
```

不带上述命令级 `PYTHONPATH` 的正式 `cu128` 调用仍加载已安装原版 A。
共同 IR2 配置必须按上述命令显式选择，原版 A0 的调用及失败记录继续独立保留。

## 已执行的精度检查

2026-11-03，独立候选通过 40×40 保留样本：最大绝对残差 `8.61e-11`，解误差 `6.88e-11`。
经完整官方 `MPCBackend + MPC.forward` 复测，同一 9 状态 B1/N15 稳定活跃集用例的成本梯度相对原生差异为 `1.33e-8`；
全坐标中心有限差分相对误差约 `1.29e-6`。原版 IR=0 的对应有限差分误差为 `1.66e-3`。
两个步长共 88 次扰动均收敛、可行且保持同一实际 KKT 活跃集。原版安装库哈希前后不变。

正式官方接口诊断命令：

```bash
bash scripts/turbompc/env.sh cu128 python scripts/turbompc/check_jerk_official.py \
  --kkt-role original --output results/turbompc-port/official-jerk-original-new.json
bash scripts/turbompc/env.sh cu128 env \
  PYTHONPATH="$PWD/build/turbompc/kkt-reference-fix/cu128/source" \
  python scripts/turbompc/check_jerk_official.py --kkt-role ir2-candidate \
  --output results/turbompc-port/official-jerk-ir2-new.json
```

以上准确性诊断本身不构成 M4 通过记录；本轮采用及完整证据核验见下文的独立汇总。

## 完整四组候选实验

`scripts/turbompc/check_candidate_acceptance.py` 复用正式验收的两端适配器、固定输入、数值/有限差分检查、AB/BA 配对计时和负载记录。
它仅改变 A 的 KKT 库选择，原生 B 使用固定验证树的构建产物。
驱动器在 `710cc8c` 中保存；预检查核对 247 个上游 tracked 文件、精确三行差异、候选与已安装原版的库哈希，以及 B 的构建来源。

```bash
bash scripts/turbompc/env.sh cu128 env \
  PYTHONPATH="$PWD/build/turbompc/kkt-reference-fix/cu128/source" \
  python scripts/turbompc/check_candidate_acceptance.py \
  --batches 1 16 64 256 --pairs 100 --warmup 10 \
  --output results/turbompc-port/candidate-ir2-official-linear-cu128-new.json
```

CPU 预检查可加 `--preflight-only`，不导入 Torch/JAX 或初始化 GPU。
每次使用新输出名；JSON 与 NPZ 均独占创建，历史失败不覆盖。
报告分开记录 `candidate_suite_passed` 和 `m4_gate_passed`：后者在该诊断入口始终为 false。
候选的数值、性能通过只为基线修正决策提供证据，不能自动改变用户规定的未修改原版 A。
原版 KKT 库必须在实验前后保持相同哈希，运行中的所有源码和构建产物也做前后核对。

### 已完成结果

`candidate-ir2-official-linear-cu128-v1.json`（及同名 NPZ）在 2026-11-03 完成。
四个 Batch 的全部数值、两步长全坐标有限差分和性能检查通过；400 对计时样本均通过数值检查。
最大成本梯度相对差异为 `5.81e-10`，初始状态梯度为 `4.14e-10`，控制误差为 `6.78e-15`，原始约束违反量为 `4.53e-13`。
P50 的 B/A 为 `0.665 / 0.878 / 0.884 / 0.871`，P95 为 `0.695 / 0.745 / 0.816 / 0.827`，均低于 1.1。
原版安装库未变，源码/构建/动态库核验全部通过；完整时间表与原始报告哈希见 `turbompc-torch-progress.md`。
这些结果支持对上述最小修正进行决策，报告仍明确保留 `m4_gate_passed=false`。

## 本轮采用共同 IR2 配置

在完整候选结果及最小源码差异披露后，本轮继续实施采用以下明确配置：

| 字段 | 独立汇总中的值 |
| --- | --- |
| `baseline_role` | `upstream_7050a4f_plus_ir2` |
| `selection_note` | `continued implementation with explicitly disclosed equal IR2 config` |
| `m4_gate_passed` | `true` |
| `original_baseline_passed` | `false` |
| `numerical_and_performance_passed` | `true` |
| `gpu_measurements_rerun` | `false` |

`scripts/turbompc/finalize_acceptance.py` 只在 CPU 上核验既有证据。它检查 323 个源码记录、
10 个构建产物、动态库实际路径及三行补丁，确认原版与候选四组 NPZ 输入逐元素相同，
有效求解参数和阈值不变，并从 400 对原始样本重新计算 P50/P95。
四组原始轨迹、成本/状态梯度及两步长 FD 数组均重新通过数值检查。

同一工具还从官方 jerk B1/N15 诊断保存的控制序列重新构造 88 次扰动的损失和中心差分，
逐一核对实际 KKT 活跃集掩码、回放求解状态和物理约束。最大重新构造的全坐标 FD
相对误差为 `1.284e-6`。原版 A0 的稳定 FD 失败通过独立验证保留，并记录原始报告及 NPZ 哈希。
所有验收阈值保持不变；性能证据保留共享 GPU 条件和完整负载记录。

复核命令（不运行 GPU；保存新汇总时使用新的输出名）：

```bash
bash scripts/turbompc/env.sh cu128 env \
  PYTHONPATH="$PWD/build/turbompc/kkt-reference-fix/cu128/source" \
  python scripts/turbompc/finalize_acceptance.py --check-only
```

生成的新汇总为 `results/turbompc-port/m4-ir2-accepted-v1.json`，SHA256：
`1ab2df7b8438b7c80dee3a16b115f0d836f3a6054bf08dec41e9b59b6ba3e03f`。
工具以独占创建方式保存汇总，不改写原版、候选或诊断报告。七项 CPU 负例与输出保护检查、
Ruff 和差异检查已通过。

## 官方配置依据

- [cuDSS 1.0 IR 配置及默认值](https://docs.nvidia.com/cuda/cudss/1.0.0/types.html)
- [cuDSS 1.0 精度诊断建议](https://docs.nvidia.com/cuda/cudss/1.0.0/tips_and_tricks.html)
- 库发现行为依据固定提交中的 `turbompc/solvers/ffi_loader.py` 与
  `turbompc/solvers/backward/backward_kkt_cudss_ffi.py`。
