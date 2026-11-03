# JAX KKT 精度修正候选：IR=2

此候选供审阅和单独诊断。它不是未修改的原版 A，不能替代正式 A 的结果，
也不代表 M4 已通过。正式 A、已安装包和环境默认值不因构建候选而改变。

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
是否采用修正基线以及如何重新定义验收，需依据正式 A 的完整结果单独决定；
原始失败记录应保留。

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

以上仍是准确性诊断，不是使用修正基线完成 M4 的授权或通过记录。

## 官方配置依据

- [cuDSS 1.0 IR 配置及默认值](https://docs.nvidia.com/cuda/cudss/1.0.0/types.html)
- [cuDSS 1.0 精度诊断建议](https://docs.nvidia.com/cuda/cudss/1.0.0/tips_and_tricks.html)
- 库发现行为依据固定提交中的 `turbompc/solvers/ffi_loader.py` 与
  `turbompc/solvers/backward/backward_kkt_cudss_ffi.py`。
