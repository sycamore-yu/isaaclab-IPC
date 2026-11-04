#!/usr/bin/env bash
# Run a command in one isolated CUDA profile. No shell startup files are changed.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/turbompc/env.sh cu128|cu130 COMMAND [ARG ...]" >&2
    exit 2
fi
profile="$1"
shift
case "$profile" in
    cu128) toolkit_version=12.8.1; cuda_major=12 ;;
    cu130) toolkit_version=13.0.2; cuda_major=13 ;;
    *) echo "Unknown CUDA profile: $profile" >&2; exit 2 ;;
esac
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
profile_root="$project_root/build/turbompc"
export VIRTUAL_ENV="$profile_root/envs/$profile"
export CUDA_HOME="$profile_root/toolchains/cuda-$toolkit_version"
export CUDA_PATH="$CUDA_HOME"
export CUDAToolkit_ROOT="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CMAKE_CUDA_COMPILER="$CUDACXX"
export CMAKE_CUDA_ARCHITECTURES=89
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR="$profile_root/torch-extensions/$profile"
export TURBOMPC_NATIVE_BUILD_DIR="$profile_root/native/$profile"
export TORCHINDUCTOR_CACHE_DIR="$profile_root/inductor/$profile"
export TRITON_CACHE_DIR="$profile_root/triton/$profile"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
export TURBOMPC_CUDA_PROFILE="$profile"
export TURBOMPC_BASELINE_SOURCE="$profile_root/upstream"
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
# Wheel-owned CUDA libraries must not be overridden by another toolkit.
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
if [[ ! -x "$VIRTUAL_ENV/bin/python" || ! -x "$CUDACXX" ]]; then
    echo "Profile $profile is incomplete; see docs/turbompc-cuda-environments.md" >&2
    exit 1
fi
export CUDSS_DIR="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu$cuda_major"
export MAX_JOBS="${MAX_JOBS:-2}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
if [[ "$profile" == cu128 ]]; then
    export JAX_PLATFORMS=cuda
    export JAX_ENABLE_X64=true
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi
exec "$@"
