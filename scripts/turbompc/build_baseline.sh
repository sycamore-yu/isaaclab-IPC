#!/usr/bin/env bash
# Build the unchanged JAX baseline from a private copy of the pinned source.
set -euo pipefail
if [[ "${TURBOMPC_CUDA_PROFILE:-}" != cu128 || $# -ne 1 ]]; then
    echo "Usage: bash scripts/turbompc/env.sh cu128 bash scripts/turbompc/build_baseline.sh UPSTREAM_REPO" >&2
    exit 2
fi
upstream="$1"
revision=7050a4f367d08504ecb6f5820d9161e704df5816
uv_command="${TURBOMPC_UV:-uv}"
if [[ ! -f "$TURBOMPC_BASELINE_SOURCE/.source-revision" ]]; then
    mkdir -p "$TURBOMPC_BASELINE_SOURCE"
    git -C "$upstream" archive "$revision" | tar -x -C "$TURBOMPC_BASELINE_SOURCE"
    printf '%s\n' "$revision" > "$TURBOMPC_BASELINE_SOURCE/.source-revision"
fi
library_paths="$(python - <<'PY'
import os
from pathlib import Path
from importlib.metadata import distribution
site = Path(distribution('torch').locate_file(''))
paths = sorted(str(p) for p in (site / 'nvidia').glob('*/lib') if p.is_dir())
paths.append(str(Path(os.environ['CUDA_HOME']) / 'lib64'))
print(';'.join(paths))
PY
)"
export TURBOMPC_CUDA=ON
export TURBOMPC_EIGEN=ON
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89 -DCUDAToolkit_ROOT=$CUDA_HOME -DCUDSS_DIR=$CUDSS_DIR -DCUDSS_LIB=$CUDSS_DIR/lib/libcudss.so.1 -DCMAKE_BUILD_RPATH=$library_paths -DCMAKE_INSTALL_RPATH=$library_paths -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--disable-new-dtags -DTURBOMPC_POST_BUILD_CHECK=OFF"
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
"$VIRTUAL_ENV/bin/python" "$script_dir/record_build.py" baseline \
    --source "$TURBOMPC_BASELINE_SOURCE" --upstream "$upstream" --revision "$revision" \
    --manifest-dir "$project_root/build/turbompc/baseline/$TURBOMPC_CUDA_PROFILE" \
    --uv "$uv_command"
