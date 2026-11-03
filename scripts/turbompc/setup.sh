#!/usr/bin/env bash
# Recreate a profile from its lockfile; generated files remain under build/.
set -euo pipefail
if [[ $# -ne 1 || ( "$1" != cu128 && "$1" != cu130 ) ]]; then
    echo "Usage: bash scripts/turbompc/setup.sh cu128|cu130" >&2
    exit 2
fi
profile="$1"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
uv_command="${TURBOMPC_UV:-uv}"
python_command="${TURBOMPC_PYTHON:-$project_root/.venv/bin/python}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-1800}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
python3 scripts/turbompc/install_toolkit.py "$profile"
environment="$project_root/build/turbompc/envs/$profile"
if [[ ! -x "$environment/bin/python" ]]; then
    "$uv_command" --no-config venv --python "$python_command" "$environment"
fi
"$uv_command" --no-config pip sync --python "$environment/bin/python" \
    --index-url https://pypi.org/simple \
    --torch-backend "$profile" \
    "scripts/turbompc/requirements-$profile.txt"
"$uv_command" --no-config pip check --python "$environment/bin/python"
