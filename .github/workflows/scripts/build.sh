#!/usr/bin/env bash

# Adapted from vLLM's build script and vllm-omni's original build logic

set -euxo pipefail

# Argument 1: Python executable (required)
# Argument 2: CUDA version (optional, ignored for now as we rely on environment or uv)
python_executable="$1"
cuda_version="${2:-}"

log() {
  local level="$1"
  shift
  printf '[%s] %s\n' "${level}" "$*"
}

if [[ -z "$python_executable" ]]; then
  echo "Usage: $0 <python_executable> [cuda_version]"
  exit 1
fi

# Ensure uv is installed (using the provided python executable to install it if missing,
# though ideally it should be pre-installed in CI)
if ! command -v uv >/dev/null 2>&1; then
    log "INFO" "uv not found, installing via pip"
    "$python_executable" -m pip install uv
fi

# Clean previous artifacts
rm -rf dist build *.egg-info

# Build using python -m build (standard for pyproject.toml)
log "INFO" "Building with $python_executable"
"$python_executable" -m pip install build
"$python_executable" -m build

log "INFO" "Build finished"
ls -lh dist
