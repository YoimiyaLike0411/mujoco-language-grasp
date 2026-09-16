#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export TMPDIR="$PWD/runtime/tmp"
export XDG_CACHE_HOME="$PWD/runtime/cache"
export XDG_CONFIG_HOME="$PWD/runtime/config"
export MPLCONFIGDIR="$PWD/runtime/config/matplotlib"
export __GL_SHADER_DISK_CACHE_PATH="$PWD/runtime/cache/nvidia"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$PWD/runtime/nvidia_egl.json"
export GRASPBENCH_SAM3_URL=http://127.0.0.1:8766/infer
export PYTHONPATH="$PWD:$PWD/src"
exec .venv/bin/python "$@"
