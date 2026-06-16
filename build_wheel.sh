#!/usr/bin/env bash
set -euo pipefail

# Build Rust lib into Python wheel using maturin
# Usage: scripts/build_wheel.sh [--release|--debug] [--out-dir DIR] [--python PYTHON_EXEC]

RELEASE=true
OUT_DIR="wheelhouse"
# 如果 wheelhouse 目录存在, 则删除
if [[ -d "$OUT_DIR" ]]; then
  rm -rf "$OUT_DIR"
fi
# Default python executable; may be overridden by --python or positional arg
PYTHON_EXEC="python"
# Flag set when the user explicitly provided a Python executable
PYTHON_EXEC_SET=""
source ./.venv/bin/activate
while [[ $# -gt 0 ]]; do
  case $1 in
    --release) RELEASE=true; shift ;;
    --debug) RELEASE=false; shift ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --python) PYTHON_EXEC="$2"; PYTHON_EXEC_SET=1; shift 2 ;;
    *) PYTHON_EXEC="$1"; PYTHON_EXEC_SET=1; shift ;;
  esac
done

# If a local .venv exists and user didn't explicitly set Python, prefer it
if [[ -d ".venv" && -x ".venv/bin/python" && -z "$PYTHON_EXEC_SET" ]]; then
  PYTHON_EXEC=".venv/bin/python"
fi

# Ensure chosen Python executable exists or is runnable
if ! command -v "$PYTHON_EXEC" >/dev/null 2>&1 && [[ ! -x "$PYTHON_EXEC" ]]; then
  echo "Error: Python executable '$PYTHON_EXEC' not found or not executable." >&2
  exit 1
fi

# Prefer a maturin on PATH, otherwise try running maturin via the chosen Python
if command -v maturin >/dev/null 2>&1; then
  MATURIN_MODE="path"
elif "$PYTHON_EXEC" -c "import maturin" >/dev/null 2>&1; then
  MATURIN_MODE="python-module"
else
  echo "Error: maturin not found. Install it in your chosen Python (e.g. '$PYTHON_EXEC -m pip install maturin')." >&2
  exit 1
fi

BUILD_ARGS=()
if $RELEASE; then BUILD_ARGS+=(--release); else BUILD_ARGS+=(--debug); fi

mkdir -p "$OUT_DIR"
echo "Building wheel into $OUT_DIR using $PYTHON_EXEC (release=$RELEASE)"
if [[ "$MATURIN_MODE" == "path" ]]; then
  maturin build "${BUILD_ARGS[@]}" -o "$OUT_DIR" -i "$PYTHON_EXEC"
else
  # run maturin as a module under the chosen Python
  "$PYTHON_EXEC" -m maturin build "${BUILD_ARGS[@]}" -o "$OUT_DIR" -i "$PYTHON_EXEC"
fi

# Locate the built wheel and install it into .venv
WHEEL=$(ls "$OUT_DIR"/*.whl 2>/dev/null | head -1)
if [[ -n "$WHEEL" && -f "$WHEEL" ]]; then
  echo "Wheel built: $WHEEL"
  if [[ -x ".venv/bin/python" ]]; then
    echo "Installing into .venv ..."
    # 使用 --no-deps 避免 uv 重新解析依赖（本地 wheel 本身已包含所有绑定代码）
    uv pip install --no-deps "$WHEEL"
    echo "Installed into .venv successfully."
  else
    echo "Warning: .venv not found; skip install. Wheels available in $OUT_DIR" >&2
  fi
  # 同步 .so 到项目内的 rsnumpy 包目录，避免本地源包遮蔽安装的 wheel
  SO_FILE=".venv/lib/python${PYTHON_VERSION:-3.13}/site-packages/rsnumpy/_core.cpython-${PYTHON_VERSION:-313}-darwin.so"
  if [[ -f "$SO_FILE" ]]; then
    cp -f "$SO_FILE" "rsnumpy/_core.cpython-${PYTHON_VERSION:-313}-darwin.so" 2>/dev/null || true
  fi
  # 通用回退：找任何 .venv 中刚生成的 _core .so
  SO_GLOB=$(ls .venv/lib/python*/site-packages/rsnumpy/_core.cpython-*-darwin.so 2>/dev/null | head -1)
  if [[ -n "$SO_GLOB" && -f "$SO_GLOB" ]]; then
    DEST="rsnumpy/$(basename "$SO_GLOB")"
    cp -f "$SO_GLOB" "$DEST"
    echo "Synced $DEST"
  fi
else
  echo "Warning: no .whl file found in $OUT_DIR; skip install." >&2
fi

echo "Done."
