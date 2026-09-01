#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="af2_catgraph"
GED_DIR="$ROOT_DIR/external/Graph_Edit_Distance"
PACKAGE_CACHE="$ROOT_DIR/.conda_package_cache"
SKIP_GED=0

usage() {
  echo "Usage: bash install_af2_catgraph.sh [--env NAME] [--ged-dir PATH] [--skip-ged]"
  echo "Requires an existing Linux Conda installation. GED source is bundled; ColabFold is not installed."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="${2:?--env requires a name}"
      shift 2
      ;;
    --ged-dir)
      GED_DIR="${2:?--ged-dir requires a path}"
      shift 2
      ;;
    --skip-ged)
      SKIP_GED=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    echo "Install it on the Linux system, then rerun this installer." >&2
    exit 1
  fi
}

require_command conda
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk 'NF > 0 && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists." >&2
  echo "Use --env with a new name, or remove the old environment explicitly." >&2
  exit 1
fi

echo "Creating the tested AF2-CatGraph environment: $ENV_NAME"
echo "Using an isolated Conda package cache: $PACKAGE_CACHE"
mkdir -p "$PACKAGE_CACHE"

# Large Conda packages may be interrupted on slow or unstable connections.
# Keep downloads isolated from the user's global cache, extend the network
# timeouts, and retry the complete transaction without changing global Conda
# configuration. Users can override these defaults for their own network.
CONDA_CONNECT_TIMEOUT="${CONDA_REMOTE_CONNECT_TIMEOUT_SECS:-60}"
CONDA_READ_TIMEOUT="${CONDA_REMOTE_READ_TIMEOUT_SECS:-300}"
CONDA_NETWORK_RETRIES="${CONDA_REMOTE_MAX_RETRIES:-10}"
CONDA_CREATE_ATTEMPTS="${AF2_CATGRAPH_CONDA_CREATE_ATTEMPTS:-3}"

environment_created=0
for ((attempt = 1; attempt <= CONDA_CREATE_ATTEMPTS; attempt++)); do
  echo "Conda environment creation attempt $attempt/$CONDA_CREATE_ATTEMPTS"
  if CONDA_PKGS_DIRS="$PACKAGE_CACHE" \
    CONDA_REMOTE_CONNECT_TIMEOUT_SECS="$CONDA_CONNECT_TIMEOUT" \
    CONDA_REMOTE_READ_TIMEOUT_SECS="$CONDA_READ_TIMEOUT" \
    CONDA_REMOTE_MAX_RETRIES="$CONDA_NETWORK_RETRIES" \
    conda create --yes --name "$ENV_NAME" \
      --override-channels --channel conda-forge \
      --file "$ROOT_DIR/requirements-conda-tested.txt"; then
    environment_created=1
    break
  fi

  if conda env list | awk 'NF > 0 && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Conda left an environment named '$ENV_NAME' after the failed transaction." >&2
    echo "Inspect or remove that environment explicitly before rerunning the installer." >&2
    exit 1
  fi

  if [[ "$attempt" -lt "$CONDA_CREATE_ATTEMPTS" ]]; then
    echo "Conda environment creation failed; retrying after a short delay." >&2
    sleep 5
  fi
done

if [[ "$environment_created" -ne 1 ]]; then
  echo "Conda environment creation failed after $CONDA_CREATE_ATTEMPTS attempts." >&2
  echo "Check network or mirror availability, then rerun this installer." >&2
  exit 1
fi

# Some Linux servers export a system-first LD_LIBRARY_PATH. The CLI injects the
# Conda runtime only into AF2-CatGraph processes; it must not globally alter the
# shell because Firefox then inherits incompatible Mesa/WebGL libraries.
ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
RUNTIME_LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Installing the recorded GUI version"
LD_LIBRARY_PATH="$RUNTIME_LD_LIBRARY_PATH" LD_PRELOAD="" \
  conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip install "streamlit==1.59.2"

echo "Installing this AF2-CatGraph source tree without changing tested dependencies"
LD_LIBRARY_PATH="$RUNTIME_LD_LIBRARY_PATH" LD_PRELOAD="" \
  conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip install --no-deps --editable "$ROOT_DIR"

GED_EXECUTABLE=""
if [[ "$SKIP_GED" -eq 0 ]]; then
  require_command make
  require_command g++

  GED_REQUIRED_FILES=(
    "makefile"
    "main.cpp"
    "Application.cpp"
    "Application.h"
    "Graph.h"
    "LICENSE.md"
  )
  for required_file in "${GED_REQUIRED_FILES[@]}"; do
    if [[ ! -f "$GED_DIR/$required_file" ]]; then
      echo "Bundled GED source is incomplete: $GED_DIR/$required_file was not found." >&2
      echo "Restore external/Graph_Edit_Distance, choose another source directory with --ged-dir, or use --skip-ged." >&2
      exit 1
    fi
  done

  echo "Compiling the bundled Graph_Edit_Distance source"
  make -C "$GED_DIR" clean
  make -C "$GED_DIR"
  GED_EXECUTABLE="$GED_DIR/ged"
  if [[ ! -x "$GED_EXECUTABLE" ]]; then
    echo "GED compilation did not produce an executable: $GED_EXECUTABLE" >&2
    exit 1
  fi

  mkdir -p "$ROOT_DIR/.catcongraph"
  {
    echo "source_directory=$GED_DIR"
    if [[ "$GED_DIR" == "$ROOT_DIR/external/Graph_Edit_Distance" && -f "$ROOT_DIR/external/GED_SOURCE_RECORD.txt" ]]; then
      cat "$ROOT_DIR/external/GED_SOURCE_RECORD.txt"
    fi
    sha256sum "$GED_DIR/LICENSE.md" "$GED_DIR/makefile" "$GED_DIR/main.cpp" "$GED_DIR/Application.cpp"
    sha256sum "$GED_EXECUTABLE"
  } > "$ROOT_DIR/.catcongraph/ged_install_record.txt"
fi

VERIFY_ARGS=()
if [[ -n "$GED_EXECUTABLE" ]]; then
  VERIFY_ARGS=(--ged "$GED_EXECUTABLE")
fi
LD_LIBRARY_PATH="$RUNTIME_LD_LIBRARY_PATH" LD_PRELOAD="" \
  conda run --no-capture-output -n "$ENV_NAME" \
  python "$ROOT_DIR/scripts/verify_install.py" "${VERIFY_ARGS[@]}"

echo
echo "Installation completed."
echo "Activate: conda activate $ENV_NAME"
echo "Start GUI: cd '$ROOT_DIR' && python -m catcongraph.cli gui --fresh"
if [[ -n "$GED_EXECUTABLE" ]]; then
  echo "GED executable for Project setup: $GED_EXECUTABLE"
fi
