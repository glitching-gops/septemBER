# P2 environment: keep every cache and temp file off "/" (root disk is ~99% full).
# Source this before any P2 stage:  source scripts/p2_env.sh
export BER_SCRATCH_ROOT="${BER_SCRATCH_ROOT:-/scratch/user4}"
export P2_ROOT="$BER_SCRATCH_ROOT/p2"
export UV_CACHE_DIR=$P2_ROOT/cache/uv
export PIP_CACHE_DIR=$P2_ROOT/cache/pip
export HF_HOME=$P2_ROOT/cache/hf
export TORCH_HOME=$P2_ROOT/cache/torch
export XDG_CACHE_HOME=$P2_ROOT/cache/xdg
export TMPDIR=$P2_ROOT/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

# Hard stop if the root filesystem is nearly full (stop condition: < 5 GB free).
_p2_root_free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
p2_check_disk() {
  local free; free=$(_p2_root_free_gb)
  if [ "$free" -lt 5 ]; then
    echo "STOP: / has ${free} GB free (< 5 GB). Refusing to run." >&2
    return 1
  fi
  echo "[p2] / free: ${free} GB"
}
p2_check_disk
