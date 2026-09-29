# Final phase environment. Source before any P4 stage:
#   source scripts/p4_env.sh
# Every cache/temp path is on /scratch. PYTHONPATH puts THIS worktree's src first: the venv
# has `ber` installed editable from the main tree, which would otherwise shadow it.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$_here/src${PYTHONPATH:+:$PYTHONPATH}"
export BER_SCRATCH_ROOT="${BER_SCRATCH_ROOT:-/scratch/user4}"
export P4_ROOT="$BER_SCRATCH_ROOT/p4"
export UV_CACHE_DIR=$P4_ROOT/cache/uv
export PIP_CACHE_DIR=$P4_ROOT/cache/pip
export HF_HOME=$P4_ROOT/cache/hf
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # models are downloaded once; no network at run time
export TORCH_HOME=$P4_ROOT/cache/torch
export XDG_CACHE_HOME=$P4_ROOT/cache/xdg
export TMPDIR=$P4_ROOT/tmp
export PY="${PY:-$(command -v python)}"
# least-loaded GPU per stage (compute.pick_gpu via BER_GPU; default.yaml's static gpu_ids: [1]
# would otherwise pin every stage to GPU 1). Override with BER_GPU=0 or 1 to steer a job.
export BER_GPU=${BER_GPU:-0,1}
export BER_BLOCKING_SYNC=${BER_BLOCKING_SYNC:-1}   # GPU syncs sleep instead of spin (ber.gpu.blocking_sync)
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-PASSIVE}   # idle OpenMP threads sleep (spinning showed up as ~20x host CPU)
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$P4_ROOT/logs"

_p4_root_free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
p4_check_disk() {
  local free; free=$(_p4_root_free_gb)
  local floor=${P4_MIN_ROOT_GB:-2}   # owner accepted / below 10 GB on 2026-09-27 ("go"); nothing P4 writes to /
  if [ "$free" -lt "$floor" ]; then
    echo "STOP: / has ${free} GB free (< ${floor} GB). Refusing to run." >&2
    return 1
  fi
  echo "[p4] / free: ${free} GB"
}
p4_check_disk
