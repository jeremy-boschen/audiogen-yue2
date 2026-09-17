#!/usr/bin/env bash
# Build the yue2 environment from a clean clone. macOS / Apple Silicon.
#
#   ./setup.sh [--venv PATH] [--dev] [--skip-models]
#
# Creates .venv/ (gitignored) from a hash-verified lockfile, then installs our
# fork of the official inference library at the commit pinned in env/pins.env.
# Idempotent: re-running re-pins everything.
#
# --dev installs the fork editable from a sibling checkout (../YuE) instead of
# from the pin, so library changes take effect without a reinstall. Never use it
# to produce a take you intend to keep: the pin is what makes a run reproducible.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env/pins.env"

VENV="$HERE/.venv"
DEV=0
SKIP_MODELS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --venv)         VENV="$2"; shift 2 ;;
    --dev)          DEV=1; shift ;;
    --skip-models)  SKIP_MODELS=1; shift ;;
    -h|--help)      sed -n '2,12p' "$0"; exit 0 ;;
    *)              echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }
need() { command -v "$1" >/dev/null || { echo "missing required tool: $1" >&2; exit 1; }; }
need uv
need git

# ---------- venv ----------
say "Python $PY_VERSION venv at $VENV"
uv venv --python "$PY_VERSION" "$VENV" >/dev/null

say "Installing locked dependencies (hash-verified)"
VIRTUAL_ENV="$VENV" uv pip install -q --require-hashes -r "$HERE/env/requirements.lock.txt"

# ---------- the library ----------
if [ "$DEV" = 1 ]; then
  SRC="$(cd "$HERE/../YuE" 2>/dev/null && pwd || true)"
  [ -n "$SRC" ] || { echo "--dev needs a checkout at $HERE/../YuE" >&2; exit 1; }
  say "Installing yue2-infer EDITABLE from $SRC (dev mode -- not reproducible)"
  VIRTUAL_ENV="$VENV" uv pip install -q --no-deps -e "$SRC"
else
  say "Installing yue2-infer pinned at ${YUE_COMMIT:0:7}"
  VIRTUAL_ENV="$VENV" uv pip install -q --no-deps \
      "yue2-infer @ git+${YUE_REPO}@${YUE_COMMIT}"
fi

say "Installing this package"
VIRTUAL_ENV="$VENV" uv pip install -q --no-deps -e "$HERE"

# ---------- weights ----------
if [ "$SKIP_MODELS" = 0 ]; then
  say "Fetching weights at pinned revisions"
  MODELS="$HERE/models"
  mkdir -p "$MODELS"
  VIRTUAL_ENV="$VENV" "$VENV/bin/python" - "$MODELS" "$HF_YUE2_3B" "$HF_YUE2_VAE" <<'PY'
import sys
from huggingface_hub import snapshot_download
root, sha_3b, sha_vae = sys.argv[1], sys.argv[2], sys.argv[3]
for repo, rev, name in (("m-a-p/YuE2-3B", sha_3b, "YuE2-3B"),
                        ("m-a-p/YuE2-Vae", sha_vae, "YuE2-Vae")):
    path = snapshot_download(repo_id=repo, revision=rev, local_dir=f"{root}/{name}")
    print(f"    {name}: {path}")
PY
else
  say "Skipping weights (--skip-models)"
fi

# ---------- verify ----------
say "Verifying"
VIRTUAL_ENV="$VENV" "$VENV/bin/python" - <<'PY'
import torch, yue2, yue2.pipeline
from yue2.protocol import GenerationConfig
print(f"    python  {__import__('sys').version.split()[0]}")
print(f"    torch   {torch.__version__}   mps={torch.backends.mps.is_available()}")
print(f"    yue2    {__import__('importlib.metadata', fromlist=['x']).version('yue2-infer')}  from {yue2.__file__}")
config = GenerationConfig()
assert hasattr(yue2.pipeline.YuE2Pipeline, "_stage_boundary"), "fork commits missing: no MPS stage guard"
assert "carry" in yue2.pipeline.YuE2Pipeline.generate_semantic.__code__.co_varnames, "fork commits missing: no carry="
assert config.rng_device == "auto", "fork commits missing: no rng_device"
print("    fork patches present: stage guard, carry=, rng_device")
PY
say "Done. Activate with:  source ${VENV#$HERE/}/bin/activate"
