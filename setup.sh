#!/usr/bin/env bash
# Build the yue2 environment from a clean clone. macOS/Metal or Windows/CUDA.
#
#   ./setup.sh [--venv PATH] [--dev] [--models PATH] [--skip-models]
#
# Weights come from MODELS_ROOT in env/pins.env unless --models says otherwise.
#
# Creates .venv/ (gitignored) from a hash-verified lockfile, then installs our
# fork of the official inference library at the commit pinned in env/pins.env.
# Idempotent: re-running rebuilds the venv from scratch and re-pins everything.
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
MODELS_SRC="${MODELS_ROOT:-}"   # default: the shared root from pins.env; --models overrides
while [ $# -gt 0 ]; do
  case "$1" in
    --venv)         VENV="$2"; shift 2 ;;
    --dev)          DEV=1; shift ;;
    --models)       MODELS_SRC="$2"; shift 2 ;;
    --skip-models)  SKIP_MODELS=1; MODELS_SRC=""; shift ;;
    -h|--help)      sed -n '2,12p' "$0"; exit 0 ;;
    *)              echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }
need() { command -v "$1" >/dev/null || { echo "missing required tool: $1" >&2; exit 1; }; }
need uv
need git

# The venv puts the interpreter in a different place on Windows, and a symlink
# there is not a symlink: MSYS ln -s deep-COPIES unless the shell was started
# with winsymlinks and developer mode, which would silently duplicate 10 GB of
# weights. A junction is the Windows equivalent and needs no privileges.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT) WINDOWS=1 ;;
  *)                               WINDOWS=0 ;;
esac

link_dir() {   # link_dir SOURCE TARGET
  if [ "$WINDOWS" = 1 ]; then
    cmd //c mklink //J "$(cygpath -w "$2")" "$(cygpath -w "$1")" >/dev/null
  else
    ln -s "$1" "$2"
  fi
}

# ---------- venv ----------
say "Python $PY_VERSION venv at $VENV"
# --clear because the script claims to be idempotent and uv refuses an existing
# venv without it. Rebuilding from the lockfile is also what makes re-running
# meaningful: a package left behind by an earlier state would otherwise survive.
uv venv --clear --python "$PY_VERSION" "$VENV" >/dev/null
[ "$WINDOWS" = 1 ] && VENV_PY="$VENV/Scripts/python.exe" || VENV_PY="$VENV/bin/python"
[ -x "$VENV_PY" ] || { echo "uv made a venv with no interpreter at $VENV_PY" >&2; exit 1; }

# Two locks, because the two machines do not run the same numbers. The mac lock
# carries the Metal attention kernel and a torch built for MPS; the Windows lock
# carries a CUDA torch from download.pytorch.org, because the torch on PyPI is
# CPU-only on Windows and would install, import and quietly never touch the GPU.
# --index-strategy is needed even here at install time: the default stops at the
# first index that has the NAME "torch", which is PyPI, which has no +cu130.
if [ "$WINDOWS" = 1 ]; then
  LOCK="requirements-win-cuda.lock.txt"; EXTRA=(--index-strategy unsafe-best-match)
elif [ "$(uname -s)" = Darwin ]; then
  LOCK="requirements.lock.txt";          EXTRA=()
else
  echo "no lockfile for $(uname -s). Only macOS and Windows are built." >&2; exit 1
fi
[ -f "$HERE/env/$LOCK" ] || { echo "missing $HERE/env/$LOCK" >&2; exit 1; }

say "Installing locked dependencies (hash-verified, $LOCK)"
VIRTUAL_ENV="$VENV" uv pip install -q --require-hashes "${EXTRA[@]}" -r "$HERE/env/$LOCK"

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
# MODELS_ROOT is one machine's shared weight store. On another machine it is
# simply a path that does not exist, and hard-failing there is wrong: the
# weights are pinned by revision and can be fetched. Only an explicit --models
# is an assertion worth failing on.
if [ -n "$MODELS_SRC" ] && [ ! -d "$MODELS_SRC" ] && [ "$MODELS_SRC" = "${MODELS_ROOT:-}" ]; then
  say "No shared weights at $MODELS_SRC on this machine -- fetching instead"
  MODELS_SRC=""
fi

if [ -n "$MODELS_SRC" ]; then
  say "Linking weights from $MODELS_SRC"
  [ -d "$MODELS_SRC" ] || { echo "no weights directory at $MODELS_SRC" >&2; exit 1; }
  SRC="$(cd "$MODELS_SRC" && pwd)"
  # Accept the model directory or its parent containing YuE2/.
  [ -d "$SRC/YuE2-3B" ] || [ ! -d "$SRC/YuE2/YuE2-3B" ] || SRC="$SRC/YuE2"
  for name in YuE2-3B YuE2-Vae; do
    [ -d "$SRC/$name" ] || { echo "missing $SRC/$name -- expected the directory holding YuE2-3B and YuE2-Vae" >&2; exit 1; }
  done
  mkdir -p "$HERE/models"
  for name in YuE2-3B YuE2-Vae; do
    rm -rf "$HERE/models/$name"
    link_dir "$SRC/$name" "$HERE/models/$name"
    echo "    $name -> $SRC/$name"
  done
elif [ "$SKIP_MODELS" = 0 ]; then
  say "Fetching weights at pinned revisions"
  MODELS="$HERE/models"
  mkdir -p "$MODELS"
  VIRTUAL_ENV="$VENV" "$VENV_PY" - "$MODELS" "$HF_YUE2_3B" "$HF_YUE2_VAE" <<'PY'
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
VIRTUAL_ENV="$VENV" "$VENV_PY" - <<'PY'
import torch, yue2, yue2.pipeline
from yue2.protocol import GenerationConfig
print(f"    python  {__import__('sys').version.split()[0]}")
# Name the backend this machine will actually use, rather than reporting mps
# on a box that has no Metal at all and calling that a verification.
backend = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
detail = torch.cuda.get_device_name(0) if backend == "cuda" else backend
print(f"    torch   {torch.__version__}   compute={backend} ({detail})")
if backend == "cuda":
    print("    profile official: no continuation, and seeds do not match the Mac")
elif backend == "cpu":
    print("    warning: no GPU here. Renders will work and will be very slow.")
print(f"    yue2    {__import__('importlib.metadata', fromlist=['x']).version('yue2-infer')}  from {yue2.__file__}")
config = GenerationConfig()
assert hasattr(yue2.pipeline.YuE2Pipeline, "_stage_boundary"), "fork commits missing: no profile stage boundary"
assert "carry" in yue2.pipeline.YuE2Pipeline.generate_semantic.__code__.co_varnames, "fork commits missing: no carry="
assert config.rng_device == "auto", "fork commits missing: no rng_device"
from yue2.profiles import OfficialProfile, ComfyUIYuE2MPSProfile
assert "profile" in yue2.pipeline.YuE2Pipeline.__init__.__code__.co_varnames
assert not OfficialProfile().supports_continuation
assert ComfyUIYuE2MPSProfile().supports_continuation
assert OfficialProfile().identity()["upstream_revision"] == "bd90e4ccae671d869b3ecaca6d7e893927d29442"
print("    engine support present: upstream official profile, opt-in compatibility policies")
PY
if [ "$WINDOWS" = 1 ]; then
  say "Done. Activate with:  ${VENV#$HERE/}/Scripts/activate"
else
  say "Done. Activate with:  source ${VENV#$HERE/}/bin/activate"
fi
