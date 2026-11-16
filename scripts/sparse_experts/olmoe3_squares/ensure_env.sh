# Sourced at the top of in-job scripts (2026-11-16): gantry's `uv pip install -e external/OLMo-core[...]` can fail on a network timeout
# (PyPI download of the build requirements) WITHOUT failing the job; the image's older packages are then used and every model build dies
# with "name 'DType' is not defined". Check that the pinned package is installed; otherwise retry the install here, and give up loudly.
ensure_env() { local i
  for i in 1 2 3 4 5 6; do
    python -c "import importlib.metadata as m; m.version('ai2-olmo-core')" 2>/dev/null && return 0
    echo "== $(date -u +%m-%d\ %H:%M) [env] ai2-olmo-core is not installed (gantry's install failed): retry $i"
    UV_HTTP_TIMEOUT=600 uv pip install --system --break-system-packages -e 'external/OLMo-core[beaker,wandb,fla]' 2>&1 | tail -n 3; sleep 20; done
  echo "== [env] could not install external/OLMo-core: giving up"; return 1; }
ensure_env || exit 1
