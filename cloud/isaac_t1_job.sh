#!/bin/bash
# Candidate independent Isaac environment; no APM environment/snapshot is modified.
# Runtime compatibility is a gate of this first workload, not a prior validated claim.
set -euo pipefail
unset MUJOCO_GL PYOPENGL_PLATFORM
export HOME=/root USER=root XDG_CACHE_HOME=/root/.cache XDG_DATA_HOME=/root/.local/share
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N
apt-get update -q
apt-get install -y -q libglu1-mesa libxt6 libxrandr2 libxi6 libxinerama1 libxcursor1 \
  libsm6 libice6 libgl1 libegl1 libvulkan1 libglvnd0 vulkan-tools python3-pip
# CUDA capability alone does not supply the RTX renderer's matching GL/Vulkan libraries.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if ! ldconfig -p | grep libGLX_nvidia > /dev/null; then
  curl -fLsS "https://us.download.nvidia.com/tesla/$DRIVER/NVIDIA-Linux-x86_64-$DRIVER.run" \
    -o /work/nvidia-userspace.run || \
    curl -fLsS "https://us.download.nvidia.com/XFree86/Linux-x86_64/$DRIVER/NVIDIA-Linux-x86_64-$DRIVER.run" \
      -o /work/nvidia-userspace.run
  sh /work/nvidia-userspace.run --no-kernel-module --silent --no-questions --accept-license \
    --no-backup --no-nouveau-check --no-x-check --install-libglvnd
fi
vulkaninfo --summary > /work/out/vulkan.txt 2>&1
grep -qi 'NVIDIA' /work/out/vulkan.txt
python3 -m pip install uv==0.9.4
uv venv --python 3.11.13 /work/isaac-venv
source /work/isaac-venv/bin/activate
uv pip install 'torch==2.7.0' 'torchvision==0.22.0' 'torchaudio==2.7.0' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install 'isaacsim[all,extscache]==5.1.0.0' --extra-index-url https://pypi.nvidia.com \
  'onnxruntime==1.23.2' 'numpy==1.26.0'
uv pip install --no-deps -e /work/DREAM
uv pip freeze > /work/out/pip-freeze.txt
cd /work/DREAM
python - <<'PY'
import hashlib, json, pathlib, urllib.request
cfg=json.loads(pathlib.Path('experiments/configs/t1_walk.json').read_text())
dst=pathlib.Path('/work/t1-policy');dst.mkdir()
for name,digest in cfg['policy_hashes'].items():
    data=urllib.request.urlopen(cfg['policy_url_root']+'/'+name, timeout=60).read()
    if hashlib.sha256(data).hexdigest()!=digest: raise ValueError('policy checksum mismatch')
    (dst/name).write_bytes(data)
PY
CASES="stand forward left"
[ "$WORKLOAD" != isaac-t1-stand ] || CASES="stand"
for CASE in $CASES; do
  LEFT=$(( CAP * 60 - 120 - $(cut -d. -f1 /proc/uptime) ))
  [ "$LEFT" -gt 0 ] || exit 124
  timeout --signal=TERM --kill-after=30s "${LEFT}s" python experiments/isaac_t1_walk.py \
    --policy-dir /work/t1-policy --case "$CASE" --out "/work/out/$CASE"
  # Kit shutdown may exit Python with status zero even during exception unwinding.
  # Persisted episode evidence is required before spending time on the next case.
  test -s "/work/out/$CASE/episode.json"
  test ! -e "/work/out/$CASE/failure.json"
  python -c 'import json,sys; assert json.load(open(sys.argv[1]))["gate_pass"] is True' "/work/out/$CASE/episode.json"
done
