#!/bin/bash
# Isaac Sim 6.2 session for the walking G1: one cold install, then committed jobs from the
# bucket (cloud/isaac_session_guest.py). Sourced by vm_startup.sh, whose EXIT trap writes
# the final manifest, syncs and deletes this VM; Google's max-run-duration bounds it anyway.
set -euo pipefail
unset MUJOCO_GL PYOPENGL_PLATFORM
export HOME=/root USER=root XDG_CACHE_HOME=/root/.cache XDG_DATA_HOME=/root/.local/share
# Running Isaac Sim requires accepting NVIDIA's EULA; telemetry consent is declined.
export OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N
apt-get update -q
apt-get install -y -q libglu1-mesa libxt6 libxrandr2 libxi6 libxinerama1 libxcursor1 \
  libsm6 libice6 libgl1 libegl1 libvulkan1 libglvnd0 vulkan-tools python3-pip
# CUDA alone does not supply the RTX renderer's matching GL/Vulkan user-space libraries.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p)
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
# No pipe into head here: under pipefail, head closing early SIGPIPEs ldd (exit 141),
# which ended the first session before installation.
ldd --version > /work/out/glibc.txt 2>&1 || true
date -u +'%Y-%m-%dT%H:%M:%SZ' > /work/out/install_started_utc.txt
python3 -m pip install uv==0.9.4
uv venv --python 3.12 /work/isaac-venv
# Isaac Sim 6.2 (docs: install_python.html; Python 3.12, torch 2.11 cu128) installed from
# cloud/isaac62.lock.txt, resolved offline with exactly these index flags, so the guest never
# resolves (sessions 2 and 3 failed in on-VM resolution). Facts from that resolution:
# isaacsim-core pins mujoco==3.12.0 and mujoco-usd-converter==0.6.1 (PyPI, not NVIDIA's
# index); a dependency needs the pre-release tinyobjloader 2.0.0rc13; Isaac ships
# onnxruntime-gpu 1.26, which the WBC actors use with the CPU provider named explicitly.
sha256sum /work/DREAM/cloud/isaac62.lock.txt > /work/out/isaac_lock_sha256.txt
uv pip install --python /work/isaac-venv --index-strategy unsafe-best-match --prerelease=allow \
  --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128 -r /work/DREAM/cloud/isaac62.lock.txt
uv pip install --python /work/isaac-venv --no-deps -e /work/DREAM
uv pip freeze --python /work/isaac-venv > /work/out/pip-freeze.txt
# Separate converter environment: its USD libraries must not mix with Isaac's.
uv venv --python 3.12 /work/conv-venv
uv pip install --python /work/conv-venv --prerelease=allow 'mujoco-usd-converter==0.8.0' \
  'mujoco==3.16.0'
uv pip freeze --python /work/conv-venv > /work/out/converter-freeze.txt
# DREAM's own pinned environment for CPU MuJoCo jobs on the same VM (EGL rendering).
uv venv --python 3.12 /work/dream-venv
uv pip install --python /work/dream-venv 'mujoco==3.16.0' 'onnxruntime==1.23.2' numpy \
  'matplotlib>=3.8' 'pillow==13.0.0' 'spark-dsg==1.1.3'
uv pip install --python /work/dream-venv --no-deps -e /work/DREAM
uv pip freeze --python /work/dream-venv > /work/out/dream-freeze.txt
cd /work/DREAM
/work/conv-venv/bin/python experiments/prepare_g1_usd.py --out /work/g1_usd > /work/out/prepare_g1_usd.log 2>&1
cp /work/g1_usd/receipt.json /work/out/g1_usd_receipt.json
DREAM_CACHE=/work/cache /work/isaac-venv/bin/python experiments/fetch_wbc_policy.py
date -u +'%Y-%m-%dT%H:%M:%SZ' > /work/out/install_finished_utc.txt
gsutil -q -m rsync -r /work/out "gs://$BUCKET/out" || true
python3 /work/DREAM/cloud/isaac_session_guest.py "$BUCKET" "$CAP" /work/isaac-venv/bin/python \
  /work/dream-venv/bin/python
