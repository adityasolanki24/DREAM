#!/bin/bash
# NERVA-style private, capped job lifecycle, isolated to this DREAM VM and bucket.
set -euo pipefail
MD=http://metadata.google.internal/computeMetadata/v1
meta() { curl -fsS -H 'Metadata-Flavor: Google' "$MD/$1"; }
# Cached images may contain a previous VM marker. Idempotency is per instance.
INSTANCE_ID=$(meta instance/id)
[ "$(cat /var/dream_started 2>/dev/null || true)" = "$INSTANCE_ID" ] && exit 0
printf '%s\n' "$INSTANCE_ID" > /var/dream_started
BUCKET=$(meta instance/attributes/dream-bucket)
NAME=$(meta instance/name)
ZONE=$(basename "$(meta instance/zone)")
PROJECT=$(meta project/project-id)
export DREAM_CODE_SHA=$(meta instance/attributes/dream-code-sha)
CAP=$(meta instance/attributes/dream-max-minutes)
WORKLOAD=$(meta instance/attributes/dream-workload)
mkdir -p /work/out
# Cached runtime images carry the prior build's filesystem. Its evidence is first
# retrieved from that run's private bucket, so each new VM starts with empty outputs.
find /work/out -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
exec > >(tee -a /work/out/vm.log) 2>&1
sync_out() { gsutil -q -m rsync -r /work/out "gs://$BUCKET/out" || true; }
finish() {
  code=$?
  trap - EXIT
  echo "$code" > /work/out/EXIT_CODE
  date -u +'%Y-%m-%dT%H:%M:%SZ' > /work/out/finished_utc.txt
  (cd /work/out && find . -type f ! -name MANIFEST.sha256 ! -name vm.log ! -name gpu.csv \
    -print0 | sort -z | xargs -0 -r sha256sum) > /work/out/MANIFEST.sha256
  sync_out
  # Only this VM has granted this account instance-level deletion permission.
  gcloud compute instances delete "$NAME" --project="$PROJECT" --zone="$ZONE" --async --quiet || shutdown -h now
  exit "$code"
}
trap finish EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' > /work/out/started_utc.txt
( while true; do sleep 60; sync_out; done ) &
for i in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /work/out/gpu.txt
( while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader \
      >> /work/out/gpu.csv 2>> /work/out/gpu_errors.log || \
      date -u +'%Y-%m-%dT%H:%M:%SZ GPU sample unavailable' >> /work/out/gpu_errors.log
    sleep 15
  done ) &
gsutil -q cp "gs://$BUCKET/code.tar.gz" /work/code.tar.gz
EXPECTED=$(meta instance/attributes/dream-archive-sha256)
echo "$EXPECTED  /work/code.tar.gz" | sha256sum -c -
mkdir -p /work/DREAM
tar -xzf /work/code.tar.gz -C /work/DREAM
export HF_HOME=/work/model-cache
export UV_CACHE_DIR=/work/uv-cache
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
if [[ "$WORKLOAD" == isaac-g1-session ]]; then
  # Isaac gets its own locked simulator environment; VLM inference reuses the
  # separately validated runtime image and must never resolve dependencies here.
  test -x /opt/dream-runtime/bin/python
  export DREAM_QWEN_PYTHON=/opt/dream-runtime/bin/python
  /opt/dream-runtime/bin/python /work/DREAM/cloud/cached_runtime.py \
    --lock /work/DREAM/cloud/requirements.lock.txt \
    --out /work/out/runtime_preflight.json --freeze /work/out/qwen-freeze.txt --cuda
  source /work/DREAM/cloud/isaac_g1_session.sh
  exit 0
fi
if [[ "$WORKLOAD" == isaac-t1-* ]]; then
  source /work/DREAM/cloud/isaac_t1_job.sh
  exit 0
fi
if [[ "$WORKLOAD" == g1-qwen || "$WORKLOAD" == g1-autonomy ]]; then
  # Validated on a prebuilt image before allocation; no apt/pip/uv download fallback.
  source /opt/dream-runtime/bin/activate
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  python /work/DREAM/cloud/cached_runtime.py --lock /work/DREAM/cloud/requirements.lock.txt \
    --out /work/out/runtime_preflight.json --freeze /work/out/pip-freeze.txt --cuda
else
apt-get update -q
apt-get install -y -q libosmesa6 python3-pip ffmpeg
if [[ "$WORKLOAD" == g1-autonomy || "$WORKLOAD" == g1-runtime-build ]]; then
  apt-get install -y -q libegl1
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
fi
export DREAM_RECORDINGS_ROOT=/work/out/recordings
python3 -m pip install uv==0.9.4
if [[ "$WORKLOAD" == g1-runtime-build ]]; then
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  uv venv --python 3.12.12 /opt/dream-runtime
  source /opt/dream-runtime/bin/activate
else
  uv venv --python 3.12.12 /work/venv
  source /work/venv/bin/activate
fi
LOCK=requirements.lock.txt
case "$WORKLOAD" in molmo-openloop|molmo-closedloop) LOCK=molmo.requirements.lock.txt ;; esac
if [[ "$WORKLOAD" == groot-openloop ]]; then
  LOCK=groot.requirements.lock.txt
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH
  mkdir -p /work/model-input
  gsutil -q -m rsync -r "gs://$BUCKET/model-input" /work/model-input
fi
uv pip install --require-hashes -r "/work/DREAM/cloud/$LOCK"
uv pip install --no-deps -e /work/DREAM
uv pip freeze > /work/out/pip-freeze.txt
fi
cd /work/DREAM
UP=$(cut -d. -f1 /proc/uptime)
LEFT=$(( CAP * 60 - 120 - UP ))
[ "$LEFT" -gt 0 ] || exit 124
case "$WORKLOAD" in
  groot-openloop)
    timeout --signal=TERM --kill-after=30s "${LEFT}s" \
      python -m experiments.cloud_groot_openloop --out /work/out/groot ;;
  g1-qwen)
    timeout --signal=TERM --kill-after=30s "${LEFT}s" \
      python -m experiments.cloud_g1_qwen --out /work/out/g1 ;;
  g1-runtime-build)
    timeout --signal=TERM --kill-after=30s "${LEFT}s" \
      python -m experiments.cloud_g1_qwen --config experiments/configs/g1_runtime_build_demo.json --out /work/out/g1
    timeout --signal=TERM --kill-after=30s "${LEFT}s" \
      python -m cloud.build_g1_runtime_image --source-disk "$NAME" --image-name "$(meta instance/attributes/dream-runtime-image-name)" --out /work/out/runtime_image.json ;;
  g1-autonomy)
    timeout --signal=TERM --kill-after=30s "${LEFT}s" \
      python -m experiments.cloud_g1_qwen --config experiments/configs/g1_autonomy_frozen.json --out /work/out/g1 ;;
  vlm-compare)
    timeout --signal=TERM --kill-after=30s "${LEFT}s" \
      python experiments/vlm_compare.py qwen-local --device cuda \
      --model-revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --out /work/out/vlm ;;
  qwen-pilot)
    timeout --signal=TERM --kill-after=30s "${LEFT}s" \
      python experiments/cloud_qwen_pilot.py --out /work/out/pilot ;;
  molmo-openloop)
    timeout --signal=TERM --kill-after=30s "${LEFT}s" \
      python experiments/cloud_molmo_openloop.py --out /work/out ;;
  molmo-closedloop)
    timeout --signal=TERM --kill-after=30s "${LEFT}s" \
      python experiments/cloud_molmo_closedloop.py --out /work/out ;;
  *) echo "Unsupported workload: $WORKLOAD"; exit 2 ;;
esac
