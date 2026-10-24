#!/bin/bash
# NERVA-style private, capped job lifecycle, isolated to this DREAM VM and bucket.
set -euo pipefail
[ -f /var/dream_started ] && exit 0
touch /var/dream_started
MD=http://metadata.google.internal/computeMetadata/v1
meta() { curl -fsS -H 'Metadata-Flavor: Google' "$MD/$1"; }
BUCKET=$(meta instance/attributes/dream-bucket)
NAME=$(meta instance/name)
ZONE=$(basename "$(meta instance/zone)")
PROJECT=$(meta project/project-id)
export DREAM_CODE_SHA=$(meta instance/attributes/dream-code-sha)
CAP=$(meta instance/attributes/dream-max-minutes)
WORKLOAD=$(meta instance/attributes/dream-workload)
mkdir -p /work/out
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
if [[ "$WORKLOAD" == isaac-t1-* ]]; then
  source /work/DREAM/cloud/isaac_t1_job.sh
  exit 0
fi
apt-get update -q
apt-get install -y -q libosmesa6 python3-pip ffmpeg
export DREAM_RECORDINGS_ROOT=/work/out/recordings
python3 -m pip install uv==0.9.4
uv venv --python 3.12.12 /work/venv
source /work/venv/bin/activate
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
