#!/usr/bin/env bash
# 폐쇄망 반입 번들 (인터넷 되는 PC에서). ./pack.sh [cpu|cu124]  → dist-offline/portrait-local-linux-x64-<variant>.tar.gz
# 포함: 소스, vendor/LivePortrait, weights/(630MB), wheels/(linux-x64 py3.11, torch 포함 → cpu ~1GB / cu124 ~3GB)
set -euo pipefail; cd "$(dirname "$0")"
V="${1:-cpu}"; STAGE="dist-offline/portrait-local-linux-x64-$V"; rm -rf "$STAGE"; mkdir -p "$STAGE/wheels"
[ -f vendor/LivePortrait/inference.py ] && [ -f weights/liveportrait/base_models/spade_generator.pth ] || { echo "먼저 bash setup.sh 로 vendor/weights 를 받으세요"; exit 1; }
IDX=(); [ "$V" = cpu ] && IDX=(--extra-index-url https://download.pytorch.org/whl/cpu) || IDX=(--extra-index-url "https://download.pytorch.org/whl/$V")
venv/bin/pip download -q --dest "$STAGE/wheels" --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --python-version 3.11 --only-binary=:all: "${IDX[@]}" -r requirements.txt torch torchvision onnxruntime
cp -r app.py gpu_pick.py ui.html selftest.py setup.sh requirements.txt README.md NOTICE LICENSE templates "$STAGE/"
rsync -a --exclude .git vendor "$STAGE/"; rsync -a --exclude .cache weights "$STAGE/"
cat > "$STAGE/INSTALL.md" <<'INS'
# 폐쇄망 설치
tar -xzf portrait-local-linux-x64-*.tar.gz && cd portrait-local-linux-x64-*
bash setup.sh            # wheels/·vendor/·weights/ 가 있으면 다운로드 없이 설치·기동
# GPU: cu124 번들을 쓰면 DEVICE=auto 가 cuda 를 잡음. CPU 번들은 수 분/영상.
# 음성: TTS_BASE_URL=http://<tts-host>:8771/v1 bash setup.sh  (kokoro 등 OpenAI 호환 /v1/audio/speech)
INS
( cd dist-offline && tar -czf "portrait-local-linux-x64-$V.tar.gz" "portrait-local-linux-x64-$V" ) && ls -lh "dist-offline/portrait-local-linux-x64-$V.tar.gz"
