#!/usr/bin/env bash
# portrait-local — 원샷 설치·실행 (macOS / Linux). bash setup.sh | bash setup.sh stop
# env: PORT(8770) DEVICE(auto) TTS_BASE_URL(선택, 예 http://localhost:8771/v1)
set -euo pipefail
cd "$(dirname "$0")"
REPO=https://github.com/gggg8657/portrait-local.git
LP_COMMIT=9b294b3d0536135442ea73cb01e6cb3ca7029dd3
PORT="${PORT:-8770}"
ok(){ printf '  ✔ %s\n' "$*"; }; die(){ printf '  ✘ %s\n' "$*" >&2; exit 1; }
if [ "${1:-}" = stop ]; then [ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null && rm -f .server.pid && ok "서버 종료" || echo "  실행 중 아님"; exit 0; fi

case "$(uname -s)" in Darwin*) OS=mac;; Linux*) OS=linux;; *) die "지원 OS: macOS/Linux";; esac; ok "OS $OS"
command -v ffmpeg >/dev/null || die "ffmpeg 필요 (mac: brew install ffmpeg / linux: apt install ffmpeg)"
PY=""; for c in python3.11 python3.12 python3.10 python3; do command -v $c >/dev/null && $c -c 'import sys;sys.exit(0 if (3,10)<=sys.version_info<(3,13) else 1)' 2>/dev/null && { PY=$c; break; }; done
[ -n "$PY" ] || die "Python 3.10~3.12 필요"; ok "$($PY --version) ($PY)"

# venv + deps (폐쇄망: wheels/ 가 있으면 거기서만 설치)
if [ ! -x venv/bin/python ]; then
  if command -v uv >/dev/null; then uv venv --python "$(command -v $PY)" venv -q; else $PY -m venv venv; fi
fi
PIP_ARGS=(); [ -d wheels ] && PIP_ARGS=(--no-index --find-links wheels)
venv/bin/python -c "import torch, cv2, tyro" 2>/dev/null || {
  echo "  · 의존성 설치 (torch 포함, 수 분)"
  if command -v uv >/dev/null; then VIRTUAL_ENV=$PWD/venv uv pip install -q "${PIP_ARGS[@]}" -r requirements.txt torch torchvision onnxruntime
  else venv/bin/pip install -q "${PIP_ARGS[@]}" -r requirements.txt torch torchvision onnxruntime; fi
}; ok "venv 준비"

# LivePortrait 소스 + 가중치 (폐쇄망 번들에는 둘 다 들어 있음)
if [ ! -f vendor/LivePortrait/inference.py ]; then
  command -v git >/dev/null || die "git 필요 (또는 pack.sh 번들 사용)"
  git clone -q https://github.com/KwaiVGI/LivePortrait.git vendor/LivePortrait && git -C vendor/LivePortrait checkout -q $LP_COMMIT
fi; ok "LivePortrait $(git -C vendor/LivePortrait rev-parse --short HEAD 2>/dev/null || echo 번들)"
if [ ! -f weights/liveportrait/base_models/spade_generator.pth ]; then
  echo "  · 가중치 다운로드 (~630MB, HF)"
  venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('KwaiVGI/LivePortrait', local_dir='weights', allow_patterns=['liveportrait/*','insightface/*'])" || die "가중치 다운로드 실패 — 폐쇄망이면 pack.sh 번들 사용"
fi
rm -rf vendor/LivePortrait/pretrained_weights && ln -s ../../weights vendor/LivePortrait/pretrained_weights; ok "가중치 $(du -sh weights | cut -f1)"

python3 selftest.py >/dev/null && ok "selftest 통과" || die "selftest 실패"
[ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null || true
PORT=$PORT nohup python3 app.py > server.log 2>&1 & echo $! > .server.pid
for _ in $(seq 1 60); do curl -fsS "http://localhost:$PORT/api/templates" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "http://localhost:$PORT/api/templates" >/dev/null || { cat server.log; die "서버 기동 실패"; }
ok "http://localhost:$PORT  (종료: bash setup.sh stop, 로그: server.log)"
case "$OS" in mac) open "http://localhost:$PORT";; linux) command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || true;; esac
