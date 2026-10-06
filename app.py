#!/usr/bin/env python3
"""portrait-local — 사진 한 장을 말하고 표정 짓게 (LivePortrait 래퍼). 서버는 stdlib, 추론은 venv 서브프로세스.

  bash setup.sh                     # venv + LivePortrait + 가중치 + 서버  http://localhost:8770
  python3 app.py --cli photo.jpg --template 말하기 [--text "안녕하세요"] -o out.mp4

env: PORT(8770) DEVICE(auto|cuda|mps|cpu) MAX_DIM(512, 작을수록 빠름) TTS_BASE_URL(OpenAI 호환 /v1, 선택) TTS_VOICE TTS_MODEL
"""
import base64, datetime, json, os, pickle, re, secrets, shutil, subprocess, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gpu_pick import env_for, label, pick

ROOT = os.path.dirname(os.path.abspath(__file__))
WS, TPL = os.environ.get("WORKSPACE") or os.path.join(ROOT, "_workspace"), os.path.join(ROOT, "templates")  # 포털이 AGENT_DATA/<도구> 로 모아 줌
LP = os.path.join(ROOT, "vendor", "LivePortrait")
PY = os.path.join(ROOT, "venv", "bin", "python") if os.path.exists(os.path.join(ROOT, "venv", "bin", "python")) else os.path.join(ROOT, "venv", "Scripts", "python.exe")
PORT = int(os.environ.get("PORT", "8770"))
DEVICE = os.environ.get("DEVICE", "auto")
MAX_DIM = os.environ.get("MAX_DIM", "512")
TTS = os.environ.get("TTS_BASE_URL", "").rstrip("/")
TTS_VOICE, TTS_MODEL = os.environ.get("TTS_VOICE", "KR"), os.environ.get("TTS_MODEL", "melo")
# LivePortrait 저장소 동봉 모션 템플릿(.pkl, 영상 없이 움직임만 → 개인정보 없음) → 한국어 라벨
LABELS = {"talking": "말하기", "laugh": "웃음", "wink": "윙크", "shy": "수줍음", "shake_face": "고개 젓기", "open_lip": "입 벌리기", "aggrieved": "억울함"}


def ensure_templates():
    """구형 템플릿은 c_d_eyes_lst/c_d_lip_lst 키가 없어 현재 파이프라인에서 KeyError → 채워서 templates/ 에 복사."""
    os.makedirs(TPL, exist_ok=True)
    src = os.path.join(LP, "assets", "examples", "driving")
    for name in LABELS:
        dst, s = os.path.join(TPL, name + ".pkl"), os.path.join(src, name + ".pkl")
        if os.path.exists(dst) or not os.path.exists(s):
            continue
        d = pickle.load(open(s, "rb"))
        for k in ("c_d_eyes_lst", "c_d_lip_lst"):
            d.setdefault(k, [None] * d["n_frames"])
        pickle.dump(d, open(dst, "wb"))


def templates():
    return [{"name": n, "label": l} for n, l in LABELS.items() if os.path.exists(os.path.join(TPL, n + ".pkl"))]


_DEV = None


def device():
    """torch 서브프로세스 탐지는 1회만 (매 호출 수 초 걸림)."""
    global _DEV
    if DEVICE != "auto":
        return DEVICE
    if _DEV:
        return _DEV
    try:
        out = subprocess.run([PY, "-c", "import torch;print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')"],
                             capture_output=True, text=True, timeout=120).stdout.strip()
        _DEV = out or "cpu"
    except Exception:
        _DEV = "cpu"
    return _DEV


def run_inference(photo, driving, out_dir, emit):
    """LivePortrait inference.py 서브프로세스. 결과 mp4 경로 반환."""
    dev = device()
    cmd = [PY, "inference.py", "-s", photo, "-d", driving, "-o", out_dir, "--source-max-dim", MAX_DIM, "--no-flag-use-half-precision"]
    if dev == "cpu":
        cmd.append("--flag-force-cpu")
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    where = dev
    if dev == "cuda":  # GPU 고정 없음 — 실행할 때마다 여유 메모리가 가장 큰 GPU 1장만 보이게 해서 띄운다
        g = pick(4000)
        env, where = env_for(g, env), label(g)
        if not g:
            cmd.append("--flag-force-cpu")
    emit({"stage": "infer", "msg": f"LivePortrait ({where}, max_dim {MAX_DIM})"})
    p = subprocess.Popen(cmd, cwd=LP, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        line = line.strip()
        if line and "warn" not in line.lower():
            tail.append(line); emit({"log": line[:200]})
    p.wait()
    base = os.path.splitext(os.path.basename(photo))[0] + "--" + os.path.splitext(os.path.basename(driving))[0]
    mp4 = os.path.join(out_dir, base + ".mp4")
    if p.returncode != 0 or not os.path.exists(mp4):
        raise RuntimeError("추론 실패: " + " | ".join(tail[-5:]))
    return mp4


def tts(text, wav):
    req = urllib.request.Request(TTS + "/audio/speech", json.dumps({"model": TTS_MODEL, "input": text, "voice": TTS_VOICE, "response_format": "wav"}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r, open(wav, "wb") as f:
        f.write(r.read())


def generate(photo, template, text="", driving_file=None, emit=lambda ev: None):
    run_id = f"{datetime.date.today()}-{secrets.token_hex(2)}"
    d = os.path.join(WS, run_id); os.makedirs(d)
    src = os.path.join(d, "source" + os.path.splitext(photo)[1].lower()); shutil.copy(photo, src)
    if driving_file:
        drv = os.path.join(d, "driving" + os.path.splitext(driving_file)[1].lower()); shutil.copy(driving_file, drv)
    else:
        drv = os.path.join(TPL, template + ".pkl")
        if not os.path.exists(drv):
            raise ValueError(f"템플릿 없음: {template} (가능: {', '.join(LABELS.values())})")
    mp4 = run_inference(src, drv, d, emit)
    final, audio = os.path.join(d, "final.mp4"), False
    tts_error = None
    if text.strip() and TTS:
        emit({"stage": "tts", "msg": "음성 합성 + 합치기 (입모양은 대사와 동기화되지 않음)"})
        try:
            wav = os.path.join(d, "speech.wav"); tts(text, wav)
            # ponytail: 영상을 반복 재생해 음성 길이에 맞춤(-shortest). 립싱크는 범위 밖 — 필요하면 SadTalker 류로 승급
            dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav], capture_output=True, text=True).stdout.strip()
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-stream_loop", "-1", "-i", mp4, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", dur, final], check=True)
            audio = True
        except Exception as e:
            emit({"log": f"TTS 건너뜀: {e}"}); tts_error = f"{type(e).__name__}: {e}"[:300]
    if not audio:
        shutil.copy(mp4, final)
    result = {"run_id": run_id, "template": template if not driving_file else "(업로드 영상)", "label": LABELS.get(template, template), "text": text,
              "audio": audio, "tts_error": tts_error, "device": device(), "ts": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(result, open(os.path.join(d, "result.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return result


def list_runs():
    out = []
    if os.path.isdir(WS):
        for n in sorted(os.listdir(WS), key=lambda n: os.path.getmtime(os.path.join(WS, n)), reverse=True)[:50]:
            p = os.path.join(WS, n, "result.json")
            if os.path.exists(p):
                try: out.append(json.load(open(p, encoding="utf-8")))
                except Exception: pass
    return out


HTML = open(os.path.join(ROOT, "ui.html"), encoding="utf-8").read() if os.path.exists(os.path.join(ROOT, "ui.html")) else "ui.html 없음"
RUN_RE = r"\d{4}-\d{2}-\d{2}-[0-9a-f]{4}"

# ── 저작권 표기 (LICENSE·NOTICE 참고) ─────────────────────────────────────
_SIG = __import__("base64").b64decode("wqkgMjAyNiBnZ2dnODY1NyDCtyBkb25nanVraW0uZGV2QGdtYWlsLmNvbQ==").decode()
_SIG_A = __import__("base64").b64decode("Z2dnZzg2NTcgPGRvbmdqdWtpbS5kZXZAZ21haWwuY29tPg==").decode()


def signed(html):
    """화면에 저작권 표기를 붙인다. ui.html 에서 지워져도 서버가 내보낼 때 다시 붙는다."""
    name, mail = _SIG.split(" · ")
    if 'name="author"' not in html:
        meta = f'<meta name="author" content="{name[7:]} <{mail}>">'
        html = html.replace("<head>", "<head>" + meta, 1) if "<head>" in html else meta + html
    if "data-sig" not in html:
        tag = (f'<!-- {_SIG} --><div data-sig title="{mail}" style="text-align:center;font-size:11px;color:#9aa0a6;'
               f'opacity:.55;margin:28px 0 8px">{name}</div>')
        html = html.replace("</body>", tag + "</body>", 1) if "</body>" in html else html + tag
    return html


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        if "/api/run" in (str(a[0]) if a else ""): super().log_message(fmt, *a)

    def _send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("X-Author", _SIG_A); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        try:
            if self.path == "/api/templates": return self._send(templates())
            if self.path == "/api/runs": return self._send(list_runs())
            m = re.fullmatch(rf"/api/runs/({RUN_RE})/final\.mp4", self.path)
            if m:
                with open(os.path.join(WS, m.group(1), "final.mp4"), "rb") as f: return self._send(f.read(), "video/mp4")
            self._send(signed(HTML.replace("%TTS%", json.dumps(bool(TTS)))).encode(), "text/html; charset=utf-8")
        except FileNotFoundError: self._send({"error": "없음"}, code=404)
        except Exception as e: self._send({"error": f"{type(e).__name__}: {e}"}, code=500)

    def do_POST(self):
        if self.path != "/api/run":
            return self._send({"error": "not found"}, code=404)
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        up = os.path.join(WS, "_upload"); os.makedirs(up, exist_ok=True)
        def save(key, name_key, allowed):
            if not req.get(key): return None
            name = re.sub(r"[^\w.\-]", "_", os.path.basename(req.get(name_key) or "file"))
            if not name.lower().endswith(allowed): raise ValueError(f"허용 확장자: {allowed}")
            p = os.path.join(up, secrets.token_hex(3) + "_" + name)
            with open(p, "wb") as f: f.write(base64.b64decode(req[key]))
            return p
        self.send_response(200); self.send_header("Content-Type", "text/event-stream; charset=utf-8"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
        def emit(ev):
            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()); self.wfile.flush()
        photo = drv = None
        try:
            photo = save("photo_b64", "photo_name", (".jpg", ".jpeg", ".png"))
            if not photo: raise ValueError("사진이 필요합니다")
            drv = save("driving_b64", "driving_name", (".mp4", ".mov", ".pkl"))
            emit({"done": generate(photo, req.get("template") or "talking", req.get("text") or "", drv, emit)})
        except Exception as e:
            emit({"error": f"{type(e).__name__}: {e}"})
        finally:
            for p in (photo, drv):
                if p and os.path.exists(p): os.remove(p)


if __name__ == "__main__":
    ensure_templates()
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        a = sys.argv[2:]; photo = a[0]
        g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
        label = g("--template", "말하기"); name = {v: k for k, v in LABELS.items()}.get(label, label)
        r = generate(photo, name, g("--text", ""), emit=lambda ev: print(ev.get("msg") or ev.get("log", ""), file=sys.stderr))
        out = g("-o", f"{r['run_id']}.mp4"); shutil.copy(os.path.join(WS, r["run_id"], "final.mp4"), out)
        print(json.dumps({**r, "output": out}, ensure_ascii=False)); sys.exit(0)
    print(f"portrait-local → http://localhost:{PORT}  (device={device()}, max_dim={MAX_DIM}, tts={TTS or '없음'}, templates={len(templates())})  {_SIG}")
    ThreadingHTTPServer(("", PORT), H).serve_forever()
