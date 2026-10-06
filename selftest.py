#!/usr/bin/env python3
"""가중치·venv 없이 파이프라인만 검증: 템플릿 변환 → (가짜 추론: ffmpeg 색상 영상) → TTS 합치기 분기 → 목록.  python3 selftest.py"""
import os, pickle, shutil, subprocess, tempfile, app

# 1) 구형 템플릿 키 보강
tmp = tempfile.mkdtemp(); app.TPL = os.path.join(tmp, "templates"); app.WS = os.path.join(tmp, "ws")
src = os.path.join(app.LP, "assets", "examples", "driving")
if os.path.isdir(src):
    app.ensure_templates(); d = pickle.load(open(os.path.join(app.TPL, "wink.pkl"), "rb"))
    assert len(d["c_d_eyes_lst"]) == d["n_frames"] and app.templates()
else:  # vendor 미설치 환경: 더미 템플릿
    os.makedirs(app.TPL); pickle.dump({"n_frames": 3, "output_fps": 25, "motion": []}, open(os.path.join(app.TPL, "talking.pkl"), "wb"))

# 2) 가짜 추론: 1초짜리 색상 영상
def fake(photo, driving, out_dir, emit):
    mp4 = os.path.join(out_dir, os.path.splitext(os.path.basename(photo))[0] + "--" + os.path.splitext(os.path.basename(driving))[0] + ".mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=1", "-pix_fmt", "yuv420p", mp4], check=True)
    emit({"log": "fake infer"}); return mp4
app.run_inference = fake; app.device = lambda: "fake"
photo = os.path.join(tmp, "face.jpg"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=white:s=64x64:d=1", "-frames:v", "1", photo], check=True)

r = app.generate(photo, "talking" if not os.path.isdir(src) else "wink", "", emit=lambda ev: None)
assert os.path.exists(os.path.join(app.WS, r["run_id"], "final.mp4")) and r["audio"] is False
# 3) TTS 켜짐 + 가짜 서버 → 합치기, TTS 실패 → 무음 폴백
app.TTS = "http://fake"; app.tts = lambda text, wav: subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=2", wav], check=True)
r = app.generate(photo, "talking" if not os.path.isdir(src) else "wink", "안녕하세요", emit=lambda ev: None); assert r["audio"] is True
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", os.path.join(app.WS, r["run_id"], "final.mp4")], capture_output=True, text=True).stdout)
assert 1.8 < dur < 2.3, dur  # 영상 1초를 반복해 음성 2초에 맞춤
app.tts = lambda text, wav: (_ for _ in ()).throw(RuntimeError("down"))
r = app.generate(photo, "talking" if not os.path.isdir(src) else "wink", "안녕", emit=lambda ev: None); assert r["audio"] is False
try: app.generate(photo, "없는템플릿", "", emit=lambda ev: None); assert False
except ValueError: pass
assert len(app.list_runs()) == 3
shutil.rmtree(tmp); print("selftest OK")
# 저작권 표기: 서버가 화면에 붙이는 코드가 있어야 한다 (LICENSE·NOTICE)
_src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "app.py"), encoding="utf-8").read()
assert "wqkgMjAyNiBnZ2dnODY1NyDCtyBkb25nanVraW0uZGV2QGdtYWlsLmNvbQ==" in _src and "signed(" in _src and "X-Author" in _src, "저작권 표기 누락"

