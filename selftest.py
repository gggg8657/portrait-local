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


# GPU 후보: 부모 후보가 0,1,2 면 3(운영 vLLM 전용)을 절대 고르지 않음 — nvidia-smi 출력은 가짜, 예약 장부는 임시 폴더
import gpu_pick as _gp, os as _os, tempfile as _tf, shutil as _sh
_smi0, _dir0, _env0 = _gp._smi, _gp.RES_DIR, dict(_os.environ)
try:
    _gp._smi = lambda: "0, 1000, 90\n1, 2000, 90\n2, 3000, 90\n3, 90000, 0\n"  # 3 이 가장 한가해 보이게
    _gp.RES_DIR = _tf.mkdtemp()
    for _k in ("GPU_POOL", "GPU_RESERVED", "VLLM_GPUS"): _os.environ.pop(_k, None)
    for _res in ("none", "3"):
        _os.environ.update(CUDA_VISIBLE_DEVICES="0,1,2", GPU_RESERVED=_res)
        assert [g["index"] for g in _gp.gpus()] == ["0", "1", "2"]
        assert _gp.pick(0, hold_s=0)["index"] == "2" and _gp.torch_device(_gp.pick(0, hold_s=0)) == "cuda:2"
        assert _gp.pool_env(_os.environ)["CUDA_VISIBLE_DEVICES"] == "0,1,2" == _gp.env_for(None, _os.environ)["CUDA_VISIBLE_DEVICES"]
    # 셸(~/.zshrc)에서 2,3 을 물려받아도 예약된 3 은 빼고 2 만 — 보이는 번호는 cuda:0
    _os.environ.update(CUDA_VISIBLE_DEVICES="2,3", GPU_RESERVED="3")
    assert _gp.pick(0, hold_s=0)["index"] == "2" and _gp.torch_device(_gp.pick(0, hold_s=0)) == "cuda:0"
    assert _gp.pool_env(_os.environ)["CUDA_VISIBLE_DEVICES"] == "2"
    _os.environ.pop("GPU_RESERVED"); _os.environ["VLLM_GPUS"] = "3"  # GPU_RESERVED 없으면 VLLM_GPUS
    assert _gp.pick(0, hold_s=0)["index"] == "2"
    _os.environ["GPU_POOL"] = "0,1"  # 교집합이 비면 GPU 를 하나도 안 보임(CPU)
    assert _gp.pick(0, hold_s=0) is None and _gp.pool_env(_os.environ)["CUDA_VISIBLE_DEVICES"] == ""
    if hasattr(app, "worker_env"):  # 작업자(그림·영상) 프로세스 환경
        assert app.worker_env({"CUDA_VISIBLE_DEVICES": "0,1,2", "GPU_RESERVED": "none"}, ["0", "1", "2", "3"])["CUDA_VISIBLE_DEVICES"] == "0,1,2"
        assert app.worker_env({"CUDA_VISIBLE_DEVICES": "2,3", "GPU_RESERVED": "3"}, ["0", "1", "2", "3"])["CUDA_VISIBLE_DEVICES"] == "2"
        assert app.worker_env({"GPU_POOL": "0,1,2,3", "VLLM_GPUS": "3"}, ["0", "1", "2", "3"])["CUDA_VISIBLE_DEVICES"] == "0,1,2"
finally:
    _sh.rmtree(_gp.RES_DIR, ignore_errors=True); _gp._smi, _gp.RES_DIR = _smi0, _dir0; _os.environ.clear(); _os.environ.update(_env0)
print("selftest GPU 후보 OK — 0,1,2 에서 3 안 고름")
