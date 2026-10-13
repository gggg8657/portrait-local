# 제3자 구성요소 (Third-Party Notices) — portrait-local

이 저장소의 원저작물(app.py, ui.html, selftest.py, setup.sh, pack.sh, gpu_pick.py, 문서)은 DongJu Kim(gggg8657)이 작성했다(`NOTICE`, `LICENSE`). 아래 제3자 구성요소는 각자의 라이선스를 따르며, 이 저장소의 라이선스(PolyForm Noncommercial 1.0.0)로 바뀌지 않는다.
조사 기준: 2026-10-12, 저장소의 실제 파일(`requirements.txt`, `setup.sh`, `pack.sh`, `app.py`, `templates/`)과 설치된 venv 의 패키지 메타데이터, HF 모델 카드, GitHub LICENSE.

> ⚠️ **얼굴 검출 모델(InsightFace buffalo_l)은 비상업 연구용**이다. 현재 구성으로는 상업 이용·외부 배포를 하지 말 것 — [#1](https://github.com/gggg8657/portrait-local/issues/1).

## 모델·소스·데이터

| 이름 | 버전 | 쓰임 | 라이선스 | 링크 | 상업 이용 | 비고 |
|---|---|---|---|---|---|---|
| LivePortrait (소스) | commit 9b294b3 | 설치(setup 때 `vendor/` 로 clone, 수정 없이 사용) | MIT (Kuaishou) | https://github.com/KwaiVGI/LivePortrait | 가능(단 아래 InsightFace 조건) | `LICENSE-LivePortrait` 사본. LICENSE 에 "상업 목적이면 InsightFace 검출 모델을 제거·교체" 명시 |
| LivePortrait 가중치 (`weights/liveportrait/`) | — | 내려받는 가중치(HF) | MIT (HF 모델 카드) | https://huggingface.co/KwaiVGI/LivePortrait | 가능 | git 에 넣지 않음. `pack.sh` 번들에는 포함 |
| **InsightFace buffalo_l** (`det_10g.onnx`, `2d106det.onnx`) | — | 내려받는 가중치(HF KwaiVGI/LivePortrait 의 `insightface/*`) | **비상업 연구용** (InsightFace 모델 정책) | https://github.com/deepinsight/insightface#license | **비상업** | git 에 넣지 않음. **`pack.sh` 번들에는 포함 → 번들 배포 금지**. #1 |
| InsightFace 코드 (LivePortrait 안 `src/utils/dependencies/insightface`) | — | 설치(vendor) | MIT | https://github.com/deepinsight/insightface | 가능 | |
| `templates/*.pkl` (7개) | — | 동봉(모션 데이터) | MIT (LivePortrait 저장소 `assets/examples/driving/*.pkl` 변환) | https://github.com/KwaiVGI/LivePortrait/tree/main/assets/examples/driving | 조건부 | 영상·얼굴 이미지 없이 움직임 값만. 원본 driving 영상 속 인물의 권리는 LivePortrait 쪽 사항(미확인) |
| TTS 서버 (kokoro-local / tts-local 등) | — | 외부 서비스(선택, `TTS_BASE_URL`) | 해당 서버·모델 라이선스 | — | 해당 서버 따름 | |
| FFmpeg (시스템 `ffmpeg` CLI) | — | 외부 프로그램(사용자 설치) | LGPL-2.1+ / 빌드에 따라 GPL (libx264 사용 시 GPL) | https://ffmpeg.org/legal.html | 조건부 | 별도 프로세스로 호출(코드 결합 없음). 번들에 넣지 않음 |

## Python 패키지 (`requirements.txt` + setup 이 추가 설치)

| 이름 | 버전 | 라이선스 | 상업 이용 | 비고 |
|---|---|---|---|---|
| torch | 미고정 (확인 시 2.14.1) | BSD-3-Clause 계열 | 가능 | |
| torchvision | 미고정 | BSD-3-Clause | 가능 | |
| onnxruntime | 미고정 | MIT | 가능 | |
| numpy | 1.26.4 | BSD-3-Clause | 가능 | |
| pyyaml | 6.0.1 | MIT | 가능 | |
| opencv-python | 4.10.0.84 | Apache-2.0 (휠에 FFmpeg LGPL 포함) | 가능 | |
| scipy | 1.13.1 | BSD-3-Clause | 가능 | |
| imageio | 2.34.2 | BSD-2-Clause | 가능 | |
| lmdb | 1.4.1 | OLDAP-2.8 | 가능 | |
| tqdm | 4.66.4 | MPL-2.0 AND MIT | 가능 | |
| rich | 13.7.1 | MIT | 가능 | |
| ffmpeg-python | 0.2.0 | Apache-2.0 | 가능 | |
| onnx | 1.16.1 | Apache-2.0 | 가능 | |
| scikit-image | 0.24.0 | BSD-3-Clause | 가능 | |
| albumentations | 1.4.10 | MIT | 가능 | |
| matplotlib | 3.9.0 | PSF 계열 (matplotlib License) | 가능 | |
| imageio-ffmpeg | 0.5.1 | BSD-2-Clause (휠에 FFmpeg 실행 파일 포함, GPL 빌드일 수 있음) | 조건부 | `pack.sh` wheels 번들 배포 시 FFmpeg 고지 확인 |
| tyro | 0.8.5 | MIT | 가능 | |
| pykalman | 0.9.7 | BSD-3-Clause | 가능 | |
| pillow | >=10.2.0 | MIT-CMU (HPND) | 가능 | |
| huggingface_hub | 미고정 | Apache-2.0 | 가능 | |
| requests | 미고정 | Apache-2.0 | 가능 | |

`ui.html` 은 외부 라이브러리·웹폰트를 쓰지 않는다.

## 확인 필요

- **[#1](https://github.com/gggg8657/portrait-local/issues/1) InsightFace buffalo_l 얼굴 모델 — 비상업 연구용**: LivePortrait 의 얼굴 검출·랜드마크에 쓴다. InsightFace 는 사전학습 모델을 비상업 연구 목적으로만 제공하고, LivePortrait LICENSE 도 상업 이용 시 교체를 요구한다. 이 이유로 agent-page-portal 목록에서 내렸다.
  회피(제안): MediaPipe Face Detector/Landmarker(Apache-2.0) 또는 OpenCV Zoo YuNet 으로 `vendor/LivePortrait/src/utils/cropper.py` 의 검출기 교체(각 모델 조건 확인 필요). 그때까지 `pack.sh` 번들을 외부로 주지 말 것.
- **딥페이크·초상권**: 사진 속 인물을 움직이게 하는 도구다. 본인 동의 없는 타인 사진, 실존 인물 사칭·허위 영상에 쓰지 말 것. 결과물이 AI 생성임을 밝힐 것.
