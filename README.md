# portrait-local — 사진 한 장을 말하고 표정 짓게 (LivePortrait, 전부 로컬)

> **한 줄 요약** — 오픈소스 [LivePortrait](https://github.com/KwaiVGI/LivePortrait)(MIT, Kuaishou)를 폐쇄망에서 쓰도록 묶은 패키지입니다.
> 인물 사진 한 장을 올리고 표정(말하기·웃음·윙크·고개 젓기 등)을 고르면 그 사람이 움직이는 mp4가 나옵니다.
> 대사를 넣으면 로컬 TTS(kokoro 등 OpenAI 호환 서버)의 음성을 붙입니다(입모양 동기화는 아님).
> 외부 통신 없음, 모델 630MB는 번들에 동봉. GPU 있으면 수 초, CPU/맥은 수 분.
> **라이선스 주의**: 코드는 MIT이지만 얼굴 검출에 쓰는 InsightFace 모델은 비상업 연구용입니다. 사내 행사·연구용으로만.
>
> - 이 맥(M1 Max)에서 실제 생성 확인: 16프레임 43초, 100프레임(말하기) 4분 47초.
> - 시무식 인사말, 수상자 축하 영상, 역사 인물 사진 애니메이션 같은 데 씁니다. 본인 동의 없는 타인 사진 사용 금지.

## 실행

```bash
bash setup.sh                                  # python3.11 venv + 의존성 + LivePortrait + 가중치 630MB + 서버 http://localhost:8770
TTS_BASE_URL=http://localhost:8771/v1 bash setup.sh   # 대사 음성 붙이기 (kokoro-local 등)
bash setup.sh stop
python3 app.py --cli 사진.jpg --template 말하기 --text "안녕하세요" -o out.mp4
python3 selftest.py                            # 가중치 없이 파이프라인만 검증 (가짜 추론)
```

| 환경변수 | 기본 | 설명 |
|---|---|---|
| `PORT` | 8770 | |
| `DEVICE` | auto | cuda → mps → cpu 자동. 강제 지정 가능. GPU 는 실행마다 여유 메모리가 가장 큰 1장을 고름(`gpu_pick.py`, `GPU_POOL` 로 후보 제한) |
| `MAX_DIM` | 512 | 입력 사진 최대 변. 작을수록 빠름 |
| `TTS_BASE_URL` | (없음) | OpenAI 호환 `/v1/audio/speech` 서버. 있으면 대사 음성을 영상에 합침 |
| `TTS_VOICE` / `TTS_MODEL` | af_heart / kokoro | TTS 서버에 넘기는 voice·model |

표정 템플릿 7종(말하기·웃음·윙크·수줍음·고개 젓기·입 벌리기·억울함)은 LivePortrait 저장소의 모션 템플릿(.pkl, 영상 없이 움직임만)이라 개인정보가 없습니다. 내 영상(mp4)을 올려 그 움직임을 따라 하게 할 수도 있습니다.

## 폐쇄망 반입

```bash
./pack.sh cpu        # 또는 ./pack.sh cu124 (NVIDIA)  → dist-offline/portrait-local-linux-x64-<v>.tar.gz
```
소스 + vendor/LivePortrait + weights/ + linux-x64 py3.11 wheels(torch 포함)를 묶습니다. 내부망에서 풀고 `bash setup.sh` 하면 다운로드 없이 설치됩니다. 이 맥에서는 번들 생성을 실행하지 않았습니다(스크립트만 작성).

## 한계
- 립싱크 아님: 대사 음성은 "말하기" 표정 위에 얹힐 뿐입니다. 입모양 동기화가 필요하면 SadTalker/MuseTalk 류를 따로 붙여야 합니다.
- 정면 단일 인물 사진이 잘 되고, 옆얼굴·안경 반사·작은 얼굴은 품질이 떨어집니다.
- 결과는 `_workspace/<run>/final.mp4`. 출처·라이선스는 `NOTICE`.

## 출처·감사 (Credits)

- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) (MIT, `LICENSE-LivePortrait`) — setup 때 `vendor/` 로 받아 수정 없이 씀. `templates/` 의 모션은 LivePortrait `assets/examples/driving/*.pkl` 을 변환한 것(모션 데이터만)
- 가중치(setup 때 받음, 재배포 안 함): [KwaiVGI/LivePortrait](https://huggingface.co/KwaiVGI/LivePortrait), InsightFace buffalo_l (InsightFace 사전학습 모델은 **비상업 연구용**)
- PyTorch, OpenCV, NumPy 등 `requirements.txt` 의 패키지 (각 라이선스)
- 이 도구는 [agent-page-portal](https://github.com/gggg8657/agent-page-portal) 에 연결해 쓰도록 만들었습니다(단독 실행도 됨).

저작권 표기·전체 목록은 `NOTICE` 를 보세요.
