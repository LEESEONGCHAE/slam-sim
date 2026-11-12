# slam-sim

차량·AMR(물류 로봇)의 동역학, 라이다 신호처리, SLAM을 **직접 구현하면서 공부하는** 2D 시뮬레이터.
라이브러리는 numpy·matplotlib만 쓰고, 알고리즘은 전부 직접 짠다.

## 실행

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py                   # 직접 운전 (센서 데이터가 실시간으로 my_slam.py 에 들어감)
python run.py --headless --seconds 60    # 자동 주행 → out/run.png
python -m pytest -q
```

`run.py` 화면: 실제 세계(운전하는 곳) | 라이다 원시값(로봇 기준) | my_slam.py 가 만든 지도와 오차.
조작: 창을 한 번 클릭한 뒤 방향키 운전, 스페이스 브레이크, `A` 자동 주행, `R` 처음부터,
`L` 지금까지의 센서 로그 저장(`data/drive_*.npz`, `python play.py data/drive_....npz` 로 다시 재생).

## 내 SLAM 돌려 보기 (센서 데이터 재생)

하드웨어 없이 SLAM을 직접 구현해 보는 곳. `play.py`가 센서 메시지를 시간순으로 [my_slam.py](my_slam.py)에 넣어 준다.
지금 `my_slam.py`는 오도메트리를 그대로 믿는 기준선이고, `on_scan`의 TODO 자리에 스캔 매칭 등을 넣으면 된다.

```bash
python get_data.py              # 실제 데이터 받기 (Intel Research Lab, 16 MB, data/에 저장)
python play.py intel            # 실제 2D 라이다 + 오도메트리 재생
python play.py sim              # 시뮬레이터 로그 (정답 위치가 있어 오차 RMSE를 보여 줌)
python play.py intel --headless # 끝까지 돌리고 out/play_intel.png 저장
```

재생 중: 스페이스 일시정지, `+`/`-` 배속.

| 데이터 | 라이다 | 오도메트리 | 정답 위치 |
| --- | --- | --- | --- |
| `intel` (실제, 약 45분) | 180빔, 정면 기준 −90°~+89° 1° 간격, 반사 없음 = 80 m 이상 | 로봇이 적분한 (x, y, θ) | 없음 (지도 모양으로 판단) |
| `sim` (240초) | 181빔, 270°, 노이즈 3 cm, 빔 2% 누락, 0.5% 엉뚱한 값, 모션 왜곡 | 휠 속도 스케일 오차 + 자이로 바이어스 | 있음 |

메시지 형식은 [slamsim/logs.py](slamsim/logs.py), 자세 계산 도우미(compose/relative 등)는 [slamsim/geometry.py](slamsim/geometry.py).
Intel 빔 각도(−90°~+89°)는 CARMEN 관례를 따른 가정이고 원 문서로 확인하지 않았다.

데이터 출처: Intel Research Lab (Seattle) 데이터셋, 프라이부르크 대학 SLAM 평가 페이지
(http://ais.informatik.uni-freiburg.de/slamevaluation/datasets.php). 저장소에는 넣지 않는다.

## 시뮬레이터 구성 (1단계)

- 차량: 저속은 기구학 자전거 모델, 고속은 동적 자전거 모델(선형 타이어 + 마찰 한계 → 미끄러짐)
- 센서: 휠 속도(스케일 오차) + 자이로(바이어스) 추측 항법, 270° 2D 라이다(레이캐스트 + 가우시안 노이즈)
- 지도: 로그 오즈 점유 격자. 실제 위치로 그린 지도와 오도메트리 위치로 그린 지도를 나란히 비교

오도메트리만으로 지도를 그리면 2분 만에 수 m씩 어긋난다. 이 문제를 푸는 게 이후 단계의 목표다.

## 강화학습 기본 예제

[rl_basics/](rl_basics/README.md): 미로 Q-learning → 막대 세우기 DQN → 진자 SAC → 직접 만든 환경(정지선, RL vs PD). 각 1~2분.

## PID vs 강화학습 (경로 따라가기)

같은 차·같은 트랙·같은 속도 규칙에서 조향만 겨룬다. 시험 트랙 10개는 학습에 쓰지 않은 모양.

```bash
python rl/compare.py                        # 비교 → rl/results.md, out/compare.png
python rl/train.py --steps 150000 --reward v2   # RL(SAC) 다시 학습 (CPU 약 35분)
python rl/tune_pid.py                       # PID 게인 탐색 (--robust: 노이즈·지연 포함)
```

직접 제어기를 짜려면 [rl/my_controller.py](rl/my_controller.py)의 `policy(obs, env)`를 고치고 `rl/compare.py`를 돌린다.
결과 표는 [rl/results.md](rl/results.md), 설명은 [docs/rl-notes.html](docs/rl-notes.html) 17장.

기본 조건 요약: PID+피드포워드 0.7 cm, RL 보상v2 4.5 cm, RL 보상v1 23.2 cm (평균 RMS 횡 오차).
보상 식만 바꿔 RL 오차가 1/5이 됐다. 센서 노이즈·조향 지연에서는 기본 튜닝 PID가 10/10 이탈했다. RL 시드 1개 기준.

## 학습 노트

- [docs/dynamics-slam-notes.html](docs/dynamics-slam-notes.html): 차량 동역학, 라이다 원시 데이터, SLAM 알고리즘 24장
- [docs/rl-notes.html](docs/rl-notes.html): 강화학습 개념·알고리즘·sim-to-real, PID vs RL 실험 18장

브라우저로 열면 된다(수식은 인터넷 연결 필요).

## 공부 로드맵

각 장은 `docs/NN-*.md`(이론·수식) + `slamsim/` 구현 + 비교 실험으로 구성한다.

| 장 | 주제 | 핵심 |
| --- | --- | --- |
| 1 | 운동 모델 | 차동구동 AMR 기구학, 자전거 기구학/동역학, 타이어 슬립. AMR은 왜 기구학으로 충분한가 |
| 2 | 오도메트리 오차 | 엔코더 체계 오차/우연 오차, 자이로 바이어스, 오차 공분산 전파 |
| 3 | 라이다 신호처리 | 빔 측정 모델(hit/short/max/random), 이상치 제거, 다운샘플, 모션 왜곡 보정(deskew), 세그멘테이션, 선분 추출(split-and-merge, RANSAC) |
| 4 | 점유 격자 | 역센서 모델, 로그 오즈 |
| 5 | 위치 추정 | EKF(특징점 지도), 파티클 필터 MCL/AMCL |
| 6 | 스캔 매칭 | ICP(point-to-point, point-to-line), 상관 스캔 매칭 |
| 7 | SLAM | EKF-SLAM, FastSLAM(그리드), 포즈 그래프 + 루프 클로저 |
| 8 | AMR 내비게이션 | 창고 지도, 코스트맵, A* 전역 경로, DWA 지역 경로 |

참고서: *Probabilistic Robotics* (Thrun, Burgard, Fox). 3~7장의 수식 대부분이 이 책에 있다.
