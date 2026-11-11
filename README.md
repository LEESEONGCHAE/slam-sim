# slam-sim

차량·AMR(물류 로봇)의 동역학, 라이다 신호처리, SLAM을 **직접 구현하면서 공부하는** 2D 시뮬레이터.
라이브러리는 numpy·matplotlib만 쓰고, 알고리즘은 전부 직접 짠다.

## 실행

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py                   # 창에서 직접 운전
python run.py --headless --seconds 120   # 자동 주행 → out/result.png
python -m pytest -q
```

조작: 방향키 운전, 스페이스 브레이크, `A` 자동 주행, `M` 지도 전환(오도메트리 위치/실제 위치), `R` 처음부터.

## 지금 되는 것 (1단계)

- 차량: 저속은 기구학 자전거 모델, 고속은 동적 자전거 모델(선형 타이어 + 마찰 한계 → 미끄러짐)
- 센서: 휠 속도(스케일 오차) + 자이로(바이어스) 추측 항법, 270° 2D 라이다(레이캐스트 + 가우시안 노이즈)
- 지도: 로그 오즈 점유 격자. 실제 위치로 그린 지도와 오도메트리 위치로 그린 지도를 나란히 비교

오도메트리만으로 지도를 그리면 2분 만에 수 m씩 어긋난다. 이 문제를 푸는 게 이후 단계의 목표다.

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
