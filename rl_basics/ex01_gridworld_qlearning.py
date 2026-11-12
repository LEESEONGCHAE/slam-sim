"""예제 1. 격자 세계에서 Q-learning  (라이브러리 없이 numpy 만)

상황
  5×5 칸 미로. 로봇이 왼쪽 위(S)에서 출발해 오른쪽 아래 목표(G)로 가야 한다.
  X 는 구덩이. 빠지면 큰 감점.

  S . . . .
  . X . X .
  . . . X .
  X . . . .
  . . X . G

배울 것
  - 상태(어느 칸), 행동(상하좌우), 보상(점수), 에피소드(출발~끝 한 번)
  - Q 표: "이 칸에서 이 방향으로 가면 앞으로 점수를 얼마나 받을까"
  - 처음엔 아무것도 모르다가, 수백 번 해 보면서 Q 표가 채워지고 길을 찾는다

실행: VS Code 에서 이 파일을 열고 ▶  (또는  .venv\\Scripts\\python rl_basics/ex01_gridworld_qlearning.py)
"""
import numpy as np

# ---------------------------------------------------------------- 1. 환경 만들기
MAP = ("S...."               # S 출발, . 빈칸, X 구덩이, G 목표
       ".X.X."
       "...X."
       "X...."
       "..X.G")              # 다섯 줄이 이어 붙어 25칸짜리 문자열 하나가 된다. 칸 번호 = 행×5 + 열
N = 5
ACTIONS = ["↑", "↓", "←", "→"]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # (행 변화, 열 변화)


def step(state, action):
    """환경: 지금 칸(state)에서 행동(action)을 하면 → 다음 칸, 보상, 끝났는지"""
    r, c = divmod(state, N)
    dr, dc = MOVES[action]
    r2, c2 = r + dr, c + dc
    if not (0 <= r2 < N and 0 <= c2 < N):    # 벽에 부딪히면 제자리
        r2, c2 = r, c
    nxt = r2 * N + c2
    cell = MAP[nxt]
    if cell == "G":
        return nxt, +10.0, True     # 목표 도착: +10, 끝
    if cell == "X":
        return nxt, -10.0, True     # 구덩이: −10, 끝
    return nxt, -0.1, False         # 그 밖: 한 걸음마다 −0.1 (빨리 가라는 뜻)


# ---------------------------------------------------------------- 2. Q 표
# Q[칸, 행동] = 그 칸에서 그 행동을 했을 때 앞으로 받을 점수 예상. 처음엔 전부 0 (아무것도 모름)
Q = np.zeros((N * N, 4))

alpha = 0.5      # 학습률: 새 경험을 얼마나 반영할지 (0~1)
gamma = 0.95     # 할인율: 미래 점수를 얼마나 쳐줄지 (0~1)
epsilon = 1.0    # 탐험 비율: 이 확률로 무작위 행동. 처음엔 100% 무작위로 시작해서 점점 줄인다

rng = np.random.default_rng(0)


def show_policy():
    """Q 표에서 칸마다 가장 점수 높은 방향을 화살표로 보여 준다"""
    for r in range(N):
        row = ""
        for c in range(N):
            s = r * N + c
            if MAP[s] in "XG":
                row += f" {MAP[s]} "
            else:
                row += f" {ACTIONS[int(np.argmax(Q[s]))]} " if Q[s].any() else " ? "
        print(row)


# ---------------------------------------------------------------- 3. 학습
for episode in range(1, 501):
    state, total, done, steps = 0, 0.0, False, 0
    while not done and steps < 100:
        # (1) 행동 고르기: ε 확률로 무작위(탐험), 아니면 Q 가 가장 큰 행동(활용)
        if rng.random() < epsilon:
            action = int(rng.integers(4))
        else:
            action = int(np.argmax(Q[state]))

        # (2) 해 보고 결과 받기
        nxt, reward, done = step(state, action)

        # (3) Q 갱신: 벨만 방정식  목표 = 보상 + γ × (다음 칸에서 가장 좋은 Q)
        target = reward if done else reward + gamma * np.max(Q[nxt])
        Q[state, action] += alpha * (target - Q[state, action])   # 목표 쪽으로 조금 당긴다

        state, total, steps = nxt, total + reward, steps + 1

    epsilon = max(0.05, epsilon * 0.99)    # 탐험을 조금씩 줄인다

    if episode in (1, 10, 50, 100, 500):
        print(f"\n=== 에피소드 {episode}  (이번 점수 {total:+.1f}, 걸음 {steps}, 탐험 ε={epsilon:.2f}) ===")
        show_policy()

# ---------------------------------------------------------------- 4. 다 배운 뒤 실제로 가 보기
print("\n=== 배운 대로 가 보기 (탐험 없이) ===")
state, path = 0, [0]
for _ in range(30):
    state, reward, done = step(state, int(np.argmax(Q[state])))
    path.append(state)
    if done:
        break
grid = [list(MAP[r * N:(r + 1) * N]) for r in range(N)]
for s in path[1:-1]:
    grid[s // N][s % N] = "*"
print("\n".join(" ".join(row) for row in grid))
print(f"\n{len(path) - 1}걸음 만에 {'목표 도착!' if MAP[path[-1]] == 'G' else '실패'}")
print("\n출발 칸의 Q 값 (↑ ↓ ← →):", np.round(Q[0], 2), " ← 가장 큰 쪽으로 간다")
