# Connect6 AlphaZero

AlphaZero 알고리즘을 Connect6(6목) 게임에 적용한 강화학습 프로젝트

---

## 프로젝트 개요

AlphaZero를 기존 바둑, 체스같은 게임이 아닌 한 턴에 2수씩 두는 Connect6 게임에 적용

### 사용한 접근법:
- Single Head: 2수를 둘 때 같은 네트워크를 연속적으로 사용하여 착수
- Dual Head: 2수를 둘 때 첫 수를 조건부로 다음 수를 예측하여 착수 

---

## 주요 기능
- Self-Play 기반 학습: 사람이 제작한 기보 없이 에이전트의 self play로 현재 상황에서의 최적의 수 학습
- 데이터 증강(8배)
- MCTS + NN(Neural Net)

---

## 설치
```
pip install -r requirements.txt
```

**Python 3.8+, PyTorch, CUDA(권장)**

---

## 빠른 시작

### 테스트(약 5분)
```
python3 train.py \
  --model.kind single --model.blocks 8 --model.channels 64 \
  --board_size 11 --selfplay.games 10 --selfplay.workers 4 \
  --mcts.sims 100 --batch 128 --train.steps 10 \
  --devices cuda:0 --epochs 3
```

### 모델 evaluation
```
python3 evaluate.py \
  --model1 ckpts/epoch_100.pt \
  --model2 ckpts/epoch_50.pt \
  --games 20 --sims 200
```

### 파일 구조

```
├── sixmoku_env.py        # 게임 환경
├── models.py             # 신경망 (Single/Dual Head)
├── mcts.py               # MCTS 탐색
├── selfplay.py           # Self-play 데이터 생성
├── train.py              # 학습 메인
├── evaluate.py           # 모델 평가
└── augmentation.py       # 데이터 증강
```

---

## 코드 예시 

1. env 사용
```
from sixmoku_env import SixMokuEnv

env = SixMokuEnv(board_size=11)
obs, info = env.reset()
action = env.legal_actions()[0]
obs, reward, done, _, _ = env.step(action)
```

2. MCTS 사용
```
from mcts import MCTS
from models import build_model

model = build_model(kind="single", blocks=12, channels=96, board_size=11)
mcts = MCTS(model, c_puct=1.5, n_simulations=200)
pi, value = mcts.search(env)
```

3. 모델 로드 
```
import torch
from models import build_model

model = build_model(kind="single", blocks=12, channels=96, board_size=11)
checkpoint = torch.load("ckpts/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 참고

- https://www.science.org/doi/10.1126/science.aar6404
- https://docs.google.com/presentation/d/1F4ITFOyzDIy7oNUkq1qeyazsu_PZ058IqoaNCmYH4SE/edit?slide=id.p4#slide=id.p4

