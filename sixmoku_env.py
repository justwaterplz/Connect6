from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  
    import gym as gym
    from gym import spaces

Player = int  


class SixMokuEnv(gym.Env):
    
    metadata = {"render.modes": ["ansi"]}

    def __init__(
        self,
        board_size: int = 11,
        max_blocking: int = 4,
        blocking_even_only: bool = True,
        step_penalty: float = 0.0,
        include_last_move: bool = True,
        fixed_blocking: Optional[List[Tuple[int,int]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert board_size >= 6
        assert max_blocking >= 0
        self.n = int(board_size)
        self.N = self.n * self.n
        self.max_blocking = int(max_blocking)
        self.blocking_even_only = bool(blocking_even_only)
        self.step_penalty = float(step_penalty)
        self.include_last_move = bool(include_last_move)
        self.fixed_blocking = fixed_blocking  

        self.rng = np.random.default_rng(seed)

        self.board = np.zeros((self.n, self.n), dtype=np.int8)
        self.current_player: Player = 1
        self.substep: int = 0
        self.first_move_done: bool = False
        self.last_move: Optional[Tuple[int,int]] = None

        self.action_space = spaces.Discrete(self.N)
        ch = 5 + (1 if self.include_last_move else 0) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(ch, self.n, self.n), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board.fill(0)
        self.current_player = 1
        self.substep = 0
        self.first_move_done = False
        self.last_move = None

        if self.fixed_blocking is not None:
            for (r, c) in self.fixed_blocking:
                self._assert_inside(r, c)
                self.board[r, c] = 2
            k = len(self.fixed_blocking)
        else:
            k = self._sample_blocking_count()
            self._place_blocking(k)

        obs = self._obs()
        info = {"blocking_count": k}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Action out of range"
        r, c = divmod(int(action), self.n)

        if not self._is_empty(r, c):
            reward = -0.05 + (-abs(self.step_penalty) if self.step_penalty != 0 else 0.0)
            return self._obs(), reward, False, False, {"illegal": True}

        self.board[r, c] = self.current_player
        self.last_move = (r, c)

        win, line = self._exact_six_win(r, c, self.current_player)
        if win:
            info = {"winner": self.current_player, "win_move": (r, c), "winning_line": line}
            return self._obs(), 1.0, True, False, info

        reward = -abs(self.step_penalty) if self.step_penalty != 0 else 0.0

        if not self.first_move_done and self.current_player == 1 and self.substep == 0:
            self.first_move_done = True
            self.current_player = -1
            self.substep = 0
        else:
            if self.substep == 0:
                self.substep = 1
            else:
                self.substep = 0
                self.current_player *= -1

        if not self._has_any_legal_move():
            return self._obs(), 0.0, True, False, {"draw": True}

        return self._obs(), reward, False, False, {}

    def legal_action_mask(self) -> np.ndarray:
        mask = (self.board.reshape(-1) == 0)
        return mask

    def legal_actions(self) -> List[int]:
        return np.flatnonzero(self.legal_action_mask()).tolist()

    def render(self, mode: str = "ansi") -> str:
        chars = {0: ".", 1: "X", -1: "O", 2: "#"}
        rows = [" ".join(chars[int(v)] for v in row) for row in self.board]
        s = "\n".join(rows)
        if mode == "ansi":
            print(s)
        return s

    def _sample_blocking_count(self) -> int:
        if self.max_blocking <= 0:
            return 0
        if self.blocking_even_only:
            candidates = [i for i in range(0, self.max_blocking + 1) if (i % 2 == 0)]
        else:
            candidates = list(range(0, self.max_blocking + 1))
        return int(self.rng.choice(candidates))

    def _place_blocking(self, k: int) -> None:
        if k <= 0:
            return
        empties = [(r, c) for r in range(self.n) for c in range(self.n) if self.board[r, c] == 0]
        self.rng.shuffle(empties)
        for i in range(min(k, len(empties))):
            r, c = empties[i]
            self.board[r, c] = 2

    def _exact_six_win(self, r: int, c: int, color: Player) -> Tuple[bool, List[Tuple[int,int]]]:
        
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in dirs:
            run, line = self._max_run_with_line(r, c, color, dr, dc)
            if run == 6:
                return True, line
        return False, []

    def _max_run_with_line(self, r: int, c: int, color: Player, dr: int, dc: int) -> Tuple[int, List[Tuple[int,int]]]:
        line = [(r, c)]
        rr, cc = r + dr, c + dc
        while self._inside(rr, cc) and self.board[rr, cc] == color:
            line.append((rr, cc))
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while self._inside(rr, cc) and self.board[rr, cc] == color:
            line.insert(0, (rr, cc))
            rr -= dr
            cc -= dc
        return len(line), line

    def _inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    def _assert_inside(self, r: int, c: int) -> None:
        if not self._inside(r, c):
            raise ValueError(f"Out of board: {(r,c)}")

    def _is_empty(self, r: int, c: int) -> bool:
        return self.board[r, c] == 0

    def _has_any_legal_move(self) -> bool:
        return np.any(self.board == 0)

    def _obs(self) -> np.ndarray:
        black = (self.board == 1).astype(np.float32)
        white = (self.board == -1).astype(np.float32)
        block = (self.board == 2).astype(np.float32)
        player = np.full_like(black, 1.0 if self.current_player == 1 else 0.0)
        sub = np.full_like(black, float(self.substep))
        chans = [black, white, block, player, sub]
        if self.include_last_move:
            last = np.zeros_like(black)
            if self.last_move is not None:
                lr, lc = self.last_move
                last[lr, lc] = 1.0
            chans.append(last)
        return np.stack(chans, axis=0)


if __name__ == "__main__":
    env = SixMokuEnv(seed=0)
    obs, info = env.reset()
    print("Blocking:", info)
    done = False
    steps = 0
    while not done and steps < 30:
        legal = env.legal_actions()
        if not legal:
            print("No legal actions → draw")
            break
        a = np.random.choice(legal)
        obs, r, done, trunc, info = env.step(int(a))
        env.render()
        print(f"step={steps} | player={'B' if env.current_player==1 else 'W'} | sub={env.substep} | reward={r} | info={info}")
        steps += 1
