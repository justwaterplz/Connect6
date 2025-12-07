from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch

class Node:
    __slots__ = ("prior", "visits", "value_sum", "children", "legal")
    def __init__(self, prior: float, legal: np.ndarray):
        self.prior = float(prior)
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[int, Node] = {}
        self.legal = legal  

    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

class MCTS:
    
    def __init__(
        self,
        model,
        kind: str = "dual",
        c_puct: float = 1.5,
        n_simulations: int = 200,
        temperature: float = 1e-8,
        device: Optional[torch.device] = None,
        proximity_alpha: float = 0.0,  
        proximity_radius: int = 2,
        proximity_decay: float = 0.7,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ):
        self.model = model.eval()
        self.kind = kind
        self.c_puct = float(c_puct)
        self.n_simulations = int(n_simulations)
        self.temperature = float(temperature)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prox_alpha = float(proximity_alpha)
        self.prox_radius = int(proximity_radius)
        self.prox_decay = float(proximity_decay)

        self.dir_alpha = float(dirichlet_alpha)
        self.dir_eps = float(dirichlet_eps)

    @torch.no_grad()
    def search(
        self, env, n_simulations: Optional[int] = None, temperature: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        sims = int(n_simulations) if n_simulations is not None else self.n_simulations
        temp = float(self.temperature if temperature is None else temperature)

        root_snap = _snapshot(env)
        root, _ = self._expand(_restore(root_snap), node=None)

        for _ in range(sims):
            self._simulate(root_snap, root)

        pi = np.zeros(env.n * env.n, dtype=np.float32)
        for a, ch in root.children.items():
            pi[a] = ch.visits
        s = pi.sum()

        if s > 0:
            if temp <= 1e-8:
                hot = np.zeros_like(pi)
                hot[int(np.argmax(pi))] = 1.0
                pi = hot
            else:
                ex = np.power(pi, 1.0 / max(1e-6, temp))
                pi = ex / np.sum(ex)
        return pi, root.value()

    def _simulate(self, snap, node: Node):
        env = _restore(snap)
        path: List[Tuple[Node, int]] = [(node, -1)]

        while node.children:
            a = self._select(node)
            path.append((node.children[a], a))
            _, r, done, _, _ = env.step(a)
            node = node.children[a]
            if done:
                self._backprop(path, r)
                return

        v = self._expand(env, node)  
        self._backprop(path, v)

    def _expand(self, env, node: Optional[Node] = None):
        obs = torch.from_numpy(env._obs()).unsqueeze(0).float().to(self.device)
        mask_np = env.legal_action_mask()
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        cond_first = None
        if self.kind == "dual":
            cond_first = torch.zeros(1, 1, env.n, env.n, dtype=torch.float32, device=self.device)
            if env.substep == 1 and env.last_move is not None:
                r, c = env.last_move
                cond_first[0, 0, r, c] = 1.0

        if self.kind == "single":
            out = self.model(obs, mask=mask)  
            logits = out["pi"]
        else:
            out = self.model(obs, cond_first=cond_first, mask1=mask, mask2=mask)
            logits = out["pi1"] if env.substep == 0 else out["pi2"]

        logits = logits.float()
        priors = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        is_root = node is None
        if is_root:
            node = Node(1.0, mask_np)

            if self.prox_alpha > 0.0:
                prox = _proximity_map(env, radius=self.prox_radius, decay=self.prox_decay).reshape(-1)
                if prox.sum() > 0:
                    priors = (1.0 - self.prox_alpha) * priors + self.prox_alpha * prox
                    s = priors.sum()
                    if s > 0:
                        priors = priors / s

            if self.dir_eps > 0.0 and self.dir_alpha > 0.0:
                legal_idx = np.flatnonzero(mask_np)
                if len(legal_idx) > 0:
                    noise = np.zeros_like(priors, dtype=np.float32)
                    g = np.random.dirichlet([self.dir_alpha] * len(legal_idx)).astype(np.float32)
                    noise[legal_idx] = g
                    priors = (1.0 - self.dir_eps) * priors + self.dir_eps * noise
                    s = priors.sum()
                    if s > 0:
                        priors = priors / s

        legal = np.flatnonzero(node.legal)
        for a in legal:
            if a not in node.children:
                node.children[a] = Node(priors[a], env.legal_action_mask())

        v = float(out["v"].item()) if isinstance(out, dict) and "v" in out else 0.0
        return (node, v) if is_root else v

    def _select(self, node: Node) -> int:
        total = max(1, node.visits)
        best_s, best_a = -1e18, None
        legal = np.flatnonzero(node.legal)
        for a in legal:
            ch = node.children.get(a)
            p = 1e-8 if ch is None else ch.prior
            q = 0.0 if ch is None else ch.value()
            u = self.c_puct * p * (math.sqrt(total) / (1 + (0 if ch is None else ch.visits)))
            s = q + u
            if s > best_s:
                best_s, best_a = s, a
        return int(best_a)

    def _backprop(self, path: List[Tuple[Node, int]], z: float):
        cur = z
        for node, _ in path:
            node.visits += 1
            node.value_sum += cur
            cur = -cur


def _proximity_map(env, radius: int = 2, decay: float = 0.7) -> np.ndarray:
    
    n = env.n
    prox = np.zeros((n, n), dtype=np.float32)
    opp = -env.current_player
    stones = np.argwhere(env.board == opp)
    if stones.size == 0:
        return prox

    for r, c in stones:
        r0, r1 = max(0, r - radius), min(n - 1, r + radius)
        c0, c1 = max(0, c - radius), min(n - 1, c + radius)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                d = abs(rr - r) + abs(cc - c)
                if d == 0 or d > radius:
                    continue
                w = decay ** (d - 1)  
                if w > prox[rr, cc]:
                    prox[rr, cc] = w

    legal = env.legal_action_mask().reshape(n, n).astype(np.float32)
    prox *= legal
    s = prox.sum()
    if s > 0:
        prox /= s
    return prox


def _snapshot(env):
    return dict(
        board=env.board.copy(),
        cur=env.current_player,
        sub=env.substep,
        first=env.first_move_done,
        last=None if env.last_move is None else tuple(env.last_move),
        n=env.n,
        maxb=env.max_blocking,
    )


def _restore(s):
    from sixmoku_env import SixMokuEnv
    e = SixMokuEnv(board_size=s["n"], max_blocking=s["maxb"])
    e.board[:, :] = s["board"]
    e.current_player = s["cur"]
    e.substep = s["sub"]
    e.first_move_done = s["first"]
    e.last_move = None if s["last"] is None else tuple(s["last"])
    return e
