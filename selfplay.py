from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import math
import copy
import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp

from sixmoku_env import SixMokuEnv
from mcts import MCTS
from models import build_model


@dataclass
class SingleSample:
    obs: np.ndarray        
    pi: np.ndarray         
    to_play: int           
    mask: np.ndarray       

@dataclass
class DualTurnSample:
    obs_first: np.ndarray      
    cond_first: np.ndarray     
    pi1: np.ndarray            
    pi2: np.ndarray            
    to_play: int               
    mask1: np.ndarray          
    mask2: np.ndarray          

class SelfPlayWorker:
    def __init__(
        self,
        model_cfg: Dict[str, Any],            
        sims: int = 200,
        c_puct: float = 1.5,
        dirichlet_alpha: Optional[float] = 0.3,
        dirichlet_eps: float = 0.25,
        temperature: float = 1.0,
        board_size: int = 11,
        max_blocking: int = 4,
        step_penalty: float = 0.0,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        temp_schedule_type: str = "constant",  
        temp_threshold_turn: int = 10,         
        temp_high: float = 1.0,                
        temp_low: float = 0.1,                 
    ) -> None:
        self.kind = model_cfg.get("kind", "single")
        self.board_size = int(board_size)
        self.model = build_model(kind=self.kind,
                                 blocks=model_cfg.get("blocks", 12),
                                 channels=model_cfg.get("channels", 96),
                                 in_channels=model_cfg.get("in_channels", 6),
                                 board_size=self.board_size)
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        self.sims = int(sims)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = float(dirichlet_eps)
        self.temperature = float(temperature)
        self.max_blocking = int(max_blocking)
        self.step_penalty = float(step_penalty)
        self.rng = np.random.default_rng(seed)
        
        self.temp_schedule_type = temp_schedule_type
        self.temp_threshold_turn = int(temp_threshold_turn)
        self.temp_high = float(temp_high)
        self.temp_low = float(temp_low)
    
    def get_temperature(self, turn_count: int) -> float:
        
        if self.temp_schedule_type == "constant":
            return self.temperature
        elif self.temp_schedule_type == "turn_based":
            if turn_count < self.temp_threshold_turn:
                return self.temp_high
            else:
                return self.temp_low
        elif self.temp_schedule_type == "move_based":
            progress = min(1.0, turn_count / (self.temp_threshold_turn * 2))
            return self.temp_high * (1 - progress) + self.temp_low * progress
        else:
            return self.temperature

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.model.load_state_dict(state_dict, strict=True)

    def _mcts(self) -> MCTS:
        return MCTS(self.model, kind=self.kind, c_puct=self.c_puct,
                    n_simulations=self.sims,
                    dirichlet_alpha=self.dirichlet_alpha, dirichlet_eps=self.dirichlet_eps,
                    device=self.device)

    def play_one_game(self) -> Dict[str, Any]:
        env = SixMokuEnv(board_size=self.board_size, max_blocking=self.max_blocking,
                         step_penalty=self.step_penalty)
        obs, info = env.reset()
        mcts = self._mcts()

        single_samples: List[SingleSample] = []
        dual_turn_samples: List[DualTurnSample] = []

        pending_dual: Optional[Dict[str, Any]] = None

        done = False
        turn_count = 0 
        while not done:
            mask = env.legal_action_mask()
            current_temp = self.get_temperature(turn_count)
            pi, _ = mcts.search(env, n_simulations=self.sims, temperature=current_temp)

            if self.kind == "single":
                single_samples.append(SingleSample(
                    obs=env._obs().copy(),
                    pi=pi.copy(),
                    to_play=env.current_player,
                    mask=mask.copy(),
                ))
            else:  
                if env.substep == 0:
                    pending_dual = {
                        "obs_first": env._obs().copy(),
                        "pi1": pi.copy(),
                        "to_play": env.current_player,
                        "mask1": mask.copy(),
                    }
                else:  
                    if pending_dual is not None and pending_dual.get("to_play") == env.current_player:
                        cond_first = _onehot_from_last_move(env)
                        dual_turn_samples.append(DualTurnSample(
                            obs_first=pending_dual["obs_first"],
                            cond_first=cond_first,
                            pi1=pending_dual["pi1"],
                            pi2=pi.copy(),
                            to_play=env.current_player,
                            mask1=pending_dual["mask1"],
                            mask2=mask.copy(),
                        ))
                        pending_dual = None
                    else:
                        pending_dual = None

            a = _select_action_from_pi(pi, mask, temperature=current_temp, rng=self.rng)
            obs, reward, done, trunc, info = env.step(a)
            
            turn_count += 1

            if done and self.kind == "dual":
                pending_dual = None

        z_game = 0
        if "winner" in info:
            z_game = 1 if info["winner"] == 1 else -1
        else:
            z_game = 0

        def signed(z, to_play):
            return float(z) if to_play == 1 else float(-z)

        num_moves = len(single_samples) if self.kind == "single" else len(dual_turn_samples) * 2
        
        out: Dict[str, Any] = {
            "kind": self.kind,
            "single": [],
            "turn": [],
            "result": z_game,
            "game_length": num_moves,
            "is_draw": (z_game == 0),
        }
        if self.kind == "single":
            for s in single_samples:
                out["single"].append({
                    "obs": s.obs, "pi": s.pi, "z": signed(z_game, s.to_play), "mask": s.mask,
                })
        else:
            for t in dual_turn_samples:
                out["turn"].append({
                    "obs": t.obs_first,
                    "cond_first": t.cond_first,
                    "pi1": t.pi1,
                    "pi2": t.pi2,
                    "z": signed(z_game, t.to_play),
                    "mask1": t.mask1,
                    "mask2": t.mask2,
                })
        return out

    def play_n_games(self, n_games: int) -> Dict[str, Any]:
        results = []
        for _ in range(int(n_games)):
            results.append(self.play_one_game())
        merged = _merge_results(results)
        if "games" not in merged:
            merged["games"] = len(results)
        return merged

class SelfPlayPool:
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        sims: int = 200,
        c_puct: float = 1.5,
        num_workers: int = 12,
        devices: Optional[List[str]] = None,   
        temperature: float = 1.0,
        board_size: int = 11,
        max_blocking: int = 4,
        step_penalty: float = 0.0,
        seed: Optional[int] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        dirichlet_alpha: Optional[float] = 0.3,
        dirichlet_eps: float = 0.25,
        temp_schedule_type: str = "constant",
        temp_threshold_turn: int = 15,
        temp_high: float = 1.0,
        temp_low: float = 0.1,
    ) -> None:
        self.model_cfg = dict(model_cfg)
        self.sims = sims
        self.c_puct = c_puct
        self.num_workers = int(num_workers)
        self.devices = devices or ["cpu"]
        self.temperature = temperature
        self.board_size = board_size
        self.max_blocking = max_blocking
        self.step_penalty = step_penalty
        self.seed = seed
        self.state_dict = state_dict
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temp_schedule_type = temp_schedule_type
        self.temp_threshold_turn = temp_threshold_turn
        self.temp_high = temp_high
        self.temp_low = temp_low

    def _spawn_args(self, rank: int) -> Dict[str, Any]:
        if self.devices and self.devices[0] != "cpu":
            dev = self.devices[rank % len(self.devices)]
        else:
            dev = "cpu"
        return {
            "rank": rank,
            "device": dev,
            "worker_args": dict(
                model_cfg=self.model_cfg,
                sims=self.sims,
                c_puct=self.c_puct,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_eps=self.dirichlet_eps,
                temperature=self.temperature,
                board_size=self.board_size,
                max_blocking=self.max_blocking,
                step_penalty=self.step_penalty,
                seed=(None if self.seed is None else self.seed + rank),
                temp_schedule_type=self.temp_schedule_type,
                temp_threshold_turn=self.temp_threshold_turn,
                temp_high=self.temp_high,
                temp_low=self.temp_low,
            ),
            "state_dict": self.state_dict,
        }

    def update_weights(self, state_dict: Dict[str, Any]) -> None:
        
        self.state_dict = state_dict

    def play(self, n_games: int) -> Dict[str, Any]:
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass  
        
        per_worker = math.ceil(n_games / self.num_workers)
        ctx = mp.get_context("spawn")
        
        manager = ctx.Manager()
        q = manager.Queue()
        
        procs = []
        for rank in range(self.num_workers):
            args = self._spawn_args(rank)
            p = ctx.Process(target=_worker_entry, args=(q, per_worker, args))
            p.start()
            procs.append(p)

        if self.sims >= 800:
            timeout_seconds = max(14400, per_worker * 10 * 60 * 2)  
        elif self.sims >= 400:
            timeout_seconds = max(7200, per_worker * 5 * 60 * 2)  
        elif self.sims >= 200:
            timeout_seconds = max(7200, per_worker * 3 * 60 * 2)  
        else:
            timeout_seconds = max(3600, per_worker * 2 * 60 * 2)  
        
        results = []
        for i in range(self.num_workers):
            try:
                result = q.get(timeout=timeout_seconds)
                results.append(result)
            except Exception as e:
                alive_workers = sum(1 for p in procs if p.is_alive())
                if alive_workers > 0:
                    print(f"[WARNING] Worker {i} timeout after {timeout_seconds}s ({timeout_seconds/3600:.1f}h), "
                          f"but {alive_workers} workers still alive. "
                          f"This indicates games are taking longer than expected. "
                          f"Current: mcts.sims={self.sims}, games={n_games}, per_worker={per_worker}. "
                          f"Consider reducing mcts.sims or selfplay.games.", flush=True)
                    try:
                        result = q.get(timeout=timeout_seconds)
                        results.append(result)
                    except:
                        raise RuntimeError(f"Worker {i} failed to return result after {timeout_seconds * 2}s ({timeout_seconds*2/3600:.1f}h). "
                                         f"Games are taking too long. Consider reducing mcts.sims (current: {self.sims}) "
                                         f"or selfplay.games (current: {n_games}).")
                else:
                    raise RuntimeError(f"Worker {i} died before returning result: {e}")
        
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        merged = _merge_results(results)
        return merged


def _worker_entry(queue: mp.Queue, per_worker_games: int, args: Dict[str, Any]):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    torch.set_num_threads(1)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        w = SelfPlayWorker(device=args["device"], **args["worker_args"])
        if args.get("state_dict") is not None:
            w.load_state_dict(args["state_dict"]) 
        data = w.play_n_games(per_worker_games)
        queue.put(data)
    except Exception as e:
        import traceback
        queue.put({"error": str(e), "traceback": traceback.format_exc(), "games": 0})
        raise


def _select_action_from_pi(pi: np.ndarray, mask: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    legal_idx = np.flatnonzero(mask)
    if len(legal_idx) == 0:
        return 0
    p = pi.copy()
    p *= 0.0
    p[legal_idx] = pi[legal_idx]
    s = p.sum()
    if s <= 0:
        return int(rng.choice(legal_idx))
    if temperature <= 1e-8:
        return int(np.argmax(p))
    p = p ** (1.0 / max(1e-8, temperature))
    p = p / p.sum()
    return int(rng.choice(np.arange(len(p)), p=p))


def _onehot_from_last_move(env: SixMokuEnv) -> np.ndarray:
    H = W = env.n
    t = np.zeros((1, H, W), dtype=np.float32)
    if env.last_move is not None:
        r, c = env.last_move
        t[0, r, c] = 1.0
    return t


def _merge_results(list_of_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "kind": None, 
        "single": [], 
        "turn": [], 
        "games": 0,
        "total_game_length": 0,
        "total_draws": 0,
        "black_wins": 0,
        "white_wins": 0,
    }
    for r in list_of_results:
        if out["kind"] is None:
            out["kind"] = r.get("kind")
        if r.get("single"):
            out["single"].extend(r["single"])
        if r.get("turn"):
            out["turn"].extend(r["turn"])
        
        if "games" in r:
            out["games"] += r["games"]
            out["total_game_length"] += r.get("total_game_length", 0)
            out["total_draws"] += r.get("total_draws", 0)
            out["black_wins"] += r.get("black_wins", 0)
            out["white_wins"] += r.get("white_wins", 0)
        else:
            out["games"] += 1
            out["total_game_length"] += r.get("game_length", 0)
            if r.get("is_draw"):
                out["total_draws"] += 1
            elif r.get("result", 0) == 1:
                out["black_wins"] += 1
            elif r.get("result", 0) == -1:
                out["white_wins"] += 1
    
    if out["games"] > 0:
        out["avg_game_length"] = out["total_game_length"] / out["games"]
        out["draw_rate"] = out["total_draws"] / out["games"]
        out["black_win_rate"] = out["black_wins"] / out["games"]
    else:
        out["avg_game_length"] = 0
        out["draw_rate"] = 0
        out["black_win_rate"] = 0
    
    return out

if __name__ == "__main__":
    model_cfg = {"kind": "single", "blocks": 8, "channels": 64}
    worker = SelfPlayWorker(model_cfg=model_cfg, sims=32, c_puct=1.5, temperature=1.0)
    data = worker.play_n_games(2)
    print("single -> samples:", len(data["single"]))

    model_cfg = {"kind": "dual", "blocks": 8, "channels": 64}
    worker2 = SelfPlayWorker(model_cfg=model_cfg, sims=32, c_puct=1.5, temperature=1.0)
    data2 = worker2.play_n_games(2)
    print("dual -> turn samples:", len(data2["turn"]))

    pool = SelfPlayPool(model_cfg={"kind":"single","blocks":8,"channels":64}, sims=16, num_workers=2, devices=["cpu"])  # 장치는 필요에 맞게
    merged = pool.play(4)
    print("pool games:", merged["games"], "samples:", len(merged["single"]))