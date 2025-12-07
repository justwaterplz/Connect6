from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from pathlib import Path

from sixmoku_env import SixMokuEnv
from mcts import MCTS
from models import build_model


class GamePlayer:
    def __init__(
        self,
        model,
        kind: str = "dual",
        sims: int = 200,
        c_puct: float = 1.5,
        temperature: float = 1e-8,  
        device: Optional[str] = None,
    ):
        self.model = model.eval()
        self.kind = kind
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model.to(self.device)
        
        self.mcts = MCTS(
            self.model,
            kind=self.kind,
            c_puct=c_puct,
            n_simulations=sims,
            temperature=temperature,
            device=self.device,
            dirichlet_eps=0.0,
        )
    
    def select_action(self, env: SixMokuEnv) -> int:
        pi, _ = self.mcts.search(env)
        mask = env.legal_action_mask()
        legal_idx = np.flatnonzero(mask)
        
        if len(legal_idx) == 0:
            return 0
        
        pi_legal = pi * mask
        action = int(np.argmax(pi_legal))
        return action


def play_game(
    player1: GamePlayer,
    player2: GamePlayer,
    board_size: int = 11,
    max_blocking: int = 4,
    max_moves: int = 200,
    verbose: bool = False,
) -> Dict[str, Any]:
    env = SixMokuEnv(board_size=board_size, max_blocking=max_blocking)
    obs, info = env.reset()
    
    moves = 0
    done = False
    
    while not done and moves < max_moves:
        if env.current_player == 1:
            player = player1
        else:
            player = player2
        
        action = player.select_action(env)
        obs, reward, done, trunc, info = env.step(action)
        moves += 1
        
        if verbose and done:
            print(f"Game ended after {moves} moves")
            env.render()
    
    if done:
        if "winner" in info:
            winner = info["winner"]
            reason = f"win by player {1 if winner == 1 else 2}"
            if verbose and "winning_line" in info:
                winning_line = info["winning_line"]
                if winning_line:
                    print(f"Winning 6-in-a-row: {winning_line[0]} → {winning_line[-1]}")
        elif "draw" in info:
            winner = 0
            reason = "draw"
            if verbose:
                black_max = _find_longest_sequence(env, 1)
                white_max = _find_longest_sequence(env, -1)
                print(f"Draw (Black max: {black_max}, White max: {white_max})")
        else:
            winner = 0
            reason = "unknown"
    else:
        winner = 0
        reason = "max_moves reached"
    
    return {
        "winner": winner,
        "moves": moves,
        "reason": reason,
    }


def _find_longest_sequence(env, color: int) -> int:
    max_len = 0
    n = env.n
    board = env.board
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for r in range(n):
        for c in range(n):
            if board[r, c] == color:
                for dr, dc in dirs:
                    length = 1
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < n and 0 <= cc < n and board[rr, cc] == color:
                        length += 1
                        rr += dr
                        cc += dc
                    max_len = max(max_len, length)
    
    return max_len


def evaluate_models(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    model_cfg: Dict[str, Any],
    model_cfg2: Optional[Dict[str, Any]] = None,
    num_games: int = 20,
    mcts_sims: int = 200,
    c_puct: float = 1.5,
    board_size: int = 11,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    kind1 = model_cfg.get("kind", "dual")
    kind2 = model_cfg2.get("kind", "dual") if model_cfg2 else kind1
    
    player1 = GamePlayer(model1, kind=kind1, sims=mcts_sims, c_puct=c_puct, device=device)
    player2 = GamePlayer(model2, kind=kind2, sims=mcts_sims, c_puct=c_puct, device=device)
    
    results = {
        "model1_wins": 0,
        "model2_wins": 0,
        "draws": 0,
        "model1_as_black_wins": 0,
        "model1_as_white_wins": 0,
        "total_games": 0,
    }
    
    for game_idx in range(num_games):
        if game_idx % 2 == 0:
            game_result = play_game(player1, player2, board_size=board_size, verbose=verbose)
            winner = game_result["winner"]
            
            if winner == 1:
                results["model1_wins"] += 1
                results["model1_as_black_wins"] += 1
            elif winner == -1:
                results["model2_wins"] += 1
            else:
                results["draws"] += 1
        else:
            game_result = play_game(player2, player1, board_size=board_size, verbose=verbose)
            winner = game_result["winner"]
            
            if winner == -1:  
                results["model1_wins"] += 1
                results["model1_as_white_wins"] += 1
            elif winner == 1:  
                results["model2_wins"] += 1
            else:
                results["draws"] += 1
        
        results["total_games"] += 1
        
        if verbose or (game_idx + 1) % 5 == 0:
            print(f"[eval] Game {game_idx+1}/{num_games} | {game_result['reason']} | moves={game_result['moves']}")
    
    results["model1_win_rate"] = results["model1_wins"] / max(1, results["total_games"])
    
    return results


def load_model_from_checkpoint(ckpt_path: str, device: Optional[str] = None) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    args = ckpt.get("args", {})
    model_cfg = {
        "kind": args.get("model_kind", "dual"),
        "blocks": args.get("model_blocks", 12),
        "channels": args.get("model_channels", 96),
        "in_channels": 6,
    }
    
    model = build_model(**model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    return model, model_cfg


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate two SixMoku models")
    parser.add_argument("--model1", type=str, required=True, help="Path to model 1 checkpoint")
    parser.add_argument("--model2", type=str, required=True, help="Path to model 2 checkpoint")
    parser.add_argument("--games", type=int, default=20, help="Number of games")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per move")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0, cpu, etc)")
    parser.add_argument("--verbose", action="store_true", help="Print game details")
    
    args = parser.parse_args()
    
    print(f"Loading models...")
    print(f"  Model 1: {args.model1}")
    print(f"  Model 2: {args.model2}")
    
    model1, cfg1 = load_model_from_checkpoint(args.model1, device=args.device)
    model2, cfg2 = load_model_from_checkpoint(args.model2, device=args.device)
    
    if cfg1["kind"] != cfg2["kind"]:
        print(f"Different architectures: {cfg1['kind']} vs {cfg2['kind']}")
        print(f"Proceeding with cross-architecture evaluation...")
    
    print(f"\nEvaluating over {args.games} games...")
    print(f"MCTS sims: {args.sims}")
    print(f"Device: {args.device or 'auto'}")
    
    results = evaluate_models(
        model1, model2, cfg1, model_cfg2=cfg2,
        num_games=args.games,
        mcts_sims=args.sims,
        device=args.device,
        verbose=args.verbose,
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total games:     {results['total_games']}")
    print(f"Model 1 wins:    {results['model1_wins']} ({results['model1_win_rate']:.1%})")
    print(f"  - as Black:    {results['model1_as_black_wins']}")
    print(f"  - as White:    {results['model1_as_white_wins']}")
    print(f"Model 2 wins:    {results['model2_wins']} ({results['model2_wins']/results['total_games']:.1%})")
    print(f"Draws:           {results['draws']} ({results['draws']/results['total_games']:.1%})")
    print("="*50)

