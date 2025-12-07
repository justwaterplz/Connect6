#!/usr/bin/env python3

import sys
import torch
from models import build_model
from evaluate import evaluate_models

print("=" * 70)
print("HEAD-TO-HEAD EVALUATION (Direct)")
print("Single vs Dual - Fair Comparison")
print("=" * 70)
print()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
print()

SINGLE_CKPT = "/app/ckpts/gpumax_20251120_123225/epoch_200.pt"
DUAL_CKPT = "/app/ckpts/fair_dual_20251120_124206/epoch_100.pt"

model_cfg_single = {'kind': 'single', 'blocks': 15, 'channels': 128, 'in_channels': 6}
model_cfg_dual = {'kind': 'dual', 'blocks': 15, 'channels': 128, 'in_channels': 6}

EVAL_GAMES = 50  # Start with 50 for faster test
MCTS_SIMS = 100

print(f"Configuration:")
print(f"  Single: epoch_200.pt (200 epochs)")
print(f"  Dual:   epoch_100.pt (100 epochs)")
print(f"  Games: {EVAL_GAMES}")
print(f"  MCTS Sims: {MCTS_SIMS}")
print()

print("Loading Single model...")
single_model = build_model(**model_cfg_single)
single_ckpt = torch.load(SINGLE_CKPT, map_location=device)
single_model.load_state_dict(single_ckpt['model'])
single_model = single_model.to(device)
print(f"Single loaded (4.53M params)")

print("Loading Dual model...")
dual_model = build_model(**model_cfg_dual)
dual_ckpt = torch.load(DUAL_CKPT, map_location=device)
dual_model.load_state_dict(dual_ckpt['model'])
dual_model = dual_model.to(device)
print(f"Dual loaded (4.58M params)")
print()

print("Starting evaluation...")
print(f"This will take approximately {EVAL_GAMES * 2} minutes on GPU")
print()

try:
    results = evaluate_models(
        single_model,
        dual_model,
        model_cfg_single,
        model_cfg2=model_cfg_dual,
        num_games=EVAL_GAMES,
        mcts_sims=MCTS_SIMS,
        c_puct=1.5,
        device=device,
        verbose=True
    )
    
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print()
    print(f"Total Games: {results['total_games']}")
    print()
    print(f"Win Rate:")
    print(f"  Single:  {results['model1_wins']:>3}/{results['total_games']} ({results['model1_win_rate']:.1%})")
    print(f"  Dual:    {results['model2_wins']:>3}/{results['total_games']} ({results['model2_wins']/results['total_games']:.1%})")
    print(f"  Draws:   {results['draws']:>3}/{results['total_games']} ({results['draws']/results['total_games']:.1%})")
    print()
    print(f"Color Performance (Single):")
    print(f"  As Black: {results['model1_as_black_wins']}/{results['total_games']//2} wins")
    print(f"  As White: {results['model1_as_white_wins']}/{results['total_games']//2} wins")
    print()
    print("=" * 70)
    print()
    
    # Verdict
    win_rate = results['model1_win_rate']
    if win_rate > 0.55:
        print("WINNER: Single! (>55% win rate)")
        print(f"   Single clearly outperforms Dual")
    elif win_rate < 0.45:
        print("WINNER: Dual! (<45% win rate)")
        print(f"Dual clearly outperforms Single")
    else:
        print("RESULT: Statistical Draw (45-55% range)")
        print(f"Both models perform similarly")
    
    print()
    print("=" * 70)
    
    # Save results
    import json
    with open('/app/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to: /app/evaluation_results.json")
    
except KeyboardInterrupt:
    print("\n\nEvaluation interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n\nError during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


