from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not installed. Logging disabled. Install: pip install wandb")

from models import build_model
from selfplay import SelfPlayPool
from evaluate import evaluate_models
from augmentation import augment_single_samples, augment_dual_samples, get_augmentation_stats

class SingleDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return (
            torch.from_numpy(s["obs"]).float(),
            torch.from_numpy(s["pi"]).float(),
            torch.tensor(s["z"]).float(),
            torch.from_numpy(s["mask"]).bool(),
        )

class DualDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return (
            torch.from_numpy(s["obs"]).float(),
            torch.from_numpy(s["cond_first"]).float(),
            torch.from_numpy(s["pi1"]).float(),
            torch.from_numpy(s["pi2"]).float(),
            torch.tensor(s["z"]).float(),
            torch.from_numpy(s["mask1"]).bool(),
            torch.from_numpy(s["mask2"]).bool(),
        )

class RingBuffer:
    def __init__(self, cap):
        self.cap, self.data, self.ptr = int(cap), [], 0
    def push_many(self, xs):
        for s in xs:
            if len(self.data) < self.cap: self.data.append(s)
            else:
                self.data[self.ptr] = s
                self.ptr = (self.ptr + 1) % self.cap
    def __len__(self): return len(self.data)
    def sample(self, n):
        import random
        n = min(n, len(self.data))
        return random.sample(self.data, n)

@torch.no_grad()
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def policy_ce(logits, target_prob):
    logp = torch.log_softmax(logits, dim=-1)
    return -(target_prob * logp).sum(dim=-1).mean()


def compute_explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    
    var_y = np.var(y_true_np)
    if var_y < 1e-8:
        return 0.0
    
    return float(1.0 - np.var(y_true_np - y_pred_np) / var_y)


def compute_kl_divergence(old_logits: torch.Tensor, new_logits: torch.Tensor) -> float:
    
    old_probs = F.softmax(old_logits, dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    new_log_probs = F.log_softmax(new_logits, dim=-1)
    
    kl = torch.sum(old_probs * (old_log_probs - new_log_probs), dim=-1).mean()
    return float(kl.detach().cpu())


def compute_policy_entropy(logits: torch.Tensor) -> float:
    
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
    return float(entropy.detach().cpu())


def compute_gradient_norm(model: torch.nn.Module) -> float:
    
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def train_single_epoch(model, opt, buffer, batch_size, num_steps, device):
    model.train()
    totals = {"p":0.0, "v":0.0, "loss":0.0, "explained_var":0.0, "policy_entropy":0.0, "grad_norm":0.0, "n":0}
    
    for step in range(num_steps):
        batch_samples = buffer.sample(min(batch_size, len(buffer)))
        loader = DataLoader(SingleDataset(batch_samples), batch_size=batch_size, shuffle=True)
        
        for obs, pi, z, mask in loader:
            obs, pi, z, mask = obs.to(device), pi.to(device), z.to(device), mask.to(device)
            out = model(obs, mask=mask)
            p = policy_ce(out["pi"], pi)
            v = F.mse_loss(out["v"], z)
            loss = p + v
            
            explained_var = compute_explained_variance(z, out["v"])
            
            policy_entropy = compute_policy_entropy(out["pi"])
            
            opt.zero_grad(set_to_none=True); loss.backward()
            
            grad_norm = compute_gradient_norm(model)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            totals["p"] += float(p.detach().cpu())
            totals["v"] += float(v.detach().cpu())
            totals["loss"] += float(loss.detach().cpu())
            totals["explained_var"] += explained_var
            totals["policy_entropy"] += policy_entropy
            totals["grad_norm"] += grad_norm
            totals["n"] += 1
    
    for k in ("p","v","loss","explained_var","policy_entropy","grad_norm"): 
        totals[k] /= max(1, totals["n"])
    return totals


def train_dual_epoch(model, opt, buffer, batch_size, num_steps, device):
    model.train()
    totals = {"p1":0.0, "p2":0.0, "v":0.0, "loss":0.0, "explained_var":0.0, "policy_entropy":0.0, "grad_norm":0.0, "n":0}
    
    for step in range(num_steps):
        batch_samples = buffer.sample(min(batch_size, len(buffer)))
        loader = DataLoader(DualDataset(batch_samples), batch_size=batch_size, shuffle=True)
        
        for obs, cond_first, pi1, pi2, z, mask1, mask2 in loader:
            obs, cond_first = obs.to(device), cond_first.to(device)
            pi1, pi2, z = pi1.to(device), pi2.to(device), z.to(device)
            mask1, mask2 = mask1.to(device), mask2.to(device)
            out = model(obs, cond_first=cond_first, mask1=mask1, mask2=mask2)
            p1 = policy_ce(out["pi1"], pi1)
            p2 = policy_ce(out["pi2"], pi2)
            v  = F.mse_loss(out["v"], z)
            loss = p1 + p2 + v
            
            explained_var = compute_explained_variance(z, out["v"])
            
            entropy1 = compute_policy_entropy(out["pi1"])
            entropy2 = compute_policy_entropy(out["pi2"])
            policy_entropy = (entropy1 + entropy2) / 2.0
            
            opt.zero_grad(set_to_none=True); loss.backward()
            
            grad_norm = compute_gradient_norm(model)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            totals["p1"] += float(p1.detach().cpu())
            totals["p2"] += float(p2.detach().cpu())
            totals["v"] += float(v.detach().cpu())
            totals["loss"] += float(loss.detach().cpu())
            totals["explained_var"] += explained_var
            totals["policy_entropy"] += policy_entropy
            totals["grad_norm"] += grad_norm
            totals["n"] += 1
    
    for k in ("p1","p2","v","loss","explained_var","policy_entropy","grad_norm"): 
        totals[k] /= max(1, totals["n"])
    return totals


def main(args):
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(kind=args.model_kind, blocks=args.model_blocks, channels=args.model_channels, in_channels=6, board_size=args.board_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    start_epoch = 1
    best_win_rate = 0.0
    best_checkpoint_path = None
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"[resume] Loading checkpoint from {resume_path}", flush=True)
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            
            if "optimizer" in checkpoint:
                opt.load_state_dict(checkpoint["optimizer"])
                print(f"[resume] Optimizer state restored", flush=True)
            
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                print(f"[resume] Resuming from epoch {start_epoch}", flush=True)
            
            if "best_win_rate" in checkpoint:
                best_win_rate = checkpoint["best_win_rate"]
                print(f"[resume] Best win rate so far: {best_win_rate:.1%}", flush=True)
            
            print(f"[resume] ✅ Checkpoint loaded successfully!", flush=True)
        else:
            print(f"[resume] ⚠️  Checkpoint not found: {resume_path}, starting from scratch", flush=True)

    if WANDB_AVAILABLE:
        run = wandb.init(project=args.project, name=args.run_name, config={
            "model.kind": args.model_kind,
            "model.blocks": args.model_blocks,
            "model.channels": args.model_channels,
            "mcts.sims": args.mcts_sims,
            "c_puct": args.c_puct,
            "temperature": args.temperature,
            "buffer.cap": args.buffer_cap,
            "buffer.min": args.min_buffer,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "devices": args.devices,
            "max_blocking": args.max_blocking,
        })
        wandb.watch(model, log="all", log_freq=200)
    else:
        run = None

    pool = SelfPlayPool(
        model_cfg={"kind": args.model_kind, "blocks": args.model_blocks, "channels": args.model_channels, "in_channels": 6},
        sims=args.mcts_sims, c_puct=args.c_puct, num_workers=args.selfplay_workers,
        devices=args.devices or ["cpu"], temperature=args.temperature,
        board_size=args.board_size,  
        max_blocking=args.max_blocking, step_penalty=args.step_penalty,
        seed=args.seed, state_dict=model.state_dict(),
        dirichlet_alpha=args.dirichlet_alpha, dirichlet_eps=args.dirichlet_eps,
        temp_schedule_type=args.temp_schedule,
        temp_threshold_turn=args.temp_threshold,
        temp_high=args.temp_high,
        temp_low=args.temp_low,
    )

    rb_single, rb_dual = RingBuffer(args.buffer_cap), RingBuffer(args.buffer_cap)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("[init] params=%.2fM | devices=%s | workers=%d" % (count_params(model)/1e6, args.devices, args.selfplay_workers), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.time()

        pool.update_weights(model.state_dict())
        
        print(f"[ep {epoch}] self-play start | games={args.selfplay_games} | sims={args.mcts_sims} | workers={args.selfplay_workers}", flush=True)
        t0 = time.time(); result = pool.play(args.selfplay_games); sp_time = time.time() - t0

        if result["kind"] == "single":
            original_samples = result["single"]
            augmented_samples = augment_single_samples(original_samples, board_size=args.board_size)
            rb_single.push_many(augmented_samples)
            num_samples = len(augmented_samples)
            aug_stats = get_augmentation_stats(len(original_samples), len(augmented_samples))
            print(f"[ep {epoch}] augmentation | {aug_stats['original_samples']} → {aug_stats['augmented_samples']} samples (×{aug_stats['multiplier']:.1f})", flush=True)
        else:
            original_samples = result["turn"]
            augmented_samples = augment_dual_samples(original_samples, board_size=args.board_size)
            rb_dual.push_many(augmented_samples)
            num_samples = len(augmented_samples)
            aug_stats = get_augmentation_stats(len(original_samples), len(augmented_samples))
            print(f"[ep {epoch}] augmentation | {aug_stats['original_samples']} → {aug_stats['augmented_samples']} samples (×{aug_stats['multiplier']:.1f})", flush=True)
        print(f"[ep {epoch}] self-play done | +{num_samples} samples | buffer.single={len(rb_single)} | buffer.dual={len(rb_dual)} | time={sp_time:.1f}s", flush=True)
        
        avg_game_length = result.get("avg_game_length", 0)
        draw_rate = result.get("draw_rate", 0)
        black_win_rate = result.get("black_win_rate", 0)
        print(f"[ep {epoch}] game stats | avg_length={avg_game_length:.1f} | draw_rate={draw_rate:.1%} | black_win={black_win_rate:.1%}", flush=True)
        
        if WANDB_AVAILABLE:
            wandb.log({
                "buffer.single": len(rb_single), 
                "buffer.dual": len(rb_dual), 
                "time/selfplay_s": sp_time, 
                "augmentation/multiplier": aug_stats.get('multiplier', 0),
                "augmentation/samples": num_samples,
                "selfplay/avg_game_length": avg_game_length,
                "selfplay/draw_rate": draw_rate,
                "selfplay/black_win_rate": black_win_rate,
                "epoch": epoch
            })

        if args.model_kind == "single":
            cur_len = len(rb_single)
            if cur_len < args.min_buffer:
                print(f"[ep {epoch}] WARMUP {cur_len}/{args.min_buffer} (collecting...)", flush=True)
                continue
            print(f"[ep {epoch}] training start | kind=single | buffer={cur_len} | batch_size={args.batch} | train_steps={args.train_steps_per_epoch}", flush=True)
            t0 = time.time()
            totals = train_single_epoch(model, opt, rb_single, args.batch, args.train_steps_per_epoch, device)
            train_time = time.time() - t0
            if WANDB_AVAILABLE:
                wandb.log({
                    "loss/p": totals["p"], 
                    "loss/v": totals["v"], 
                    "loss/total": totals["loss"], 
                    "metrics/explained_variance": totals["explained_var"],
                    "metrics/policy_entropy": totals["policy_entropy"],
                    "metrics/grad_norm": totals["grad_norm"],
                    "time/train_s": train_time, 
                    "epoch": epoch
                })
            print(f"[ep {epoch}] loss: p={totals['p']:.4f} v={totals['v']:.4f} total={totals['loss']:.4f} | explained_var={totals['explained_var']:.3f} | entropy={totals['policy_entropy']:.3f} | grad_norm={totals['grad_norm']:.2f} | time={train_time:.1f}s", flush=True)
        else:
            cur_len = len(rb_dual)
            if cur_len < args.min_buffer:
                print(f"[ep {epoch}] WARMUP {cur_len}/{args.min_buffer} (collecting...)", flush=True)
                continue
            print(f"[ep {epoch}] training start | kind=dual | buffer={cur_len} | batch_size={args.batch} | train_steps={args.train_steps_per_epoch}", flush=True)
            t0 = time.time()
            totals = train_dual_epoch(model, opt, rb_dual, args.batch, args.train_steps_per_epoch, device)
            train_time = time.time() - t0
            if WANDB_AVAILABLE:
                wandb.log({
                    "loss/p1": totals.get("p1",0), 
                    "loss/p2": totals.get("p2",0), 
                    "loss/v": totals["v"], 
                    "loss/total": totals["loss"], 
                    "metrics/explained_variance": totals["explained_var"],
                    "metrics/policy_entropy": totals["policy_entropy"],
                    "metrics/grad_norm": totals["grad_norm"],
                    "time/train_s": train_time, 
                    "epoch": epoch
                })
            print(f"[ep {epoch}] loss: p1={totals.get('p1',0):.4f} p2={totals.get('p2',0):.4f} v={totals['v']:.4f} total={totals['loss']:.4f} | explained_var={totals['explained_var']:.3f} | entropy={totals['policy_entropy']:.3f} | grad_norm={totals['grad_norm']:.2f} | time={train_time:.1f}s", flush=True)

        if epoch % args.ckpt_interval == 0:
            path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "best_win_rate": best_win_rate,
                "args": vars(args)
            }, path)
            if WANDB_AVAILABLE:
                try:
                    art = wandb.Artifact(f"sixmoku-{args.model_kind}-ckpt", type="model")
                    art.add_file(str(path)); wandb.log_artifact(art)
                    print(f"[ep {epoch}] checkpoint saved → {path} (artifact uploaded)", flush=True)
                except Exception as e:
                    print(f"[ep {epoch}] checkpoint saved → {path} (artifact upload failed: {e})", flush=True)
            else:
                print(f"[ep {epoch}] checkpoint saved → {path}", flush=True)

        if epoch % args.eval_interval == 0 and epoch >= args.ckpt_interval:
            prev_epoch = epoch - args.ckpt_interval
            prev_ckpt_path = ckpt_dir / f"epoch_{prev_epoch}.pt"
            
            if prev_ckpt_path.exists():
                print(f"[ep {epoch}] evaluation start | comparing to epoch {prev_epoch} | games={args.eval_games}", flush=True)
                t0 = time.time()
                
                try:
                    eval_model_current = build_model(
                        kind=args.model_kind,
                        blocks=args.model_blocks,
                        channels=args.model_channels,
                        board_size=args.board_size,
                        in_channels=6
                    )
                    eval_model_current.load_state_dict(model.state_dict())
                    eval_model_current.to(device)
                    eval_model_current.eval()
                    
                    prev_ckpt = torch.load(prev_ckpt_path, map_location=device)
                    eval_model_prev = build_model(
                        kind=args.model_kind,
                        blocks=args.model_blocks,
                        channels=args.model_channels,
                        board_size=args.board_size,
                        in_channels=6
                    )
                    eval_model_prev.load_state_dict(prev_ckpt["model"])
                    eval_model_prev.to(device)
                    eval_model_prev.eval()
                    
                    model_cfg = {"kind": args.model_kind, "blocks": args.model_blocks, "channels": args.model_channels}
                    eval_mcts_sims = getattr(args, 'eval_mcts_sims', args.mcts_sims * 4)
                    eval_results = evaluate_models(
                        eval_model_current,
                        eval_model_prev,
                        model_cfg,
                        num_games=args.eval_games,
                        mcts_sims=eval_mcts_sims,
                        c_puct=args.c_puct,
                        board_size=args.board_size,  
                        device=str(device),
                        verbose=False,
                    )
                    
                    eval_time = time.time() - t0
                    win_rate = eval_results["model1_win_rate"]
                    
                    if WANDB_AVAILABLE:
                        wandb.log({
                            "eval/win_rate": win_rate,
                            "eval/wins": eval_results["model1_wins"],
                            "eval/losses": eval_results["model2_wins"],
                            "eval/draws": eval_results["draws"],
                            "eval/as_black_wins": eval_results["model1_as_black_wins"],
                            "eval/as_white_wins": eval_results["model1_as_white_wins"],
                            "time/eval_s": eval_time,
                            "epoch": epoch,
                        })
                    
                    print(f"[ep {epoch}] evaluation done | win_rate={win_rate:.1%} ({eval_results['model1_wins']}/{eval_results['total_games']}) | time={eval_time:.1f}s", flush=True)
                    
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_checkpoint_path = str(path)
                        best_path = ckpt_dir / "best_model.pt"
                        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "win_rate": win_rate}, best_path)
                        print(f"[ep {epoch}] NEW BEST MODEL! win_rate={win_rate:.1%} → saved to {best_path}", flush=True)
                        if WANDB_AVAILABLE:
                            wandb.log({"eval/best_win_rate": best_win_rate, "eval/best_epoch": epoch})
                    
                    del eval_model_current, eval_model_prev
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"[ep {epoch}] evaluation failed: {e}", flush=True)
            else:
                print(f"[ep {epoch}] evaluation skipped (no previous checkpoint found)", flush=True)

        if WANDB_AVAILABLE:
            wandb.log({"time/epoch_s": time.time() - t_ep, "params": count_params(model)})

    if WANDB_AVAILABLE and run is not None:
        run.finish()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path (e.g., ckpts/fast/epoch_50.pt)")
    p.add_argument("--model.kind", dest="model_kind", choices=["single","dual"], required=True)
    p.add_argument("--model.blocks", dest="model_blocks", type=int, default=12)
    p.add_argument("--model.channels", dest="model_channels", type=int, default=96)
    p.add_argument("--selfplay.games", dest="selfplay_games", type=int, default=64)
    p.add_argument("--selfplay.workers", dest="selfplay_workers", type=int, default=12)
    p.add_argument("--mcts.sims", dest="mcts_sims", type=int, default=400)
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temp_schedule", type=str, default="constant", 
                   choices=["constant", "turn_based", "move_based"],
                   help="Temperature schedule type")
    p.add_argument("--temp_threshold", type=int, default=15,
                   help="Turn threshold for temperature schedule")
    p.add_argument("--temp_high", type=float, default=1.0,
                   help="High temperature for exploration")
    p.add_argument("--temp_low", type=float, default=0.1,
                   help="Low temperature for exploitation")
    p.add_argument("--dirichlet.alpha", dest="dirichlet_alpha", type=float, default=0.3)
    p.add_argument("--dirichlet.eps", dest="dirichlet_eps", type=float, default=0.25)
    p.add_argument("--board_size", type=int, default=11, help="Board size (e.g., 11)")
    p.add_argument("--max_blocking", type=int, default=4)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--train.steps", dest="train_steps_per_epoch", type=int, default=50, help="Number of training steps per epoch")
    p.add_argument("--buffer.cap", dest="buffer_cap", type=int, default=200000)
    p.add_argument("--buffer.min", dest="min_buffer", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--devices", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--project", type=str, default="sixmoku")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--ckpt.dir", dest="ckpt_dir", type=str, default="ckpts/default")
    p.add_argument("--ckpt.interval", dest="ckpt_interval", type=int, default=50)
    p.add_argument("--eval.interval", dest="eval_interval", type=int, default=50, help="Evaluation interval (epochs)")
    p.add_argument("--eval.games", dest="eval_games", type=int, default=20, help="Number of games for evaluation")
    p.add_argument("--eval.mcts_sims", dest="eval_mcts_sims", type=int, default=None, help="MCTS sims for evaluation (default: 4x selfplay)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0); np.random.seed(0)
    main(args)
