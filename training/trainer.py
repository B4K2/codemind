import os
import time
import torch
import wandb
from tqdm import tqdm

from config.model_config import CodeMindConfig
from tokenizer.tokenizer import CodeMindTokenizer
from data.dataloader import get_dataloader
from model.codemind import CodeMindSLM
from training.optimizer import setup_optimizer
from training.scheduler import CodeMindLRScheduler
from training.losses import calculate_ntp_loss
from utils.checkpoint import save_checkpoint, load_checkpoint

# trainer.py — full updated trainer with tokens_seen

MAX_STEPS        = 10000
GRAD_ACCUM_STEPS = 2          # adjusted for cloud batch size
LOG_FREQ         = 10
SAVE_FREQ        = 500
MAX_GRAD_NORM    = 1.0

def main():
    wandb.init(project="CodeMind-SLM", name="nano-a40-run1")

    config    = CodeMindConfig()
    tokenizer = CodeMindTokenizer()
    wandb.config.update(config.__dict__)

    print("Loading CodeMindSLM...")
    model = CodeMindSLM(config).to("cuda").to(torch.bfloat16)
    model.use_gradient_checkpointing = True
    model = torch.compile(model, mode="default")
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {num_params:.1f}M")

    optimizer = setup_optimizer(model, muon_lr=0.02, adam_lr=3e-4)
    scheduler = CodeMindLRScheduler(optimizer, warmup_steps=200, total_steps=MAX_STEPS)

    # ── Resume ────────────────────────────────────────────────────────────────
    RESUME = True
    if RESUME and os.path.exists("checkpoints"):
        current_step, tokens_seen = load_checkpoint(model, optimizer, save_dir="checkpoints")
        for _ in range(current_step):
            scheduler.step()
        print(f"Resumed from step {current_step} ({tokens_seen/1e6:.1f}M tokens seen)")
    else:
        current_step = 0
        tokens_seen  = 0          

    dataloader = get_dataloader(config, tokenizer,
                                skip_batches=current_step * GRAD_ACCUM_STEPS)
    model.train()
    optimizer.zero_grad()

    tokens_per_step = config.max_batch_size * config.max_seq_len * GRAD_ACCUM_STEPS

    accumulated_loss = 0.0
    t0 = time.perf_counter()

    print("\n🚀 Starting Training...\n")

    try:
        for batch_idx, batch in enumerate(dataloader):
            if current_step >= MAX_STEPS:
                break

            inputs  = batch[:, :-1].to("cuda", non_blocking=True)
            targets = batch[:, 1:].to("cuda",  non_blocking=True)

            logits, all_router_logits = model(inputs)
            loss, ntp_loss, lb_loss, z_loss = calculate_ntp_loss(
                logits, targets,
                all_router_logits = all_router_logits,
                n_experts         = config.n_routed_experts,
                lb_weight         = 0.05,
                z_weight          = 0.001,
            )

            (loss / GRAD_ACCUM_STEPS).backward()
            accumulated_loss += ntp_loss.item()

            tokens_seen += config.max_batch_size * config.max_seq_len

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                current_lr = scheduler.get_last_lr()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                current_step += 1

                if current_step % LOG_FREQ == 0:
                    t1          = time.perf_counter()
                    avg_loss    = accumulated_loss / (LOG_FREQ * GRAD_ACCUM_STEPS)
                    tok_per_sec = (tokens_per_step * LOG_FREQ) / (t1 - t0)
                    perplexity  = torch.exp(torch.tensor(avg_loss)).item()

                    # Human readable token count
                    if tokens_seen < 1e6:
                        tok_str = f"{tokens_seen/1e3:.1f}K"
                    elif tokens_seen < 1e9:
                        tok_str = f"{tokens_seen/1e6:.2f}M"
                    else:
                        tok_str = f"{tokens_seen/1e9:.3f}B"

                    print(
                        f"Step {current_step:5d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"PPL: {perplexity:.1f} | "
                        f"Tok/s: {tok_per_sec:,.0f} | "
                        f"Tokens: {tok_str} | "       
                        f"LR: {current_lr[0]:.5f}"
                    )
                    wandb.log({
                        "train/loss":        avg_loss,
                        "train/perplexity":  perplexity,
                        "train/tok_per_sec": tok_per_sec,
                        "train/tokens_seen": tokens_seen,   
                        "train/lb_loss":     lb_loss.item() if lb_loss is not None else 0,
                        "train/z_loss":      z_loss.item()  if z_loss  is not None else 0,
                        "train/muon_lr":     current_lr[0],
                        "step":              current_step,
                    })
                    accumulated_loss = 0.0
                    t0 = t1

                if current_step % SAVE_FREQ == 0:
                    save_checkpoint(model, optimizer, current_step,
                                    tokens_seen=tokens_seen)  # ← save tokens too

    except KeyboardInterrupt:
        print("\n🛑 Interrupted — saving emergency checkpoint...")
        save_checkpoint(model, optimizer, current_step,
                        tokens_seen=tokens_seen,
                        save_dir="checkpoints/emergency")
    except Exception:
        import traceback
        print("\n❌ CRASH:")
        traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()