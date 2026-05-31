import os
import time
import torch
import wandb
import argparse

from config.model_config import CodeMindConfig
from tokenizer.tokenizer import CodeMindTokenizer
from data.sft_dataset import SFTDataset
from model.codemind import CodeMindSLM
from training.optimizer import setup_optimizer
from training.scheduler import CodeMindLRScheduler
from training.sft_losses import calculate_sft_loss
from utils.checkpoint import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader

# ── SFT Hyperparameters ───────────────────────────────────────────────────────
MAX_STEPS        = 3000       # SFT needs far fewer steps than pre-training
GRAD_ACCUM_STEPS = 8
LOG_FREQ         = 10
SAVE_FREQ        = 500
MAX_GRAD_NORM    = 0.3
MAX_SEQ_LEN      = 512      # can use full length for SFT
BATCH_SIZE       = 1         # per GPU — SFT examples are longer than pretraining

def str2bool(v):
    if isinstance(v, bool): return v
    return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",       type=str2bool, default=False)
    parser.add_argument("--base_ckpt",    type=str,      default=None,
                        help="Path to pre-trained checkpoint dir to start from")
    parser.add_argument("--max_samples",  type=int,      default=None,
                        help="Limit dataset size for testing")
    args = parser.parse_args()

    wandb.init(project="CodeMind-SLM", name="1B-SFT-Truerun2")

    # ── Config ────────────────────────────────────────────────────────────────
    config = CodeMindConfig(
        max_batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )
    tokenizer = CodeMindTokenizer()
    wandb.config.update({**config.__dict__, "phase": "SFT"})

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading CodeMindSLM for SFT...")
    torch.set_float32_matmul_precision('high')
    model = CodeMindSLM(config).to("cuda").to(torch.bfloat16)
    model.use_gradient_checkpointing = True
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {num_params:.1f}M")

    # ── Optimizer — much lower LR than pre-training ───────────────────────────
    # SFT needs gentle updates — too high LR causes catastrophic forgetting
    optimizer = setup_optimizer(
        model,
        muon_lr=0.0005,    # was 0.008 in pre-training — 4x lower
        adam_lr=3e-5,     # was 3e-4 — 10x lower
    )
    scheduler = CodeMindLRScheduler(
        optimizer,
        warmup_steps=100,         # shorter warmup for SFT
        total_steps=MAX_STEPS,
    )

    # ── Load base checkpoint ──────────────────────────────────────────────────
    current_step = 0
    tokens_seen  = 0

    if args.base_ckpt and not args.resume:
        # Load pre-trained weights as starting point
        print(f"Loading pre-trained weights from: {args.base_ckpt}")
        from safetensors.torch import load_file
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        weights = load_file(os.path.join(args.base_ckpt, "model.safetensors"))
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
        raw_model.load_state_dict(cleaned, strict=False)
        print("Pre-trained weights loaded ✅")

    elif args.resume and os.path.exists("checkpoints_sft"):
        current_step, tokens_seen = load_checkpoint(
            model, optimizer, save_dir="checkpoints_sft"
        )
        for _ in range(current_step):
            scheduler.step()
        print(f"Resumed SFT from step {current_step}")

    # Compile AFTER loading weights
    model = torch.compile(model, mode="default")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Building SFT dataset...")
    dataset = SFTDataset(
        tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,           # SFT uses shuffle — dataset fits in RAM
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    accumulated_loss = 0.0
    tokens_per_step  = BATCH_SIZE * MAX_SEQ_LEN * GRAD_ACCUM_STEPS
    t0 = time.perf_counter()

    print(f"\n🚀 Starting SFT Training ({len(dataset)} examples)...\n")

    try:
        epoch = 0
        while current_step < MAX_STEPS:
            epoch += 1
            for batch_idx, (input_ids, loss_mask) in enumerate(dataloader):
                if current_step >= MAX_STEPS:
                    break

                # SFT input — full sequence (no shift here, loss handles it)
                input_ids = input_ids.to("cuda", non_blocking=True)  # [B, S]
                loss_mask = loss_mask.to("cuda", non_blocking=True)  # [B, S]

                # For model input we use all but last token
                # targets are all but first token (standard LM shift)
                inputs  = input_ids[:, :-1]    # [B, S-1]
                targets = input_ids[:, 1:]     # [B, S-1]
                mask    = loss_mask[:, :-1]    # [B, S-1] — align with inputs

                # Forward — no MTP during SFT
                logits, _, _ = model(inputs, return_mtp=False)

                # Loss only on assistant tokens
                loss = calculate_sft_loss(logits, targets, mask)
                (loss / GRAD_ACCUM_STEPS).backward()

                accumulated_loss += loss.item()
                tokens_seen += BATCH_SIZE * MAX_SEQ_LEN

                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                    grad_norm  = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), MAX_GRAD_NORM
                    )
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

                        print(
                            f"Step {current_step:5d} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"PPL: {perplexity:.1f} | "
                            f"Tok/s: {tok_per_sec:,.0f} | "
                            f"Epoch: {epoch} | "
                            f"LR: {current_lr[0]:.6f}"
                        )
                        wandb.log({
                            "sft/loss":       avg_loss,
                            "sft/perplexity": perplexity,
                            "sft/tok_per_sec": tok_per_sec,
                            "sft/grad_norm":  grad_norm.item(),
                            "sft/muon_lr":    current_lr[0],
                            "step":           current_step,
                        })
                        accumulated_loss = 0.0
                        t0 = t1

                    if current_step % SAVE_FREQ == 0:
                        save_checkpoint(
                            model, optimizer, current_step,
                            tokens_seen=tokens_seen,
                            save_dir="checkpoints_sft",
                        )

    except KeyboardInterrupt:
        print("\n🛑 Interrupted — saving checkpoint...")
        save_checkpoint(model, optimizer, current_step,
                        tokens_seen=tokens_seen,
                        save_dir="checkpoints_sft")
    except Exception:
        import traceback
        print("\n❌ CRASH:")
        traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()