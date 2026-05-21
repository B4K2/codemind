import os
import torch
from safetensors.torch import save_file, load_file

def save_checkpoint(model, optimizer, step, tokens_seen=0, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"step_{step:06d}")
    os.makedirs(path, exist_ok=True)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state = {k: v.contiguous().cpu() for k, v in raw_model.state_dict().items()}
    save_file(state, os.path.join(path, "model.safetensors"))

    torch.save({
        "optimizer":   optimizer.state_dict(),
        "step":        step,
        "tokens_seen": tokens_seen,
    }, os.path.join(path, "optimizer.pt"))

    print(f"Checkpoint saved at step {step} ({tokens_seen/1e6:.1f}M tokens) -> {path}")


def load_checkpoint(model, optimizer, save_dir="checkpoints", step=None):
    if step is None:
        checkpoints = sorted([
            d for d in os.listdir(save_dir)
            if d.startswith("step_") and os.path.isdir(os.path.join(save_dir, d))
        ])
        if not checkpoints:
            print("No checkpoints found — starting from scratch.")
            return 0, 0
        latest = checkpoints[-1]
        step   = int(latest.split("_")[1])

    path      = os.path.join(save_dir, f"step_{step:06d}")
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    weights = load_file(os.path.join(path, "model.safetensors"))
    raw_model.load_state_dict(weights)

    opt_state    = torch.load(os.path.join(path, "optimizer.pt"), map_location="cpu")
    optimizer.load_state_dict(opt_state["optimizer"])
    resumed_step = opt_state["step"]
    tokens_seen  = opt_state.get("tokens_seen", 0)

    return resumed_step, tokens_seen