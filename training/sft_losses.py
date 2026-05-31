import torch
import torch.nn.functional as F


def calculate_sft_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, S, V = logits.shape

    shift_logits  = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    shift_mask    = loss_mask[:, :-1].contiguous()

    flat_logits  = shift_logits.view(-1, V)
    flat_targets = shift_targets.view(-1)
    flat_mask    = shift_mask.view(-1).float()

    n_assistant_tokens = flat_mask.sum().clamp(min=1)

    # Collect chunk losses as a list — keeps grad_fn intact
    chunk_losses = []

    for start in range(0, flat_logits.size(0), chunk_size):
        end = min(start + chunk_size, flat_logits.size(0))

        chunk_logits  = flat_logits[start:end].float()
        chunk_targets = flat_targets[start:end]
        chunk_mask    = flat_mask[start:end]

        if not chunk_mask.any():
            continue

        chunk_loss = F.cross_entropy(
            chunk_logits,
            chunk_targets,
            reduction="none",
        )  # [chunk] — has grad_fn ✅

        # Mask and sum — grad_fn preserved
        chunk_losses.append((chunk_loss * chunk_mask).sum())

    if not chunk_losses:
        # All tokens are prompt tokens — no assistant tokens in this batch
        # Return zero loss WITH grad_fn by using logits
        return logits.sum() * 0.0

    # Stack and sum — single tensor with grad_fn ✅
    total_loss = torch.stack(chunk_losses).sum()
    return total_loss / n_assistant_tokens