import math
import torch
import torch.nn.functional as F
from typing import Optional

def calculate_mtp_loss(
    mtp_logits: list,       # list of [B, S, V] tensors, one per head
    targets: torch.Tensor,  # [B, S] — the ORIGINAL shifted targets (predicting t+1)
    mtp_weight: float = 0.1,
) -> torch.Tensor:
    """
    Each MTP head k predicts tokens at offset k+2 from input position.
    
    head 0 (offset=0): predicts t+2  → targets shifted by +1 from main targets
    head 1 (offset=1): predicts t+3  → targets shifted by +2 from main targets
    """
    if not mtp_logits:
        return torch.tensor(0.0, device=targets.device)

    B, S    = targets.shape
    total   = torch.tensor(0.0, device=targets.device)

    for k, head_logits in enumerate(mtp_logits):
        extra_shift = k + 1
        shifted_targets = torch.roll(targets, shifts=-extra_shift, dims=1)

        valid_len = S - extra_shift
        valid_logits  = head_logits[:, :valid_len, :].contiguous()   
        valid_targets = shifted_targets[:, :valid_len].contiguous()   

        V = valid_logits.size(-1)
        head_loss = chunked_cross_entropy(
            valid_logits.view(-1, V),
            valid_targets.view(-1),
        )
        total = total + head_loss

    return mtp_weight * (total / len(mtp_logits))

def calculate_z_loss(all_router_logits, z_weight=0.001):
    total = torch.tensor(0.0, device=all_router_logits[0].device)
    for router_logits in all_router_logits:
        clamped = router_logits.clamp(-10, 10)
        z_loss  = torch.logsumexp(clamped, dim=-1).pow(2).mean()
        total   = total + z_loss
    return total / len(all_router_logits)

def chunked_cross_entropy(logits, targets, chunk_size=256):
    N, V    = logits.shape
    total   = torch.zeros(1, device=logits.device, dtype=torch.float32)
    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        chunk = F.cross_entropy(logits[start:end].float(), targets[start:end], reduction="sum")
        total = total + chunk
    return total / N


def calculate_load_balance_loss(all_router_logits, n_experts):
    total = torch.tensor(0.0, device=all_router_logits[0].device)
    for router_logits in all_router_logits:
        probs        = torch.softmax(router_logits, dim=-1)  
        expert_usage = probs.mean(dim=0)                     
        entropy = -(expert_usage * (expert_usage + 1e-9).log()).sum()
        max_entropy = math.log(n_experts)
        lb_loss = 1.0 - (entropy / max_entropy)
        total   = total + lb_loss
    return total / len(all_router_logits)


def calculate_ntp_loss(logits, targets, all_router_logits=None, mtp_logits=None,
                       n_experts=6, lb_weight=0.05, z_weight=0.001, mtp_weight=0.1):
    B, S, V  = logits.shape
    ntp_loss = chunked_cross_entropy(
        logits.contiguous().view(-1, V),
        targets.contiguous().view(-1),
    )
    total_loss = ntp_loss
    lb_loss    = None    # ← initialise
    z_loss     = None    # ← initialise
    mtp_loss   = None    # ← initialise

    if all_router_logits:
        lb_loss     = calculate_load_balance_loss(all_router_logits, n_experts)
        z_loss      = calculate_z_loss(all_router_logits, z_weight)
        total_loss  = total_loss + lb_weight * lb_loss + z_weight * z_loss

    if mtp_logits:
        mtp_loss   = calculate_mtp_loss(mtp_logits, targets, mtp_weight)
        total_loss = total_loss + mtp_loss

    return total_loss, ntp_loss.detach(), lb_loss, z_loss, mtp_loss