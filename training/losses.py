import torch
import torch.nn.functional as F
from typing import Optional

def calculate_z_loss(all_router_logits, z_weight=0.001):
    total = torch.tensor(0.0, device=all_router_logits[0].device)
    for router_logits in all_router_logits:
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        total  = total + z_loss
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
        probs        = torch.softmax(router_logits, dim=-1)   # [N_tokens, E]
        expert_usage = probs.mean(dim=0)                      # [E]
        ideal        = torch.full_like(expert_usage, 1.0 / n_experts)
        total        = total + ((expert_usage - ideal) ** 2).mean()
    return total / len(all_router_logits)


def calculate_ntp_loss(logits, targets, all_router_logits=None, n_experts=6, lb_weight=0.05, z_weight=0.001):
    B, S, V  = logits.shape
    ntp_loss = chunked_cross_entropy(
        logits.contiguous().view(-1, V),
        targets.contiguous().view(-1),
    )

    if all_router_logits:
        lb_loss    = calculate_load_balance_loss(all_router_logits, n_experts)
        z_loss     = calculate_z_loss(all_router_logits, z_weight)
        total_loss = ntp_loss + lb_weight * lb_loss + z_weight * z_loss
        return total_loss, ntp_loss.detach(), lb_loss.detach(), z_loss.detach()

    return ntp_loss, ntp_loss.detach(), None, None