from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_config import CodeMindConfig


class Expert(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.dim, config.moe_inter_dim, bias=False)
        self.w_up   = nn.Linear(config.dim, config.moe_inter_dim, bias=False)
        self.w_down = nn.Linear(config.moe_inter_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Gate(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.n_routed_experts    = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.router = nn.Linear(config.dim, self.n_routed_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        self.register_buffer("expert_bias", torch.zeros(config.n_routed_experts))

    def forward(self, x: torch.Tensor):
        logits = self.router(x)                                     
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)    

        biased = scores + self.expert_bias
        _, top_k_indices = torch.topk(biased, self.n_activated_experts, dim=-1)  

        top_k_weights = scores.gather(1, top_k_indices)            
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return top_k_weights.to(x.dtype), top_k_indices, logits, scores


class MoE(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.n_routed_experts    = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.gate           = Gate(config)
        self.routed_experts = nn.ModuleList([
            Expert(config) for _ in range(self.n_routed_experts)
        ])
        self.shared_expert  = Expert(config)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        N       = B * S
        x_flat  = x.view(N, D)

        routing_weights, expert_indices, router_logits, router_scores = self.gate(x_flat)

        final_output = torch.zeros_like(x_flat)

        for expert_idx in range(self.n_routed_experts):
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  
            if not expert_mask.any():
                continue

            expert_input = x_flat[expert_mask]
            expert_out   = self.routed_experts[expert_idx](expert_input)

            slot_mask    = (expert_indices[expert_mask] == expert_idx)  
            slot_weights = routing_weights[expert_mask][slot_mask]      

            final_output[expert_mask] += expert_out * slot_weights.unsqueeze(-1)

        final_output += self.shared_expert(x_flat)

        lb_loss = self._load_balance_loss(expert_indices, router_scores, N)

        with torch.no_grad():
            expert_usage = self._get_expert_usage(expert_indices, N)

        if self.training:
            self._update_expert_bias(expert_indices)

        return final_output.view(B, S, D), router_logits, lb_loss, expert_usage

    def _load_balance_loss(
        self,
        expert_indices: torch.Tensor,  
        router_scores: torch.Tensor,   
        N: int,
    ) -> torch.Tensor:
        """
        Switch Transformer load balancing loss.

        f_i = fraction of tokens dispatched to expert i
        P_i = mean routing probability for expert i
        L   = n_experts * sum_i(f_i * P_i)

        Minimum when all f_i = P_i = 1/n_experts (perfectly balanced).
        f_i is not differentiable but P_i is — gradient flows through P_i.
        """
        E = self.n_routed_experts

        one_hot = F.one_hot(expert_indices, num_classes=E).float()  
        tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)           
        f = tokens_per_expert / (N * self.n_activated_experts)      

        P = router_scores.mean(dim=0)                               

        # L = E * sum(f * P)
        lb_loss = E * (f.detach() * P).sum()

        return lb_loss

    @torch.no_grad()
    def _get_expert_usage(
        self,
        expert_indices: torch.Tensor,  
        N: int,
    ) -> torch.Tensor:
        """
        Returns per-expert usage fraction [E].
        expert_usage[i] = fraction of token-slots going to expert i.
        Sums to n_activated_experts / n_routed_experts.
        """
        E = self.n_routed_experts
        counts = expert_indices.flatten().bincount(minlength=E).float()
        return counts / (N * self.n_activated_experts) 

    @torch.no_grad()
    def _update_expert_bias(self, expert_indices: torch.Tensor):
        usage = expert_indices.flatten().bincount(
            minlength=self.n_routed_experts
        ).float()
        usage = usage / usage.sum()
        ideal = 1.0 / self.n_routed_experts
        self.gate.expert_bias.add_(0.001 * (ideal - usage))