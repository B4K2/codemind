from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
        nn.init.normal_(self.router.weight, std=0.01)  # small init → stable logits
        self.register_buffer("expert_bias", torch.zeros(config.n_routed_experts))

    def forward(self, x: torch.Tensor):
        logits  = self.router(x)                                    
        scores  = F.softmax(logits, dim=-1, dtype=torch.float32)   

        biased  = scores + self.expert_bias
        _, top_k_indices = torch.topk(biased, self.n_activated_experts, dim=-1)

        top_k_weights = scores.gather(1, top_k_indices)            
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return top_k_weights.to(x.dtype), top_k_indices, logits


class MoE(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.n_routed_experts    = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.gate           = Gate(config)
        self.routed_experts = nn.ModuleList([Expert(config) for _ in range(self.n_routed_experts)])
        self.shared_expert  = Expert(config)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        N       = B * S
        x_flat  = x.view(N, D)                                    

        routing_weights, expert_indices, router_logits = self.gate(x_flat)

        one_hot = F.one_hot(expert_indices, num_classes=self.n_routed_experts).to(x_flat.dtype)

        dispatch_weights = (routing_weights.unsqueeze(-1) * one_hot).sum(dim=1)

        expert_outs = torch.stack(
            [expert(x_flat) for expert in self.routed_experts],
            dim=0
        )
        
        routed_out = torch.einsum("ne, end -> nd", dispatch_weights, expert_outs)  

        final_out = routed_out + self.shared_expert(x_flat)                        

        if self.training:
            self._update_expert_bias(expert_indices)

        return final_out.view(B, S, D), router_logits

    @torch.no_grad()
    def _update_expert_bias(self, expert_indices: torch.Tensor):
        """Update expert bias outside the forward graph — never touches autograd."""
        usage = expert_indices.flatten().bincount(
            minlength=self.n_routed_experts
        ).float()
        usage = usage / usage.sum()
        ideal = 1.0 / self.n_routed_experts
        self.gate.expert_bias.add_(0.001 * (ideal - usage))

# Quick local test
if __name__ == "__main__":
    print("Testing CodeMind MoE...")
    config = CodeMindConfig(max_batch_size=2, max_seq_len=512)
    model = MoE(config).to("cuda").to(torch.bfloat16)
    
    # Dummy input
    x = torch.randn(2, 512, config.dim, dtype=torch.bfloat16, device="cuda")
    
    # Forward pass
    out, router_logits = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert x.shape == out.shape
    print("MoE forward pass successful!")