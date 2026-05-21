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
        self.w_up = nn.Linear(config.dim, config.moe_inter_dim, bias=False)
        self.w_down = nn.Linear(config.moe_inter_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        out = self.w_down(gate * up)

        if weights is not None:
            return out * weights
        return out

class Gate(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.router = nn.Linear(config.dim, self.n_routed_experts, bias=False)
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.n_routed_experts)
        )

    def forward(self, x: torch.Tensor):
        logits = self.router(x)                                        # raw logits
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        biased_scores = scores + self.expert_bias             # [N, E]
        _, top_k_indices = torch.topk(biased_scores, self.n_activated_experts, dim=-1)
        top_k_weights = scores.gather(1, top_k_indices)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_weights.to(x.dtype), top_k_indices, logits


class MoE(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.n_routed_experts  = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.gate          = Gate(config)
        self.routed_experts = nn.ModuleList([Expert(config) for _ in range(self.n_routed_experts)])
        self.shared_expert  = Expert(config)

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        x_flat  = x.view(-1, D)

        routing_weights, expert_indices, router_logits = self.gate(x_flat)

        final_output        = torch.zeros_like(x_flat)
        flat_expert_indices = expert_indices.flatten()
        flat_inputs         = x_flat.repeat_interleave(self.n_activated_experts, dim=0)
        expert_counts       = torch.bincount(flat_expert_indices, minlength=self.n_routed_experts)
        expert_inputs_split = torch.split(flat_inputs, expert_counts.tolist(), dim=0)

        expert_outputs_split = []
        for i in range(self.n_routed_experts):
            if expert_inputs_split[i].shape[0] > 0:
                expert_outputs_split.append(self.routed_experts[i](expert_inputs_split[i]))

        expert_outputs   = torch.cat(expert_outputs_split, dim=0)
        weighted_outputs = expert_outputs * routing_weights.flatten().unsqueeze(1)
        token_indices    = torch.arange(x_flat.size(0), device=x.device).repeat_interleave(self.n_activated_experts)
        final_output.scatter_add_(0, token_indices.unsqueeze(1).expand_as(weighted_outputs), weighted_outputs)
        final_output    += self.shared_expert(x_flat)

        # if self.training:
        #     with torch.no_grad():
        #         usage = torch.bincount(
        #             expert_indices.flatten(),
        #             minlength=self.n_routed_experts
        #         ).float()
        #         usage = usage / usage.sum()                   
        #         ideal = 1.0 / self.n_routed_experts
        #         self.gate.expert_bias += 0.001 * (ideal - usage)

        return final_output.view(B, S, D), router_logits   

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