from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import CodeMindConfig
from model.components.norms import RMSNorm
from model.components.rope import precompute_freqs_cis
from model.components.attn_residuals import AttnResOperator
from model.blocks.transformer_block import TransformerBlock
from model.components.mtp import MTPHead

from torch.utils.checkpoint import checkpoint

class CodeMindSLM(nn.Module):
    """
    The complete CodeMind Small Language Model.
    
    This model fuses the DeepSeek-V4 architecture (MLA, MoE) with the
    Kimi Attention Residuals (Block AttnRes) paradigm for connecting layers.
    """
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.config = config

        self.use_gradient_checkpointing = True
        
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        self.attn_res_operators = nn.ModuleList([AttnResOperator(config.dim, config.norm_eps) for _ in range(config.n_layers)])

        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.token_emb.weight
        
        self.mtp_heads = nn.ModuleList([
            MTPHead(config, offset=i) for i in range(config.n_mtp_layers)
        ])

        for head in self.mtp_heads:
            head.lm_head   = self.lm_head   
            head.token_emb = self.token_emb 

        rope_freqs = precompute_freqs_cis(
            config.rope_head_dim, config.max_seq_len + 10,
            config.original_seq_len, config.rope_theta,
            config.rope_factor, config.beta_fast, config.beta_slow,
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)
        self._init_weights()

    def _init_weights(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, AttnResOperator):
                nn.init.zeros_(module.pseudo_query)

    def _forward_block_attnres(self, x: torch.Tensor):
        B, S, D   = x.shape
        rope_freqs = self.rope_freqs[:S].to(x.device)

        block_reps        = [x]
        partial_block     = None
        last_output       = None
        all_router_logits = []
        all_lb_losses     = []       
        all_expert_usages = []        

        for i, layer in enumerate(self.layers):
            source_list = block_reps + ([partial_block] if partial_block is not None else [])
            sources     = torch.stack(source_list, dim=0)
            h_l         = self.attn_res_operators[i](sources)

            if self.use_gradient_checkpointing and self.training:
                def make_layer_fn(l):
                    def layer_fn(h, freqs):
                        out, _, lb, usage = l(h, freqs, use_sliding_window=True)
                        return out
                    return layer_fn

                block_output = checkpoint(
                    make_layer_fn(layer),
                    h_l, rope_freqs,
                    use_reentrant=True,    
                )
                with torch.no_grad():
                    _, router_logits, lb_loss, expert_usage = layer(
                        h_l.detach(), rope_freqs, use_sliding_window=True
                    )
            else:
                block_output, router_logits, lb_loss, expert_usage = layer(
                    h_l, rope_freqs, use_sliding_window=True
                )

            all_router_logits.append(router_logits)
            all_lb_losses.append(lb_loss)
            all_expert_usages.append(expert_usage)

            last_output   = block_output
            partial_block = block_output if partial_block is None else partial_block + block_output

            is_boundary   = (i + 1) % self.config.block_size == 0
            is_last_layer = i == len(self.layers) - 1
            if is_boundary and not is_last_layer:
                block_reps.append(partial_block)
                partial_block = None

        return last_output, all_router_logits, all_lb_losses, all_expert_usages


    def forward(self, tokens: torch.Tensor, return_mtp: bool = True):
        x = self.token_emb(tokens)
        h, all_router_logits, all_lb_losses, all_expert_usages = \
            self._forward_block_attnres(x)
        h_norm = self.final_norm(h)
        logits  = self.lm_head(h_norm)

        mtp_logits = []
        if return_mtp and len(self.mtp_heads) > 0 and self.training:
            rope_freqs = self.rope_freqs[:tokens.size(1)].to(x.device)
            for head in self.mtp_heads:
                mtp_logits.append(head(h, tokens, rope_freqs))

        return logits, all_router_logits, all_lb_losses, all_expert_usages, mtp_logits