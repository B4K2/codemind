from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import CodeMindConfig
from model.components.norms import RMSNorm
from model.components.rope import precompute_freqs_cis
from model.components.attn_residuals import AttnResOperator
from model.blocks.transformer_block import TransformerBlock

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
        
        # Token embedding layer
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Kimi AttnRes operators (one for each transformer block)
        self.attn_res_operators = nn.ModuleList([AttnResOperator(config.dim, config.norm_eps) for _ in range(config.n_layers)])

        # Final normalization and output head
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Precompute RoPE frequencies and register as a buffer
        rope_freqs = precompute_freqs_cis(
            config.rope_head_dim, config.max_seq_len + 10, config.original_seq_len, 
            config.rope_theta, config.rope_factor, 
            config.beta_fast, config.beta_slow
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, AttnResOperator):
                # CRITICAL: zero init for pseudo-query ensures uniform
                # initial attention -> equivalent to equal-weight average
                nn.init.zeros_(module.pseudo_query)

    def _forward_block_attnres(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        rope_freqs = self.rope_freqs[:S].to(x.device)

        block_reps: List[torch.Tensor] = [x]       # b_0 = token embedding
        partial_block: Optional[torch.Tensor] = None
        last_output: Optional[torch.Tensor] = None
        all_router_logits = []  

        for i, layer in enumerate(self.layers):
            # Assemble sources: completed block summaries + current partial sum
            source_list = block_reps + ([partial_block] if partial_block is not None else [])
            sources = torch.stack(source_list, dim=0)  # [N_src, B, S, D]

            # Depth-wise attention over sources → this layer's input
            h_l = self.attn_res_operators[i](sources)

            # Run through transformer block
            if self.use_gradient_checkpointing and self.training:
                def make_layer_fn(l):
                    def layer_fn(h, freqs):
                        out, _ = l(h, freqs, use_sliding_window=True)
                        return out
                    return layer_fn

                block_output = checkpoint(
                    make_layer_fn(layer),
                    h_l, rope_freqs,
                    use_reentrant=False,
                )
                with torch.no_grad():
                    _, router_logits = layer(h_l.detach(), rope_freqs, use_sliding_window=True)
            else:
                block_output, router_logits = layer(h_l, rope_freqs, use_sliding_window=True)
            all_router_logits.append(router_logits)
            last_output = block_output  # always track the real last output

            # Accumulate into partial block sum
            partial_block = block_output if partial_block is None else partial_block + block_output

            # At block boundary: save summary, reset partial
            is_boundary = (i + 1) % self.config.block_size == 0
            is_last_layer = (i == len(self.layers) - 1)

            if is_boundary and not is_last_layer:
                # Save completed block summary for future layers to attend to
                block_reps.append(partial_block)
                partial_block = None

        # Final hidden state = actual last layer output (not a block summary)
        return last_output, all_router_logits 

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass from token IDs to logits.
        """
        # 1. Get token embeddings
        x = self.token_emb(tokens)
        
        # 2. Run through the transformer layers with Block AttnRes
        h, all_router_logits = self._forward_block_attnres(x)
        
        # 3. Final normalization and language model head
        h = self.final_norm(h)
        logits = self.lm_head(h)
        
        return logits, all_router_logits 

# Quick local test
if __name__ == "__main__":
    print("Testing full CodeMindSLM model assembly...")
    config = CodeMindConfig(max_batch_size=2, max_seq_len=512)
    model = CodeMindSLM(config).to("cuda").to(torch.bfloat16)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Dummy input
    tokens = torch.randint(0, config.vocab_size, (2, 512), device="cuda")
    
    # Forward pass
    logits = model(tokens)
    
    print(f"Input shape:  {tokens.shape}")
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (2, 512, config.vocab_size)
    print("CodeMindSLM full forward pass successful!")