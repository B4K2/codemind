import torch
import torch.nn as nn

from config.model_config import CodeMindConfig
from model.components.norms import RMSNorm
from model.components.mla import MultiHeadLatentAttention
from model.components.moe import MoE

class TransformerBlock(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = MultiHeadLatentAttention(config)

        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = MoE(config)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor, 
        use_sliding_window: bool
    ) -> torch.Tensor:
        attn_output = self.attn(
            self.attn_norm(x), 
            freqs_cis=freqs_cis, 
            use_sliding_window=use_sliding_window
        )
        h = x + attn_output
        
        ffn_output, router_logits = self.ffn(self.ffn_norm(h))
        output = h + ffn_output
        
        return output, router_logits

# Quick local test
if __name__ == "__main__":
    print("Testing CodeMind TransformerBlock...")
    config = CodeMindConfig(max_batch_size=2, max_seq_len=512)
    model = TransformerBlock(config).to("cuda").to(torch.bfloat16)
    
    from model.components.rope import precompute_freqs_cis
    
    # Dummy inputs
    x = torch.randn(2, 512, config.dim, dtype=torch.bfloat16, device="cuda")
    freqs_cis = precompute_freqs_cis(
        config.rope_head_dim, 512, config.original_seq_len, 
        config.rope_theta, config.rope_factor, 
        config.beta_fast, config.beta_slow
    ).cuda()
    
    # Forward pass
    out = model(x, freqs_cis, use_sliding_window=True)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert x.shape == out.shape
    print("TransformerBlock forward pass successful!")