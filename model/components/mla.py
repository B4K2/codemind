import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

from config.model_config import CodeMindConfig
from model.components.norms import RMSNorm
from model.components.rope import apply_rotary_emb

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.window_size = config.window_size

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        
        self.wq_a = nn.Linear(self.dim, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, config.norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * (self.head_dim + self.rope_head_dim), bias=False)
        
        self.wkv_a = nn.Linear(self.dim, config.kv_lora_rank + self.rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank, config.norm_eps)
        self.wkv_b = nn.Linear(config.kv_lora_rank, self.head_dim + self.head_dim, bias=False)
        
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, use_sliding_window: bool = True) -> torch.Tensor:
        B, S, _ = x.shape

        q_c = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_c)
        q = q.view(B, S, self.n_heads, self.head_dim + self.rope_head_dim)

        q_content, q_rope = q.split([self.head_dim, self.rope_head_dim], dim=-1)

        kv_c_and_rope = self.wkv_a(x)
        kv_c, k_rope = kv_c_and_rope.split([self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv_c = self.kv_norm(kv_c)

        kv = self.wkv_b(kv_c)
        kv = kv.view(B, S, 1, self.head_dim * 2) # 1 KV head
        k_content, v = kv.split([self.head_dim, self.head_dim], dim=-1)
        k_rope = k_rope.view(B, S, 1, self.rope_head_dim)

        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        k_rope = apply_rotary_emb(k_rope, freqs_cis)

        q = torch.cat([q_content, q_rope], dim=-1) 
        k = torch.cat([k_content, k_rope], dim=-1)
        
        if HAS_FLASH_ATTN and x.is_cuda and x.dtype in [torch.float16, torch.bfloat16]:
            window = (self.window_size, 0) if use_sliding_window else (-1, -1)
            o = flash_attn_func(q, k, v, dropout_p=0.0, causal=True, window_size=window)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            if use_sliding_window:
                from model.components.sliding_window import get_sliding_window_mask
                attn_mask = get_sliding_window_mask(S, self.window_size, x.device, x.dtype)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
            else:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
                
            o = o.transpose(1, 2)
            
        o = o.reshape(B, S, self.n_heads * self.head_dim)
        return self.wo(o)

# Quick local test
if __name__ == "__main__":
    from config.model_config import CodeMindConfig
    from model.components.rope import precompute_freqs_cis
    
    print("Testing CodeMind MLA...")
    config = CodeMindConfig(max_batch_size=2, max_seq_len=1024)
    model = MultiHeadLatentAttention(config).to(torch.bfloat16).cuda()
    
    # Dummy inputs
    x = torch.randn(2, 512, config.dim, dtype=torch.bfloat16, device="cuda")
    
    # Precompute RoPE
    freqs_cis = precompute_freqs_cis(
        config.rope_head_dim, 512, config.original_seq_len, 
        config.rope_theta, config.rope_factor, 
        config.beta_fast, config.beta_slow
    ).cuda()
    
    # Forward pass
    out = model(x, freqs_cis, use_sliding_window=True)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print("MLA forward pass successful!")