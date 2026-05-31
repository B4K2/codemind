import math
from dataclasses import dataclass
from typing import Literal, Tuple

# For Local Testing
@dataclass
class CodeMindConfig:
    # ------------------ Global & Hardware ------------------
    max_batch_size: int = 2
    max_seq_len: int = 1024        # 8K context for Phase 1
    dtype: str = "bfloat16"        # Native BF16 for RTX 5060 + Muon

    # ------------------ Core Dimensions --------------------
    vocab_size: int = 100352        
    dim: int = 768                 # Hidden dimension (d_model)
    n_layers: int = 16             # Total transformer blocks
    n_heads: int = 12              # Query heads (768 / 12 = 64 dim per head)
    
    # ------------------ DeepSeek MLA (Latent Attention) ----
    q_lora_rank: int = 512         # Query compression rank
    kv_lora_rank: int = 512        # KV compression rank
    head_dim: int = 64             # Dimension per attention head
    rope_head_dim: int = 32        # Dimension dedicated to RoPE (uncompressed)
    window_size: int = 4096        # Sliding window size (local attention)
    
    # YaRN RoPE settings for long context
    rope_theta: float = 10000.0
    compress_rope_theta: float = 40000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 4096

    # ------------------ DeepSeek MoE -----------------------
    moe_inter_dim: int = 512      # Hidden dim inside the expert FFN
    n_routed_experts: int = 10      # Total number of routed experts (Nano spec)
    n_activated_experts: int = 2   # Top-2 routing (Nano spec)
    n_shared_experts: int = 1      # 1 always-on shared expert
    score_func: str = "softmax"    # Routing activation function
    route_scale: float = 1.0       # Routing score multiplier
    
    # ------------------ Kimi Attention Residuals -----------
    attn_res_mode: Literal["none", "full", "block"] = "block"
    n_blocks: int = 4              # 16 layers / 4 blocks = 4 layers per block
    
    # ------------------ MTP (Multi-Token Prediction) -------
    n_mtp_layers: int = 1          # Predict next 1 auxiliary token (t+2)
    
    # ------------------ Optimizations & Norms --------------
    norm_eps: float = 1e-6
    hc_mult: int = 1               # Disable Hyper-Connections for Phase 1 to simplify AttnRes integration
    
    def __post_init__(self):
        # Sanity Checks
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_layers % self.n_blocks == 0, "n_layers must be evenly divisible by n_blocks for AttnRes"
        
    @property
    def block_size(self) -> int:
        """Layers per AttnRes block"""
        return self.n_layers // self.n_blocks
    
# For Cloud Training
# @dataclass
# class CodeMindConfig:
#     # ------------------ Global & Hardware ------------------
#     max_batch_size: int = 8        # was 2 — A40 48GB handles this
#     max_seq_len: int = 2048        # was 1024 — 2x context
#     dtype: str = "bfloat16"

#     # ------------------ Core Dimensions --------------------
#     vocab_size: int = 100352
#     dim: int = 1024                # was 768 — bigger model
#     n_layers: int = 16
#     n_heads: int = 16              # was 12 — must divide dim (1024/16=64 ✅)

#     # ------------------ MLA --------------------------------
#     q_lora_rank: int = 512
#     kv_lora_rank: int = 512
#     head_dim: int = 64
#     rope_head_dim: int = 32
#     window_size: int = 4096

#     rope_theta: float = 10000.0
#     compress_rope_theta: float = 40000.0
#     rope_factor: float = 40.0
#     beta_fast: int = 32
#     beta_slow: int = 1
#     original_seq_len: int = 4096

#     # ------------------ MoE --------------------------------
#     moe_inter_dim: int = 1024      # was 512
#     n_routed_experts: int = 16     # was 10
#     n_activated_experts: int = 2
#     n_shared_experts: int = 1
#     score_func: str = "softmax"
#     route_scale: float = 1.0

#     # ------------------ AttnRes ----------------------------
#     attn_res_mode: str = "block"
#     n_blocks: int = 4

#     # ------------------ MTP --------------------------------
#     n_mtp_layers: int = 1

#     # ------------------ Norms ------------------------------
#     norm_eps: float = 1e-6
#     hc_mult: int = 1

#     def __post_init__(self):
#         assert self.dim % self.n_heads == 0
#         assert self.n_layers % self.n_blocks == 0

#     @property
#     def block_size(self): return self.n_layers // self.n_blocks