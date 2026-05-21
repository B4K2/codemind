import math
from functools import lru_cache
import torch

@lru_cache(2)
def precompute_freqs_cis(
    dim: int, 
    seqlen: int, 
    original_seq_len: int, 
    base: float, 
    factor: float, 
    beta_fast: int, 
    beta_slow: int
) -> torch.Tensor:
    """
    Precomputes complex exponentials for rotary embeddings with YaRN scaling.
    From DeepSeek V4: Applies frequency interpolation with a smooth linear ramp.
    """
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.
    This is an OUT-OF-PLACE operation.
    """
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    
    if inverse:
        freqs_cis = freqs_cis.conj()
        
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
        
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    
    return x_rotated.to(x.dtype)