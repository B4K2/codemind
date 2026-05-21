import torch

def get_sliding_window_mask(seq_len: int, window_size: int, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Creates a causal sliding window mask for native PyTorch SDPA fallback.
    (FlashAttention handles this internally, but we need this if FlashAttention isn't available).
    
    Args:
        seq_len: Current sequence length.
        window_size: Number of previous tokens to attend to.
    
    Returns:
        Boolean or float mask tensor of shape (1, 1, seq_len, seq_len)
    """
    # Create a standard causal mask (True for valid positions, False for future)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    
    # Create a window mask (False for positions that are TOO far in the past)
    window_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=-window_size + 1)
    
    # Combine them: Must be <= current token AND within window
    valid_mask = causal_mask & window_mask
    
    # Convert to attention bias (-inf for masked, 0 for valid)
    bias = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    bias.masked_fill_(~valid_mask, float("-inf"))
    
    return bias.unsqueeze(0).unsqueeze(0) # [1, 1, S, S]