import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNormNoWeight(nn.Module):
    """
    RMSNorm without a learnable scale parameter. Used inside the AttnRes
    operator to normalize keys so that layers with large-magnitude outputs
    do not dominate the softmax.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype)

class AttnResOperator(nn.Module):
    """
    Depth-wise Attention Residual operator from the Kimi paper.

    Computes softmax attention over a set of source representations (e.g.,
    block-level summaries) using a single learned pseudo-query vector.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(dim))
        self.key_norm = RMSNormNoWeight(eps=eps)

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted aggregation of source representations.

        Args:
            sources: Tensor of shape (Num_Sources, B, S, D) containing the
                     source representations to attend over.

        Returns:
            Aggregated representation of shape (B, S, D).
        """
        # K = RMSNorm(sources)
        K = self.key_norm(sources)

        # Logits: dot product of pseudo-query with each normalized source
        # pseudo_query: [D], K: [N, B, S, D] -> logits: [N, B, S]
        logits = torch.einsum("d, n b s d -> n b s", self.pseudo_query, K)

        # Softmax over sources (the "depth" dimension)
        weights = F.softmax(logits, dim=0)

        # Weighted sum to produce the layer input h_l
        # weights: [N, B, S], sources: [N, B, S, D] -> out: [B, S, D]
        out = torch.einsum("n b s, n b s d -> b s d", weights, sources)
        return out