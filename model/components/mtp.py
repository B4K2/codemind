import torch
import torch.nn as nn
from config.model_config import CodeMindConfig
from model.components.norms import RMSNorm


class MTPHead(nn.Module):
    """
    Multi-Token Prediction head for one future offset.
    
    For offset k, predicts token at position t+k+1 from hidden state at t.
    
    Architecture (from DeepSeek V3 paper):
      - Takes main hidden state h and previous MTP head's hidden state
      - Projects and combines them
      - Runs through a lightweight transformer layer
      - Projects to vocab logits
    """
    def __init__(self, config: CodeMindConfig, offset: int):
        super().__init__()
        self.offset = offset 

        self.enorm  = RMSNorm(config.dim, config.norm_eps)
        self.hnorm  = RMSNorm(config.dim, config.norm_eps)
        self.proj   = nn.Linear(config.dim * 2, config.dim, bias=False)

        from model.blocks.transformer_block import TransformerBlock
        self.layer  = TransformerBlock(config)

        self.norm   = RMSNorm(config.dim, config.norm_eps)

        self.lm_head: nn.Linear = None   
        self.token_emb: nn.Embedding = None  

    def forward(
        self,
        h: torch.Tensor,           
        input_ids: torch.Tensor,   
        rope_freqs: torch.Tensor,
    ) -> torch.Tensor:
        assert self.lm_head is not None, "lm_head not set on MTPHead"
        assert self.token_emb is not None, "token_emb not set on MTPHead"

        shifted_ids = torch.roll(input_ids, shifts=-(self.offset + 1), dims=1)
        e = self.token_emb(shifted_ids)                    # [B, S, D]

        combined = torch.cat([self.hnorm(h), self.enorm(e)], dim=-1)  
        x = self.proj(combined)                            

        x, *_ = self.layer(x, rope_freqs, use_sliding_window=True)

        logits = self.lm_head(self.norm(x))                
        return logits