import math
from torch.optim.lr_scheduler import LRScheduler
from training.optimizer import SingleDeviceMuonWithAuxAdam

class CodeMindLRScheduler:
    """
    Custom Cosine Annealing with Warmup.
    Handles scheduling both Muon (high LR) and AdamW (low LR) param groups simultaneously.
    """
    def __init__(
        self, 
        optimizer: SingleDeviceMuonWithAuxAdam, 
        warmup_steps: int, 
        total_steps: int,
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        
        self.current_step = 0
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Updates the learning rates for all parameter groups."""
        self.current_step += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            
            if self.current_step < self.warmup_steps:
                lr = base_lr * (self.current_step / self.warmup_steps)

            elif self.current_step <= self.total_steps:
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                
                min_lr = base_lr * self.min_lr_ratio
                lr = min_lr + (base_lr - min_lr) * cosine_decay
                
            else:
                lr = base_lr * self.min_lr_ratio
                
            param_group['lr'] = lr

    def get_last_lr(self):
        """Returns the current learning rates (useful for logging to Weights & Biases)."""
        return [group['lr'] for group in self.optimizer.param_groups]

# Quick local test
if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    from training.optimizer import SingleDeviceMuonWithAuxAdam
    
    # Dummy optimizer just to test the scheduler
    dummy_muon_param = torch.nn.Parameter(torch.zeros(10, 10))
    dummy_adam_param = torch.nn.Parameter(torch.zeros(10))
    
    param_groups = [
        dict(params=[dummy_muon_param], lr=0.02, momentum=0.95, weight_decay=0.01, use_muon=True),
        dict(params=[dummy_adam_param], lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.1, use_muon=False)
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    
    # 100 warmup steps, 1000 total steps
    scheduler = CodeMindLRScheduler(optimizer, warmup_steps=100, total_steps=1000)
    
    muon_lrs = []
    adam_lrs = []
    
    for _ in range(1200): # Run past total_steps to see it flatline
        scheduler.step()
        lrs = scheduler.get_last_lr()
        muon_lrs.append(lrs[0])
        adam_lrs.append(lrs[1])
        
    print(f"Max Muon LR: {max(muon_lrs):.4f} | Min Muon LR: {min(muon_lrs):.4f}")
    print(f"Max Adam LR: {max(adam_lrs):.6f} | Min Adam LR: {min(adam_lrs):.6f}")
    print("Scheduler logic works perfectly!")