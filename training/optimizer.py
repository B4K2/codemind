import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Core Muon Math (Newton-Schulz Iteration)
# -----------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2 
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A 
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

# -----------------------------------------------------------------------------
# Single Device Hybrid Optimizer (Muon + AdamW)
# -----------------------------------------------------------------------------

class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    Perfect for local RTX 5060 Phase 1 testing.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.01)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.1)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss

# -----------------------------------------------------------------------------
# CodeMind Parameter Router
# -----------------------------------------------------------------------------

def setup_optimizer(model: torch.nn.Module, muon_lr: float = 0.02, adam_lr: float = 3e-4) -> SingleDeviceMuonWithAuxAdam:
    """
    Scans the CodeMindSLM model and routes 2D hidden matrices to Muon, 
    and 1D parameters/embeddings/norms to AdamW.
    """
    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        if "emb" in name or "lm_head" in name:
            adam_params.append(p)
            
        elif p.ndim < 2:
            adam_params.append(p)
            
        else:
            muon_params.append(p)

    print(f"Optimizer Setup: {len(muon_params)} tensors routed to Muon, {len(adam_params)} tensors routed to AdamW.")

    param_groups = [
        dict(params=muon_params, lr=muon_lr, momentum=0.95, weight_decay=0.01, use_muon=True),
        dict(params=adam_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.1, use_muon=False)
    ]
    
    return SingleDeviceMuonWithAuxAdam(param_groups)

if __name__ == "__main__":
    from model.codemind import CodeMindSLM
    from config.model_config import CodeMindConfig
    
    config = CodeMindConfig(n_layers=4)
    model = CodeMindSLM(config).to(torch.bfloat16).cuda()
    
    optimizer = setup_optimizer(model)
    print("Optimizer initialized successfully!")