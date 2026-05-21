import os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file # Changed from load_model to load_file

from config.model_config import CodeMindConfig
from tokenizer.tokenizer import CodeMindTokenizer
from model.codemind import CodeMindSLM

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=10):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device="cuda")
    
    print(f"\n[Prompt]: {prompt}\n" + "-"*40)
    print("[Model Output]:", end="", flush=True)
    
    for _ in range(max_new_tokens):
        # We cap seq_len to prevent out-of-bounds (RoPE limit)
        context = input_ids[:, -model.config.max_seq_len:]
        
        # Unpack the tuple since your new forward pass returns (logits, all_router_logits)
        logits, _ = model(context)
        
        # Only take the last position
        next_token_logits = logits[0, -1, :] / temperature
        
        # Top-K Sampling
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
        probs = F.softmax(top_k_logits, dim=-1)
        next_token_id = top_k_indices[torch.multinomial(probs, 1)]
        
        word = tokenizer.decode([next_token_id.item()])
        print(word, end="", flush=True)
        
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
        if next_token_id.item() == tokenizer.eos_id:
            break
    print("\n" + "-"*40)

if __name__ == "__main__":
    config = CodeMindConfig(
    )
    tokenizer = CodeMindTokenizer()
    model = CodeMindSLM(config).to("cuda").to(torch.bfloat16)
    

    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        steps = [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")]
        if steps:
            latest_step = max(steps, key=lambda x: int(x.split("_")[1]))
            model_path = os.path.join(checkpoint_dir, latest_step, "model.safetensors")
            print(f"Loading weights from: {model_path}")
            
            state_dict = load_file(model_path)
            
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
                cleaned_state_dict[new_key] = v
                
            model.load_state_dict(cleaned_state_dict, strict=False)
            print("Weights successfully loaded!")
        else:
            print("No checkpoints found! Using random weights.")
    else:
        print("No checkpoints directory found! Using random weights.")

    prompts_example  = [
        "def fibonacci(n):",
        "# Sort a list using bubble sort\ndef bubble_sort(",
        "class BinaryTree:\n    def __init__(self):",
        "import numpy as np\n\ndef matrix_multiply(A, B):",
    ]

    for i in prompts_example: 
        prompt = i
        generate(model, tokenizer, prompt, max_new_tokens=100)