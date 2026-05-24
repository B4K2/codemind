"""
CodeMind Generation Script
Usage:
  Interactive mode:    python -m eval.generate
  Single prompt:       python -m eval.generate --prompt "def fibonacci(n):"
  From checkpoint:     python -m eval.generate --step 5000
  Benchmark all:       python -m eval.generate --benchmark
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from config.model_config import CodeMindConfig
from tokenizer.tokenizer import CodeMindTokenizer
from model.codemind import CodeMindSLM


# ── Sampling ──────────────────────────────────────────────────────────────────

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
    """
    Apply top-k and top-p (nucleus) filtering to logits.
    top_k=0 disables top-k. top_p=1.0 disables top-p.
    """
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        top_k_vals = torch.topk(logits, k).values
        logits = logits.masked_fill(logits < top_k_vals[..., -1, None], float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

    return logits


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.1) -> torch.Tensor:
    """Penalise tokens that have already appeared in the context."""
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, input_ids)
    # Positive logits get divided, negative get multiplied
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, input_ids, score)
    return logits


# ── Main generation function ──────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: CodeMindSLM,
    tokenizer: CodeMindTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
    repetition_penalty: float = 1.1,
    stream: bool = True,
    stop_at_eos: bool = True,
) -> str:
    """
    Generate text from a prompt.

    Args:
        model:              Loaded CodeMindSLM
        tokenizer:          CodeMindTokenizer
        prompt:             Input text
        max_new_tokens:     Maximum tokens to generate
        temperature:        Sampling temperature. Lower = more deterministic.
                            0.0 = greedy (argmax), 1.0 = raw distribution
        top_k:              Keep only top-k tokens. 0 = disabled
        top_p:              Nucleus sampling threshold. 1.0 = disabled
        repetition_penalty: > 1.0 penalises repeated tokens
        stream:             Print tokens as they're generated
        stop_at_eos:        Stop when EOS token is produced

    Returns:
        Generated text (not including the prompt)
    """
    model.eval()
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=next(model.parameters()).device
    )

    generated = []
    t0 = time.perf_counter()

    for step in range(max_new_tokens):
        # Cap context to max_seq_len
        context = input_ids[:, -model.config.max_seq_len:]

        # Forward pass — return_mtp=False for speed during generation
        logits, _, _ = model(context, return_mtp=False)

        # Take logits at the last position
        next_logits = logits[0, -1, :].float()  # [V]

        # Repetition penalty
        if repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(
                next_logits.unsqueeze(0),
                input_ids,
                repetition_penalty
            ).squeeze(0)

        # Temperature scaling
        if temperature == 0.0:
            # Greedy — no sampling
            next_token_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            next_logits = top_k_top_p_filter(next_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        token_id = next_token_id.item()
        generated.append(token_id)

        # Decode and stream
        token_str = tokenizer.decode([token_id])
        if stream:
            print(token_str, end="", flush=True)

        # Append to context
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

        # Stop conditions
        if stop_at_eos and token_id == tokenizer.eos_id:
            break

    elapsed   = time.perf_counter() - t0
    tok_per_s = len(generated) / elapsed if elapsed > 0 else 0

    output_text = tokenizer.decode(generated)
    return output_text, tok_per_s


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_model(config: CodeMindConfig, checkpoint_dir: str = "checkpoints", step: int = None) -> CodeMindSLM:
    model = CodeMindSLM(config).to("cuda").to(torch.bfloat16)

    if not os.path.exists(checkpoint_dir):
        print("⚠ No checkpoints directory found — using random weights.")
        return model

    # Find checkpoint
    all_steps = sorted([
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("step_") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ])

    if not all_steps:
        print("⚠ No checkpoints found — using random weights.")
        return model

    if step is not None:
        target = f"step_{step:06d}"
        if target not in all_steps:
            print(f"⚠ Step {step} not found. Available: {[s.split('_')[1] for s in all_steps]}")
            print(f"  Using latest: {all_steps[-1]}")
            target = all_steps[-1]
    else:
        target = all_steps[-1]

    model_path = os.path.join(checkpoint_dir, target, "model.safetensors")
    opt_path   = os.path.join(checkpoint_dir, target, "optimizer.pt")

    print(f"Loading from: {model_path}")
    state_dict = load_file(model_path)

    # Strip _orig_mod. prefix from torch.compile
    cleaned = {
        k.replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=False)

    # Print training progress if available
    if os.path.exists(opt_path):
        meta = torch.load(opt_path, map_location="cpu", weights_only=True)
        step_num     = meta.get("step", "?")
        tokens_seen  = meta.get("tokens_seen", 0)
        tok_str      = f"{tokens_seen/1e6:.1f}M" if tokens_seen < 1e9 else f"{tokens_seen/1e9:.2f}B"
        print(f"✅ Loaded step {step_num} | Tokens seen: {tok_str}")

    return model


# ── Benchmark prompts ─────────────────────────────────────────────────────────

BENCHMARK_PROMPTS = [
    # Code completion
    ("Python function",     "def fibonacci(n):"),
    ("Sorting algorithm",   "# Sort a list using bubble sort\ndef bubble_sort(arr):"),
    ("Class definition",    "class BinaryTree:\n    def __init__(self):\n        self.root = None\n\n    def insert(self, value):"),
    ("NumPy usage",         "import numpy as np\n\ndef matrix_multiply(A, B):"),
    ("Recursive function",  "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr"),
    ("File I/O",            "def read_csv(filepath):\n    \"\"\"Read a CSV file and return a list of dicts.\"\"\"\n"),
    # General knowledge (tests degeneration)
    ("General knowledge",   "The capital of France is"),
    ("Science",             "The speed of light is approximately"),
]


def run_benchmark(model, tokenizer, max_new_tokens=150, **kwargs):
    """Run all benchmark prompts and display results."""
    print("\n" + "="*60)
    print("  CodeMind Benchmark")
    print("="*60)

    total_toks = 0
    total_time = 0

    for label, prompt in BENCHMARK_PROMPTS:
        print(f"\n[{label}]")
        print(f"Prompt: {prompt[:60]}{'...' if len(prompt)>60 else ''}")
        print("Output: ", end="")

        output, tok_per_s = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            stream=True,
            **kwargs
        )
        print(f"\n[{tok_per_s:.0f} tok/s]")
        print("-"*60)


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive_mode(model, tokenizer, **kwargs):
    """REPL for interactive generation."""
    print("\n" + "="*60)
    print("  CodeMind Interactive Mode")
    print("  Type your prompt and press Enter twice to generate.")
    print("  Commands: :quit :benchmark :temp=0.8 :tokens=200")
    print("="*60)

    gen_kwargs = {
        "max_new_tokens":    200,
        "temperature":       0.8,
        "top_k":             50,
        "top_p":             0.92,
        "repetition_penalty": 1.1,
        **kwargs,
    }

    while True:
        print("\nPrompt (blank line to generate, :quit to exit):")
        lines = []
        try:
            while True:
                line = input()
                if line == "":
                    break
                # Handle commands
                if line.startswith(":quit"):
                    print("Goodbye!")
                    sys.exit(0)
                elif line.startswith(":benchmark"):
                    run_benchmark(model, tokenizer, **gen_kwargs)
                    break
                elif line.startswith(":temp="):
                    gen_kwargs["temperature"] = float(line.split("=")[1])
                    print(f"Temperature set to {gen_kwargs['temperature']}")
                    break
                elif line.startswith(":tokens="):
                    gen_kwargs["max_new_tokens"] = int(line.split("=")[1])
                    print(f"Max tokens set to {gen_kwargs['max_new_tokens']}")
                    break
                else:
                    lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not lines:
            continue

        prompt = "\n".join(lines)
        print("\n--- Output ---")
        _, tok_per_s = generate(model, tokenizer, prompt, stream=True, **gen_kwargs)
        print(f"\n--- {tok_per_s:.0f} tok/s ---")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CodeMind text generation")
    parser.add_argument("--prompt",      type=str,   default=None,
                        help="Single prompt to generate from")
    parser.add_argument("--step",        type=int,   default=None,
                        help="Checkpoint step to load (default: latest)")
    parser.add_argument("--checkpoint",  type=str,   default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--max_tokens",  type=int,   default=200,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.0=greedy)")
    parser.add_argument("--top_k",       type=int,   default=50,
                        help="Top-k sampling (0=disabled)")
    parser.add_argument("--top_p",       type=float, default=0.92,
                        help="Nucleus sampling threshold")
    parser.add_argument("--rep_penalty", type=float, default=1.1,
                        help="Repetition penalty (1.0=disabled)")
    parser.add_argument("--benchmark",   action="store_true",
                        help="Run all benchmark prompts")
    parser.add_argument("--greedy",      action="store_true",
                        help="Use greedy decoding (temperature=0)")
    args = parser.parse_args()

    # Config and model
    config    = CodeMindConfig()
    tokenizer = CodeMindTokenizer()
    model     = load_model(config, checkpoint_dir=args.checkpoint, step=args.step)
    model.eval()

    gen_kwargs = {
        "max_new_tokens":     args.max_tokens,
        "temperature":        0.0 if args.greedy else args.temperature,
        "top_k":              args.top_k,
        "top_p":              args.top_p,
        "repetition_penalty": args.rep_penalty,
        "stream":             True,
    }

    if args.benchmark:
        run_benchmark(model, tokenizer, **gen_kwargs)

    elif args.prompt:
        print(f"\nPrompt: {args.prompt}\n" + "-"*40)
        _, tok_per_s = generate(model, tokenizer, args.prompt, **gen_kwargs)
        print(f"\n[{tok_per_s:.0f} tok/s]")

    else:
        interactive_mode(model, tokenizer, **gen_kwargs)


if __name__ == "__main__":
    main()