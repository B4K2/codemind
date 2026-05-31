"""
CodeMind SFT Generation Script
Tests the instruction-tuned model with proper chat format.

Usage:
  Interactive:  python -m eval.generate_sft
  Benchmark:    python -m eval.generate_sft --benchmark
  Single:       python -m eval.generate_sft --prompt "Write a binary search function"
  Greedy:       python -m eval.generate_sft --greedy
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


SYSTEM_PROMPT = """You are CodeMind, an expert coding assistant.
Think through problems step by step, then provide clean working code."""

# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_prompt(user: str, system: str = SYSTEM_PROMPT) -> str:
    """Format input using the same chat template used during SFT training."""
    return (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user}<|end|>\n"
        f"<|assistant|>\n"
    )


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if temperature == 0.0:
        return logits.argmax(dim=-1).item()

    logits = logits / temperature

    # Top-k
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals = torch.topk(logits, k).values
        logits[logits < topk_vals[-1]] = float("-inf")

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    unique_ids = input_ids.unique()
    score = logits[unique_ids]
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits[unique_ids] = score
    return logits


# ── Main generation ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: CodeMindSLM,
    tokenizer: CodeMindTokenizer,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.92,
    repetition_penalty: float = 1.15,
    stream: bool = True,
    stop_sequences: list = None,
) -> tuple[str, float]:
    """
    Generate a response to a user prompt using the SFT chat format.
    Returns (generated_text, tokens_per_second).
    """
    model.eval()
    device = next(model.parameters()).device

    # Format with chat template
    prompt     = format_prompt(user_prompt, system_prompt)
    prompt_ids = tokenizer.encode(prompt)
    input_ids  = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    stop_seqs  = stop_sequences or ["<|end|>", "<|user|>"]
    generated  = []
    t0         = time.perf_counter()
    buffer     = ""  # for stop sequence detection

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.max_seq_len:]
        logits, _, _ = model(context, return_mtp=False)
        next_logits  = logits[0, -1, :].float()

        # Repetition penalty
        next_logits = apply_repetition_penalty(
            next_logits, input_ids[0], repetition_penalty
        )

        token_id = sample_next_token(next_logits, temperature, top_k, top_p)
        generated.append(token_id)

        token_str = tokenizer.decode([token_id])
        buffer   += token_str

        if stream:
            print(token_str, end="", flush=True)

        # Check stop sequences
        should_stop = False
        for stop in stop_seqs:
            if stop in buffer:
                # Print up to stop sequence then halt
                if stream:
                    pass  # already printed
                should_stop = True
                break

        if should_stop or token_id == tokenizer.eos_id:
            break

        input_ids = torch.cat([
            input_ids,
            torch.tensor([[token_id]], device=device)
        ], dim=1)

    elapsed    = time.perf_counter() - t0
    tok_per_s  = len(generated) / elapsed if elapsed > 0 else 0

    # Clean up stop tokens from output
    output = tokenizer.decode(generated)
    for stop in stop_seqs:
        output = output.split(stop)[0]

    return output.strip(), tok_per_s


# ── Benchmark prompts ─────────────────────────────────────────────────────────

BENCHMARK_PROMPTS = [
    # Core coding tasks
    ("Fibonacci",
     "Write an efficient Python function to compute the nth Fibonacci number."),

    ("Bubble Sort",
     "Implement bubble sort in Python with an explanation of how it works."),

    ("Binary Search",
     "Write a binary search function in Python. Include edge cases."),

    ("Linked List",
     "Implement a singly linked list in Python with insert, delete, and search methods."),

    ("File Reading",
     "Write a Python function that reads a CSV file and returns a list of dictionaries."),

    ("Recursion",
     "Write merge sort in Python using recursion. Explain the time complexity."),

    ("OOP",
     "Create a Python class for a Stack data structure with push, pop, peek, and is_empty methods."),

    ("Bug Fix",
     "Fix the bug in this code:\n```python\ndef factorial(n):\n    if n = 0:\n        return 1\n    return n * factorial(n-1)\n```"),

    # Thinking/reasoning tasks
    ("Algorithm Choice",
     "When would you use a hash table vs a binary search tree? Give examples."),

    ("General",
     "What is the difference between a process and a thread?"),
]


def run_benchmark(model, tokenizer, **kwargs):
    print("\n" + "="*65)
    print("  CodeMind SFT Benchmark")
    print("="*65)

    for label, prompt in BENCHMARK_PROMPTS:
        print(f"\n[{label}]")
        print(f"Q: {prompt[:80]}{'...' if len(prompt)>80 else ''}")
        print(f"A: ", end="")

        output, tok_per_s = generate(model, tokenizer, prompt, **kwargs)
        print(f"\n[{tok_per_s:.0f} tok/s]")
        print("-"*65)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(
    config: CodeMindConfig,
    checkpoint_dir: str = "checkpoints_sft",
    step: int = None,
) -> CodeMindSLM:
    model = CodeMindSLM(config).to("cuda").to(torch.bfloat16)

    if not os.path.exists(checkpoint_dir):
        print(f"⚠ Directory '{checkpoint_dir}' not found — using random weights.")
        return model

    all_steps = sorted([
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("step_") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ])

    if not all_steps:
        print("⚠ No checkpoints found — using random weights.")
        return model

    target = f"step_{step:06d}" if step else all_steps[-1]
    if target not in all_steps:
        print(f"⚠ Step {step} not found, using latest: {all_steps[-1]}")
        target = all_steps[-1]

    model_path = os.path.join(checkpoint_dir, target, "model.safetensors")
    opt_path   = os.path.join(checkpoint_dir, target, "optimizer.pt")

    print(f"Loading SFT checkpoint: {model_path}")
    state_dict = load_file(model_path)
    cleaned    = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    raw_model  = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(cleaned, strict=False)

    if os.path.exists(opt_path):
        try:
            meta = torch.load(opt_path, map_location="cpu", weights_only=True)
            print(f"✅ SFT step {meta.get('step','?')} | "
                  f"Tokens: {meta.get('tokens_seen',0)/1e6:.1f}M")
        except Exception as e:
            print(f"⚠️ Optimizer file corrupted, skipping metadata print.")
            print("✅ Weights loaded successfully though! Proceeding...")

    return model


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive(model, tokenizer, **kwargs):
    print("\n" + "="*65)
    print("  CodeMind SFT — Interactive Mode")
    print("  Commands: :quit  :benchmark  :temp=0.7  :tokens=512")
    print("="*65)

    gen_kwargs = {
        "max_new_tokens":    512,
        "temperature":       0.7,
        "top_k":             50,
        "top_p":             0.92,
        "repetition_penalty": 1.15,
        **kwargs,
    }

    while True:
        print("\nYou: ", end="")
        try:
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                if line == ":quit":
                    sys.exit(0)
                elif line == ":benchmark":
                    run_benchmark(model, tokenizer, **gen_kwargs)
                    break
                elif line.startswith(":temp="):
                    gen_kwargs["temperature"] = float(line.split("=")[1])
                    print(f"Temperature → {gen_kwargs['temperature']}")
                    break
                elif line.startswith(":tokens="):
                    gen_kwargs["max_new_tokens"] = int(line.split("=")[1])
                    print(f"Max tokens → {gen_kwargs['max_new_tokens']}")
                    break
                else:
                    lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not lines:
            continue

        prompt = "\n".join(lines)
        print("\nCodeMind: ", end="")
        _, tok_per_s = generate(model, tokenizer, prompt,
                                stream=True, **gen_kwargs)
        print(f"\n[{tok_per_s:.0f} tok/s]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",      type=str,   default=None)
    parser.add_argument("--checkpoint",  type=str,   default="checkpoints_sft")
    parser.add_argument("--step",        type=int,   default=None)
    parser.add_argument("--max_tokens",  type=int,   default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k",       type=int,   default=50)
    parser.add_argument("--top_p",       type=float, default=0.92)
    parser.add_argument("--rep_penalty", type=float, default=1.15)
    parser.add_argument("--benchmark",   action="store_true")
    parser.add_argument("--greedy",      action="store_true")
    args = parser.parse_args()

    config    = CodeMindConfig()
    tokenizer = CodeMindTokenizer()
    model     = load_model(config, args.checkpoint, args.step)
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
        print(f"\nQ: {args.prompt}\nA: ", end="")
        _, tok_per_s = generate(model, tokenizer, args.prompt, **gen_kwargs)
        print(f"\n[{tok_per_s:.0f} tok/s]")
    else:
        interactive(model, tokenizer, **gen_kwargs)


if __name__ == "__main__":
    main()