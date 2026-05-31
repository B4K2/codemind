import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizer.tokenizer import CodeMindTokenizer

SYSTEM_PROMPT = """You are CodeMind, an expert coding assistant.
Think through problems step by step, then provide clean working code."""

def tokenize_with_loss_mask(
    tokenizer: CodeMindTokenizer,
    system: str,
    user: str,
    assistant: str,
    max_seq_len: int,
):
    prompt = (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user}<|end|>\n"
        f"<|assistant|>\n"
    )
    prompt_ids   = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(f"{assistant}<|end|>")
    response_ids.append(tokenizer.eos_id)

    input_ids = prompt_ids + response_ids
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

    input_ids = input_ids[:max_seq_len]
    loss_mask = loss_mask[:max_seq_len]

    pad_len   = max_seq_len - len(input_ids)
    input_ids = input_ids + [tokenizer.eos_id] * pad_len
    loss_mask = loss_mask + [0] * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(loss_mask, dtype=torch.bool),
    )


def parse_openthoughts_assistant(conversation: list) -> tuple[str, str]:
    """
    Extract user prompt and assistant response from OpenThoughts conversation.
    
    Conversation format:
    [
      {"from": "user",      "value": "problem text"},
      {"from": "assistant", "value": "<|begin_of_thought|>...<|end_of_thought|>
                                      <|begin_of_solution|>...<|end_of_solution|>"}
    ]
    
    Returns (user_prompt, formatted_assistant_response)
    """
    user_text      = ""
    assistant_text = ""

    for turn in conversation:
        role  = turn.get("from", "")
        value = turn.get("value", "")

        if role == "user":
            user_text = value

        elif role == "assistant":
            # Extract thinking and solution from OpenThoughts format
            thought  = ""
            solution = ""

            if "<|begin_of_thought|>" in value and "<|end_of_thought|>" in value:
                thought_start = value.index("<|begin_of_thought|>") + len("<|begin_of_thought|>")
                thought_end   = value.index("<|end_of_thought|>")
                thought       = value[thought_start:thought_end].strip()

            if "<|begin_of_solution|>" in value and "<|end_of_solution|>" in value:
                sol_start = value.index("<|begin_of_solution|>") + len("<|begin_of_solution|>")
                sol_end   = value.index("<|end_of_solution|>")
                solution  = value[sol_start:sol_end].strip()

            # Convert to CodeMind's <think> format
            if thought and solution:
                assistant_text = f"<think>\n{thought}\n</think>\n\n{solution}"
            elif solution:
                assistant_text = solution
            else:
                # Fallback — use raw value if parsing fails
                assistant_text = value

    return user_text, assistant_text


class SFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: CodeMindTokenizer,
        max_seq_len: int = 2048,
        max_samples: int = None,   # set to ~500 for quick local test
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.examples    = []

        print("Loading SFT datasets...")
        self._load_magicoder(max_samples)
        self._load_openthoughts(max_samples)
        print(f"Total SFT examples: {len(self.examples)}")

    def _load_magicoder(self, max_samples=None):
        """
        Columns: lang, problem, solution
        Filter to Python only.
        """
        try:
            ds = load_dataset(
                "ise-uiuc/Magicoder-OSS-Instruct-75K",
                split="train",
                streaming=False,
            )
            count = 0
            for ex in ds:
                if max_samples and count >= max_samples:
                    break

                # Filter Python only
                if ex.get("lang", "").lower() != "python":
                    continue

                problem  = ex.get("problem", "").strip()
                solution = ex.get("solution", "").strip()

                if not problem or not solution:
                    continue

                self.examples.append({
                    "system":    SYSTEM_PROMPT,
                    "user":      problem,
                    "assistant": solution,
                })
                count += 1

            print(f"  Magicoder (Python only): {count} examples")
        except Exception as e:
            print(f"  Magicoder failed: {e}")

    def _load_openthoughts(self, max_samples=None):
        """
        Using default subset.
        Columns: system, conversation
        conversation is a list of {from, value} dicts.
        """
        try:
            ds = load_dataset(
                "open-thoughts/OpenThoughts-114k",
                name="default",      # ← explicit subset
                split="train",
                streaming=False,
            )
            count = 0
            for ex in ds:
                if max_samples and count >= max_samples:
                    break

                conversation = ex.get("conversations", [])
                if not conversation:
                    continue

                # Use dataset's system prompt if available, else ours
                system = ex.get("system", "").strip() or SYSTEM_PROMPT

                user_text, assistant_text = parse_openthoughts_assistant(conversation)

                if not user_text or not assistant_text:
                    continue

                # Skip very short responses — likely parsing failures
                if len(assistant_text) < 50:
                    continue

                self.examples.append({
                    "system":    system,
                    "user":      user_text,
                    "assistant": assistant_text,
                })
                count += 1

            print(f"  OpenThoughts: {count} examples")
        except Exception as e:
            print(f"  OpenThoughts failed: {e}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_ids, loss_mask = tokenize_with_loss_mask(
            self.tokenizer,
            ex["system"],
            ex["user"],
            ex["assistant"],
            self.max_seq_len,
        )
        return input_ids, loss_mask