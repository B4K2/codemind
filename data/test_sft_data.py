# test_sft_data.py — run this first
from tokenizer.tokenizer import CodeMindTokenizer
from data.sft_dataset import SFTDataset, parse_openthoughts_assistant

tokenizer = CodeMindTokenizer()

# Test parser directly
fake_conversation = [
    {"from": "user", "value": "Write a binary search function."},
    {"from": "assistant", "value": (
        "<|begin_of_thought|>\n"
        "I need to implement binary search. It works by repeatedly halving the search space.\n"
        "Base case: if left > right, return -1.\n"
        "<|end_of_thought|>\n"
        "<|begin_of_solution|>\n"
        "def binary_search(arr, target):\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] == target: return mid\n"
        "        elif arr[mid] < target: left = mid + 1\n"
        "        else: right = mid - 1\n"
        "    return -1\n"
        "<|end_of_solution|>"
    )},
]

user, assistant = parse_openthoughts_assistant(fake_conversation)
print("User:", user[:80])
print("Assistant preview:", assistant[:200])
print("Has <think>:", "<think>" in assistant)
print()

# Test full dataset with small sample
ds = SFTDataset(tokenizer, max_seq_len=512, max_samples=100)
input_ids, loss_mask = ds[0]
print(f"input_ids shape: {input_ids.shape}")
print(f"loss_mask shape: {loss_mask.shape}")
print(f"Assistant tokens: {loss_mask.sum().item()} / {len(loss_mask)}")
print()

# Decode to verify it looks right
prompt_part    = tokenizer.decode(input_ids[~loss_mask].tolist()[:50])
assistant_part = tokenizer.decode(input_ids[loss_mask].tolist()[:100])
print(f"Prompt start:    {prompt_part[:100]}")
print(f"Assistant start: {assistant_part[:100]}")