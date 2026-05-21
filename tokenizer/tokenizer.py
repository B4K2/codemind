import tiktoken
from typing import List, Union

class CodeMindTokenizer:
    def __init__(self):
        base_encoding = tiktoken.get_encoding("cl100k_base")
        
        base_vocab = base_encoding._mergeable_ranks
        
        base_vocab_size = len(base_vocab)
        self.custom_special_tokens = {
            # Standard formatting
            "<|endoffile|>": base_vocab_size,
            "<|repo_name|>": base_vocab_size + 1,
            "<|pad|>": base_vocab_size + 2,
            
            # Reasoning (DeepSeek R1 / GRPO style)
            "<think>": base_vocab_size + 3,
            "</think>": base_vocab_size + 4,
            "<answer>": base_vocab_size + 5,
            "</answer>": base_vocab_size + 6,
            
            # Memory (Evicted KV cache summarization)
            "<mem_summary>": base_vocab_size + 7,
            "</mem_summary>": base_vocab_size + 8,
            
            # FIM (Fill-in-the-Middle for Code Completion)
            "<fim_prefix>": base_vocab_size + 9,
            "<fim_suffix>": base_vocab_size + 10,
            "<fim_middle>": base_vocab_size + 11,
        }
        
        self.tokenizer = tiktoken.Encoding(
            name="codemind_encoding",
            pat_str=base_encoding._pat_str,
            mergeable_ranks=base_vocab,
            special_tokens=self.custom_special_tokens,
        )
        
        self.pad_id = self.custom_special_tokens["<|pad|>"]
        self.eos_id = self.custom_special_tokens["<|endoffile|>"]
        
        raw_vocab_size = base_vocab_size + len(self.custom_special_tokens)
        self.vocab_size = (raw_vocab_size + 127) // 128 * 128

    def encode(self, text: str, allowed_special: Union[str, set] = "all") -> List[int]:
        """Convert a string to a list of token IDs."""
        return self.tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back to a string."""
        return self.tokenizer.decode(token_ids)

# Test
if __name__ == "__main__":
    tokenizer = CodeMindTokenizer()
    
    # Code and text
    sample = "def hello_world():\n    print('Hello')\n<|endoffile|>"
    encoded = tokenizer.encode(sample)
    
    # Reasoning block
    reasoning_sample = "<think>\nLet me think about this...\n</think>\n<answer>Done!</answer>"
    encoded_reasoning = tokenizer.encode(reasoning_sample)
    
    print(f"Vocab Size: {tokenizer.vocab_size} (Rounded for Tensor Cores)")
    print(f"Encoded code: {encoded}")
    print(f"Decoded code: {repr(tokenizer.decode(encoded))}")
    print(f"Encoded reasoning: {encoded_reasoning}")