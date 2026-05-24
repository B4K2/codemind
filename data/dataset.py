import torch
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset

from tokenizer.tokenizer import CodeMindTokenizer
from data.fim import apply_fim
from data.packing import pack_token_stream

class CodeMindDataset(IterableDataset):
    def __init__(self, tokenizer: CodeMindTokenizer, max_seq_len: int, split: str = "train"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # 1. Load GitHub Code (Filtered for Python)
        code_ds = load_dataset(
            "codeparrot/codeparrot-clean",
            streaming=True,
            split="train"
        )

        code_ds = code_ds.select_columns(
            ["content"]
        ).rename_column(
            "content",
            "text"
        )

        # 2. Load Open Web Math
        math_ds = load_dataset(
            "open-web-math/open-web-math", 
            streaming=True, 
            split="train"
        )
        # Drop url, date, metadata, etc. Keep only 'text'
        math_ds = math_ds.select_columns(["text"])

        # 3. Load FineWeb-Edu (10B Token Sample)
        web_ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name="sample-10BT", 
            streaming=True, 
            split="train"
        )
        # Drop id, dump, url, score, etc. Keep only 'text'
        web_ds = web_ds.select_columns(["text"])

        # 4. Interleave them 
        # Using 70% Python, 15% Math, 15% General Edu Web
        self.mixed_dataset = interleave_datasets(
            [code_ds, math_ds, web_ds],
            probabilities=[0.70, 0.15, 0.15],
            seed=42,
            stopping_strategy="all_exhausted"
        )

        # 5. Shuffle buffer (50,000 documents in RAM) to ensure good randomization
        self.mixed_dataset = self.mixed_dataset.shuffle(buffer_size=100000, seed=42)

    def _process_stream(self):
        """Generator that pulls text, applies FIM to code, and tokenizes."""
        for example in self.mixed_dataset:
            text = example["text"]
            
            # Apply FIM (the function internally handles the 20% probability)
            # FIM is mostly useful for code, but applying it to math/text occasionally doesn't hurt.
            text = apply_fim(text, fim_rate=0.2)
            
            # Tokenize and append <|endoffile|>
            tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_id]
            yield tokens

    def __iter__(self):
        """Yields packed PyTorch tensors of exactly max_seq_len + 2."""
        # We pack to max_seq_len + 2 because Multi-Token Prediction (MTP) 
        # needs target tokens at t+1 and t+2
        pack_len = self.max_seq_len + 2 
        
        token_stream = self._process_stream()
        packed_stream = pack_token_stream(token_stream, pack_len)
        
        for packed_chunk in packed_stream:
            # Yield as PyTorch long tensors mapped to CPU memory
            yield torch.tensor(packed_chunk, dtype=torch.long)