from torch.utils.data import DataLoader
from data.dataset import CodeMindDataset
from tokenizer.tokenizer import CodeMindTokenizer
from config.model_config import CodeMindConfig

def get_dataloader(config: CodeMindConfig, tokenizer: CodeMindTokenizer, skip_batches=0):
    dataset = CodeMindDataset(
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len
    )
    
    loader = DataLoader(dataset, batch_size=config.max_batch_size, pin_memory=True, num_workers=2)
    
    if skip_batches > 0:
        print(f"Skipping {skip_batches} batches to resume data position...")
        loader = iter(loader)
        for _ in range(skip_batches):
            next(loader)
        
    return loader

if __name__ == "__main__":
    print("Initializing Config and Tokenizer...")
    config = CodeMindConfig(max_batch_size=2, max_seq_len=1024) # Small for quick test
    tokenizer = CodeMindTokenizer()
    
    print("Building Streaming Dataloader (This might take a few seconds to connect to Hugging Face)...")
    dataloader = get_dataloader(config, tokenizer)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"Batch Shape: {batch.shape} (Batch Size, Seq Len + 2)")
        
        sample_tokens = batch[0][:500].tolist()
        decoded = tokenizer.decode(sample_tokens)
        print("\n--- Decoded Sample (Notice the <|endoffile|> and <fim> tags!) ---")
        print(decoded)
        
        break