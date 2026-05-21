from typing import Iterator, List

def pack_token_stream(token_stream: Iterator[List[int]], max_seq_len: int) -> Iterator[List[int]]:
    """
    Takes an iterator of tokenized documents and yields exact chunks of max_seq_len.
    Leftover tokens are kept in a buffer for the next chunk.
    """
    buffer = []
    
    for tokens in token_stream:
        buffer.extend(tokens)
        
        while len(buffer) >= max_seq_len:
            chunk = buffer[:max_seq_len]
            buffer = buffer[max_seq_len:]
            
            yield chunk