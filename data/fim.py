import random
from typing import Tuple

def apply_fim(text: str, fim_rate: float = 0.2) -> str:
    """
    With probability `fim_rate`, splits the document into Prefix, Middle, and Suffix,
    and rearranges it for Fill-in-the-Middle (FIM) training.
    """
    if random.random() > fim_rate:
        return text

    if len(text) < 3:
        return text

    split1 = random.randint(1, len(text) - 2)
    split2 = random.randint(split1 + 1, len(text) - 1)

    prefix = text[:split1]
    middle = text[split1:split2]
    suffix = text[split2:]

    return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}"