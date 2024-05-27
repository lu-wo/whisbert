from transformers import AutoTokenizer


def get_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """Loads tokenizer from HuggingFace's transformers library."""
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
