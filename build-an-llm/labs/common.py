"""Shared setup for the text-to-answer labs: load Qwen2.5-0.5B (base) once.
Needs: pip install torch transformers. First run downloads ~1 GB from Hugging Face."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "What is the capital of India?"


def load(eager_attention=False):
    tok = AutoTokenizer.from_pretrained(NAME)
    kw = {"dtype": torch.float32}
    if eager_attention:
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(NAME, **kw).eval()
    return tok, model
