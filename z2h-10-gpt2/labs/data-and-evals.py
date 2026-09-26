# Card: data-and-evals.html · Lecture 9 · datasets used in GPT-2, GPT-3, FineWeb (EDU) / validation split / evaluation: HellaSwag · https://www.youtube.com/watch?v=l8pRSuU81PU&t=11421s
# Run from the course folder: python labs/data-and-evals.py   (--shard 1000000 tokens per mini shard, --hella 20 examples; --hella 10042 is the full eval)
import os
import sys
import time
import numpy as np
import tiktoken
import torch
from torch.nn import functional as F
from common import DATA, pick_device, arg, flag, iterate_examples, render_example

# ----------------------------------------------------------------------------- 1) FineWeb-Edu, a small taste
# fineweb.py downloads all of sample-10BT (28.5 GB of parquet) into 100 shards of 100M tokens.
# Here we *stream* the first documents and write two mini shards in the same format:
# shard 0 = val, shard 1 = train, uint16 .npy, every document starts with <|endoftext|>.
shard_size = arg('--shard', 1_000_000)
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
print(f"<|endoftext|> = {eot}; vocab {enc.n_vocab} < 2**16 = {2**16}, so uint16 is enough")
print(f"full dataset: 10e9 tokens / 1e8 per shard = {int(10e9 / 1e8)} shards x {int(1e8) * 2 / 1e6:.0f} MB = {10e9 * 2 / 1e9:.0f} GB on disk")


def tokenize(doc):
    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


if not flag('--skip-fineweb'):
    from datasets import load_dataset
    out_dir = os.path.join(DATA, "edu_fineweb10B")
    os.makedirs(out_dir, exist_ok=True)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    t0 = time.time()
    docs = iter(fw)
    buf, n_docs, tok_time = np.empty((shard_size,), dtype=np.uint16), 0, 0.0
    for shard_index in range(2):
        count = 0
        while count < shard_size:
            doc = next(docs)
            t1 = time.time()
            tokens = tokenize(doc)
            tok_time += time.time() - t1
            n_docs += 1
            take = min(len(tokens), shard_size - count)  # the rest of the doc is dropped here
            buf[count:count + take] = tokens[:take]
            count += take
            if n_docs == 1:
                print(f"first doc: {doc['text'][:80]!r}... score {doc['score']:.2f}")
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(out_dir, f"edufineweb_{split}_{shard_index:06d}")
        np.save(filename, buf)
        print(f"wrote {filename}.npy: {shard_size:,} tokens, {os.path.getsize(filename + '.npy') / 1e6:.1f} MB")
    print(f"{n_docs} docs -> {2 * shard_size:,} tokens ({2 * shard_size / n_docs:.0f} tokens/doc) in {time.time() - t0:.1f} s")
    rate = 2 * shard_size / tok_time
    print(f"tokenizing alone: {rate:,.0f} tokens/s on 1 core -> 10B tokens = {10e9 / rate / 3600:.1f} h on 1 core, "
          f"{10e9 / rate / 3600 / (os.cpu_count() // 2):.1f} h with fineweb.py's {os.cpu_count() // 2} processes (plus the 28.5 GB download)")
    del docs, fw
    x = np.load(filename + ".npy").astype(np.int32)  # uint16 -> int32 -> long (errata)
    print("first 12 train tokens:", x[:12].tolist(), "->", repr(enc.decode(x[:12].tolist())))

print(f"steps per epoch: 10e9 / 2**19 = {10e9 / 2**19:,.1f} -> max_steps 19073")

# ----------------------------------------------------------------------------- 2) HellaSwag, completion style
# hellaswag.py downloads from github.com/rowanz/hellaswag, now DMCA-blocked (HTTP 451).
# common.py's iterate_examples reads the same validation split from Hugging Face (Rowan/hellaswag).


@torch.no_grad()
def evaluate(n, device):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    num_correct_norm = num_correct = num_total = 0
    t0 = time.time()
    for example in iterate_examples(n):
        tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        logits = model(tokens).logits
        # autoregressive loss at every position, then average over the ending only
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        shift_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_tokens.view(-1), reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        shift_mask = (mask[..., 1:]).contiguous()
        sum_loss = (shift_losses * shift_mask).sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        pred, pred_norm = sum_loss.argmin().item(), avg_loss.argmin().item()
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        if num_total == 1:
            print(f"Context:\n {example['ctx']}")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")
        if num_total % 1000 == 0:
            print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm / num_total:.4f}")
    dt = time.time() - t0
    print(f"gpt2 on {num_total} examples: acc {num_correct / num_total:.4f} | acc_norm {num_correct_norm}/{num_total} = {num_correct_norm / num_total:.4f}")
    print(f"eval time: {dt:.1f} s ({dt / num_total * 1000:.0f} ms/example; all 10,042 would take ~{dt / num_total * 10042 / 60:.0f} min)")
    print("hellaswag.py reference on all 10,042: acc 0.2859, acc_norm 0.2955")


evaluate(arg('--hella', 20), pick_device())
# the streaming parquet reader leaves background threads that can stop Python from exiting
sys.stdout.flush()
os._exit(0)
