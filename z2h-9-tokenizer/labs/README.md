# Labs: Let's build the GPT tokenizer (z2h-9-tokenizer)

Runnable companions to Lecture 8 of Karpathy's Neural Networks: Zero to Hero
([video](https://www.youtube.com/watch?v=zduSFxRajkE),
[Colab](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L),
[minbpe](https://github.com/karpathy/minbpe)).
One lab per card. Everything is plain Python on the CPU and deterministic, so no GPU or seed is involved.

## Setup

```bash
uv venv
uv pip install tiktoken regex sentencepiece torch
```

Run every lab from the course folder (`z2h-9-tokenizer/`), e.g. `python labs/stats-and-merge.py`.
On first use, the labs download what they need into `labs/data/` (gitignored):

- the lecture's Colab notebook, which holds the tiktokenizer example string and the Unicode blog text;
- `taylorswift.txt` from minbpe;
- tiny Shakespeare;
- GPT-2's `encoder.json` and `vocab.bpe` (about 1.5 MB).

tiktoken also downloads and caches its gpt2/cl100k files the first time.

`common.py` holds those loaders plus the lecture's `get_stats`, `merge`, `train`, `encode` and `decode`.
Some labs write these functions out in full because writing them is the card's exercise.

## Run order (times measured on an M3 while other jobs were running)

| # | Lab | What it prints | Time |
|---|---|---|---|
| 1 | `labs/why-tokenization.py` | Shakespeare: 65 characters vs GPT-2's 50,257 tokens; 1,115,394 chars → 338,025 GPT-2 tokens | < 1 s |
| 2 | `labs/tiktokenizer-tour.py` | the example string: 300 GPT-2 vs 185 cl100k tokens; ids of "Tokenization", " is", the numbers, Egg/egg/EGG; Korean and FizzBuzz costs | < 1 s |
| 3 | `labs/unicode-utf8.py` | `ord()`, the Korean string's code points, UTF-8/16/32 byte counts, the blog paragraph (533 chars → 616 bytes) | < 1 s |
| 4 | `labs/bpe-by-hand.py` | aaabdaaabac → ZabdZabac → ZYdZYac → XdXac; the byte version gives `[258, 100, 258, 97, 99]` | < 1 s |
| 5 | `labs/stats-and-merge.py` | top pairs of the paragraph: (101, 32) = 'e ' ×20; `merge` takes 616 → 596 | < 1 s |
| 6 | `labs/train-loop.py` | the 20 merges on the full post, 24,597 → 19,438 (1.27X), and ratios at vocab 356 and 512 | ~5 s |
| 7 | `labs/decode.py` | `vocab` by concatenation; `decode([128])` fails strictly, and gives � with `errors="replace"` | < 1 s |
| 8 | `labs/encode.py` | `encode` with the min-over-merge-index loop; round-trips; `encode(decode([128])) != [128]` | < 1 s |
| 9 | `labs/regex-presplit.py` | GPT-2 and GPT-4 regex chunks (contractions, digits, spaces); `"    hello world!!!"` in gpt2 vs cl100k | < 1 s |
| 10 | `labs/encoder-py-special-tokens.py` | `len(encoder)` 50257, 50,000 merges, `<\|endoftext\|>` = 50256; the cl100k special tokens; `allowed_special` | < 1 s (plus the first download) |
| 11 | `labs/build-gpt4-tokenizer.py` | the Basic and Regex tokenizers at vocab 512 on taylorswift.txt (2.36 / 2.13); recovered GPT-4 merges and byte_shuffle; matches tiktoken | ~30 s |
| 12 | `labs/sentencepiece-llama.py` | trains `tok400` with the Llama 2 options; "hello 안녕하세요" with byte fallback and without it (`<unk>`) | < 1 s |
| 13 | `labs/vocab-size-new-tokens.py` | gpt.py's parameters at vocab 65 / 50,257 / 100,277; adding one token by model surgery | ~8 s |
| 14 | `labs/quirks-explained.py` | DefaultCellStyle, Korean, number splits, Python indents, `<\|endoftext\|>` as text, trailing space (220), SolidGoldMagikarp (43453), JSON vs YAML | < 1 s |

tiktoken's regex in `openai_public.py` today is written differently from the one in the video (it uses possessive quantifiers), but the two are equivalent.
The labs quote minbpe's `GPT4_SPLIT_PATTERN`.
