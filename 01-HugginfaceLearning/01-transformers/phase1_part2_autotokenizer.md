# Phase 1 — Transformers
# Part 2: AutoTokenizer

---

## What is a Tokenizer

A tokenizer converts raw text into numbers that a model can process.
It also converts numbers back into text. Every model has its own tokenizer —
you must always use the tokenizer that matches your model.

```
"Hello world"  →  tokenizer  →  [101, 7592, 2088, 102]  →  model
    text                             token IDs
```

---

## Installation

```bash
pip install transformers tokenizers
# tokenizers is auto-installed with transformers
```

---

## Loading a Tokenizer

```python
from transformers import AutoTokenizer

# always use AutoTokenizer — it detects the correct tokenizer class automatically
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# from a local folder
tokenizer = AutoTokenizer.from_pretrained("./my-model-folder")

# specific revision
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", revision="main")
```

---

## Tokenizer Properties

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(tokenizer.vocab_size)        # 30522 for bert-base-uncased
print(tokenizer.model_max_length)  # 512 — max tokens this model accepts
print(tokenizer.is_fast)           # True if Rust-based fast tokenizer

# special tokens
print(tokenizer.cls_token)         # [CLS] — start of sequence
print(tokenizer.sep_token)         # [SEP] — end of sequence / separator
print(tokenizer.pad_token)         # [PAD] — padding token
print(tokenizer.unk_token)         # [UNK] — unknown words
print(tokenizer.mask_token)        # [MASK] — masked token for MLM
print(tokenizer.bos_token)         # BOS — beginning of sequence (GPT-style)
print(tokenizer.eos_token)         # EOS — end of sequence (GPT-style)

# special token IDs
print(tokenizer.cls_token_id)      # 101
print(tokenizer.sep_token_id)      # 102
print(tokenizer.pad_token_id)      # 0
print(tokenizer.unk_token_id)      # 100

# full vocabulary
vocab = tokenizer.get_vocab()      # dict of token → id
print(len(vocab))                  # vocab size
```

---

## Basic Encoding

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# simplest — returns a list of token ids
ids = tokenizer.encode("Hello world")
print(ids)
# [101, 7592, 2088, 102]  ← [CLS] hello world [SEP]

# tokenize — split into tokens (strings) without converting to IDs
tokens = tokenizer.tokenize("Hello world")
print(tokens)
# ['hello', 'world']

# convert tokens to IDs manually
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
# [7592, 2088]

# convert IDs to tokens manually
tokens = tokenizer.convert_ids_to_tokens(ids)
print(tokens)
# ['hello', 'world']
```

---

## Full Encoding — The Correct Way

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# calling the tokenizer directly — returns a BatchEncoding dict
encoding = tokenizer("Hello, how are you?")
print(encoding)
# {
#   'input_ids':      [101, 7592, 1010, 2129, 2024, 2017, 1029, 102],
#   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0],
#   'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]
# }

print(encoding["input_ids"])       # the token IDs
print(encoding["attention_mask"])  # 1 = real token, 0 = padding
print(encoding["token_type_ids"]) # 0 = sentence A, 1 = sentence B
```

---

## return_tensors — Format Control

```python
# default — plain Python lists
encoding = tokenizer("Hello world")

# PyTorch tensors — use this when feeding to PyTorch models
encoding = tokenizer("Hello world", return_tensors="pt")

# TensorFlow tensors
encoding = tokenizer("Hello world", return_tensors="tf")

# NumPy arrays
encoding = tokenizer("Hello world", return_tensors="np")

# feed directly to model
import torch
model = AutoModel.from_pretrained("bert-base-uncased")
encoding = tokenizer("Hello world", return_tensors="pt")
output = model(**encoding)
```

---

## Padding — Handling Different Length Inputs

All inputs in a batch must be the same length. Padding adds special tokens
to make them equal.

```python
# batch of different lengths
texts = ["Short text.", "This is a much longer sentence with more words."]

# no padding (default) — each is its own length
encoding = tokenizer(texts)
# input_ids[0] has 6 tokens, input_ids[1] has 12 tokens — cannot batch

# pad to longest in the batch
encoding = tokenizer(texts, padding=True)
encoding = tokenizer(texts, padding="longest")

# pad to model's maximum length (512 for BERT)
encoding = tokenizer(texts, padding="max_length")

# pad to a specific length
encoding = tokenizer(texts, padding="max_length", max_length=128)

# pad on the right (default for most models)
encoding = tokenizer(texts, padding=True, padding_side="right")

# pad on the left (needed for some decoder-only models)
encoding = tokenizer(texts, padding=True, padding_side="left")
tokenizer.padding_side = "left"   # set globally
```

---

## Truncation — Handling Long Inputs

Models have a maximum token limit (512 for BERT, 1024 for GPT-2, etc.).
Truncation cuts text that is too long.

```python
long_text = "This is a very long text..." * 100   # definitely too long

# truncate to model's max length
encoding = tokenizer(long_text, truncation=True)

# truncate to a specific max length
encoding = tokenizer(long_text, truncation=True, max_length=128)

# truncation strategies for sentence pairs:
# "longest_first" — remove tokens from the longest sequence first (default)
# "only_first" — truncate only the first sentence
# "only_second" — truncate only the second sentence
# "do_not_truncate" — raise error if too long

encoding = tokenizer(
    sentence1, sentence2,
    truncation="longest_first",
    max_length=256
)
```

---

## Padding and Truncation Together

```python
# the correct pattern for training — always use both together
texts = ["short", "a medium length sentence", "very long text" * 50]

encoding = tokenizer(
    texts,
    padding=True,         # pad shorter sequences
    truncation=True,      # truncate longer sequences
    max_length=128,       # fixed length for all
    return_tensors="pt"   # PyTorch tensors for model input
)

print(encoding["input_ids"].shape)      # torch.Size([3, 128])
print(encoding["attention_mask"].shape) # torch.Size([3, 128])
```

---

## Encoding Sentence Pairs

Many tasks use two sentences — NLI (does A entail B?), QA (question + context),
next sentence prediction. Pass them as two arguments.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# two sentences — separated by [SEP], identified by token_type_ids
encoding = tokenizer(
    "The cat sat on the mat.",   # sentence A
    "The mat had a cat on it."   # sentence B
)
print(encoding["input_ids"])
# [101, the cat sat on the mat. 102 the mat had a cat on it. 102]
#       ←—— sentence A ——→         ←——— sentence B ————→

print(encoding["token_type_ids"])
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#  ←————— all 0 for A ————→    ←—————— all 1 for B ————————→

# batch of sentence pairs
encoding = tokenizer(
    ["Question 1", "Question 2"],   # list of sentence A
    ["Context 1",  "Context 2"],    # list of sentence B
    padding=True,
    truncation=True
)
```

---

## Decoding — IDs Back to Text

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

ids = [101, 7592, 2088, 102]

# decode — convert IDs back to string
text = tokenizer.decode(ids)
print(text)
# "[CLS] hello world [SEP]"

# skip special tokens for clean output
text = tokenizer.decode(ids, skip_special_tokens=True)
print(text)
# "hello world"

# batch decode
batch_ids = [[101, 7592, 102], [101, 2088, 102]]
texts = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
print(texts)
# ["hello", "world"]
```

---

## offset_mapping — Token to Character Alignment

Maps each token back to the character positions in the original string.
Essential for NER, QA, and any task where you need to know where a token came from.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hello, World!"
encoding = tokenizer(text, return_offsets_mapping=True)

print(encoding["offset_mapping"])
# [(0, 0),   ← [CLS] — no position (special token)
#  (0, 5),   ← "Hello"  — characters 0 to 5
#  (5, 6),   ← ","      — character 5 to 6
#  (7, 12),  ← "World"  — characters 7 to 12
#  (12, 13), ← "!"      — character 12 to 13
#  (0, 0)]   ← [SEP]    — no position (special token)

# use case: map a predicted token span back to original text
start_token, end_token = 1, 3   # model predicted tokens 1 to 3
char_start = encoding["offset_mapping"][start_token][0]
char_end   = encoding["offset_mapping"][end_token][1]
answer     = text[char_start:char_end]
print(answer)  # "Hello,"
```

---

## word_ids — Token to Word Alignment

Maps each token to the word index it came from. Useful when words are split
into multiple subword tokens.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "New York City is amazing"
encoding = tokenizer(text)

print(encoding.tokens())
# ['[CLS]', 'new', 'york', 'city', 'is', 'amazing', '[SEP]']

print(encoding.word_ids())
# [None, 0, 1, 2, 3, 4, None]
# None = special token, number = which word it belongs to

# example with subword splitting
text = "unbelievable"
encoding = tokenizer(text)
print(encoding.tokens())
# ['[CLS]', 'un', '##believable', '[SEP]']
print(encoding.word_ids())
# [None, 0, 0, None]
# both 'un' and '##believable' map to word 0
```

---

## Handling Long Documents — Stride and Overflow

For documents longer than the model's max length, split into overlapping chunks.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

long_text = "Long document text..." * 100

# return_overflowing_tokens — creates multiple windows
# stride — overlap between windows so context is preserved
encoding = tokenizer(
    long_text,
    max_length=512,
    stride=128,                          # 128 token overlap between chunks
    return_overflowing_tokens=True,      # return all chunks
    return_offsets_mapping=True,         # char positions
    padding="max_length",
    truncation=True
)

print(len(encoding["input_ids"]))        # number of chunks
print(encoding["overflow_to_sample_mapping"])  # which chunk belongs to which sample

# iterate over chunks
for i in range(len(encoding["input_ids"])):
    chunk_ids = encoding["input_ids"][i]
    chunk_mask = encoding["attention_mask"][i]
```

---

## Special Tokens

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# add special tokens manually to encoding
encoding = tokenizer(
    "Hello world",
    add_special_tokens=True    # default True — adds [CLS] and [SEP]
)

encoding = tokenizer(
    "Hello world",
    add_special_tokens=False   # raw tokens only, no [CLS] or [SEP]
)

# add new tokens to an existing tokenizer
tokenizer.add_tokens(["[PRODUCT]", "[COMPANY]", "[PRICE]"])
model.resize_token_embeddings(len(tokenizer))   # resize model embeddings too

# add new special tokens
tokenizer.add_special_tokens({"additional_special_tokens": ["<START>", "<END>"]})

# check token ID of a specific token
id = tokenizer.convert_tokens_to_ids("[MASK]")
print(id)  # 103 for BERT
```

---

## Fast vs Slow Tokenizers

```python
# fast tokenizer — Rust-based, default when available
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.is_fast)   # True

# slow tokenizer — Python-based, fallback
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    use_fast=False
)
print(tokenizer.is_fast)   # False

# speed difference — fast is 5-10x faster for large batches
# fast also supports offset_mapping and word_ids natively
```

---

## Chat Templates — Modern LLMs

Modern instruction-following LLMs expect a specific message format.
Chat templates format a list of messages into the correct prompt string.

```python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user",      "content": "And what about Germany?"},
]

# apply the model's chat template
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,         # return string, not token IDs
    add_generation_prompt=True  # add the assistant turn start token
)
print(formatted)

# tokenize the formatted prompt
encoding = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt"
)

# Qwen2.5 example
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```

---

## Saving and Loading Custom Tokenizers

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# add custom tokens
tokenizer.add_tokens(["mycustomtoken1", "mycustomtoken2"])

# save to disk
tokenizer.save_pretrained("./my-tokenizer")
# creates: tokenizer_config.json, vocab.txt, tokenizer.json, special_tokens_map.json

# load it back
tokenizer = AutoTokenizer.from_pretrained("./my-tokenizer")

# push to Hub
tokenizer.push_to_hub("username/my-tokenizer")
```

---

## Training a Tokenizer from Scratch

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# BPE tokenizer (GPT-style)
raw_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
raw_tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# train on your text files
raw_tokenizer.train(files=["corpus.txt"], trainer=trainer)

# wrap in HuggingFace interface
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

tokenizer.save_pretrained("./my-custom-tokenizer")
```

---

## Tokenizer Behavior by Model Family

| Model Family | CLS Token | SEP Token | Mask Token | Padding Side |
|---|---|---|---|---|
| BERT | [CLS] | [SEP] | [MASK] | right |
| RoBERTa | <s> | </s> | <mask> | right |
| GPT-2 | none | none | none | left |
| T5 | none | </s> | <extra_id_0> | right |
| LLaMA | \<s\> | \</s\> | none | left |
| Mistral | \<s\> | \</s\> | none | left |
| ALBERT | [CLS] | [SEP] | [MASK] | right |

---

## Common Errors and Fixes

```python
# Error: Token indices sequence length is longer than the specified maximum
# Fix: add truncation=True
encoding = tokenizer(long_text, truncation=True, max_length=512)

# Error: GPT-2 has no padding token
# Fix: set pad token manually
tokenizer.pad_token = tokenizer.eos_token
# or
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))

# Error: Batch sizes differ (can't stack tensors)
# Fix: add padding=True
encoding = tokenizer(texts, padding=True, return_tensors="pt")
```

---

## Quick Reference

```python
# full standard pattern for training data preparation
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

encoding = tokenizer(
    texts,                          # list of strings
    padding=True,                   # pad to same length
    truncation=True,                # cut if too long
    max_length=128,                 # enforce fixed length
    return_tensors="pt",            # PyTorch tensors
    return_attention_mask=True,     # include attention mask
    return_token_type_ids=True,     # include segment ids
)

input_ids      = encoding["input_ids"]        # shape: (batch, seq_len)
attention_mask = encoding["attention_mask"]   # shape: (batch, seq_len)
token_type_ids = encoding["token_type_ids"]   # shape: (batch, seq_len)
```

---

## What is Next

Part 3 covers AutoModel — all task-specific model classes for text, vision,
audio, and multimodal tasks. We go from pipeline (black box) to AutoModel
(full control over inputs and outputs).