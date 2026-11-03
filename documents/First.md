# **HuggingFace Learning Roadmap**

### **From Zero to Production — Python ML Engineer Path**

## **What We Are Building Toward**

```
datasets → transformers → fine-tune → optimize → deploy
```

Every library in this roadmap feeds the next one.
You are not learning tools in isolation. You are building one complete pipeline.

This roadmap covers the core ML stack only: data, models, training, evaluation, optimization, and low-level performance.

## **Read This First — Library Versions and Breaking Changes**

Most tutorials online were written for older versions. The libraries below had major releases, and old code will fail or warn. This roadmap is written against the current versions. Always check your installed versions before debugging.

```bash
pip list | grep -i -E "transformers|datasets|huggingface|trl|peft|accelerate|sentence|evaluate|kernels"
```

| Library | Current major line | What changed (the things that break old tutorials) |
|---|---|---|
| transformers | 5.x | PyTorch only (TensorFlow and Flax removed). The summarization, translation, question-answering and text2text-generation pipelines were removed. `torch_dtype` became `dtype`. `Trainer(tokenizer=...)` became `processing_class=...`. `encode_plus()` removed, call the tokenizer directly. `TRANSFORMERS_CACHE` ignored, use `HF_HOME`. Requires huggingface_hub 1.x |
| huggingface_hub | 1.x | `huggingface-cli` removed, use `hf`. HTTP backend moved from requests to httpx. `hf_xet` is the default transfer backend, `hf_transfer` removed. `Repository` and `HfFolder` removed. Python 3.9 or newer |
| datasets | 4.x | Loading scripts and `trust_remote_code` removed (datasets must be in Parquet, CSV, JSON etc.). `ds["col"]` returns a lazy `Column` object, not a list. `Sequence` replaced by `List`. Audio and video decoding uses torchcodec |
| trl | 1.x | Stable trainers: SFT, DPO, Reward, RLOO, GRPO. Most other methods (KTO, ORPO, CPO, IPO, GKD and more) live under `trl.experimental`. `SFTConfig(max_seq_length=...)` became `max_length=...`. `DataCollatorForCompletionOnlyLM` replaced by `completion_only_loss` and `assistant_only_loss` options |
| sentence-transformers | 5.x | Training uses `SentenceTransformerTrainer`. The old `model.fit()` path is deprecated. v5 adds `SparseEncoder`. From v5.4 internal modules moved (for example `sentence_transformers.sentence_transformer.losses`), top-level imports are unchanged |
| evaluate | 0.4.x | Still works for classic metrics. Hugging Face now recommends LightEval for LLM evaluation |
| kernels | 0.x | Fast moving. Kernel builds are tied to your torch and CUDA version, so check the kernel repo if loading fails |

Install the base environment once:

```bash
pip install -U huggingface_hub datasets transformers accelerate evaluate
pip install -U peft trl safetensors tokenizers
```

## **PHASE 0 — Platform Foundation**

### **1. HuggingFace Hub + huggingface_hub**

**What it is:** The GitHub of ML. Central place for models, datasets, and Spaces. The `huggingface_hub` library is the Python and CLI client for all of it.

**What we covered:**
- All 5 login methods: `login()`, `notebook_login()`, `interpreter_login()`, env var `HF_TOKEN`, CLI
- `hf auth login` CLI (the old `huggingface-cli` no longer exists in v1)
- Downloading files: `hf_hub_download()`, `snapshot_download()`
- Uploading files: `upload_file()`, `upload_folder()`
- Creating and managing repos: `create_repo()`, `delete_repo()`
- Gated model access and `GatedRepoError` fix
- `CommitOperationAdd` for atomic multi-file commits
- Space management: `get_space_runtime()`, `pause_space()`, `restart_space()`

**Topic checklist:**

**Authentication**
- `login()`, `notebook_login()`, `interpreter_login()`
- Environment variable `HF_TOKEN`
- CLI: `hf auth login`, `hf auth whoami`
- Switching between multiple saved tokens
- `logout()` and `whoami()`
- Fine-grained tokens versus read and write tokens (give each project the least access it needs)

**Downloading**
- `hf_hub_download()` for a single file
- `snapshot_download()` for an entire repo
- Specific revisions and branches
- `allow_patterns` and `ignore_patterns` filters
- Gated model access and `GatedRepoError` handling
- Offline mode with cached files (`HF_HUB_OFFLINE=1`)
- CLI: `hf download`
- Cache location is controlled by `HF_HOME`
- Fast transfers use `hf_xet` automatically (the old `hf_transfer` is gone)

**Uploading**
- `upload_file()` for a single file
- `upload_folder()` for a directory
- `CommitOperationAdd` for atomic multi-file commits
- `create_commit()` for full commit control
- Ignoring files during upload
- CLI: `hf upload`

**Repository management**
- `create_repo()` for model, dataset, space
- `delete_repo()`
- `list_models()`, `list_datasets()`, `list_spaces()`
- `model_info()`, `dataset_info()`
- `repo_exists()`

**Spaces management (API only)**
- `get_space_runtime()`
- `pause_space()`, `restart_space()`
- Space hardware management

**Error handling in v1**
- Catch `HfHubHTTPError` (and subclasses such as `GatedRepoError`, `RepositoryNotFoundError`) instead of `requests` exceptions

**Key install:**
```bash
pip install -U huggingface_hub
```

**Key imports:**
```python
from huggingface_hub import (
    login, logout, whoami,
    hf_hub_download, snapshot_download,
    upload_file, upload_folder,
    create_repo, delete_repo,
    HfApi, CommitOperationAdd
)
```

**CLI quick reference:**
```bash
hf auth login
hf auth whoami
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./qwen
hf upload username/my-model ./my-model
```

### **2. Datasets**

**What it is:** The data layer. Load, process, and share datasets for any ML task. Built on Apache Arrow, so large datasets are memory-mapped from disk instead of loaded into RAM.

**What we covered:**

**Loading — all sources:**
```python
from datasets import load_dataset, Dataset, load_from_disk

load_dataset("stanfordnlp/imdb")                  # from Hub
load_dataset("stanfordnlp/imdb", split="train")   # a specific split
load_dataset("stanfordnlp/imdb", split="train[:10%]")  # split slice
load_dataset("csv", data_files="data.csv")        # local CSV
load_dataset("json", data_files="data.jsonl")     # local JSONL
load_dataset("parquet", data_files="data.parquet")  # local Parquet
load_dataset("text", data_files="notes.txt")      # plain text, one row per line
load_dataset("csv", data_files={"train": "tr.csv", "test": "te.csv"})  # files mapped to splits
Dataset.from_dict({...})                          # from Python dict
Dataset.from_pandas(df)                           # from Pandas
Dataset.from_list([...])                          # from list of dicts
Dataset.from_generator(gen_fn)                    # from a generator
load_from_disk("./saved")                         # from disk (save_to_disk output)
```

Important in datasets 4.x: dataset loading scripts are gone. If you see "Dataset scripts are no longer supported", look for a Parquet version of the dataset on the Hub or convert it yourself. Do not pass `trust_remote_code`.

**Inspecting:**
```python
ds.features         # schema
ds.column_names
ds.shape, ds.num_rows, ds.num_columns
ds[0]               # first row as dict
ds[-1]              # last row
ds[:5]              # slice of rows (dict of lists)
ds["text"]          # lazy Column object (datasets 4.x), not a list
list(ds["text"])    # materialize the full column as a list
ds[0]["text"]       # one value
```

**Row operations:**
```python
ds.sort("label")
ds.sort("label", reverse=True)
ds.shuffle(seed=42)
ds.select([0, 1, 2])
ds.filter(lambda x: x["label"] == 1)
ds.filter(lambda x, idx: idx % 2 == 0, with_indices=True)
ds.train_test_split(test_size=0.2, seed=42)
ds.train_test_split(test_size=0.2, stratify_by_column="label")
ds.shard(num_shards=4, index=0)
```

**Column operations:**
```python
from datasets import Value, ClassLabel, List

ds.rename_column("text", "review")
ds.rename_columns({"text": "review", "label": "target"})
ds.remove_columns(["label"])
ds.add_column("length", values)
ds.cast_column("label", Value("float32"))
ds.flatten()
ds.select_columns(["text", "label"])
ds.align_labels_with_mapping(label2id, "label")
```

Schema types to know: `Value`, `ClassLabel`, `List` (replaces the old `Sequence` in 4.x), `Image`, `Audio`.

**Transformation:**
```python
ds.map(fn)                                  # row by row
ds.map(fn, batched=True)                    # batch processing (much faster)
ds.map(fn, batched=True, batch_size=1000)
ds.map(fn, num_proc=4)                      # multiprocessing
ds.map(fn, remove_columns=["text"])         # map and drop columns
ds.map(fn, with_indices=True)
ds.with_transform(fn)                       # on-the-fly, applied when rows are accessed
ds.with_format("torch")                     # non-destructive format change
```

A batched `map` can also return more rows than it received (generate several rows from one), as long as every column grows consistently.

**Combining:**
```python
from datasets import concatenate_datasets, interleave_datasets

concatenate_datasets([ds1, ds2])
interleave_datasets([ds1, ds2], probabilities=[0.7, 0.3], seed=42)
```

**Export and format:**
```python
ds.to_pandas()
ds.to_dict()
ds.to_list()
ds.to_csv("out.csv")
ds.to_json("out.jsonl")
ds.to_parquet("out.parquet")
ds.set_format("torch", columns=["input_ids", "label"])
ds.set_format("numpy")
ds.set_format("pandas")
ds.reset_format()
```

**Save and push:**
```python
ds.save_to_disk("./my_dataset")
load_from_disk("./my_dataset")
ds.push_to_hub("username/my-dataset")
ds.push_to_hub("username/my-dataset", private=True)
ds.push_to_hub("username/my-dataset", num_proc=8)   # faster upload in 4.x
dataset_dict.push_to_hub("username/my-dataset")     # DatasetDict: all splits at once
```

Also learn: dataset cards and README metadata for your pushed datasets.

**Streaming (large datasets):**
```python
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
for example in ds:
    print(example["text"])
list(ds.take(10))
ds.skip(1000).take(10)
ds.map(fn)            # lazy, runs as you iterate
ds.filter(fn)
ds.shuffle(seed=42, buffer_size=10_000)
```

Streaming returns an `IterableDataset`. It supports `map`, `filter`, `shuffle`, `take`, `skip`, and `interleave_datasets`. To keep a subset, build a regular `Dataset` from it with `Dataset.from_generator`.

**Python dict deep dive:**
```python
from collections import Counter

ds[0]                   # first row as dict
ds[:5]                  # dict of lists
ds.features.keys()      # column names
ds.features.items()     # name + type pairs
ds.to_dict()            # full dict
ds.to_list()            # list of row dicts
"text" in ds.features   # check column exists
Counter(ds["label"])    # count values
[row for row in ds if row["label"] == 1]   # comprehension
```

**Key install:**
```bash
pip install -U datasets
```

## **PHASE 1 — Core ML**

Status: NEXT

### **3. Transformers**

**What it is:** The most important library in the ecosystem. Access to 1M+ pretrained models for text, vision, audio, and multimodal tasks. Version 5 is PyTorch only.

**Learning path inside Transformers:**

**Step A — Pipeline API (start here, 2 lines of code for common tasks):**
```python
from transformers import pipeline

# text
classifier = pipeline("text-classification")
generator  = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
ner        = pipeline("token-classification", aggregation_strategy="simple")
fill       = pipeline("fill-mask", model="bert-base-uncased")
zero_shot  = pipeline("zero-shot-classification")
embedder   = pipeline("feature-extraction")

# vision
img_class  = pipeline("image-classification")
obj_detect = pipeline("object-detection")

# audio
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

# run
result = classifier("HuggingFace is amazing!")
```

Pipeline tasks that no longer exist in Transformers 5: `summarization`, `translation`, `question-answering`, `text2text-generation` (and `conversational` before that). Check what your version supports:

```python
import transformers
from transformers.pipelines import get_supported_tasks

print(transformers.__version__)
print(get_supported_tasks())
```

Two replacements for the removed pipelines:

```python
# Option 1: a modern chat model through the text-generation pipeline
chat = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
messages = [{"role": "user", "content": "Summarize in one sentence: " + text}]
out = chat(messages, max_new_tokens=100)
print(out[0]["generated_text"][-1]["content"])

# Option 2: a seq2seq model called directly with generate()
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok   = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
inputs = tok("summarize: " + text, return_tensors="pt", truncation=True)
ids = model.generate(**inputs, max_new_tokens=80)
print(tok.decode(ids[0], skip_special_tokens=True))
```

If you must follow an old tutorial that uses the removed pipelines, install the older line in a separate environment: `pip install -U "transformers<5"`.

**Pipeline task overview (verify with `get_supported_tasks()`):**

Text
- text-classification — sentiment, topic, intent
- token-classification — NER, POS tagging
- text-generation — open-ended generation and chat
- fill-mask — masked language modeling
- zero-shot-classification — no training needed
- feature-extraction — embeddings from any model

Vision
- image-classification
- object-detection
- image-segmentation
- zero-shot-image-classification
- zero-shot-object-detection
- depth-estimation
- image-to-text — captioning

Audio
- automatic-speech-recognition — Whisper and others
- audio-classification
- text-to-audio

Multimodal
- image-text-to-text — vision-language chat models

**Step B — AutoTokenizer + AutoModel (understand what pipeline does internally):**
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

inputs = tokenizer("Hello world", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
probs = outputs.logits.softmax(-1)
```

**Step C — Task-specific model classes:**

Natural language understanding
```python
AutoModelForSequenceClassification   # text classification, sentiment
AutoModelForTokenClassification      # NER, POS tagging
AutoModelForQuestionAnswering        # extractive span QA
AutoModelForMultipleChoice           # multiple choice
```

Natural language generation
```python
AutoModelForCausalLM                 # GPT-style decoder models (text generation)
AutoModelForMaskedLM                 # BERT-style fill-mask
AutoModelForSeq2SeqLM                # encoder-decoder: T5, BART (summarization, translation)
AutoModelForSpeechSeq2Seq            # Whisper-style ASR
```

Vision
```python
AutoModelForImageClassification
AutoModelForObjectDetection
AutoModelForSemanticSegmentation
AutoModelForDepthEstimation
```

Vision plus language (multimodal)
```python
AutoModelForImageTextToText          # LLaVA, Qwen-VL, PaliGemma, SmolVLM style models
AutoModelForVisualQuestionAnswering
AutoModelForDocumentQuestionAnswering
```

Audio
```python
AutoModelForAudioClassification
AutoModelForCTC                      # Wav2Vec2-style ASR
```

Base and preprocessing
```python
AutoModel                # raw hidden states, embeddings
AutoConfig               # architecture config
AutoImageProcessor       # image preprocessing
AutoFeatureExtractor     # audio preprocessing
AutoProcessor            # tokenizer + image/audio processor for multimodal models
```

**Step D — Trainer (fine-tuning with a built-in training loop):**
```python
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)

ds    = load_dataset("stanfordnlp/imdb")
tok   = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=256)

tokenized = ds.map(tokenize, batched=True)
accuracy  = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=True,                    # mixed precision on Ampere or newer GPUs, use fp16=True on older GPUs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    processing_class=tok,         # replaces the old tokenizer= argument
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./my-model")
trainer.push_to_hub("username/my-model")
```

**Trainer topic checklist:**
- `TrainingArguments` — all training hyperparameters
- `Trainer` — standard supervised training loop
- `Seq2SeqTrainer` and `Seq2SeqTrainingArguments` — for seq2seq models that use `generate()`
- `TrainerCallback` and `EarlyStoppingCallback`
- `compute_metrics` — custom evaluation function
- Data collators:
  - `DataCollatorWithPadding` — classification
  - `DataCollatorForSeq2Seq` — summarization, translation
  - `DataCollatorForLanguageModeling` — causal and masked LM
  - `DataCollatorForTokenClassification` — NER

**Model loading and saving:**
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    dtype="auto",          # load in the saved precision (v5 name, was torch_dtype)
    device_map="auto",     # place layers across available GPUs and CPU
)
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
model.push_to_hub("username/my-model")
tokenizer.push_to_hub("username/my-model")
```
Also write a model card (README.md) for every model you push.

**Generation (text and sequences):**
```python
from transformers import GenerationConfig, TextStreamer

inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=100,       # prefer this over max_length
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))

# reusable settings
gen_cfg = GenerationConfig(max_new_tokens=100, do_sample=True, temperature=0.7)
model.generate(**inputs, generation_config=gen_cfg)

# stream tokens as they are produced
model.generate(**inputs, max_new_tokens=100, streamer=TextStreamer(tokenizer))
```

Decoding strategies to know: greedy, beam search (`num_beams`), sampling (`do_sample=True`), temperature, top-k, top-p (nucleus), repetition penalty, length penalty, constrained generation.

Chat models: format conversations with `tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")`.

**Popular model architectures to know:**

Encoder-only (understanding tasks)
- BERT, RoBERTa, DistilBERT, ALBERT
- DeBERTa — strong on NLU benchmarks
- XLM-RoBERTa — multilingual

Decoder-only (generation tasks)
- GPT-2 — classic open generation
- Llama 3 family — Meta
- Mistral — efficient 7B
- Qwen family (Qwen2.5, Qwen3) — Alibaba
- Gemma family — Google
- Phi family — Microsoft small models

Encoder-decoder (seq2seq tasks)
- T5 and Flan-T5
- BART
- mT5 — multilingual
- MarianMT — translation

Vision
- ViT, DeiT, Swin Transformer
- DINOv2 — self-supervised vision
- CLIP and SigLIP — vision-language alignment

Multimodal
- LLaVA, Idefics, PaliGemma, Qwen-VL, SmolVLM

Audio
- Whisper — ASR across many languages
- Wav2Vec 2.0, HuBERT — speech representation

**What to practice:**
- Sentiment analysis on IMDB with BERT or DistilBERT
- Text generation with GPT-2 and a small chat model
- Named entity recognition
- Question answering with `AutoModelForQuestionAnswering`
- Summarization with T5 using `generate()`
- Speech recognition with Whisper
- Image classification with ViT

**Key install:**
```bash
pip install -U transformers torch
pip install -U "transformers[torch]"   # all torch deps
```

### **4. Tokenizers**

**What it is:** The fast tokenization library (written in Rust) that powers all `AutoTokenizer` classes. Understanding it deeply shows you how models actually read text.

**What to learn:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# encoding
enc = tokenizer(
    "Hello world how are you?",
    truncation=True,
    padding="max_length",
    max_length=128,
    return_tensors="pt",
)

print(enc.input_ids)         # token IDs
print(enc.attention_mask)    # 1 for real tokens, 0 for padding
print(enc.token_type_ids)    # segment IDs for sentence pairs

# sentence pairs
tokenizer("question text", "context text", truncation="only_second")

# decoding
tokenizer.decode(enc.input_ids[0], skip_special_tokens=True)

# special tokens
tokenizer.cls_token     # [CLS]
tokenizer.sep_token     # [SEP]
tokenizer.pad_token     # [PAD]
tokenizer.unk_token     # [UNK]
tokenizer.mask_token    # [MASK]

# batch encoding
tokenizer(["sentence one", "sentence two"], padding=True, truncation=True)

# map tokens back to the source text
enc = tokenizer("Hello world", return_offsets_mapping=True)
print(enc["offset_mapping"])   # character spans per token
print(enc.word_ids())          # token to word index

# long documents: split into overlapping chunks
tokenizer(long_text, truncation=True, max_length=512,
          stride=128, return_overflowing_tokens=True)

# add new tokens (then resize the model embeddings)
tokenizer.add_tokens(["new_token"])
tokenizer.add_special_tokens({"additional_special_tokens": ["tool_token"]})
# model.resize_token_embeddings(len(tokenizer))
```

In v5, call the tokenizer directly. `encode_plus()` and `batch_encode_plus()` are removed.

**Train your own tokenizer from scratch:**
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)
tok.train(files=["corpus.txt"], trainer=trainer)
tok.save("tokenizer.json")

# wrap it so it works with Transformers
fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
fast.save_pretrained("./my-tokenizer")
```

**Key concepts to understand:**
- BPE (Byte-Pair Encoding) — GPT-style models
- Byte-level BPE — GPT-2, RoBERTa
- WordPiece — BERT
- SentencePiece and Unigram — T5, LLaMA 1 and 2, many multilingual models
- Character-level tokenization
- Vocabulary and `vocab_size`
- Subword tokenization — why words get split
- Why padding and truncation exist
- What `attention_mask` does and why it matters
- `token_type_ids` for sentence pairs
- `offset_mapping` and `word_ids()` for alignment (essential for NER labels)
- Padding strategies: `longest`, `max_length`, `do_not_pad`
- Truncation strategies: `longest_first`, `only_first`, `only_second`
- Overflow tokens and stride for long documents
- Chat templates for instruction-tuned models
- Fast (Rust) versus slow (Python) tokenizers. In v5 the separate slow and fast files are unified around the fast implementation

**Key install:**
```bash
pip install -U tokenizers
# installed automatically with transformers
```

### **5. Evaluate**

**What it is:** Standardized metrics for measuring model performance. Good for classic NLP and vision metrics. For benchmarking LLMs, also look at LightEval (see the end of this section).

**What to learn:**
```python
import evaluate

# load metrics
accuracy  = evaluate.load("accuracy")
f1        = evaluate.load("f1")
rouge     = evaluate.load("rouge")
bleu      = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# compute
result = accuracy.compute(
    predictions=[0, 1, 1, 0],
    references= [0, 1, 0, 1]
)
print(result)  # {"accuracy": 0.5}

# streaming evaluation over batches
for preds, refs in batches:
    accuracy.add_batch(predictions=preds, references=refs)
accuracy.compute()

# use inside Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return accuracy.compute(predictions=predictions, references=labels)

# combine multiple metrics
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
clf_metrics.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 1])

# list available metrics
evaluate.list_evaluation_modules()
```

**Metric guide:**

Classification
- accuracy, precision, recall
- f1 — binary, macro, micro, weighted averaging
- matthews_correlation (MCC)

Question answering
- exact_match
- squad — exact match plus F1

Generation and translation
- bleu and sacrebleu — translation
- rouge (rouge1, rouge2, rougeL) — summarization
- meteor, ter

Semantic metrics
- bertscore — semantic similarity using BERT embeddings
- bleurt — learned metric

Code
- code_eval — for code generation (runs generated code, only use in a sandbox)

Also learn: creating custom metrics and sharing them on the Hub.

**LLM evaluation:** Hugging Face recommends LightEval for evaluating LLMs on benchmarks. Learn it after you finish fine-tuning.

**Key install:**
```bash
pip install -U evaluate
pip install scikit-learn rouge_score sacrebleu   # dependencies for common metrics
```

## **PHASE 2 — Training and Fine-tuning**

Learn after Phase 1.

### **6. Accelerate**

**What it is:** Run the exact same PyTorch training code on a single GPU, multi-GPU, TPU, or CPU with minimal code changes. Powers the Trainer inside Transformers.

**What to learn:**
```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",         # or "fp16" for older GPUs
    gradient_accumulation_steps=4,
)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)          # replaces loss.backward()
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

# checkpoint and resume across distributed setups
accelerator.save_state("./checkpoint")
accelerator.load_state("./checkpoint")
```

**Topic checklist:**

Core concepts
- `Accelerator` object — central controller
- `accelerator.prepare()` — wraps model, optimizer, dataloader
- `accelerator.backward()`
- Mixed precision (fp16, bf16)
- Gradient accumulation and gradient clipping

Training setups
- Single GPU, multi-GPU (DDP), multi-node
- TPU and CPU-only
- DeepSpeed integration (ZeRO stages 1, 2, 3)
- FSDP (Fully Sharded Data Parallel)

State management
- `save_state()` and `load_state()`
- `wait_for_everyone()` — sync across processes
- `gather()` — collect tensors from all GPUs
- `is_main_process` — run code only on rank 0

Utilities
- `find_executable_batch_size` — automatic batch size search on OOM
- `notebook_launcher` — run distributed training from a notebook
- `PartialState` — lightweight distributed state

**CLI tools:**
```bash
# configure your machine once
accelerate config

# verify the setup
accelerate test

# launch distributed training
accelerate launch train.py

# launch with specific settings
accelerate launch --num_processes=4 --mixed_precision=bf16 train.py
```

**Key install:**
```bash
pip install -U accelerate
```

### **7. PEFT — Parameter Efficient Fine-Tuning**

**What it is:** Fine-tune massive models (7B, 13B, 70B parameters) on limited GPUs by training only a tiny fraction of the weights. LoRA is the most used technique.

**What to learn:**

**LoRA — Low Rank Adaptation (most important):**
```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="auto")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                  # rank: lower means fewer params trained
    lora_alpha=32,                         # scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # or "all-linear" to adapt every linear layer
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# prints: trainable params: ... || all params: ... || trainable%: ...
```

Target module names differ per architecture. Print `model` and look at the layer names, or use `target_modules="all-linear"`.

**QLoRA — quantized LoRA (fine-tune a 7B model on a single consumer GPU):**
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM", r=16, lora_alpha=32,
    lora_dropout=0.05, target_modules="all-linear",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

**Save, load, and merge LoRA adapters:**
```python
model.save_pretrained("./lora-adapter")        # saves only the small adapter
model.push_to_hub("username/my-lora-adapter")

# load adapter on top of the base model later
from peft import PeftModel
base  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="auto")
model = PeftModel.from_pretrained(base, "username/my-lora-adapter")

merged = model.merge_and_unload()   # bake the adapter into the base weights
merged.save_pretrained("./merged-model")
```

To produce a merged model from a QLoRA run, reload the base model in 16-bit precision (not 4-bit), attach the adapter, then merge.

**Topic checklist:**

LoRA details
- Concept: train only low-rank decomposition matrices
- `r`, `lora_alpha`, `lora_dropout`, `target_modules`
- `LoraConfig`, `get_peft_model()`, `print_trainable_parameters()`
- Saving adapters separately, `PeftModel.from_pretrained()`, `merge_and_unload()`

QLoRA details
- 4-bit NF4 quantization and double quantization
- `BitsAndBytesConfig` integration
- `prepare_model_for_kbit_training()`

Other PEFT methods (know they exist)
- Prefix Tuning, Prompt Tuning, P-Tuning
- IA3
- AdaLoRA
- LoftQ — quantization-aware LoRA initialization

Task types
- `SEQ_CLS`, `TOKEN_CLS`, `CAUSAL_LM`, `SEQ_2_SEQ_LM`, `QUESTION_ANS`

Multi-adapter usage
- Load several adapters, switch between them, and combine them

**Key install:**
```bash
pip install -U peft
```

### **8. TRL — Transformer Reinforcement Learning**

**What it is:** Post-training for LLMs: supervised fine-tuning, preference optimization, and reinforcement learning. TRL reached v1.0 in March 2026 with a two-tier design:

- Stable (semver guarantees): SFT, DPO, Reward modeling, RLOO, GRPO and their close variants
- Experimental (`trl.experimental`, APIs can change): more than 75 other methods such as KTO, ORPO, CPO, IPO, GKD and PPO. Some get promoted to stable over time, so check the docs

```python
from trl import SFTTrainer                    # stable
from trl.experimental.orpo import ORPOTrainer # experimental
```

**What to learn:**

**SFTTrainer — Supervised Fine-Tuning (most common):**
```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")   # conversational "messages" format

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",                 # a model id string is enough
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./sft-model",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,                    # typical for LoRA
        bf16=True,
        max_length=1024,                       # was max_seq_length in old versions
        packing=True,                          # combine short examples for efficiency
    ),
    peft_config=LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"
    ),
)

trainer.train()
trainer.save_model("./my-sft-model")
```

SFT dataset formats: plain text (`dataset_text_field`), prompt and completion pairs, or conversational `messages`. Chat templates are applied for you. For plain text with custom structure use `formatting_func`.

Training only on the answer: set `completion_only_loss=True` for prompt and completion data, or `assistant_only_loss=True` for conversations (needs a chat template that marks assistant turns). These replace the old `DataCollatorForCompletionOnlyLM`.

Memory options in `SFTConfig`: `gradient_checkpointing`, `packing` with `max_length`, `use_liger_kernel`, `activation_offloading`, and the chunked loss options. See the TRL "Reducing Memory Usage" guide.

**DPO — Direct Preference Optimization:**
```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

# dataset needs: prompt, chosen, rejected
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=DPOConfig(output_dir="./dpo-model", beta=0.1),
    train_dataset=dataset,
)
trainer.train()
```

The reference model is created for you (or derived from the adapter when using PEFT). Truncation is controlled by `max_length` only. The old `max_prompt_length` and `max_completion_length` options were removed, so filter overlong examples beforehand.

**RewardTrainer — train a reward model from preference data:**
```python
from trl import RewardConfig, RewardTrainer
# same chosen / rejected data format as DPO
```

**GRPOTrainer — Group Relative Policy Optimization (reasoning-style RL, used by DeepSeek-R1 style training):**
```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")   # prompts only

def reward_len(completions, **kwargs):
    # toy reward: prefer completions close to 20 characters
    return [-abs(20 - len(c)) for c in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_len,                # a function, a reward model, or a list of them
    args=GRPOConfig(output_dir="./grpo-model"),
    train_dataset=dataset,
)
trainer.train()
```

**Topic checklist:**

Supervised fine-tuning
- `SFTTrainer`, `SFTConfig`, `formatting_func`
- Chat templates and `apply_chat_template()`
- Packing, `max_length`, PEFT integration
- Completion-only and assistant-only loss

Preference learning
- DPO (`DPOTrainer`, `DPOConfig`, `beta`, loss types)
- Dataset format: prompt, chosen, rejected
- Experimental: IPO, KTO (binary good or bad feedback), CPO, ORPO

Reinforcement learning
- `RewardTrainer` and `RewardConfig`
- GRPO (`GRPOTrainer`, `GRPOConfig`) with verifiable reward functions
- RLOO (stable)
- PPO (`PPOTrainer`) — older approach, now in the experimental area

Utilities and ideas
- Using vLLM for fast generation during online RL (`trl vllm-serve`)
- Keeping a frozen reference model for KL divergence

**Key install:**
```bash
pip install -U trl
```

### **9. Safetensors**

**What it is:** The standard format for storing neural network weights. Safe, fast, and with no arbitrary code execution risk (unlike pickle-based `.bin` and `.pt` files). All modern models on the Hub use it.

**What to learn:**
```python
import torch
from safetensors.torch import save_file, load_file

# save model weights
tensors = {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}
save_file(tensors, "model.safetensors", metadata={"format": "pt"})

# load model weights
tensors = load_file("model.safetensors")

# with transformers: automatic
model.save_pretrained("./model")                 # writes model.safetensors
model = AutoModel.from_pretrained("./model")     # loads it automatically

# inspect without loading everything (lazy, memory-mapped)
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    print(f.keys())                     # all tensor names
    print(f.metadata())                 # file metadata
    weight = f.get_tensor("weight")     # load one tensor only
```

**Topic checklist:**
- Why it exists: no code execution, unlike pickle
- Format: a JSON header followed by raw tensor bytes
- Zero-copy, memory-mapped loading
- Framework support: PyTorch, NumPy, JAX
- Sharded checkpoints for large models (`model.safetensors.index.json`)
- `save_pretrained()` uses safetensors by default and `from_pretrained()` prefers it
- Converting an old `.bin` checkpoint: load it once, then `save_pretrained()` again

**Key install:**
```bash
pip install -U safetensors
```

## **PHASE 3 — Specialized Libraries**

Pick based on your goal. Not all are required.

### **10. Sentence Transformers**

For: RAG, semantic search, embeddings, similarity, clustering, reranking.

**What it is:** The go-to library for text embeddings and semantic similarity. Sentence-BERT and all its variants live here. It also covers rerankers (`CrossEncoder`) and, since v5, sparse embedding models (`SparseEncoder`).

**What to learn:**
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")   # fast and good quality

# encode text to embeddings
sentences = ["This is a cat", "This is a dog", "Paris is beautiful"]
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)   # (3, 384)

# semantic similarity
print(model.similarity(embeddings, embeddings))   # 3 x 3 similarity matrix
sim = util.cos_sim(embeddings[0], embeddings[1])

# semantic search (find the most similar entries to a query)
query = model.encode("What is a pet?", normalize_embeddings=True)
hits = util.semantic_search(query, embeddings, top_k=2)
```

**Fine-tune on your own pairs (current Trainer-based API):**
```python
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

model = SentenceTransformer("all-MiniLM-L6-v2")

train_ds = Dataset.from_dict({
    "anchor":   ["How do I reset my password?", "Where is my order?"],
    "positive": ["Steps to reset your account password", "Track your package status"],
})

loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="./my-embedding-model",
    num_train_epochs=3,
    per_device_train_batch_size=32,
)

trainer = SentenceTransformerTrainer(
    model=model, args=args, train_dataset=train_ds, loss=loss,
)
trainer.train()
model.save_pretrained("./my-embedding-model")
model.push_to_hub("username/my-embedding-model")
```

Note on imports: the code above uses the pre-5.4 import path for losses. From v5.4 the recommended path is `sentence_transformers.sentence_transformer.losses`. Top-level imports such as `SentenceTransformer` are unchanged. The old `model.fit()` with `InputExample` and `DataLoader` still appears in old tutorials but is the deprecated route.

**Topic checklist:**

Core concepts
- Sentence embeddings versus token embeddings
- Cosine similarity and dot score
- Bi-encoder versus cross-encoder
- Pooling strategies: mean, CLS, max

Popular models
- `all-MiniLM-L6-v2` — fast, small, general purpose
- `all-mpnet-base-v2` — higher quality general purpose
- `BAAI/bge` series and `intfloat/e5` series — strong retrieval models
- Multilingual models such as `paraphrase-multilingual`
- Check the MTEB leaderboard for the current best model for your task

Operations
- `encode()`, `similarity()`, `util.semantic_search()`, `util.cos_sim()`
- Paraphrase mining and clustering embeddings

Cross encoders (rerankers)
- `CrossEncoder` for reranking
- Bi-encoder retrieval followed by cross-encoder reranking for RAG

Training
- Losses: `CosineSimilarityLoss`, `ContrastiveLoss`, `TripletLoss`, `MultipleNegativesRankingLoss` (best default for retrieval), `MatryoshkaLoss` (flexible embedding sizes)
- Evaluators: `EmbeddingSimilarityEvaluator`, `InformationRetrievalEvaluator`, `RerankingEvaluator`
- Hard negative mining
- `push_to_hub()` to share models

**Key install:**
```bash
pip install -U sentence-transformers
pip install -U "sentence-transformers[train]"   # training extras
```

### **11. Diffusers**

For: image generation, video generation, audio generation.

**What it is:** The Transformers equivalent for diffusion models. Stable Diffusion, SDXL, Flux, and ControlNet all live here.

**What to learn:**
```python
import torch
from diffusers import DiffusionPipeline

# text to image (the generic loader picks the right pipeline class)
pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("a photo of an astronaut riding a horse").images[0]
image.save("output.png")

# newer, higher quality open models
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",       # gated: accept the license on the Hub first
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()           # fit on smaller GPUs

# image to image
from diffusers import AutoPipelineForImage2Image
pipe = AutoPipelineForImage2Image.from_pretrained(...)
output = pipe(prompt="...", image=init_image, strength=0.75).images[0]

# inpainting
from diffusers import AutoPipelineForInpainting

# ControlNet (guided generation)
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
```

The original `runwayml/stable-diffusion-v1-5` repo is no longer the home of that model. Use the community-hosted `stable-diffusion-v1-5/stable-diffusion-v1-5`. If `torch_dtype` raises a warning in your version, check the Diffusers release notes, since Transformers renamed this argument to `dtype`.

**Topic checklist:**

Pipeline types
- Text to image, image to image, inpainting, outpainting
- Text to video and image to video
- Text to audio

Core concepts
- Forward noising and reverse denoising
- Denoising steps (more steps: better quality, slower)
- Guidance scale, negative prompts, seeds for reproducibility
- Latent diffusion: working in a compressed latent space

Schedulers
- DDPM, DDIM, DPMSolverMultistep, Euler, EulerAncestral, UniPC
- Swapping schedulers on an existing pipeline

Notable models and adapters
- Stable Diffusion 1.5, SDXL, SD3, FLUX.1
- ControlNet: canny edges, depth, pose, segmentation, scribble
- IP-Adapter for image prompts
- LoRA for diffusion models and textual inversion

Memory and speed
- `enable_attention_slicing()`, VAE slicing and tiling
- `enable_model_cpu_offload()`
- `torch.compile()`
- float16 versus bfloat16

**Key install:**
```bash
pip install -U diffusers
pip install -U "diffusers[torch]"
```

### **12. timm — PyTorch Image Models**

For: image classification, object detection backbones, feature extraction, transfer learning.

**What it is:** 1000+ state-of-the-art pretrained vision models. ResNet, ViT, EfficientNet, Swin Transformer, and ConvNeXt all in one place.

**What to learn:**
```python
import timm

# list available models
timm.list_models("resnet*")
timm.list_models("vit*", pretrained=True)

# load pretrained model
model = timm.create_model("resnet50", pretrained=True)
model = timm.create_model("vit_base_patch16_224", pretrained=True)

# custom classifier (transfer learning)
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=10,        # your number of classes
)

# feature extraction (no classifier head)
model = timm.create_model("resnet50", pretrained=True, num_classes=0)
features = model(image_tensor)   # (batch, 2048) embeddings

# intermediate feature maps for detection and segmentation backbones
backbone = timm.create_model("resnet50", pretrained=True,
                             features_only=True, out_indices=(1, 2, 3, 4))

# correct preprocessing config for each model
data_config = timm.data.resolve_model_data_config(model)
transforms  = timm.data.create_transform(**data_config, is_training=False)
```

**Topic checklist:**
- Architecture families: ResNet, EfficientNet, ViT, Swin, ConvNeXt, DenseNet, MobileNet, RegNet, MaxViT, DINOv2-based models
- Partial fine-tuning: freeze the trunk, train the head
- `global_pool` options
- Correct mean and std normalization and input size per model (always use `resolve_model_data_config`)
- Using timm models as backbones for other tasks

**Key install:**
```bash
pip install -U timm
```

## **PHASE 4 — Low Level and Advanced**

Learn when you hit GPU memory or speed limitations.

### **13. Bitsandbytes**

Used with PEFT for QLoRA. Learn it when you need to load big models on a small GPU.

**What to learn:**
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit quantization (most memory efficient)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4, best quality for 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,    # quantize the quantization constants too
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto",
)

# 8-bit quantization (less memory, better quality than 4-bit)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```

**8-bit and paged optimizers (save memory during training):**
```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)
# in TrainingArguments you can also set: optim="paged_adamw_8bit"
```

**Topic checklist:**
- 8-bit versus 4-bit quantization
- NF4 versus FP4, double quantization
- Compute dtypes: float16 for older GPUs, bfloat16 for A100, H100 and newer
- `BitsAndBytesConfig` with `device_map="auto"`
- `prepare_model_for_kbit_training()` before attaching PEFT adapters
- Quantized and paged optimizers

**Key install:**
```bash
pip install -U bitsandbytes
```

### **14. Kernels + CUDA + Triton**

Advanced. Learn when you need maximum speed or want to understand model internals.

**What it is:** The HuggingFace `kernels` library lets you load optimized compute kernels (written in CUDA or Triton) directly from the Hub, instead of compiling them yourself. This is where raw GPU performance lives.

**The learning path for CUDA + Kernels:**

**Step A — Understand what a kernel is:**
```
A GPU kernel is a function that runs in parallel across thousands of GPU threads.
When you call model(input), internally it runs dozens of kernels:
  - matrix multiplication kernel (the most called)
  - softmax kernel
  - layer norm kernel
  - attention kernel (Flash Attention is the famous example)
```

**Step B — Triton (Python-based GPU programming, learn before raw CUDA):**
```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                      # which block am I
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                      # guard the tail
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out
```

**Step C — HuggingFace Kernels library:**
```python
from kernels import get_kernel

# download a pre-built, optimized kernel from the Hub
kernel = get_kernel("kernels-community/flash-attn2")

# each kernel repo documents its own functions on its model card,
# for example flash_attn_func and flash_attn_varlen_func
# out = kernel.flash_attn_func(q, k, v, causal=True)
```

Kernels are built for specific torch and CUDA versions. If `get_kernel` fails, check the kernel repo's discussions for compatibility with your torch version, or pin the `kernels` package version. Transformers can also load community kernels directly, for example through `attn_implementation="kernels-community/flash-attn2"` in `from_pretrained`. Check the Transformers docs for the supported options in your version.

To write and publish your own kernels, study the `kernel-builder` tooling in the Kernels docs.

**Step D — Raw CUDA with PyTorch (deepest level):**
```python
from torch.utils.cpp_extension import load_inline

# the GPU kernel itself
cuda_kernel = """
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""
```

To make this callable from Python you also need a small C++ wrapper function that allocates the output tensor, computes the grid and block sizes, and launches the kernel, then expose that wrapper with `load_inline(name=..., cpp_sources=..., cuda_sources=..., functions=[...])`. Follow the official PyTorch custom C++ and CUDA extension tutorial for the full wrapper, since the launch syntax is easiest to learn from a complete working example.

**Notable optimized kernels to know about:**
- **Flash Attention** (and Flash Attention 2) — fast, memory-efficient attention used in modern LLMs
- **Fused LayerNorm and RMSNorm** — combine normalization ops into one kernel
- **Rotary Embeddings (RoPE)** — position encoding used by Llama, Mistral, Qwen
- **Paged Attention** — vLLM's memory management trick for serving
- **Fused cross-entropy** — lower memory during training
- **Triton matmul** — can beat cuBLAS for certain shapes

**Key GPU concepts to understand:**
- Thread, block, grid hierarchy
- Memory types: global, shared, registers, constant
- Memory coalescing — why access patterns matter
- Occupancy — how many threads run simultaneously
- Warp — 32 threads that execute together
- Bank conflicts in shared memory
- Roofline model — memory bound versus compute bound

**Triton topic checklist:**
- `@triton.jit`, `tl.program_id()`, `tl.arange()`
- `tl.load()` and `tl.store()` with masks
- `BLOCK_SIZE` as `tl.constexpr`
- `triton.autotune()` for automatic config search
- `triton.testing.do_bench()` to benchmark
- Fusion: combining multiple ops into one kernel
- Practice projects: vector addition, softmax, matrix multiplication, then Flash Attention from scratch

**Raw CUDA topic checklist:**
- `__global__`, `__device__`, `__host__`
- Launching kernels with grid and block dimensions
- `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- Shared memory (`__shared__`) and `__syncthreads()`
- Atomic operations
- cuBLAS and cuDNN
- PyTorch C++ extensions and pybind11

**What to study for CUDA:**
```
1. CUDA by Example (book)
2. Triton tutorials: https://triton-lang.org/main/getting-started/tutorials
3. HuggingFace Kernels docs: https://huggingface.co/docs/kernels
4. PyTorch custom C++ and CUDA extension guide
5. NVIDIA CUDA Programming Guide
6. Flash Attention paper (Dao et al, 2022), the most important paper to read
```

**Key install:**
```bash
pip install -U kernels
pip install -U triton        # for writing custom GPU kernels in Python
```

## **The Full Pipeline — How Everything Connects**

```
FOUNDATION
  huggingface_hub  (auth, download, upload, repos)
  datasets         (load, process, map, filter, push_to_hub)
        |
        v
CORE ML
  transformers     (pipeline, AutoModel, AutoTokenizer, Trainer, generate)
  tokenizers       (how text becomes numbers)
  evaluate         (accuracy, F1, ROUGE, BLEU)
        |
        v
TRAINING
  accelerate       (multi-GPU, mixed precision, DeepSpeed, FSDP)
  peft             (LoRA, QLoRA, adapters)
  trl              (SFT, DPO, GRPO, reward models)
  safetensors      (save and load weights safely)
        |
        v
SPECIALIZED (pick based on goal)
  sentence-transformers   (embeddings, search, RAG)
  diffusers               (image generation)
  timm                    (vision models and features)
        |
        v
LOW LEVEL (when you hit limits)
  bitsandbytes     (quantization for big models)
  kernels          (optimized GPU ops from the Hub)
  triton + CUDA    (custom GPU kernels)
        |
        v
SHARE
  huggingface_hub  (push model, model card, versioned releases)
```

## **Recommended Learning Order — Week by Week**

```
Weeks 01-02   huggingface_hub                       DONE
Weeks 03-04   datasets, all operations              DONE
Weeks 05-06   transformers: pipeline API
Weeks 07-08   transformers: AutoModel, Trainer, generate()
Week  09      tokenizers deep dive
Week  10      evaluate + first full fine-tune with Trainer
Week  11      accelerate + multi-GPU training
Weeks 12-13   peft + LoRA + QLoRA
Week  14      trl: SFTTrainer on custom data
Week  15      trl: DPO, then GRPO and reward models
Week  16      safetensors + model saving and sharing
Week  17      pick one: sentence-transformers OR diffusers OR timm
Week  18      bitsandbytes (quantization, 8-bit optimizers)
Weeks 19+     kernels, triton, CUDA
```

## **GitHub Repo Structure for All Learnings**

```
huggingface-learning/
├── 01_hub/
│   ├── login_methods.py
│   ├── download_upload.py
│   └── spaces_runtime.py
├── 02_datasets/
│   ├── loading_all_sources.py
│   ├── row_column_ops.py
│   ├── map_filter_transform.py
│   ├── streaming.py
│   └── dict_methods.py
├── 03_transformers/
│   ├── pipeline/
│   ├── automodel/
│   │   ├── text/
│   │   ├── vision/
│   │   ├── audio/
│   │   └── multimodal/
│   ├── generation/
│   └── trainer/
├── 04_tokenizers/
│   └── tokenization_deep_dive.py
├── 05_evaluate/
│   └── metrics.py
├── 06_accelerate/
│   └── distributed_training.py
├── 07_peft/
│   ├── lora/
│   └── qlora/
├── 08_trl/
│   ├── sft/
│   ├── dpo/
│   └── grpo/
├── 09_safetensors/
│   └── save_load_weights.py
├── 10_specialized/
│   ├── sentence_transformers/
│   ├── diffusers/
│   └── timm/
├── 11_advanced/
│   ├── bitsandbytes/
│   ├── kernels/
│   ├── triton/
│   └── cuda/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## **Key Rule**

**Code goes to GitHub. Model weights go to HuggingFace Hub.**

Never push `.bin`, `.safetensors`, or `.pt` files to GitHub.
Push them with `model.push_to_hub("username/my-model")` instead.

Never commit your `HF_TOKEN`. Keep it in `.env` and list `.env` in `.gitignore`.

*Last updated: November 2026*
*Covers: huggingface_hub, datasets, transformers, tokenizers, evaluate,*
*accelerate, peft, trl, safetensors, sentence-transformers, diffusers,*
*timm, bitsandbytes, kernels, triton, CUDA*