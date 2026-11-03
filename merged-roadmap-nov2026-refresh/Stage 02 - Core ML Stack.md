# **Stage 02 — Core ML Stack**

**Libraries:** `transformers` 5.22.1 · `tokenizers` 0.23.3 · `evaluate` 0.4.6 · `lighteval` 0.13.0 · `lm-evaluation-harness` 0.4.13
**Outcome:** run any model, fine-tune it with the built-in trainer, and measure it with a metric that matches the task.

## **Transformers 5.22.1**

The most important library in the ecosystem: 1M+ pretrained models for text, vision, audio, and multimodal tasks. Version 5 is PyTorch-only and requires **Python ≥3.11** and `huggingface_hub` in the 1.31–2.x range [1][2].

**Step A — Pipeline API**
```python
from transformers import pipeline

# text
classifier = pipeline("text-classification")
generator  = pipeline("text-generation", model="Qwen/Qwen3-1.7B")
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

**Pipeline task inventory (verify with `get_supported_tasks()`)**

```python
import transformers
from transformers.pipelines import get_supported_tasks

print(transformers.__version__)
print(get_supported_tasks())
```

Text
- text-classification — sentiment, topic, intent (`sentiment-analysis` alias)
- token-classification — NER, POS tagging (`ner` alias)
- question-answering — extractive QA
- table-question-answering
- text-generation — open-ended generation and chat
- fill-mask — masked language modeling
- zero-shot-classification — no training needed
- feature-extraction — embeddings from any model

Vision
- image-classification
- object-detection
- image-segmentation
- mask-generation
- zero-shot-image-classification
- zero-shot-object-detection
- depth-estimation
- image-to-image
- image-feature-extraction
- keypoint-matching
- video-classification

Audio
- automatic-speech-recognition — Whisper and others
- audio-classification
- zero-shot-audio-classification
- text-to-audio
- text-to-speech

Multimodal
- image-text-to-text — vision-language chat, captioning
- visual-question-answering
- document-question-answering
- any-to-any

**Pipelines removed in Transformers 5** — `summarization`, `translation`, `text2text-generation`, `image-to-text` (replaced by `image-text-to-text`), and `conversational` (removed earlier). Two replacements:

```python
# Option 1: a modern chat model through the text-generation pipeline
chat = pipeline("text-generation", model="Qwen/Qwen3-1.7B")
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

**Step B — AutoTokenizer + AutoModel**
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

**Step C — Task-specific model classes**

Natural language understanding
```python
AutoModelForSequenceClassification   # text classification, sentiment
AutoModelForTokenClassification      # NER, POS tagging
AutoModelForQuestionAnswering        # extractive span QA
AutoModelForMultipleChoice           # multiple choice
```

Natural language generation
```python
AutoModelForCausalLM                 # GPT-style decoder models
AutoModelForMaskedLM                 # BERT-style fill-mask
AutoModelForSeq2SeqLM                # encoder-decoder: T5, BART
AutoModelForSpeechSeq2Seq            # Whisper-style ASR
```

Vision
```python
AutoModelForImageClassification
AutoModelForObjectDetection
AutoModelForSemanticSegmentation
AutoModelForInstanceSegmentation
AutoModelForDepthEstimation
AutoModelForImageToImage
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

**Step D — Trainer**
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

**Trainer topic checklist**
- `TrainingArguments` — all training hyperparameters
- `Trainer` — standard supervised training loop
- `Seq2SeqTrainer` and `Seq2SeqTrainingArguments` — for seq2seq models that use `generate()`
- `TrainerCallback` and `EarlyStoppingCallback`
- `compute_metrics` — custom evaluation function
- Mixed precision: `bf16=True` on newer GPUs, `fp16=True` on older GPUs
- Data collators:
  - `DataCollatorWithPadding` — classification
  - `DataCollatorForSeq2Seq` — summarization, translation
  - `DataCollatorForLanguageModeling` — causal and masked LM
  - `DataCollatorForTokenClassification` — NER

**Model loading and saving**
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    dtype="auto",          # load in the saved precision (v5 name, was torch_dtype)
    device_map="auto",     # place layers across available GPUs and CPU
)
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
model.push_to_hub("username/my-model")
tokenizer.push_to_hub("username/my-model")
```
Also write a model card (README.md) for every model you push, and load community kernels through `attn_implementation` (for example `attn_implementation="kernels-community/flash-attn2"`).

**Generation**
```python
from transformers import GenerationConfig, TextStreamer, TextIteratorStreamer

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

gen_cfg = GenerationConfig(max_new_tokens=100, do_sample=True, temperature=0.7)
model.generate(**inputs, generation_config=gen_cfg)

model.generate(**inputs, max_new_tokens=100, streamer=TextStreamer(tokenizer))
```

Decoding strategies: greedy, beam search (`num_beams`), sampling (`do_sample=True`), temperature, top-k, top-p (nucleus), repetition penalty, length penalty, constrained generation, and token-by-token streaming with `TextStreamer` / `TextIteratorStreamer`.

Chat models: format conversations with `tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")`.

**Architectures to know**

Encoder-only (understanding): BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa, XLM-RoBERTa.

Decoder-only (generation): GPT-2, Llama 3/4 family, Mistral, Qwen family (Qwen2.5, Qwen3, Qwen3.5, Qwen3.8), Gemma family, Phi family, Falcon TII.

Encoder-decoder (seq2seq): T5 and Flan-T5, BART, Pegasus, mT5, MarianMT, NLLB.

Vision: ViT, DeiT, Swin Transformer, DINOv2, CLIP and SigLIP.

Multimodal: LLaVA, Idefics, PaliGemma, Qwen-VL, SmolVLM.

Audio: Whisper, Wav2Vec 2.0, HuBERT.

**What to practice**
- Sentiment analysis on IMDB with BERT or DistilBERT
- Text generation with GPT-2 and a small chat model
- Named entity recognition
- Question answering with `AutoModelForQuestionAnswering`
- Summarization with T5 using `generate()`
- Speech recognition with Whisper
- Image classification with ViT

**Key install**
```bash
pip install -U transformers torch
pip install -U "transformers[torch]"   # all torch deps
```

## **Tokenizers 0.23.3**

The fast tokenization library (written in Rust) that powers all `AutoTokenizer` classes. It is how models actually read text [28].

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

tokenizer("question text", "context text", truncation="only_second")
tokenizer.decode(enc.input_ids[0], skip_special_tokens=True)

tokenizer.cls_token     # [CLS]
tokenizer.sep_token     # [SEP]
tokenizer.pad_token     # [PAD]
tokenizer.unk_token     # [UNK]
tokenizer.mask_token    # [MASK]

tokenizer(["sentence one", "sentence two"], padding=True, truncation=True)

enc = tokenizer("Hello world", return_offsets_mapping=True)
print(enc["offset_mapping"])   # character spans per token
print(enc.word_ids())          # token to word index

tokenizer(long_text, truncation=True, max_length=512,
          stride=128, return_overflowing_tokens=True)

tokenizer.add_tokens(["new_token"])
tokenizer.add_special_tokens({"additional_special_tokens": ["tool_token"]})
# model.resize_token_embeddings(len(tokenizer))
```

In v5, call the tokenizer directly — `encode_plus()` and `batch_encode_plus()` are removed.

**Train your own tokenizer from scratch**
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

fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
fast.save_pretrained("./my-tokenizer")
```

**Key concepts**
- BPE (Byte-Pair Encoding) — GPT-style models
- Byte-level BPE — GPT-2, RoBERTa
- WordPiece — BERT
- SentencePiece and Unigram — T5, LLaMA, many multilingual models
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
- Fast (Rust) versus slow (Python) tokenizers — unified around the fast implementation in v5, with parallel tokenization

**Key install**
```bash
pip install -U tokenizers
```

## **Evaluation stack**

Three libraries with three different jobs. The source notes blurred them into "evaluate, and LightEval for LLMs". The accurate split matters because each one answers a different question [50][51][52].

| Library | Version | Job | Reach |
|---|---|---|---|
| `evaluate` | 0.4.6 (Sep 2025) | Classic metric implementations: accuracy, F1, ROUGE, BLEU | Stable but no longer the recommended path for LLM benchmarks |
| `lighteval` | 0.13.0 (Nov 2025) | HF task frameworks with inspect-ai / accelerate / nanotron / vLLM entry points, 1000+ tasks | No release since Nov 2025 — treat as stable-but-static |
| `lm-evaluation-harness` | 0.4.13 (Aug 2026) | Standard benchmark harness: MMLU, GSM8K, HellaSwag, IFEval… | The one to reach for when a JD names a benchmark |

**`evaluate` usage**
```python
import evaluate

accuracy  = evaluate.load("accuracy")
f1        = evaluate.load("f1")
rouge     = evaluate.load("rouge")
bleu      = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

result = accuracy.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 1])
print(result)  # {"accuracy": 0.5}

for preds, refs in batches:
    accuracy.add_batch(predictions=preds, references=refs)
accuracy.compute()

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
clf_metrics.compute(predictions=[0, 1, 1, 0], references=[0, 1, 0, 1])

evaluate.list_evaluation_modules()
```

**Metric guide**

Classification: `accuracy`, `precision`, `recall`, `f1` (binary / macro / micro / weighted), `matthews_correlation` (MCC).

Question answering: `exact_match`, `squad` (exact match plus F1).

Generation and translation: `bleu` and `sacrebleu`, `rouge` (rouge1, rouge2, rougeL), `meteor`, `ter`.

Semantic: `bertscore`, `bleurt`.

Code: `code_eval` — executes generated code; sandbox only.

Also learn: creating custom metrics and sharing them on the Hub. The full surface is `evaluate.load()`, `evaluate.combine()`, `metric.compute()`, and `metric.add_batch()` for streaming evaluation.

**lm-eval command line**
```bash
lm-eval run --model hf --model_args pretrained=Qwen/Qwen3-1.7B,dtype=bfloat16 --tasks hellaswag
lm-eval run --model vllm --model_args pretrained=./my-model --tasks ifeval
lm-eval ls tasks
```
Backend extras: `pip install "lm_eval[hf]"`, `"lm_eval[vllm]"`, `"lm_eval[api]"`.

**Decontamination — the real API.** The harness does not expose a `decontaminate(train_dataset=..., eval_datasets=[...], ngram_size=8)` function, and there is no `--decontamination_ngram_size=8` flag; both snippets in the source notes describe an API that does not exist. Contamination detection is implemented in `lm_eval/decontaminate.py` over **precomputed n-gram files**: you generate n-gram files plus `info.json` for your training set, pass the directory with `--decontamination_ngrams_path`, and the evaluator excludes contaminated documents and reports a second metric with a `decontaminate` suffix. It is **disabled by default for new tasks** unless the task overrides `should_decontaminate` and `doc_to_decontamination_query`. The harness uses 13-grams as its default basis [53][54].

**Key install**
```bash
pip install -U evaluate
pip install -U lighteval
pip install "lm_eval[hf]"
pip install scikit-learn rouge_score sacrebleu
```

---

## **Attention architectures**

The 2026 model landscape is mostly a story about attention. Every open frontier release changed how the KV cache is stored, and that change determines what the serving stack can do with long context.

**Standard multi-head attention:**

```
Attention(Q,K,V) = softmax(QK^T / √d_k) × V
Multi-head: Concat(head_1, ..., head_h) × W_O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Memory complexity is O(n²d) — quadratic in sequence length.

**Grouped-query attention (GQA):** multiple Q heads share the same K and V heads. Llama-3 uses 32 Q heads against 8 KV heads, a 4× KV cache reduction. GQA is the standard in every 2026 open LLM.

**Multi-query attention (MQA):** all Q heads share a single K/V head. Maximum savings, with a quality trade-off.

**Multi-head latent attention (MLA) — DeepSeek:** the KV cache stores only a compressed latent (`c^KV = W^{DKV} h`), decompressed at attention time, which is roughly a 50× cache reduction against MHA. Native in vLLM and SGLang.

**DeepSeek V4 hybrid attention (CSA + HCA):** Compressed Sparse Attention adds a sparse pattern with compression; Heavily Compressed Attention compresses aggressively in the tail layers. Together they reach ~27% of single-token inference FLOPs versus V3.2 at 1M-token context, which is what makes 1M context viable without linear approximations.

**FlashAttention 1 / 2 / 3:** FA1 tiles the computation so the full n×n matrix is never materialised, giving HBM reads of O(n). FA2 improves parallelism and is roughly 2× faster than FA1 on A100. FA3 is H100-optimised — warp-specialised, FP8, TMA async memory — at roughly 1.5–2× FA2, and it is the default in vLLM from 0.8 onward.

**Sparse and linear attention (frontier):** Mamba (SSM-based, O(n) complexity), RWKV (O(1) inference, parallelisable training), and the Qwen3.5/3.8 Gated Delta Network hybrid of SSM and Transformer, which is what gives those models their high inference throughput.

The practical consequence for this roadmap: KV cache size is not a serving detail, it is an architectural property that sets your context ceiling before you write a single line of serving code. Stage 10 quantifies it.

## **Parallelism for training and inference**

Data parallelism replicates the model and all-reduces gradients. FSDP shards parameters, gradients and optimizer states and is functionally equivalent to ZeRO-3. Tensor parallelism splits individual layers and needs NVLink. Pipeline parallelism splits layers across stages and pays a bubble of `(p-1)/m`. Expert parallelism splits MoE experts and pays an all-to-all.

The full taxonomy, the code, and the failure signatures are in Stage 8.

---

## **Worksheet additions**

## **Scope**
Stage 1 is where the stack stops being a list of APIs and becomes a model. Three things have to work together: a tokenizer that decides what the model sees, a model class that decides what the model does, and an evaluation harness that decides whether any of it mattered.

**Exit criterion:** fine-tune a pretrained model on a public dataset, push it to the Hub, and produce a metric you can defend in an interview.

## **Environment**
```bash
pip install -U transformers torch tokenizers
pip install -U "transformers[torch]"
pip install -U evaluate lighteval
pip install "lm_eval[hf]"
pip list | grep -i -E "transformers|tokenizers|evaluate|lighteval|lm-eval"
```

`transformers` 5.22.1 requires **Python ≥3.11** and `huggingface-hub<3.0,>=1.31.0`. The `transformers[torch]` extra pulls `torch>=2.6` and `accelerate>=1.1.0`.

---

### **Pipeline API**
- [ ] Text: `text-classification`, `token-classification`, `question-answering`, `table-question-answering`, `text-generation`, `fill-mask`, `zero-shot-classification`, `feature-extraction`
- [ ] Vision: `image-classification`, `object-detection`, `image-segmentation`, `mask-generation`, `zero-shot-image-classification`, `zero-shot-object-detection`, `depth-estimation`, `image-to-image`, `image-feature-extraction`, `keypoint-matching`, `video-classification`
- [ ] Audio: `automatic-speech-recognition`, `audio-classification`, `zero-shot-audio-classification`, `text-to-audio`, `text-to-speech`
- [ ] Multimodal: `image-text-to-text`, `visual-question-answering`, `document-question-answering`, `any-to-any`
- [ ] Confirming the live list with `get_supported_tasks()`
- [ ] Replacing the removed pipelines: `summarization`, `translation`, `text2text-generation`, `image-to-text`, `conversational`
- [ ] Isolating an old tutorial in `pip install -U "transformers<5"`

### **AutoTokenizer**
- [ ] `from_pretrained()` for any tokenizer
- [ ] Encoding a single sentence, sentence pairs, and batches
- [ ] Calling the tokenizer directly (`encode_plus()` is removed)
- [ ] Padding strategies — `longest`, `max_length`, `do_not_pad`
- [ ] Truncation strategies — `longest_first`, `only_first`, `only_second`
- [ ] `return_tensors` — `pt`, `np`
- [ ] Special tokens — CLS, SEP, PAD, UNK, MASK, BOS, EOS
- [ ] `offset_mapping` — map tokens back to characters
- [ ] `word_ids()` — map tokens back to words
- [ ] Decoding with `skip_special_tokens`
- [ ] Chat templates via `apply_chat_template()`
- [ ] Saving and loading custom tokenizers
- [ ] Adding new special tokens and resizing model embeddings
- [ ] Fast (Rust) tokenizers as the default, with parallel tokenization

### **AutoModel classes**
- [ ] NLU: `AutoModelForSequenceClassification`, `AutoModelForMultipleChoice`, `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering`
- [ ] NLG: `AutoModelForCausalLM`, `AutoModelForMaskedLM`, `AutoModelForSeq2SeqLM`, `AutoModelForSpeechSeq2Seq`
- [ ] Vision: `AutoModelForImageClassification`, `AutoModelForObjectDetection`, `AutoModelForSemanticSegmentation`, `AutoModelForInstanceSegmentation`, `AutoModelForDepthEstimation`, `AutoModelForImageToImage`
- [ ] Multimodal: `AutoModelForImageTextToText`, `AutoModelForVisualQuestionAnswering`, `AutoModelForDocumentQuestionAnswering`
- [ ] Audio: `AutoModelForAudioClassification`, `AutoModelForCTC`
- [ ] Base and utility: `AutoModel`, `AutoConfig`, `AutoImageProcessor`, `AutoFeatureExtractor`, `AutoProcessor`

### **Trainer**
- [ ] `TrainingArguments` — all training hyperparameters
- [ ] `Trainer` — the standard supervised loop
- [ ] `processing_class` (replaces the old `tokenizer` argument)
- [ ] `Seq2SeqTrainer` and `Seq2SeqTrainingArguments`
- [ ] `TrainerCallback`, `EarlyStoppingCallback`
- [ ] `compute_metrics`
- [ ] Mixed precision: `bf16=True` on newer GPUs, `fp16=True` on older
- [ ] Collators: `DataCollatorWithPadding`, `DataCollatorForSeq2Seq`, `DataCollatorForLanguageModeling`, `DataCollatorForTokenClassification`

### **Configuration, generation, architectures**
- [ ] `AutoConfig` and `from_pretrained()` with `device_map="auto"` and `dtype="auto"`
- [ ] `save_pretrained()`, `push_to_hub()`, model cards
- [ ] Loading community kernels through `attn_implementation`
- [ ] `model.generate()`, `max_new_tokens`, greedy, beam search, sampling, temperature, top-k, top-p, repetition penalty, length penalty, constrained generation
- [ ] Streaming with `TextStreamer` and `TextIteratorStreamer`
- [ ] `GenerationConfig` for reusable settings
- [ ] Encoder-only: BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa, XLM-RoBERTa
- [ ] Decoder-only: GPT-2, Llama, Mistral, Qwen, Gemma, Phi, Falcon
- [ ] Seq2seq: T5/Flan-T5, BART, Pegasus, mT5, MarianMT
- [ ] Vision: ViT, DeiT, Swin, DINOv2, CLIP, SigLIP
- [ ] Multimodal: LLaVA, Idefics, PaliGemma, Qwen-VL, SmolVLM
- [ ] Audio: Whisper, Wav2Vec 2.0, HuBERT

---

### **Pipeline, and what to do when the task is gone**
```python
from transformers import pipeline
from transformers.pipelines import get_supported_tasks
import transformers

print(transformers.__version__)
print(get_supported_tasks())

classifier = pipeline("text-classification")
ner        = pipeline("token-classification", aggregation_strategy="simple")
embedder   = pipeline("feature-extraction")

# summarization pipeline no longer exists in v5
chat = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")
out = chat([{"role": "user", "content": "Summarize in one sentence: " + text}],
           max_new_tokens=100)
print(out[0]["generated_text"][-1]["content"])

# seq2seq fallback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok   = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
ids = model.generate(**tok("summarize: " + text, return_tensors="pt", truncation=True),
                     max_new_tokens=80)
print(tok.decode(ids[0], skip_special_tokens=True))
```

### **Trainer with a custom metric**
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
    "distilbert-base-uncased", num_labels=2)

tokenized = ds.map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)
accuracy  = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return accuracy.compute(predictions=logits.argmax(-1), references=labels)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
    ),
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    processing_class=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./my-model")
trainer.push_to_hub("username/my-model")
```

### **Generation controls**
```python
from transformers import GenerationConfig, TextStreamer

inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=100, do_sample=True,
                     temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1)

cfg = GenerationConfig(max_new_tokens=100, do_sample=True, temperature=0.7)
model.generate(**inputs, generation_config=cfg)

model.generate(**inputs, max_new_tokens=100, streamer=TextStreamer(tokenizer))
```

### **Tokenizer deep dive**
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()
tok.train(files=["corpus.txt"], trainer=BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
))
tok.save("tokenizer.json")

fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
fast.save_pretrained("./my-tokenizer")
```

### **Evaluation, three ways**
```python
import evaluate

acc = evaluate.load("accuracy")
acc.add_batch(predictions=[0, 1], references=[0, 1])
print(acc.compute())
print(evaluate.list_evaluation_modules()[:5])
```

```bash
# lm-eval: standard benchmarks, CLI subcommands since 0.4.10
lm-eval run --model hf --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,dtype=bfloat16 --tasks hellaswag
lm-eval run --model vllm --model_args pretrained=./my-model --tasks ifeval
lm-eval ls tasks
```

```python
# lighteval: HF task frameworks with inspect-ai / accelerate / vllm entry points
# see https://github.com/huggingface/lighteval for the current task and backend names
```

### **Decontamination, the way the harness actually implements it**
```bash
# 1. Generate n-gram files + info.json for your training set
python scripts/pile_ngram_generation/build_ngrams.py --train-data ./train.jsonl \
  --output-dir ./ngrams --ngram-size 13 --split train

# 2. Run the harness pointing at those files
lm-eval run --model hf --model_args pretrained=./my-model \
  --tasks mmlu --decontamination_ngrams_path ./ngrams
```

The harness marks contaminated documents and reports a second metric with a `decontaminate` suffix. New tasks are **not** decontaminated by default: a task must override `should_decontaminate` and `doc_to_decontamination_query`. There is no `decontaminate(train_dataset=..., eval_datasets=[...], ngram_size=8)` function — that API never existed, and code written against it will fail on import.

---

### **Lab 1-1 — Pipeline inventory**
- Print `get_supported_tasks()` and diff it against the checklist above.
- Run one pipeline per modality (text, vision, audio, multimodal).
- Replace a removed pipeline two ways: chat model and seq2seq `generate()`.

### **Lab 1-2 — Trainer fine-tune on IMDB**
- Fine-tune DistilBERT for sentiment with `Trainer`.
- Record loss curve, eval accuracy, training wall time, peak memory.
- Push the model and the tokenizer to the Hub with a model card.
- Re-load it in a fresh process with `pipeline("sentiment-analysis", model=...)`.

### **Lab 1-3 — Tokenizer forensics**
- Tokenize a multilingual sample; explain each subword split.
- Map tokens back to characters with `offset_mapping` and `word_ids()`.
- Build a NER alignment by hand for three sentences.
- Train a 30k-vocab BPE tokenizer and wrap it with `PreTrainedTokenizerFast`.

### **Lab 1-4 — Generation sweep**
- Compare greedy, beam, temperature, top-k, top-p on the same prompt.
- Measure tokens/sec and output length.
- Stream with `TextStreamer` and with `TextIteratorStreamer`.

### **Lab 1-5 — Three eval harnesses on one model**
- Score the same checkpoint with `evaluate`, `lm-eval`, and `lighteval`.
- Record which tasks each harness supports and what the output format is.
- Run a decontamination pass and report how many eval documents were flagged.

---

## **Pitfalls**
- **`torch_dtype` → `dtype`.** The old argument name warns or errors in v5.
- **`Trainer(tokenizer=...)` → `processing_class=...`.** Passing the old argument raises a deprecation error.
- **`encode_plus()` is gone.** Call the tokenizer directly.
- **Removed pipelines fail with `KeyError`, not a helpful message.** Check `get_supported_tasks()` first.
- **`ds["col"]` is lazy.** `list(ds["col"])` materializes it; for a large column this is where your RAM goes.
- **`hf eval` / lighteval are not interchangeable with `evaluate`.** A metric that exists in one may be missing in another; check `evaluate.list_evaluation_modules()` and the harness task list.
- **lighteval has had no release since 0.13.0 (Nov 2025).** Pin it, and read the source for new entry points rather than trusting blog posts.
- **`lighteval` and `lm-eval` have different task names.** Copy-pasting a task name from one into the other silently evaluates nothing.

---

## **Acceptance gate**
- [ ] Every task class in the pipeline inventory has been run at least once.
- [ ] A `Trainer` fine-tune completed and was pushed to the Hub with a card.
- [ ] The pushed model reloads in a fresh process and produces the same output as the local copy.
- [ ] A tokenizer was trained from scratch and round-tripped through `PreTrainedTokenizerFast`.
- [ ] One metric was computed with `evaluate`, one benchmark with `lm-eval`, and one with `lighteval`, with the outputs compared.
- [ ] A decontamination run was executed against a benchmark and the flagged-document count recorded.

---

## **Resources**
- Transformers v5.22.1 release — https://github.com/huggingface/transformers/releases/tag/v5.22.1
- transformers on PyPI — https://pypi.org/project/transformers/
- tokenizers on PyPI — https://pypi.org/project/tokenizers/
- Evaluate v0.4.6 — https://github.com/huggingface/evaluate/releases/tag/v0.4.6
- LightEval v0.13.0 — https://github.com/huggingface/lighteval/releases/tag/v0.13.0
- lm-evaluation-harness v0.4.13 — https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13
- lm-eval decontamination guide — https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md
- lm-eval CLI interface — https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md

**Next:** Stage 03 — Data Engineering.
