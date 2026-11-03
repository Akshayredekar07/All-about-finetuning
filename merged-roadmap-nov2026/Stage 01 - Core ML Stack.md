# **Stage 01 — Core ML Stack**
**Outcome:** run any model, fine-tune it with the built-in trainer, and measure it with a metric that matches the task.**Libraries:** `transformers` 5.22.1 · `tokenizers` 0.23.3 · `evaluate` 0.4.6 · `lighteval` 0.13.0 · `lm-evaluation-harness` 0.4.13


---

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

## **Topic checklist — Transformers**

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

## **Code**

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

# **summarization pipeline no longer exists in v5**
chat = pipeline("text-generation", model="Qwen/Qwen3-4B-Instruct-2507")
out = chat([{"role": "user", "content": "Summarize in one sentence: " + text}],
           max_new_tokens=100)
print(out[0]["generated_text"][-1]["content"])

# **seq2seq fallback**
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
# **lm-eval: standard benchmarks, CLI subcommands since 0.4.10**
lm-eval run --model hf --model_args pretrained=Qwen/Qwen3-4B-Instruct-2507,dtype=bfloat16 --tasks hellaswag
lm-eval run --model vllm --model_args pretrained=./my-model --tasks ifeval
lm-eval ls tasks
```

```python
# **lighteval: HF task frameworks with inspect-ai / accelerate / vllm entry points**
# **see https://github.com/huggingface/lighteval for the current task and backend names**
```

### **Decontamination, the way the harness actually implements it**

```bash
# **1. Generate n-gram files + info.json for your training set**
python scripts/pile_ngram_generation/build_ngrams.py --train-data ./train.jsonl \
  --output-dir ./ngrams --ngram-size 13 --split train

# **2. Run the harness pointing at those files**
lm-eval run --model hf --model_args pretrained=./my-model \
  --tasks mmlu --decontamination_ngrams_path ./ngrams
```

The harness marks contaminated documents and reports a second metric with a `decontaminate` suffix. New tasks are **not** decontaminated by default: a task must override `should_decontaminate` and `doc_to_decontamination_query`. There is no `decontaminate(train_dataset=..., eval_datasets=[...], ngram_size=8)` function — that API never existed, and code written against it will fail on import.

---

## **Labs**

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

**Next:** Stage 02 — Data Engineering.
