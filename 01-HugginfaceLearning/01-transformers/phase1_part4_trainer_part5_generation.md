# Phase 1 — Transformers
# Part 4: Trainer + TrainingArguments
# Part 5: Generation

---

# PART 4: Trainer

---

## What is Trainer

Trainer is HuggingFace's complete training loop. It handles everything:
gradient computation, optimizer step, evaluation, logging, checkpointing,
mixed precision, distributed training. You only write the data and the metric function.

```
your dataset + your model + TrainingArguments → Trainer → trained model
```

---

## Installation

```bash
pip install transformers torch datasets evaluate accelerate
```

---

## TrainingArguments — Every Parameter

```python
from transformers import TrainingArguments

args = TrainingArguments(

    # ── Output ──────────────────────────────────────────────────────────
    output_dir="./results",          # where to save checkpoints and model

    # ── Training Duration ────────────────────────────────────────────────
    num_train_epochs=3,              # number of full passes over training data
    max_steps=-1,                    # if > 0, overrides num_train_epochs

    # ── Batch Size ───────────────────────────────────────────────────────
    per_device_train_batch_size=16,  # batch size per GPU during training
    per_device_eval_batch_size=64,   # batch size per GPU during evaluation
    gradient_accumulation_steps=1,   # accumulate gradients before stepping
    # effective batch size = per_device_train_batch_size × num_gpus × gradient_accumulation_steps

    # ── Optimizer ────────────────────────────────────────────────────────
    learning_rate=5e-5,              # peak learning rate
    weight_decay=0.01,               # L2 regularization on all params except bias and LayerNorm
    adam_beta1=0.9,                  # Adam optimizer beta1
    adam_beta2=0.999,                # Adam optimizer beta2
    adam_epsilon=1e-8,               # Adam optimizer epsilon
    max_grad_norm=1.0,               # gradient clipping threshold

    # ── Learning Rate Schedule ───────────────────────────────────────────
    lr_scheduler_type="linear",      # "linear", "cosine", "cosine_with_restarts",
                                     # "polynomial", "constant", "constant_with_warmup"
    warmup_steps=500,                # number of warmup steps
    warmup_ratio=0.0,                # warmup as fraction of total steps (alternative to warmup_steps)

    # ── Evaluation ───────────────────────────────────────────────────────
    eval_strategy="epoch",           # "no", "steps", "epoch"
    eval_steps=500,                  # evaluate every N steps (when eval_strategy="steps")
    eval_delay=0,                    # wait N steps/epochs before starting evaluation

    # ── Saving ───────────────────────────────────────────────────────────
    save_strategy="epoch",           # "no", "steps", "epoch" — must match eval_strategy to use load_best
    save_steps=500,                  # save every N steps (when save_strategy="steps")
    save_total_limit=3,              # keep only last N checkpoints (deletes older ones)
    load_best_model_at_end=True,     # load best checkpoint after training finishes

    # ── Best Model ───────────────────────────────────────────────────────
    metric_for_best_model="f1",      # which metric to use when selecting best model
    greater_is_better=True,          # is a higher metric better?

    # ── Logging ──────────────────────────────────────────────────────────
    logging_dir="./logs",            # TensorBoard log directory
    logging_steps=10,                # log every N steps
    logging_strategy="steps",        # "no", "steps", "epoch"
    report_to="tensorboard",         # "tensorboard", "wandb", "none", "all"
    run_name="my-training-run",      # name for the run (shown in W&B, etc.)

    # ── Mixed Precision ──────────────────────────────────────────────────
    fp16=False,                      # use float16 mixed precision (NVIDIA Volta+)
    bf16=False,                      # use bfloat16 mixed precision (NVIDIA Ampere+)
    fp16_opt_level="O1",             # Apex AMP optimization level

    # ── Speed and Memory ─────────────────────────────────────────────────
    dataloader_num_workers=0,        # CPU workers for data loading
    dataloader_pin_memory=True,      # pin memory for faster GPU transfer
    group_by_length=False,           # group similar-length sequences in same batch
    length_column_name="length",     # column to use for grouping
    torch_compile=False,             # use torch.compile() for speed
    optim="adamw_torch",             # optimizer: "adamw_torch", "adamw_8bit", "sgd", etc.

    # ── Distributed Training ─────────────────────────────────────────────
    local_rank=-1,                   # set by launcher for distributed training
    ddp_find_unused_parameters=False, # DDP unused param detection
    deepspeed=None,                  # path to DeepSpeed config JSON
    fsdp="",                         # Fully Sharded Data Parallel config

    # ── Hub ──────────────────────────────────────────────────────────────
    push_to_hub=False,               # push model to Hub after training
    hub_model_id="username/model",   # Hub repo to push to
    hub_strategy="every_save",       # "end", "every_save", "checkpoint"
    hub_token=None,                  # HF token (uses cached by default)

    # ── Misc ─────────────────────────────────────────────────────────────
    seed=42,                         # random seed for reproducibility
    data_seed=42,                    # separate seed for data operations
    no_cuda=False,                   # force CPU even if GPU is available
    resume_from_checkpoint=None,     # path to checkpoint to resume from
    ignore_data_skip=False,          # skip data already seen when resuming
)
```

---

## Data Collators

Data collators batch individual examples together and apply dynamic padding.
They sit between the dataset and the model during training.

### DataCollatorWithPadding

For classification tasks. Pads sequences to the longest in the batch.

```python
from transformers import DataCollatorWithPadding, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# dynamically pads each batch to the longest sequence in that batch
# more efficient than padding everything to max_length
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,          # "longest", "max_length", True=longest
    max_length=None,       # only used if padding="max_length"
    return_tensors="pt"
)
```

---

### DataCollatorForSeq2Seq

For summarization, translation, and any encoder-decoder model.
Pads both encoder inputs and decoder labels.

```python
from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,               # needed to get decoder_start_token_id
    padding=True,
    label_pad_token_id=-100,   # -100 is ignored by cross-entropy loss
    pad_to_multiple_of=None    # pad to multiple of this number (GPU efficiency)
)
```

---

### DataCollatorForLanguageModeling

For pre-training causal (CLM) or masked (MLM) language models.

```python
from transformers import DataCollatorForLanguageModeling, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Causal Language Modeling (GPT-style)
# labels = input_ids shifted by 1 (next-token prediction)
collator_clm = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False     # False = CLM, True = MLM
)

# Masked Language Modeling (BERT-style)
# randomly masks 15% of tokens and predicts them
collator_mlm = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15   # fraction of tokens to mask
)
```

---

### DataCollatorForTokenClassification

For NER and other token-level classification tasks.
Aligns labels with tokens, pads with -100.

```python
from transformers import DataCollatorForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,   # padding labels ignored in loss
    return_tensors="pt"
)
```

---

## compute_metrics — Custom Evaluation

```python
import evaluate
import numpy as np

accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    # eval_pred is a namedtuple (predictions, labels)
    logits, labels = eval_pred

    # for classification — take argmax of logits
    predictions = np.argmax(logits, axis=-1)

    # compute metrics
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1  = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"]
    }

# for seq2seq — need to decode first
import evaluate
rouge = evaluate.load("rouge")

def compute_metrics_seq2seq(eval_pred):
    predictions, labels = eval_pred

    # decode token IDs to text
    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # replace -100 (padding) with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # strip whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return result
```

---

## Callbacks — Hook Into Training

```python
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class MyCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        print("Training started")

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} started")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            print(f"Step {state.global_step}, loss: {state.log_history[-1]}")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"Eval metrics: {metrics}")

    def on_save(self, args, state, control, **kwargs):
        print(f"Model saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        print("Training finished!")
```

**Built-in callbacks:**
```python
from transformers import (
    EarlyStoppingCallback,        # stop when metric stops improving
    TensorBoardCallback,          # log to TensorBoard
    WandbCallback,                # log to Weights & Biases
    ProgressCallback,             # print progress bar
    PrinterCallback,              # print logs to console
)

early_stop = EarlyStoppingCallback(
    early_stopping_patience=3,   # stop if no improvement for 3 evals
    early_stopping_threshold=0.001  # minimum improvement to count
)
```

---

## Full Training Example — Text Classification

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import load_dataset
import evaluate
import numpy as np

# 1. load data
dataset   = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. tokenize
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized = tokenized.rename_column("label", "labels")

# 3. model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1}
)

# 4. metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# 5. training arguments
args = TrainingArguments(
    output_dir="./bert-imdb",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    logging_steps=50,
)

# 6. trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 7. train
trainer.train()

# 8. evaluate
results = trainer.evaluate()
print(results)

# 9. save and push
trainer.save_model("./bert-imdb-final")
trainer.push_to_hub("username/bert-imdb-sentiment")
```

---

## Seq2SeqTrainer — For Summarization and Translation

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="./summarizer",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,   # use model.generate() during evaluation
    generation_max_length=150,    # max tokens for generated summaries
    generation_num_beams=4,       # beam search during evaluation
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge2"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics_seq2seq
)

trainer.train()
```

---

## Resuming Training from Checkpoint

```python
# resume from latest checkpoint in output_dir
trainer.train(resume_from_checkpoint=True)

# resume from specific checkpoint
trainer.train(resume_from_checkpoint="./results/checkpoint-1000")
```

---

## Hyperparameter Search with Trainer

```python
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

trainer = Trainer(
    model_init=model_init,   # use model_init instead of model=
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# search with Optuna or Ray Tune
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    n_trials=10,
    hp_space=lambda trial: {
        "learning_rate": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("batch", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("epochs", 1, 5)
    }
)
print(best_run.hyperparameters)
```

---

---

# PART 5: Generation

---

## What is generate()

`model.generate()` is the main method for text generation in decoder-only
and encoder-decoder models. It handles all decoding strategies.

```
inputs → model.generate() → output token IDs → tokenizer.decode() → text
```

---

## Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model     = AutoModelForCausalLM.from_pretrained("gpt2")

inputs     = tokenizer("The weather today is", return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=50)
text       = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)
```

---

## Decoding Strategies

---

### Greedy Decoding

Always pick the single most likely next token. Fast, deterministic, but
often repetitive and boring.

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,    # greedy = no sampling (default)
    num_beams=1         # greedy = 1 beam
)
```

---

### Beam Search

Explore multiple token sequences in parallel and return the best overall sequence.
Better quality than greedy, slower, still deterministic.

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=4,          # number of beams to track
    early_stopping=True,  # stop when all beams reach EOS
    no_repeat_ngram_size=3  # prevent repeating 3-gram sequences
)

# return multiple candidates
output_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=5,
    num_return_sequences=3,  # return 3 different beam search results
    early_stopping=True
)
```

---

### Sampling

Randomly sample from the probability distribution. Creative, varied output.

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,           # enable random sampling
    temperature=1.0,          # 1.0 = unchanged probabilities
                              # < 1.0 = more focused/conservative
                              # > 1.0 = more random/creative
)
```

---

### Top-k Sampling

Only sample from the top k most likely next tokens. Reduces low-probability tokens.

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_k=50,        # only consider top 50 tokens at each step
    temperature=0.8
)
```

---

### Top-p (Nucleus) Sampling

Only sample from the smallest set of tokens whose cumulative probability ≥ p.
Adapts the number of candidates per step.

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_p=0.92,      # use tokens whose probs sum to 0.92
    temperature=0.8
)
```

---

### Combined (Best Practice for Creative Generation)

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2   # penalize repeating tokens
)
```

---

### Contrastive Search (2022)

Produces coherent and non-repetitive text without sampling.

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=200,
    penalty_alpha=0.6,  # contrastive search penalty
    top_k=4             # candidate pool size
)
```

---

## Generation Control Parameters

```python
output_ids = model.generate(
    **inputs,

    # ── Length Control ────────────────────────────────────────────────────
    max_new_tokens=100,         # max NEWLY GENERATED tokens (recommended)
    max_length=200,             # max TOTAL tokens (prompt + generated)
    min_new_tokens=10,          # minimum new tokens to generate
    min_length=20,              # minimum total length

    # ── When to Stop ─────────────────────────────────────────────────────
    eos_token_id=tokenizer.eos_token_id,   # stop at EOS token
    forced_eos_token_id=None,              # force EOS at max_length
    early_stopping=True,                   # stop beam search when beams hit EOS

    # ── Repetition ───────────────────────────────────────────────────────
    repetition_penalty=1.2,     # > 1.0 penalizes tokens that appeared before
    no_repeat_ngram_size=3,     # block repeating 3-gram sequences
    encoder_no_repeat_ngram_size=0,  # for seq2seq — no repeat from input

    # ── Diversity ────────────────────────────────────────────────────────
    num_beam_groups=1,          # diverse beam search — split beams into groups
    diversity_penalty=0.0,      # penalize beams that are too similar

    # ── Multiple Outputs ─────────────────────────────────────────────────
    num_return_sequences=1,     # how many sequences to return

    # ── Constrained Generation ───────────────────────────────────────────
    forced_bos_token_id=None,   # force first token
    force_words_ids=None,       # force specific words to appear

    # ── Length Penalty (Beam Search) ─────────────────────────────────────
    length_penalty=1.0,         # > 1.0 favors longer, < 1.0 favors shorter

    # ── Output Options ────────────────────────────────────────────────────
    return_dict_in_generate=True,    # return GenerateOutput instead of IDs
    output_scores=True,              # return token scores at each step
    output_attentions=False,         # return attention weights
    output_hidden_states=False,      # return hidden states

    # ── Misc ─────────────────────────────────────────────────────────────
    pad_token_id=tokenizer.eos_token_id,  # GPT-2 has no pad token
    bos_token_id=tokenizer.bos_token_id,
    use_cache=True,              # use key-value cache for speed
)
```

---

## GenerationConfig — Reusable Settings

```python
from transformers import GenerationConfig

# define once
gen_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)

# use it
output_ids = model.generate(**inputs, generation_config=gen_config)

# save to disk with model
gen_config.save_pretrained("./my-model")

# load it back
gen_config = GenerationConfig.from_pretrained("./my-model")

# push to Hub
gen_config.push_to_hub("username/my-model")
```

---

## Streaming — Token by Token Output

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer
import threading

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model     = AutoModelForCausalLM.from_pretrained("gpt2")
inputs    = tokenizer("Once upon a time", return_tensors="pt")

# TextStreamer — prints each token as it is generated
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
model.generate(**inputs, max_new_tokens=100, streamer=streamer)

# TextIteratorStreamer — iterate over tokens in your own code
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# must run generate in a thread
thread = threading.Thread(
    target=model.generate,
    kwargs={"input_ids": inputs["input_ids"], "max_new_tokens": 100, "streamer": streamer}
)
thread.start()

# now iterate over streamed tokens
generated_text = ""
for token in streamer:
    generated_text += token
    print(token, end="", flush=True)   # print as they arrive
```

---

## Seq2Seq Generation — Summarization and Translation

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

inputs = tokenizer(
    "Long article text here...",
    max_length=1024,
    truncation=True,
    return_tensors="pt"
)

# generate with beam search — standard for summarization
output_ids = model.generate(
    **inputs,
    max_new_tokens=150,
    min_length=40,
    num_beams=4,
    length_penalty=2.0,      # favor longer summaries
    no_repeat_ngram_size=3,
    early_stopping=True
)

summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(summary)
```

---

## Batch Generation

```python
# pad all inputs in a batch
inputs = tokenizer(
    ["Prompt 1", "Prompt 2 that is longer", "Short 3"],
    padding=True,
    return_tensors="pt"
)

# generate for the whole batch at once
output_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

# decode each separately
for ids in output_ids:
    print(tokenizer.decode(ids, skip_special_tokens=True))
```

---

## Strategy Decision Guide

| Goal | Strategy | Key Parameters |
|---|---|---|
| Fastest, deterministic | Greedy | do_sample=False, num_beams=1 |
| Best quality, deterministic | Beam search | num_beams=4-8 |
| Creative, varied | Sampling | do_sample=True, temperature=0.8 |
| Balanced creative | Top-k + top-p | do_sample=True, top_k=50, top_p=0.95 |
| No repetition + coherent | Contrastive | penalty_alpha=0.6, top_k=4 |
| Multiple diverse outputs | Diverse beam | num_beam_groups=4, diversity_penalty=0.5 |
| Summarization | Beam search | num_beams=4, length_penalty=2.0 |
| Chatbot | Sampling | do_sample=True, temperature=0.7, top_p=0.95 |
| Translation | Beam search | num_beams=5, no_repeat_ngram_size=3 |

---

## What is Next

Phase 2 starts with Accelerate — running the exact same training code on
multiple GPUs, TPU, mixed precision, and DeepSpeed with zero code changes.