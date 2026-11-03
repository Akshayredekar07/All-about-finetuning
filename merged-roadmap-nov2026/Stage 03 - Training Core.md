# **Stage 03 — Training Core**
**Outcome:** fine-tune a base model on your own data with LoRA or QLoRA, resume it, and save a checkpoint any other tool can load.**Libraries:** `accelerate` 1.15.0 · `peft` 0.21.2 · `trl` 1.17.1 · `safetensors` 0.8.0 · `bitsandbytes` 0.50.2


---

## **Scope**

Stage 3 is the training core: the four libraries that turn a dataset and a base model into weights you can serve. PEFT decides what is trainable, TRL decides what the loss is, Accelerate decides how the work is distributed, and safetensors decides how the result is stored.

**Exit criterion:** a public LoRA adapter on a domain dataset, with a training curve, an eval number, and a checkpoint that loads in a different process without conversion.

## **Environment**

```bash
pip install -U accelerate peft trl safetensors tokenizers
pip install -U "trl[peft,kernels]"
```

TRL 1.17.1 requires Python ≥3.11, transformers ≥4.56.2, datasets ≥4.7.0, accelerate ≥1.11.0, torch ≥2.6.0, numpy ≥1.26.0. The `peft` extra pulls PEFT ≥0.17; the `kernels` extra pulls `transformers[kernels]`.

---

## **Topic checklist — Accelerate**

### **Core concepts**

- [ ] The `Accelerator` object as central controller
- [ ] `accelerator.prepare()` wrapping model, optimizer, dataloader
- [ ] `accelerator.backward()`
- [ ] Mixed precision (fp16, bf16)
- [ ] Gradient accumulation and gradient clipping

### **Training setups**

- [ ] Single GPU
- [ ] Multi-GPU (DDP)
- [ ] Multi-node
- [ ] TPU and CPU-only
- [ ] DeepSpeed integration (ZeRO stages 1, 2, 3)
- [ ] FSDP (Fully Sharded Data Parallel)

### **State management**

- [ ] `save_state()` and `load_state()`
- [ ] `wait_for_everyone()` — sync across processes
- [ ] `gather()` — collect tensors from all GPUs
- [ ] `is_main_process` — run code only on rank 0

### **Utilities and CLI**

- [ ] `find_executable_batch_size` — automatic batch size search on OOM
- [ ] `notebook_launcher` — distributed training from a notebook
- [ ] `PartialState` — lightweight distributed state
- [ ] `accelerate config`, `accelerate test`, `accelerate launch`

---

## **Topic checklist — PEFT**

### **LoRA**

- [ ] Concept: train only the low-rank decomposition matrices
- [ ] `r` — rank, controls how many parameters are trained
- [ ] `lora_alpha` — scaling factor (keep α = 2×r)
- [ ] `lora_dropout`
- [ ] `target_modules` — which layers get LoRA
- [ ] Choosing `target_modules` for different architectures
- [ ] `target_modules="all-linear"`
- [ ] `LoraConfig`, `get_peft_model()`, `print_trainable_parameters()`
- [ ] Saving adapters separately from the base model
- [ ] `PeftModel.from_pretrained()`
- [ ] `merge_and_unload()`

### **QLoRA**

- [ ] 4-bit quantization with the NF4 dtype
- [ ] Double quantization
- [ ] `BitsAndBytesConfig` integration
- [ ] `prepare_model_for_kbit_training()`
- [ ] Fine-tuning 7B and larger models on a single consumer GPU
- [ ] Merging a QLoRA adapter: reload the base in 16-bit, attach, then merge

### **The wider family**

- [ ] Prefix Tuning, Prompt Tuning, P-Tuning
- [ ] IA3
- [ ] AdaLoRA
- [ ] LoftQ — quantization-aware LoRA initialization
- [ ] DoRA, GaLore, rsLoRA, OLoRA, LoHa, LoKr, OFT, VeRA, X-LoRA

### **Task types and multi-adapter use**

- [ ] `TaskType.SEQ_CLS`, `TOKEN_CLS`, `CAUSAL_LM`, `SEQ_2_SEQ_LM`, `QUESTION_ANS`
- [ ] Loading multiple adapters
- [ ] Switching between adapters
- [ ] Combining adapters — `add`, `cat`, `svd`, `linear`

---

## **Topic checklist — TRL SFT**

- [ ] `SFTTrainer`, `SFTConfig`, `formatting_func`
- [ ] Passing a model id string directly to the trainer
- [ ] Dataset formats — plain text, prompt/completion, conversational `messages`
- [ ] Chat templates and `apply_chat_template()`
- [ ] `packing` — combine short examples for efficiency
- [ ] `max_length` control (was `max_seq_length`)
- [ ] `completion_only_loss` — train only on the completion
- [ ] `assistant_only_loss` — train only on assistant turns (needs a chat template with generation markers)
- [ ] Integration with PEFT and LoRA
- [ ] Memory options — `gradient_checkpointing`, `activation_offloading`, chunked loss
- [ ] `trl vllm-serve` for fast generation during online RL
- [ ] Frozen reference model for KL divergence
- [ ] Reading training logs for collapsed reward signals

### **TRL version discipline**

- [ ] Stable trainers in 1.17.1: SFT, DPO, KTO, Reward, RLOO, GRPO, Distillation
- [ ] Experimental trainers under `trl.experimental` — APIs can change in any minor release
- [ ] PPO was removed in 1.13.0 (`PPOTrainer`, `PPOConfig`, `modeling_value_head.py`)
- [ ] `trl.losses` was removed in 1.14.0
- [ ] The fused Triton logprob + entropy kernel in `trl.kernels` is the default
- [ ] `use_liger_kernel` is deprecated in trainers; removal is scheduled for TRL 2.0.0
- [ ] Pinning `trl` and reading release notes before every upgrade

---

## **Topic checklist — Safetensors**

- [ ] Why it exists: no arbitrary code execution, unlike pickle
- [ ] Format: a JSON header followed by raw tensor bytes
- [ ] Zero-copy, memory-mapped loading
- [ ] Framework support: PyTorch, NumPy, JAX
- [ ] `save_file()` — save tensors to disk
- [ ] `load_file()` — load all tensors
- [ ] `safe_open()` — inspect and load selectively without a full load
- [ ] Listing tensor names without loading the weights
- [ ] Loading a single specific tensor
- [ ] Metadata in safetensors files
- [ ] Sharded checkpoints and `model.safetensors.index.json`
- [ ] Converting an old `.bin` checkpoint by loading and saving again

---

## **Code**

### **Accelerate, manual loop**

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=4,
)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

accelerator.save_state("./checkpoint")
accelerator.load_state("./checkpoint")
```

### **LoRA, in two variants**

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype="auto")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

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
    "Qwen/Qwen3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    task_type="CAUSAL_LM", r=16, lora_alpha=32,
    lora_dropout=0.05, target_modules="all-linear"))
```

### **Save, load, merge**

```python
model.save_pretrained("./lora-adapter")
model.push_to_hub("username/my-lora-adapter")

from peft import PeftModel
base  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype="auto")
model = PeftModel.from_pretrained(base, "username/my-lora-adapter")
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
```

To merge a QLoRA adapter, reload the base in **16-bit** (not 4-bit), attach the adapter, then merge.

### **SFT on your own dataset**

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("username/billing-support-v1.0", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen3-4B-Instruct-2507",
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./sft-model",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        max_length=2048,
        packing=True,
        assistant_only_loss=True,     # train only on assistant turns
        gradient_checkpointing=True,
    ),
    peft_config=LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"
    ),
)

trainer.train()
trainer.save_model("./my-sft-model")
```

### **Safetensors inspection**

```python
import torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open

save_file({"weight": torch.zeros(2, 2), "bias": torch.zeros(2)},
          "model.safetensors", metadata={"format": "pt"})

with safe_open("model.safetensors", framework="pt") as f:
    print(f.keys())
    print(f.metadata())
    weight = f.get_tensor("weight")
```

### **8-bit optimizers**

```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)
# **or in TrainingArguments: optim="paged_adamw_8bit"**
```

---

## **Hyperparameter reference**

| Knob | Default | Notes |
|---|---|---|
| `r` | 16 | 8 when memory-bound, 64 when quality-bound |
| `lora_alpha` | 32 | Keep α = 2×r |
| `target_modules` | all-linear | q, k, v, o, gate, up, down |
| `lora_dropout` | 0.05–0.1 | Raise on overfitting |
| Learning rate | 1e-4 to 5e-4 | 10–100× full FT |
| Optimizer | AdamW, or PagedAdamW8bit for QLoRA | |
| Epochs | 2–3 | Stop when eval loss plateaus |
| `max_length` | 2048 typical | Truncate from the left for long chats |
| Effective batch size | 16+ | per-device × accumulation |

---

## **Memory reference**

| Model | Full FT (fp16 + Adam) | LoRA r=16 | QLoRA NF4 |
|---|---|---|---|
| 7B | ~84 GB | ~16 GB | ~6–8 GB |
| 13B | ~156 GB | ~30 GB | ~10–12 GB |
| 70B | ~600 GB | ~120 GB | ~40–48 GB |

Full FT needs roughly 2× model size for weights plus gradients, plus 4–8× for optimizer state, plus activations. QLoRA fit by rule of thumb: a 3–4B model trains in 8–12 GB, a 13B in about 12 GB, a 33B in 24 GB.

---

## **Labs**

### **Lab 3-1 — LoRA from scratch in PyTorch (~150 lines)**
- Implement `LoRALinear` with a frozen base and a low-rank `A`/`B` update.
- Verify that the output matches the base model exactly at initialization (`B` is zero-initialized).
- Attach it to a real `q_proj` and confirm `print_trainable_parameters()` shows the expected 0.1–0.5% trainable ratio.
- Train briefly and compare the loss curve against PEFT's implementation on identical data.

### **Lab 3-2 — SFT on your own dataset**
- Take your Stage 2 dataset and train LoRA on it.
- Sweep `r` ∈ {8, 16, 32} and learning rate ∈ {1e-4, 2e-4, 5e-4}; record the eval metric for each.
- Try `completion_only_loss` versus `assistant_only_loss` and record the difference in loss scale.
- Push the adapter to the Hub with a model card.

### **Lab 3-3 — QLoRA on a consumer GPU**
- Train a 7B/8B model with NF4 + double quantization on a 16–24 GB card.
- Record peak memory, tokens/sec, and wall time.
- Merge the adapter in 16-bit and confirm the merged model's eval score matches the adapter's.

### **Lab 3-4 — Distributed run with Accelerate**
- `accelerate config`, `accelerate test`, then a two-process DDP run on one node.
- Add gradient accumulation and confirm the effective batch size is what you think it is.
- Save and resume from `accelerator.save_state()` and confirm the loss curve is continuous.

### **Lab 3-5 — Checkpoint hygiene**
- Save with `safetensors` and confirm no `.bin` files appear.
- Convert an old `.bin` checkpoint to safetensors and verify identical outputs.
- Inspect a sharded checkpoint through `model.safetensors.index.json`.

---

## **Pitfalls**

- **`max_seq_length` → `max_length`.** The old name raises an error in TRL 1.x.
- **`DataCollatorForCompletionOnlyLM` is gone.** Use `completion_only_loss` or `assistant_only_loss`.
- **Assistant-only loss needs a chat template with generation markers.** Without them, the mask is wrong and the loss is meaningless.
- **`packing=True` changes the effective loss.** Short examples are concatenated; verify your metrics still mean what you think.
- **Merging a 4-bit model produces garbage.** Reload the base in 16-bit before merging.
- **`target_modules` names differ per architecture.** `q_proj`/`v_proj` is a Qwen/Llama convention; a Mistral checkpoint may need `q_proj`, `k_proj`, `v_proj`, `o_proj` explicitly.
- **PPO is gone from TRL.** A tutorial importing `PPOTrainer` will fail on import; that path is verl or OpenRLHF now.
- **Experimental trainers disappear.** TRL 1.14 deleted six; 1.17 removed TPO. Pin the version for anything longer than a weekend.
- **The `use_liger_kernel` flag is deprecated** and slated for removal in TRL 2.0.0.
- **`bf16=True` on a pre-Ampere GPU** produces overflow rather than a clear error.

---

## **Acceptance gate**

- [ ] A LoRA implementation written from scratch matches PEFT numerically.
- [ ] A QLoRA fine-tune completed on your own dataset and was pushed to the Hub.
- [ ] A merged model scores the same as the adapter it came from.
- [ ] A two-process Accelerate run trained and resumed cleanly.
- [ ] Checkpoints are safetensors; the repo contains no `.bin` or `.pt` files.
- [ ] The hyperparameter sweep table (r × learning rate) is attached to the model card.

---

## **Resources**

- TRL v1.17.1 release — https://github.com/huggingface/trl/releases/tag/v1.17.1
- trl on PyPI — https://pypi.org/project/trl/
- PEFT v0.21.2 release — https://github.com/huggingface/peft/releases/tag/v0.21.2
- accelerate on PyPI — https://pypi.org/project/accelerate/
- safetensors on PyPI — https://pypi.org/project/safetensors/
- bitsandbytes on PyPI — https://pypi.org/project/bitsandbytes/

**Next:** Stage 04 — Alignment and Post-Training.
