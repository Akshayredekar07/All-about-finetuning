# **Stage 04 — Training Core**

**Libraries:** `accelerate` 1.15.0 · `peft` 0.21.2 · `trl` 1.17.1 · `safetensors` 0.8.0 · `bitsandbytes` 0.50.2
**Outcome:** fine-tune a base model on your own data with LoRA or QLoRA, resume it, and save a checkpoint any other tool can load.

## **Accelerate 1.15.0**

Runs the same PyTorch training code on a single GPU, multi-GPU, TPU, or CPU with minimal changes. It powers the `Trainer` inside Transformers [30].

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

accelerator.save_state("./checkpoint")
accelerator.load_state("./checkpoint")
```

**Core concepts**
- `Accelerator` object — central controller
- `accelerator.prepare()` — wraps model, optimizer, dataloader
- `accelerator.backward()`
- Mixed precision (fp16, bf16)
- Gradient accumulation and gradient clipping

**Training setups**
- Single GPU, multi-GPU (DDP), multi-node
- TPU and CPU-only
- DeepSpeed integration (ZeRO stages 1, 2, 3)
- FSDP (Fully Sharded Data Parallel)

**State management**
- `accelerator.save_state()` and `accelerator.load_state()`
- `accelerator.wait_for_everyone()` — sync across processes
- `accelerator.gather()` — collect tensors from all GPUs
- `accelerator.is_main_process` — run code only on rank 0

**Utilities**
- `find_executable_batch_size` — automatic batch size search on OOM
- `notebook_launcher` — run distributed training from a notebook
- `PartialState` — lightweight distributed state

**CLI**
```bash
accelerate config
accelerate test
accelerate launch train.py
accelerate launch --num_processes=4 --mixed_precision=bf16 train.py
```

**Key install**
```bash
pip install -U accelerate
```

## **PEFT 0.21.2 — parameter-efficient fine-tuning**

Fine-tune massive models (7B, 13B, 70B) on limited GPUs by training a tiny fraction of the weights. LoRA is the dominant technique [9].

**LoRA — the core idea.** Freeze `W ∈ R^(d×k)` and learn a low-rank update `ΔW = B @ A`, with `A ∈ R^(r×k)`, `B ∈ R^(d×r)`, and `r << min(d, k)`. Only A and B receive gradients.

```python
class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base_linear  # frozen
        for p in self.base.parameters():
            p.requires_grad = False
        in_dim = base_linear.in_features
        out_dim = base_linear.out_features
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)  # Gaussian init
        self.B = nn.Parameter(torch.zeros(out_dim, r))         # zero init
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base_out = self.base(x)  # no grad
        lora_out = self.dropout(x) @ self.A.T @ self.B.T * self.scaling
        return base_out + lora_out
```

**The default LoRA configuration**

| Hyperparameter | Default | When to change |
|---|---|---|
| rank (`r`) | 16 | r=8 if memory-bound, r=64 if quality-bound |
| `lora_alpha` | 32 | Keep α = 2×r; α/r is the scaling factor |
| `target_modules` | all-linear (q, k, v, o, gate, up, down) | The original paper used only q, v; all-linear is the 2026 default |
| `lora_dropout` | 0.05–0.1 | Raise if overfitting |
| learning rate | 1e-4 to 5e-4 | 10–100× larger than full FT |
| optimizer | AdamW (PagedAdamW8bit for QLoRA) | — |
| epochs | 2–3 | Stop when eval loss plateaus |

**Memory savings.** A 7B full fine-tune in fp16 needs roughly 84 GB (14 GB weights + 14 GB gradients + 56 GB Adam state) plus activations; LoRA r=16 needs about 16 GB — a ~5× reduction. For 70B the same ratio holds: ~600 GB full FT versus ~120 GB LoRA.

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype="auto")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # or "all-linear"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: ... || all params: ... || trainable%: ...
```

Target module names differ per architecture. Print the model and read the layer names, or use `target_modules="all-linear"`.

**QLoRA — a 4-bit base with BF16 adapters**

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

The 4-bit quantization applies only to the frozen base; adapters train in BF16, which is why QLoRA matches fp16 LoRA closely (Dettmers et al., 2023). Reference timings from the source notes: Llama-3 8B QLoRA r=16 on an RTX 4090 takes ~4 hours versus 8–10 hours with vanilla TRL; Llama-3 70B QLoRA r=16 on a single 80GB H100 takes ~24 hours; Qwen3-30B-A3B QLoRA on H100 runs up to 1.77× faster with Unsloth than vanilla.

**Save, load, and merge adapters**
```python
model.save_pretrained("./lora-adapter")        # saves only the small adapter
model.push_to_hub("username/my-lora-adapter")

from peft import PeftModel
base  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype="auto")
model = PeftModel.from_pretrained(base, "username/my-lora-adapter")

merged = model.merge_and_unload()   # bake the adapter into the base weights
merged.save_pretrained("./merged-model")
```
To merge a QLoRA adapter, reload the base in **16-bit** (not 4-bit), attach the adapter, then merge.

**LoRA family — know the alternatives, switch only on measurement**

| Variant | Mechanism | When it wins |
|---|---|---|
| DoRA | Decomposes `W = m * V / ||V||` where `m` is magnitude and `V/||V||` is unit direction, then trains `m` and `V` separately with LoRA on the direction | Measured LoRA underperformance at the same rank; ~1–3% on reasoning benchmarks in some studies |
| GaLore | Projects gradients into a low-rank subspace before the optimizer step | Full-rank updates needed but full-FT memory is unaffordable |
| rsLoRA | Scales by `1/sqrt(r)` instead of `α/r` | Higher ranks (r ≥ 64) |
| OLoRA | Orthogonal init for `A` | Faster convergence in some cases |
| LoHa | Low-rank + Hadamard product | More expressive at the same rank |
| LoKr | Low-rank + Kronecker product | More expressive at the same rank |
| OFT | Orthogonal update matrices | Better preservation of base behavior |
| VeRA | Shared frozen random matrices; train only scaling vectors | Smallest memory of the family |
| X-LoRA | Mixture of LoRA experts | Multi-task fine-tuning |
| AdaLoRA | Adaptive per-layer rank allocation | Unknown right rank per layer |
| IA3 | Learned activation rescaling | Few-shot adaptation, smaller than LoRA |
| LoftQ | Quantization-aware LoRA init | QLoRA from a quantized starting point |

The honest verdict from the source material, unchanged by any version drift: start with vanilla LoRA r=16, all-linear, α=32, and switch only when you have measured a problem. Most "X-LoRA beats LoRA" claims do not replicate at scale.

**Other PEFT methods** — Prefix Tuning, Prompt Tuning, P-Tuning, IA3, AdaLoRA, LoftQ.

**Task types** — `TaskType.SEQ_CLS`, `TaskType.TOKEN_CLS`, `TaskType.CAUSAL_LM`, `TaskType.SEQ_2_SEQ_LM`, `TaskType.QUESTION_ANS`.

**Multi-adapter usage** — load several adapters, switch between them, and combine them (`add`, `cat`, `svd`, `linear`).

**Key install**
```bash
pip install -U peft
```

## **TRL 1.17.1 — supervised fine-tuning**

TRL is the post-training primitive layer: SFT, preference optimization, and RL [6][7].

The tier model changed under you
TRL v1.x splits trainers into stable (semver-guaranteed) and `trl.experimental` (APIs can change in any minor release). **PPO was removed from TRL entirely in 1.13.0** (`PPOTrainer`, `PPOConfig`, `modeling_value_head.py`), and `trl.losses` was removed in 1.14.0. The current stable set is SFT, DPO, KTO (stable since 1.8), Reward, RLOO, GRPO, and Distillation (stable since 1.10). Experimental trainers have been deleted in minor releases before — v1.14 removed six of them and v1.17 removed TPO — so pin `trl` and check release notes before upgrading.

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")   # conversational "messages" format

trainer = SFTTrainer(
    model="Qwen/Qwen3-1.7B",                 # a model id string is enough
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

**SFT checklist**
- `SFTTrainer`, `SFTConfig`, `formatting_func`
- Dataset formats: plain text (`dataset_text_field`), prompt/completion `(prompt, completion)` pairs, conversational `messages`
- Chat templates and `apply_chat_template()`
- `packing` — combine short examples for efficiency; `max_length` control (was `max_seq_length`)
- `completion_only_loss` — train only on the completion (replaces `DataCollatorForCompletionOnlyLM`)
- `assistant_only_loss` — train only on assistant turns; needs a chat template with generation markers
- Integration with PEFT and LoRA
- Memory options: `gradient_checkpointing`, `activation_offloading`, chunked loss options
- `use_liger_kernel` is **deprecated** in trainers and scheduled for removal in TRL 2.0.0

The kernel that matters
TRL now defaults to a **fused Triton logprob + entropy kernel** in `trl.kernels`. It is not a micro-optimization: with it, DPO's maximum sequence length rises from 10,240 to 59,392 tokens on a 79 GiB GPU. Long-context preference tuning is newly practical.

**Package extras that decide your install:** `trl[peft]` pulls PEFT ≥0.17; `trl[kernels]` pulls `transformers[kernels]`; `trl[harbor]` pulls Harbor ≥0.13 on Python ≥3.12; `trl[vlm]` pulls torchvision. Base requirements are transformers ≥4.56.2, datasets ≥4.7.0, accelerate ≥1.11.0, torch ≥2.6.0, numpy ≥1.26.0, Python ≥3.11.

**Utilities**
- `trl vllm-serve` — fast generation during online RL
- Frozen reference model for KL divergence
- Training logs and warnings for collapsed reward signals

**Key install**
```bash
pip install -U trl
pip install -U "trl[peft,kernels]"
```

## **Safetensors 0.8.0**

The standard format for storing neural network weights: safe, fast, and with no arbitrary code execution risk, unlike pickle-based `.bin` and `.pt` files. All modern models on the Hub use it [31].

```python
import torch
from safetensors.torch import save_file, load_file

tensors = {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}
save_file(tensors, "model.safetensors", metadata={"format": "pt"})

tensors = load_file("model.safetensors")

model.save_pretrained("./model")                 # writes model.safetensors
model = AutoModel.from_pretrained("./model")     # loads it automatically

from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    print(f.keys())                     # all tensor names
    print(f.metadata())                 # file metadata
    weight = f.get_tensor("weight")     # load one tensor only
```

**Checklist**
- Why it exists: no code execution, unlike pickle
- Format: a JSON header followed by raw tensor bytes
- Zero-copy, memory-mapped loading
- Framework support: PyTorch, NumPy, JAX
- Sharded checkpoints (`model.safetensors.index.json`)
- `save_pretrained()` uses safetensors by default; `from_pretrained()` prefers it
- Converting an old `.bin` checkpoint: load it once, then `save_pretrained()` again
- Listing tensor names without loading the weights

**Key install**
```bash
pip install -U safetensors
```

---

## **Optimizers**

**AdamW (standard):**

```
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
θ_t = θ_{t-1} - α(m̂_t/(√v̂_t+ε) + λθ_{t-1})
β₁=0.9, β₂=0.999, ε=1e-8, λ=0.01
```

**8-bit Adam (bitsandbytes):** optimizer states in 8 bits, ~75% memory saving on optimizer state.

**Paged Adam (QLoRA):** optimizer states live in CUDA unified memory, so the optimizer cannot OOM the trainer the way a naive implementation can.

**Muon (2025–2026):** orthogonalises momentum with a Newton–Schulz iteration; used in several 2026 pretraining runs.

**LR schedule:** warmup (100–500 steps) → cosine decay. LoRA learning rates sit at 1e-4 to 5e-5; full fine-tuning at 1e-5 to 2e-5. Gemma 4's own tooling suggests 2e-4 for QLoRA and 2e-5 for full fine-tuning, with LoRA rank 16 or 32 and `lora_alpha` at twice the rank.

## **SFT data preparation**

- `tokenizer.apply_chat_template()` — standardised across Llama, Qwen, Gemma, DeepSeek, and it is the only correct way to render training text.
- Multi-turn format: `system → user → assistant → tool_call → tool_response → assistant`.
- `DataCollatorForCompletionOnlyLM` is gone in TRL 1.17.1; use `completion_only_loss` and `assistant_only_loss` instead.
- **Packing** concatenates short examples and buys 2–4× throughput.
- Quality beats quantity: 1K curated examples often beats 100K noisy ones.

```python
# LoRA target modules by architecture
# Llama / Mistral / Gemma 4 / DeepSeek V4-Flash
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Qwen3 / Qwen3.5 / Qwen3.8 (dense)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Qwen3.5 MoE — same modules; Unsloth handles expert routing automatically

# BERT / RoBERTa (encoder-only)
target_modules = ["query", "value"]

# Whisper (ASR)
target_modules = ["q_proj", "v_proj"]  # encoder layers
```

## **VRAM budget for supervised fine-tuning**

The 8B reference row is the classical baseline; the Gemma 4 and Qwen rows are what you will actually use in Nov 2026.

| Config | Approx VRAM | Notes |
|---|---|---|
| Full FT, BF16, bs=1 (8B) | ~80 GB | A100/H100 required |
| Full FT + FSDP, 8×A100 (8B) | ~10 GB/GPU | Standard multi-GPU |
| LoRA r=64, BF16 (8B) | ~28 GB | Single A100 |
| QLoRA 4-bit r=32 (8B) | ~10 GB | RTX 3090 / 4090 |
| Unsloth QLoRA 4-bit (8B) | ~7–8 GB | Optimised kernels |
| Gemma 4 E2B, Unsloth QLoRA | ~8 GB | Smallest model that still does image + audio |
| Gemma 4 E4B, Unsloth QLoRA | ~10 GB | **The recommended default for a consumer GPU** |
| Gemma 4 31B, Unsloth QLoRA | ~22 GB | Fits a 24 GB card |
| Gemma 4 26B-A4B, LoRA | >40 GB | 4-bit MoE is blocked by bitsandbytes |
| Qwen3.5-2B, bf16 LoRA | ~5 GB | No 4-bit for Qwen3.5 |
| Qwen3.5-35B-A3B, bf16 LoRA | ~74 GB | MoE; bf16 only |
| Qwen3.8-27B, QLoRA | ~24 GB | LoRA needs >36 GB; full FT 4× more |
| Mistral Large 4, FP8 LoRA | 8×B300 node | TRL 1.17.0 ships the recipe |

## **SFT framework selection**

| Framework | Best for | Key feature |
|---|---|---|
| **Unsloth** | Single GPU; consumer hardware | ~1.5× faster and ~50–60% less VRAM than FA2 setups; MoE support; GRPO/GSPO/DAPO |
| **Axolotl** | Production pipelines | YAML config; FSDP + FlashAttention + QLoRA. **Python ≥3.12** |
| **LlamaFactory** | Rapid iteration | Web UI + CLI; broadest model support; **unpatched RCE in the WebUI** |
| **TRL SFTTrainer** | Hugging Face native | Most flexible; programmatic; supports `environment_factory` for agentic rollouts |
| **Tunix (Google)** | TPU clusters | JAX-native; Gemma/Llama/Qwen; GRPO. Pointer only — not re-verified in this refresh |
| **Haven (YC 2025)** | Serverless | Hundreds of LoRA adapters; claimed <1s cold start. Pointer only — not re-verified in this refresh |

## **Thinking-mode handling**

Qwen3, Qwen3.5, Qwen3.6+ and the Gemma 4 reasoning variants all expose a thinking mode, and mishandling it during SFT is one of the most common ways to destroy a model's reasoning ability.

```python
# Qwen3.5 — enable thinking per turn
messages = [{"role": "user", "content": "/think Solve this math problem: ..."}]
# or add to system prompt: "enable_thinking: true"

# To fully disable thinking at inference
tokenizer.apply_chat_template(messages, enable_thinking=False)

# preserve_thinking for multi-turn agentic (Qwen3.6+):
# pass preserve_thinking=True in extra_body to retain CoT across turns
```

Qwen3.8 adds a third control: `reasoning_effort` with `xhigh` (default), `medium` and `low`, alongside `enable_thinking=False` to turn reasoning off entirely for a turn, and `preserve_thinking=True` as the default that retains prior-turn reasoning across a conversation [70].

## **Catastrophic forgetting**

| Strategy | When to use |
|---|---|
| PEFT (LoRA) — default | Base frozen; only adapters change |
| Replay buffers | Mix 10–30% general data in a full fine-tune |
| Multi-task training | High-data regimes |
| Conservative LR | Full fine-tune; reduces the forgetting rate |
| Layer-wise LR decay | Standard practice for full fine-tuning |
| Thinking-mode preservation | Mix ≥75% reasoning-style examples to preserve `<think>` capability |

For the thinking-mode row specifically: mixing at least three-quarters reasoning-style examples into a general SFT mixture is what keeps a Qwen3.5-style model's reasoning behaviour intact. Below that threshold the model usually still works, but silently stops emitting `<think>` blocks, and you will not notice until the benchmark drops.

---

## **Worksheet additions**

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
# or in TrainingArguments: optim="paged_adamw_8bit"
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

**Next:** Stage 05 — Alignment and Post-Training.
