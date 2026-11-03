## **Stage 04: Training Core**

**Libraries:** `accelerate`, `peft`, `trl`, `safetensors`, `bitsandbytes`

**Outcome:** fine-tune a base model on your own data with LoRA or QLoRA, resume it, and save a checkpoint any other tool can load.

### **1. Scope and Exit Criterion**

### **1.1 Scope**

This stage is the training core: the four libraries that turn a dataset and a base model into weights you can serve. PEFT decides what is trainable, TRL decides what the loss is, Accelerate decides how the work is distributed, and safetensors decides how the result is stored.

### **1.2 Exit Criterion**

A public LoRA adapter on a domain dataset, with a training curve, an eval number, and a checkpoint that loads in a different process without conversion.

### **2. Environment**

```bash
pip install -U accelerate peft trl safetensors tokenizers
pip install -U "trl[peft,kernels]"
```

TRL requires Python 3.11 or newer, plus recent releases of transformers, datasets, accelerate, torch, and numpy. The package extras decide your install:

- `trl[peft]` pulls a recent PEFT.
- `trl[kernels]` pulls `transformers[kernels]`.
- `trl[harbor]` pulls Harbor and needs Python 3.12 or newer.
- `trl[vlm]` pulls torchvision.

### **3. Accelerate**

### **3.1 Overview**

Accelerate runs the same PyTorch training code on a single GPU, multi-GPU, TPU, or CPU with minimal changes. It powers the `Trainer` inside Transformers.

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

### **3.2 Core Concepts**

- The `Accelerator` object is the central controller
- `accelerator.prepare()` wraps the model, optimizer, and dataloader
- `accelerator.backward()`
- Mixed precision (fp16, bf16)
- Gradient accumulation and gradient clipping

### **3.3 Training Setups**

- Single GPU
- Multi-GPU (DDP)
- Multi-node
- TPU and CPU-only
- DeepSpeed integration (ZeRO stages 1, 2, 3)
- FSDP (Fully Sharded Data Parallel)

### **3.4 State Management**

- `accelerator.save_state()` and `accelerator.load_state()`
- `accelerator.wait_for_everyone()`: sync across processes
- `accelerator.gather()`: collect tensors from all GPUs
- `accelerator.is_main_process`: run code only on rank 0

### **3.5 Utilities**

- `find_executable_batch_size`: automatic batch size search on OOM
- `notebook_launcher`: run distributed training from a notebook
- `PartialState`: lightweight distributed state

### **3.6 CLI**

```bash
accelerate config
accelerate test
accelerate launch train.py
accelerate launch --num_processes=4 --mixed_precision=bf16 train.py
```

### **3.7 Install**

```bash
pip install -U accelerate
```

### **4. PEFT: Parameter-Efficient Fine-Tuning**

### **4.1 The Core Idea of LoRA**

PEFT lets you fine-tune massive models (7B, 13B, 70B) on limited GPUs by training a tiny fraction of the weights. LoRA is the dominant technique.

Freeze $W \in \mathbb{R}^{d \times k}$ and learn a low-rank update:

$$\Delta W = B A, \quad A \in \mathbb{R}^{r \times k}, \quad B \in \mathbb{R}^{d \times r}, \quad r \ll \min(d, k)$$

Only $A$ and $B$ receive gradients. The adapted output is:

$$y = W x + \frac{\alpha}{r} \, B A x$$

### **4.2 Worked Example: LoRA Forward Pass**

Tanvi's layer has input size $k = 3$, output size $d = 2$, rank $r = 2$, and $\alpha = 4$, so the scaling factor is $\alpha / r = 2$.

$$x = \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix}, \quad A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 2 \end{bmatrix}, \quad B = \begin{bmatrix} 1 & 2 \\ 3 & 0 \end{bmatrix}$$

Step 1: compute $h = A x$, row by row.

- Row 1: $1 \cdot 2 + 0 \cdot 1 + 1 \cdot 3 = 2 + 0 + 3 = 5$
- Row 2: $0 \cdot 2 + 1 \cdot 1 + 2 \cdot 3 = 0 + 1 + 6 = 7$

$$h = \begin{bmatrix} 5 \\ 7 \end{bmatrix}$$

Step 2: compute $B h$, row by row.

- Row 1: $1 \cdot 5 + 2 \cdot 7 = 5 + 14 = 19$
- Row 2: $3 \cdot 5 + 0 \cdot 7 = 15 + 0 = 15$

$$B h = \begin{bmatrix} 19 \\ 15 \end{bmatrix}$$

Step 3: apply the scaling factor of 2.

- Row 1: $2 \cdot 19 = 38$
- Row 2: $2 \cdot 15 = 30$

$$\boxed{\frac{\alpha}{r} B A x = \begin{bmatrix} 38 \\ 30 \end{bmatrix}}$$

This is added to the frozen base output $W x$. At initialization $B = 0$, so the added term is zero and the model starts identical to the base.

### **4.3 LoRA from Scratch**

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

### **4.4 Default LoRA Configuration and Hyperparameters**

| **Hyperparameter** | **Default** | **When to change** |
|---|---|---|
| rank (`r`) | 16 | r=8 if memory-bound, r=64 if quality-bound |
| `lora_alpha` | 32 | Keep $\alpha = 2r$; $\alpha / r$ is the scaling factor |
| `target_modules` | all-linear (q, k, v, o, gate, up, down) | The original paper used only q, v; all-linear is the current default |
| `lora_dropout` | 0.05 to 0.1 | Raise if overfitting |
| learning rate | 1e-4 to 5e-4 | 10 to 100 times larger than full fine-tuning |
| optimizer | AdamW (PagedAdamW8bit for QLoRA) | No change needed |
| epochs | 2 to 3 | Stop when eval loss plateaus |
| `max_length` | 2048 typical | Truncate from the left for long chats |
| effective batch size | 16 or more | Per-device batch size times accumulation steps |

### **4.5 Memory Savings**

A 7B full fine-tune in fp16 needs roughly the following, plus activations:

$$14\ \text{GB (weights)} + 14\ \text{GB (gradients)} + 56\ \text{GB (Adam state)} = \boxed{84\ \text{GB}}$$

LoRA with r=16 needs about 16 GB, a reduction of about 5 times. For 70B the same ratio holds: about 600 GB for full fine-tuning versus about 120 GB for LoRA.

### **4.6 LoRA with PEFT**

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

A variant that uses all-linear on a different base model:

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

Target module names differ per architecture. Print the model and read the layer names, or use `target_modules="all-linear"`.

### **4.7 LoRA Topic List**

- Concept: train only the low-rank decomposition matrices
- `r`: rank, controls how many parameters are trained
- `lora_alpha`: scaling factor (keep $\alpha = 2r$)
- `lora_dropout`
- `target_modules`: which layers get LoRA
- Choosing `target_modules` for different architectures
- `target_modules="all-linear"`
- `LoraConfig`, `get_peft_model()`, `print_trainable_parameters()`
- Saving adapters separately from the base model
- `PeftModel.from_pretrained()`
- `merge_and_unload()`

### **4.8 QLoRA: 4-bit Base with BF16 Adapters**

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

The same recipe on a different base model, with the LoRA config inline:

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

The 4-bit quantization applies only to the frozen base. Adapters train in BF16, which is why QLoRA matches fp16 LoRA closely (Dettmers et al., 2023).

Reference timings from the source notes:

- Llama-3 8B QLoRA r=16 on an RTX 4090 takes about 4 hours, versus 8 to 10 hours with vanilla TRL.
- Llama-3 70B QLoRA r=16 on a single 80GB H100 takes about 24 hours.
- Qwen3-30B-A3B QLoRA on H100 runs up to 1.77 times faster with Unsloth than vanilla.

QLoRA topic list:

- 4-bit quantization with the NF4 dtype
- Double quantization
- `BitsAndBytesConfig` integration
- `prepare_model_for_kbit_training()`
- Fine-tuning 7B and larger models on a single consumer GPU
- Merging a QLoRA adapter: reload the base in 16-bit, attach, then merge

### **4.9 Save, Load, and Merge Adapters**

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

### **4.10 The LoRA Family**

Know the alternatives, and switch only on measurement.

| **Variant** | **Mechanism** | **When it wins** |
|---|---|---|
| DoRA | Decomposes $W = m \cdot V / \lVert V \rVert$, where $m$ is magnitude and $V / \lVert V \rVert$ is unit direction, then trains $m$ and $V$ separately with LoRA on the direction | Measured LoRA underperformance at the same rank; about 1 to 3% on reasoning benchmarks in some studies |
| GaLore | Projects gradients into a low-rank subspace before the optimizer step | Full-rank updates needed but full fine-tuning memory is unaffordable |
| rsLoRA | Scales by $1 / \sqrt{r}$ instead of $\alpha / r$ | Higher ranks (r of 64 or more) |
| OLoRA | Orthogonal init for `A` | Faster convergence in some cases |
| LoHa | Low-rank plus Hadamard product | More expressive at the same rank |
| LoKr | Low-rank plus Kronecker product | More expressive at the same rank |
| OFT | Orthogonal update matrices | Better preservation of base behavior |
| VeRA | Shared frozen random matrices; train only scaling vectors | Smallest memory of the family |
| X-LoRA | Mixture of LoRA experts | Multi-task fine-tuning |
| AdaLoRA | Adaptive per-layer rank allocation | Unknown right rank per layer |
| IA3 | Learned activation rescaling | Few-shot adaptation, smaller than LoRA |
| LoftQ | Quantization-aware LoRA init | QLoRA from a quantized starting point |

The honest verdict: start with vanilla LoRA r=16, all-linear, $\alpha = 32$, and switch only when you have measured a problem. Most "X-LoRA beats LoRA" claims do not replicate at scale.

### **4.11 Other PEFT Methods, Task Types, and Multi-Adapter Use**

**Other PEFT methods:** Prefix Tuning, Prompt Tuning, P-Tuning, IA3, AdaLoRA, LoftQ.

**Task types:** `TaskType.SEQ_CLS`, `TaskType.TOKEN_CLS`, `TaskType.CAUSAL_LM`, `TaskType.SEQ_2_SEQ_LM`, `TaskType.QUESTION_ANS`.

**Multi-adapter usage:**

- Load several adapters
- Switch between adapters
- Combine adapters (`add`, `cat`, `svd`, `linear`)

### **4.12 Install**

```bash
pip install -U peft
```

### **5. TRL: Supervised Fine-Tuning**

### **5.1 Overview**

TRL is the post-training primitive layer: SFT, preference optimization, and RL.

### **5.2 Trainer Tiers and Version Discipline**

The tier model has changed. TRL now splits trainers into stable (semver-guaranteed) and `trl.experimental` (APIs can change in any minor release).

- **PPO was removed from TRL entirely** (`PPOTrainer`, `PPOConfig`, `modeling_value_head.py`).
- `trl.losses` was removed.
- The current stable set is SFT, DPO, KTO, Reward, RLOO, GRPO, and Distillation.
- Experimental trainers have been deleted in minor releases before: one release removed six of them and a later one removed TPO. Pin `trl` and check release notes before upgrading.
- `use_liger_kernel` is **deprecated** in trainers and scheduled for removal in the next major release.

### **5.3 The Kernel That Matters**

TRL now defaults to a **fused Triton logprob and entropy kernel** in `trl.kernels`. It is not a micro-optimization: with it, DPO's maximum sequence length rises from 10,240 to 59,392 tokens on a 79 GiB GPU. Long-context preference tuning is newly practical.

### **5.4 SFT Example**

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
        max_length=1024,                       # formerly max_seq_length
        packing=True,                          # combine short examples for efficiency
    ),
    peft_config=LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"
    ),
)

trainer.train()
trainer.save_model("./my-sft-model")
```

### **5.5 SFT on Your Own Dataset**

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

### **5.6 SFT Topic List**

- `SFTTrainer`, `SFTConfig`, `formatting_func`
- Passing a model id string directly to the trainer
- Dataset formats: plain text (`dataset_text_field`), prompt/completion `(prompt, completion)` pairs, conversational `messages`
- Chat templates and `apply_chat_template()`
- `packing`: combine short examples for efficiency
- `max_length` control (formerly `max_seq_length`)
- `completion_only_loss`: train only on the completion (replaces `DataCollatorForCompletionOnlyLM`)
- `assistant_only_loss`: train only on assistant turns; needs a chat template with generation markers
- Integration with PEFT and LoRA
- Memory options: `gradient_checkpointing`, `activation_offloading`, chunked loss options
- `trl vllm-serve`: fast generation during online RL
- Frozen reference model for KL divergence
- Reading training logs and warnings for collapsed reward signals

### **5.7 TRL Version Discipline Checklist**

- Stable trainers: SFT, DPO, KTO, Reward, RLOO, GRPO, Distillation
- Experimental trainers under `trl.experimental`: APIs can change in any minor release
- PPO was removed (`PPOTrainer`, `PPOConfig`, `modeling_value_head.py`)
- `trl.losses` was removed
- The fused Triton logprob and entropy kernel in `trl.kernels` is the default
- `use_liger_kernel` is deprecated in trainers; removal is scheduled for the next major release
- Pin `trl` and read release notes before every upgrade

### **5.8 Install**

```bash
pip install -U trl
pip install -U "trl[peft,kernels]"
```

### **6. Safetensors**

### **6.1 Overview**

Safetensors is the standard format for storing neural network weights: safe, fast, and with no arbitrary code execution risk, unlike pickle-based `.bin` and `.pt` files. All modern models on the Hub use it.

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

### **6.2 Topic List**

- Why it exists: no arbitrary code execution, unlike pickle
- Format: a JSON header followed by raw tensor bytes
- Zero-copy, memory-mapped loading
- Framework support: PyTorch, NumPy, JAX
- `save_file()`: save tensors to disk
- `load_file()`: load all tensors
- `safe_open()`: inspect and load selectively without a full load
- Listing tensor names without loading the weights
- Loading a single specific tensor
- Metadata in safetensors files
- Sharded checkpoints (`model.safetensors.index.json`)
- `save_pretrained()` uses safetensors by default; `from_pretrained()` prefers it
- Converting an old `.bin` checkpoint: load it once, then `save_pretrained()` again

### **6.3 Install**

```bash
pip install -U safetensors
```

### **7. Optimizers**

### **7.1 AdamW (Standard)**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

Default values:

$$\boxed{\beta_1 = 0.9, \quad \beta_2 = 0.999, \quad \epsilon = 10^{-8}, \quad \lambda = 0.01}$$

### **7.2 Other Optimizers**

- **8-bit Adam (bitsandbytes):** optimizer states in 8 bits, about 75% memory saving on optimizer state.
- **Paged Adam (QLoRA):** optimizer states live in CUDA unified memory, so the optimizer cannot OOM the trainer the way a naive implementation can.
- **Muon (2025 to 2026):** orthogonalises momentum with a Newton-Schulz iteration; used in several recent pretraining runs.

```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)
# or in TrainingArguments: optim="paged_adamw_8bit"
```

### **7.3 Learning Rate Schedule**

Warmup (100 to 500 steps), then cosine decay. LoRA learning rates sit at 1e-4 to 5e-5; full fine-tuning at 1e-5 to 2e-5. Gemma 4's own tooling suggests 2e-4 for QLoRA and 2e-5 for full fine-tuning, with LoRA rank 16 or 32 and `lora_alpha` at twice the rank.

### **8. SFT Data Preparation**

- `tokenizer.apply_chat_template()` is standardised across Llama, Qwen, Gemma, and DeepSeek, and it is the only correct way to render training text.
- Multi-turn format: `system → user → assistant → tool_call → tool_response → assistant`.
- `DataCollatorForCompletionOnlyLM` is gone in current TRL. Use `completion_only_loss` and `assistant_only_loss` instead.
- **Packing** concatenates short examples and buys 2 to 4 times the throughput.
- Quality beats quantity: 1K curated examples often beat 100K noisy ones.

```python
# LoRA target modules by architecture
# Llama / Mistral / Gemma 4 / DeepSeek V4-Flash
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Qwen3 / Qwen3.5 / Qwen3.8 (dense)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Qwen3.5 MoE: same modules; Unsloth handles expert routing automatically

# BERT / RoBERTa (encoder-only)
target_modules = ["query", "value"]

# Whisper (ASR)
target_modules = ["q_proj", "v_proj"]  # encoder layers
```

### **9. VRAM Budget for Supervised Fine-Tuning**

The 8B reference rows are the classical baseline. The Gemma 4 and Qwen rows are what you will actually use in November 2026.

| **Config** | **Approx VRAM** | **Notes** |
|---|---|---|
| Full FT, BF16, bs=1 (8B) | ~80 GB | A100/H100 required |
| Full FT + FSDP, 8xA100 (8B) | ~10 GB/GPU | Standard multi-GPU |
| LoRA r=64, BF16 (8B) | ~28 GB | Single A100 |
| QLoRA 4-bit r=32 (8B) | ~10 GB | RTX 3090 / 4090 |
| Unsloth QLoRA 4-bit (8B) | ~7 to 8 GB | Optimised kernels |
| Gemma 4 E2B, Unsloth QLoRA | ~8 GB | Smallest model that still does image and audio |
| Gemma 4 E4B, Unsloth QLoRA | ~10 GB | **The recommended default for a consumer GPU** |
| Gemma 4 31B, Unsloth QLoRA | ~22 GB | Fits a 24 GB card |
| Gemma 4 26B-A4B, LoRA | >40 GB | 4-bit MoE is blocked by bitsandbytes |
| Qwen3.5-2B, bf16 LoRA | ~5 GB | No 4-bit for Qwen3.5 |
| Qwen3.5-35B-A3B, bf16 LoRA | ~74 GB | MoE; bf16 only |
| Qwen3.8-27B, QLoRA | ~24 GB | LoRA needs >36 GB; full FT 4 times more |
| Mistral Large 4, FP8 LoRA | 8xB300 node | TRL ships the recipe |

### **10. SFT Framework Selection**

| **Framework** | **Best for** | **Key feature** |
|---|---|---|
| **Unsloth** | Single GPU; consumer hardware | About 1.5 times faster and 50 to 60% less VRAM than FA2 setups; MoE support; GRPO/GSPO/DAPO |
| **Axolotl** | Production pipelines | YAML config; FSDP + FlashAttention + QLoRA. Needs Python 3.12 or newer |
| **LlamaFactory** | Rapid iteration | Web UI + CLI; broadest model support; **unpatched RCE in the WebUI** |
| **TRL SFTTrainer** | Hugging Face native | Most flexible; programmatic; supports `environment_factory` for agentic rollouts |
| **Tunix (Google)** | TPU clusters | JAX-native; Gemma/Llama/Qwen; GRPO. Pointer only, not independently verified |
| **Haven (YC 2025)** | Serverless | Hundreds of LoRA adapters; claimed under 1s cold start. Pointer only, not independently verified |

### **11. Thinking-Mode Handling**

Qwen3, Qwen3.5, Qwen3.6+ and the Gemma 4 reasoning variants all expose a thinking mode, and mishandling it during SFT is one of the most common ways to destroy a model's reasoning ability.

```python
# Qwen3.5: enable thinking per turn
messages = [{"role": "user", "content": "/think Solve this math problem: ..."}]
# or add to system prompt: "enable_thinking: true"

# To fully disable thinking at inference
tokenizer.apply_chat_template(messages, enable_thinking=False)

# preserve_thinking for multi-turn agentic (Qwen3.6+):
# pass preserve_thinking=True in extra_body to retain CoT across turns
```

Qwen3.8 adds a third control: `reasoning_effort` with `xhigh` (default), `medium` and `low`. It also offers `enable_thinking=False` to turn reasoning off entirely for a turn, and `preserve_thinking=True` as the default that retains prior-turn reasoning across a conversation.

### **12. Catastrophic Forgetting**

| **Strategy** | **When to use** |
|---|---|
| PEFT (LoRA), the default | Base frozen; only adapters change |
| Replay buffers | Mix 10 to 30% general data in a full fine-tune |
| Multi-task training | High-data regimes |
| Conservative learning rate | Full fine-tune; reduces the forgetting rate |
| Layer-wise learning rate decay | Standard practice for full fine-tuning |
| Thinking-mode preservation | Mix at least 75% reasoning-style examples to preserve `<think>` capability |

For the thinking-mode row specifically: mixing at least three-quarters reasoning-style examples into a general SFT mixture is what keeps a Qwen3.5-style model's reasoning behaviour intact. Below that threshold the model usually still works, but silently stops emitting `<think>` blocks, and you will not notice until the benchmark drops.

$$\boxed{\text{reasoning-style share} \geq 75\%}$$

### **13. Memory Reference**

| **Model** | **Full FT (fp16 + Adam)** | **LoRA r=16** | **QLoRA NF4** |
|---|---|---|---|
| 7B | ~84 GB | ~16 GB | ~6 to 8 GB |
| 13B | ~156 GB | ~30 GB | ~10 to 12 GB |
| 70B | ~600 GB | ~120 GB | ~40 to 48 GB |

Full fine-tuning needs roughly 2 times the model size for weights plus gradients, plus 4 to 8 times for optimizer state, plus activations. QLoRA fit by rule of thumb: a 3 to 4B model trains in 8 to 12 GB, a 13B in about 12 GB, and a 33B in 24 GB.

### **14. Labs**

### **14.1 Lab 1: LoRA from Scratch in PyTorch (about 150 lines)**

- Implement `LoRALinear` with a frozen base and a low-rank `A`/`B` update.
- Verify that the output matches the base model exactly at initialization (`B` is zero-initialized).
- Attach it to a real `q_proj` and confirm `print_trainable_parameters()` shows the expected 0.1 to 0.5% trainable ratio.
- Train briefly and compare the loss curve against PEFT's implementation on identical data.

### **14.2 Lab 2: SFT on Your Own Dataset**

- Take your dataset from Stage 03 and train LoRA on it.
- Sweep `r` in {8, 16, 32} and learning rate in {1e-4, 2e-4, 5e-4}; record the eval metric for each.
- Try `completion_only_loss` versus `assistant_only_loss` and record the difference in loss scale.
- Push the adapter to the Hub with a model card.

### **14.3 Lab 3: QLoRA on a Consumer GPU**

- Train a 7B/8B model with NF4 and double quantization on a 16 to 24 GB card.
- Record peak memory, tokens/sec, and wall time.
- Merge the adapter in 16-bit and confirm the merged model's eval score matches the adapter's.

### **14.4 Lab 4: Distributed Run with Accelerate**

- Run `accelerate config`, `accelerate test`, then a two-process DDP run on one node.
- Add gradient accumulation and confirm the effective batch size is what you think it is.
- Save and resume from `accelerator.save_state()` and confirm the loss curve is continuous.

### **14.5 Lab 5: Checkpoint Hygiene**

- Save with `safetensors` and confirm no `.bin` files appear.
- Convert an old `.bin` checkpoint to safetensors and verify identical outputs.
- Inspect a sharded checkpoint through `model.safetensors.index.json`.

### **15. Pitfalls**

- **`max_seq_length` became `max_length`.** The old name raises an error in current TRL.
- **`DataCollatorForCompletionOnlyLM` is gone.** Use `completion_only_loss` or `assistant_only_loss`.
- **Assistant-only loss needs a chat template with generation markers.** Without them, the mask is wrong and the loss is meaningless.
- **`packing=True` changes the effective loss.** Short examples are concatenated; verify your metrics still mean what you think.
- **Merging a 4-bit model produces garbage.** Reload the base in 16-bit before merging.
- **`target_modules` names differ per architecture.** `q_proj`/`v_proj` is a Qwen/Llama convention; a Mistral checkpoint may need `q_proj`, `k_proj`, `v_proj`, `o_proj` explicitly.
- **PPO is gone from TRL.** A tutorial importing `PPOTrainer` will fail on import; that path is verl or OpenRLHF now.
- **Experimental trainers disappear.** One release deleted six and a later one removed TPO. Pin the version for anything longer than a weekend.
- **The `use_liger_kernel` flag is deprecated** and slated for removal in the next major release.
- **`bf16=True` on a pre-Ampere GPU** produces overflow rather than a clear error.

### **16. Acceptance Gate**

- A LoRA implementation written from scratch matches PEFT numerically.
- A QLoRA fine-tune completed on your own dataset and was pushed to the Hub.
- A merged model scores the same as the adapter it came from.
- A two-process Accelerate run trained and resumed cleanly.
- Checkpoints are safetensors; the repo contains no `.bin` or `.pt` files.
- The hyperparameter sweep table (r by learning rate) is attached to the model card.

### **17. Resources**

- TRL release notes: https://github.com/huggingface/trl/releases/tag/v1.17.1
- trl on PyPI: https://pypi.org/project/trl/
- PEFT release notes: https://github.com/huggingface/peft/releases/tag/v0.21.2
- accelerate on PyPI: https://pypi.org/project/accelerate/
- safetensors on PyPI: https://pypi.org/project/safetensors/
- bitsandbytes on PyPI: https://pypi.org/project/bitsandbytes/

**Next:** Stage 05: Alignment and Post-Training.