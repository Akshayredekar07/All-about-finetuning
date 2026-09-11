# LLM Fine-Tuning & Inference Engineering Roadmap (2026)

> **Purpose:** Complete post-HuggingFace-stack roadmap covering all model types (SLM, LLM, MoE, VLM, Diffusion, Audio, Embedding), all theoretical foundations, and the full 2026 production stack — fine-tuning through inference.
>
> **Target Role:** LLM Fine-Tuning Engineer / Inference Engineer / Applied AI Engineer
>
> **Prerequisites Assumed:** Full HuggingFace stack completed (Hub, Datasets, Transformers, Tokenizers, Evaluate, Accelerate, PEFT, TRL, Safetensors, Gradio, smolagents, bitsandbytes)
>
> **Last Updated:** May 2026 — includes all Jan–May 2026 model releases (Gemma 4, Qwen3.5, DeepSeek V4, Kimi K2.6, GLM-5/5.1, MiMo-V2.5, Llama 4, Nemotron Ultra), TurboQuant (ICLR 2026), KVTC (ICLR 2026), DART speculative decoding, GRPO-λ, GSPO, and all major 2026 production-stack updates.

---

## Part 0 — Model Type Landscape

Knowing what you are fine-tuning before you touch training code.

### 0.0 Open-Weight Leaderboard Snapshot (May 2026)

The HF Open LLM Leaderboard v3 uses harder anti-contamination benchmarks (GPQA Diamond, MMLU-Pro, MATH-500, IFEval, BBH, MUSR). Chinese labs now dominate the top. Western standouts: Gemma 4 (Google) and Nemotron Ultra (NVIDIA).

| Rank | Model | Lab | Total / Active | Context | Key Strength | License |
|------|-------|-----|---------------|---------|--------------|---------|
| ~1 | **Kimi K2.6** | Moonshot AI | 1T / 32B | 262K | Coding agents; PARL swarm; GPQA ~90% | Modified MIT |
| ~2 | **MiMo-V2.5-Pro** | Xiaomi | 1.02T / 42B | 1M | Long-horizon agentic; engineering | Open |
| ~3 | **DeepSeek V4-Pro** | DeepSeek | 1.6T / 49B | 1M | Math/coding; Hybrid Attention; MIT | MIT |
| ~4 | **GLM-5.1** | Z.AI (Zhipu) | 754B / 40B | 200K | SWE-Bench Pro leader; MIT license | MIT |
| ~5 | **Qwen3.5-397B-A17B** | Alibaba | 397B / 17B | 262K | GPQA 88.4%; multilingual; multimodal | Apache 2.0 |
| ~6 | **Gemma 4 31B** | Google | 31B dense | 256K | Best efficiency; AIME 89.2%; Apache 2.0 | Apache 2.0 |
| ~7 | **Llama 4 Scout** | Meta | 109B / 17B | 10M | Extreme context; multimodal | Meta Custom |
| ~8 | **Nemotron Ultra** | NVIDIA | 253B dense | 3M | Enterprise; reasoning; efficient | Open |
| ~9 | **MiniMax M2.7** | MiniMax | ~230B MoE | 1M | Productivity; tool use | Open |
| ~10 | **DeepSeek V4-Flash** | DeepSeek | 284B / 13B | 1M | Best price-perf; $0.14/1M in | MIT |

> **Trend:** MoE at massive scale is standard; MIT/Apache-2.0 is becoming the norm for top open models; reasoning mode (+15 pts) is universal at frontier.

---

### 0.1 SLMs — Small Language Models (< 4B parameters)

| Model | Params | Release | Notes |
|-------|--------|---------|-------|
| Phi-4-mini | 3.8B | 2025 | Microsoft; strong reasoning per param |
| Gemma 4 E2B | ~2.3B | Apr 2026 | Google; multimodal; audio; Apache 2.0 |
| Gemma 4 E4B | ~4.5B | Apr 2026 | Google; edge + multimodal; best on-device 2026 |
| Gemma-3-1B / 2B | 1–2B | 2025 | Google; CPU/mobile baseline |
| Qwen3.5-0.8B / 2B | 0.8–2B | Feb 2026 | Alibaba; native multimodal; thinking mode |
| Qwen3-1.7B | 1.7B | 2025 | GRPO-friendly; hybrid reasoning |
| SmolLM2 | 135M–1.7B | 2024 | HuggingFace; edge-optimized |
| Llama-3.2-1B/3B | 1–3B | 2024 | Meta; instruction-tuned |

**Characteristics:**
- Single consumer GPU (8–16 GB VRAM) with QLoRA or bf16 LoRA
- Gemma 4 E2B: Unsloth QLoRA runs in **< 5 minutes** on RTX 3060 12 GB for small datasets
- Qwen3.5-0.8B: bf16 LoRA needs only ~3 GB VRAM (Unsloth)
- Target for on-device, edge, and serverless deployment
- GRPO on SLMs is viable — Gemma 4 E2B RL works on 9 GB (Unsloth)
- Qwen3.5 family: **not recommended for QLoRA 4-bit** (higher-than-normal quantization errors); use bf16 LoRA instead

**Fine-Tuning Example — Gemma 4 E4B (Unsloth, QLoRA):**
```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "google/gemma-4-e4b",
    max_seq_length= 4096,
    dtype         = None,   # auto-detect BF16
    load_in_4bit  = True,   # QLoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r              = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 16,
    lora_dropout   = 0,        # Unsloth-optimised; keep at 0
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
)
```

**Fine-Tuning Example — Qwen3.5-2B (Unsloth, bf16 LoRA — no 4-bit):**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "Qwen/Qwen3.5-2B-Instruct",
    max_seq_length= 2048,
    dtype         = torch.bfloat16,  # bf16; NOT load_in_4bit for Qwen3.5
    load_in_4bit  = False,
)

# Qwen3.5 uses transformers v5 — older versions will not work
# thinking mode: /think and /no_think tokens in user prompts
# preserve_thinking=True in API for multi-turn agentic workflows (Qwen3.6+)
```

---

### 0.2 LLMs — Large Language Models (7B–72B Dense)

| Model | Params | Release | Notes |
|-------|--------|---------|-------|
| Gemma 4 31B | 31B dense | Apr 2026 | GPQA 84.3%; AIME 89.2%; Apache 2.0; best Western open |
| Qwen3.5-9B / 27B | 9–27B | Feb 2026 | Natively multimodal; thinking/non-thinking mode |
| Qwen3-7B / 14B / 32B / 72B | 7–72B | 2025 | Hybrid thinking mode; strong reasoning |
| Mistral-7B / Small 4 | 7B / ~22B | 2023–2026 | Small 4: vision + reasoning + Apache 2.0 |
| DeepSeek-R1 distills | 1.5B–70B | 2025 | RL-trained reasoning |
| Nemotron Ultra 253B | 253B | 2025 | NVIDIA; fits 8×H100; FP8 3M ctx; enterprise |
| Llama 4 Scout | 109B / 17B active | Apr 2026 | MoE hybrid; 10M context; multimodal |

**Characteristics:**
- QLoRA required for consumer hardware; Gemma 4 31B needs A100 80 GB for bf16 LoRA
- Gemma 4 26B MoE (A4B active): fits in ~12 GB with Unsloth QLoRA — extremely efficient
- `transformers v5` required for Qwen3.5; Qwen3.5 on vLLM requires ≥ 0.17.0 (not 0.16.0)
- Llama 4 Scout fine-tuning: focus on LoRA on attention layers; 10M context impractical for training — use 8K–32K during SFT

**Fine-Tuning Example — Gemma 4 31B (LoRA, A100):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-31b-it",
    torch_dtype  = torch.bfloat16,
    device_map   = "auto",
)

lora_config = LoraConfig(
    r              = 64,
    lora_alpha     = 128,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~160M / 31B = 0.5%

# Keras alternative (official Google tooling):
# import keras_hub
# gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma4_31b_instruct")
# gemma_lm.backbone.enable_lora(rank=16)
```

---

### 0.3 MoE — Mixture of Experts (Updated 2026)

| Model | Total Params | Active Params | Release | Notes |
|-------|-------------|---------------|---------|-------|
| **DeepSeek V4-Pro** | 1.6T | 49B | Apr 2026 | Hybrid Attention (CSA + HCA); 1M ctx; MIT |
| **DeepSeek V4-Flash** | 284B | 13B | Apr 2026 | $0.14/1M; MIT; day-0 vLLM support |
| **Kimi K2.6** | ~1T | 32B | Apr 2026 | PARL agent swarm; Mod MIT; open weights |
| **GLM-5.1** | 754B | 40B | Apr 2026 | Huawei Ascend 910B trained; zero NVIDIA; MIT |
| **Qwen3.5-397B-A17B** | 397B | 17B | Feb 2026 | Native multimodal; Gated Delta + sparse MoE |
| **Qwen3.5-122B-A10B** | 122B | 10B | Feb 2026 | Strong mid-tier; 262K context |
| **MiMo-V2.5-Pro** | ~1.02T | 42B | Apr 2026 | 1M ctx; agentic; long-horizon engineering |
| **MiMo-V2.5-Flash** | 309B | 15B | Mar 2026 | Efficient variant |
| **MiniMax M2.7** | ~230B | — | 2026 | 1M context; productivity |
| **Llama 4 Maverick** | ~400B | 17B | Apr 2026 | Multimodal; Meta Custom license |
| **Qwen3-235B-A22B** | 235B | 22B | 2025 | SOTA open MoE (pre-Qwen3.5) |
| **DeepSeek-V3** | 671B | 37B | 2024 | MLA + FP8 training |
| **Mixtral-8×7B** | 47B | 12.9B | 2023 | Baseline MoE reference |

**2026 MoE Architecture Advances:**
- **DeepSeek V4 Hybrid Attention:** Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA) — achieves 27% of single-token inference FLOPs vs V3.2 at 1M context. CSA + HCA together make long-context prefill far cheaper.
- **GLM-5.1 (Huawei Ascend):** First top-tier open model with zero NVIDIA dependency; trained on 100K Huawei Ascend 910B chips. Key for hardware-diversity awareness.
- **Qwen3.5 Gated Delta Network + Sparse MoE:** Hybrid SSM+Transformer architecture; early fusion vision-language; high-throughput inference with minimal latency overhead.
- **Kimi K2.6 PARL (Parallel-Agent RL):** Orchestrator model trained to decompose tasks + delegate to frozen sub-agent copies; BrowseComp jumps from 60.6% → 78.4% vs single-agent.

**Fine-Tuning MoE (2026 Practice):**
- **Qwen3.5 MoE (35B-A3B / 122B-A10B):** Use Unsloth's ~12× faster MoE training; >35% less VRAM; `bf16 LoRA` (not QLoRA 4-bit — BitsandBytes limitations on MoE); `Qwen3.5-35B-A3B bf16 LoRA` runs on 74 GB VRAM
- **DeepSeek V4-Flash fine-tuning:** Use Flash (284B/13B active) not Pro (1.6T/49B); Pro needs 4× H200 141 GB; Flash fits 2× A100 80 GB at QLoRA; `--trust-remote-code` required (custom attention kernels not yet in upstream Transformers)
- **Llama 4 Maverick fine-tuning:** 8× H100 minimum even with QLoRA due to large expert tensors; most teams tune Scout (109B) and serve Maverick stock
- Expert collapse risk: monitor per-expert utilization histogram during training; alert if any expert sees <1% utilization
- GSPO (sequence-level policy gradient) is preferred over GRPO for MoE RL training (stability)

```python
# Qwen3.5 MoE fine-tune with Unsloth (bf16, NOT 4-bit)
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "Qwen/Qwen3.5-35B-A3B-Instruct",
    max_seq_length= 4096,
    dtype         = torch.bfloat16,  # bf16 — do NOT use load_in_4bit for MoE
    load_in_4bit  = False,
)
# Unsloth's MoE-specific optimisation: ~12x faster, >35% less VRAM vs standard
```

---

### 0.4 VLMs — Vision-Language Models (Updated 2026)

| Model | Notes |
|-------|-------|
| **Qwen3.5-VL / Qwen3.5-Omni** | Native multimodal (early fusion); vision + audio; 262K ctx |
| Qwen2.5-VL-7B / 72B | 2025 SOTA VLM; dynamic resolution; video; still widely fine-tuned |
| InternVL3-8B / 78B | MPO training; strong multimodal reasoning |
| **Gemma 4 31B** | Text + image; Gemma 4 E2B/E4B also support audio |
| **Llama 4 Scout/Maverick** | Open-weight multimodal; Scout 10M ctx; Maverick video-capable |
| LLaVA-NeXT | Any-resolution; widely documented |
| PaliGemma 2 | Google; MLP adapter; compact and strong |
| DeepSeek-VL2 | Dynamic tiling; MoE backbone |
| UI-TARS-1.5 (ByteDance) | GUI agent; browser/computer/phone control |

**Architecture:** Vision encoder → projection layer (linear / MLP / Q-Former) → LLM decoder

**Early Fusion (2026 frontier):** Qwen3.5's approach — train multimodal tokens jointly from pretraining, not bolt-on adapter. Cross-generational parity with dedicated VLMs on reasoning, coding, agents. More complex to fine-tune but stronger base.

**Fine-Tuning Strategies:**
- **Stage 1 — Projector only:** Freeze vision encoder + LLM; train only projection layer (fastest, least data)
- **Stage 2 — LLM LoRA + Projector:** Freeze vision encoder; LoRA on LLM + train projector (standard)
- **Stage 3 — Full fine-tune:** Unfreeze everything; multi-GPU required; best quality
- **Image-LoRA (2026):** Apply LoRA specifically to image-relevant attention weights; preserves text reasoning better than standard LoRA on VLMs
- **GRPO on VLMs:** Verifiable rewards for VLM reasoning (wildfire detection, driving analysis); DGX Spark playbook from NVIDIA

**Gemma 4 VLM Fine-Tuning (Unsloth):**
```python
from unsloth import FastVisionModel

# Gemma 4 E4B multimodal — vision fine-tuning
model, tokenizer = FastVisionModel.from_pretrained(
    "google/gemma-4-e4b-it",
    load_in_4bit = True,
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers  = False,  # freeze vision encoder by default
    finetune_language_layers= True,
    finetune_attention_modules = True,
    finetune_mlp_modules    = True,
    r = 16, lora_alpha = 16,
)
```

**Qwen2.5-VL (standard approach still valid):**
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto"
)
lora_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
for param in model.model.visual.parameters():
    param.requires_grad = False  # freeze vision encoder
```

---

### 0.5 Diffusion Models

| Model | Notes |
|-------|-------|
| FLUX.1-dev / schnell | Black Forest Labs; flow matching; best open T2I 2026 |
| FLUX.1-Kontext | Edit-in-context; instruction-following image editing |
| Wan2.1 | Video generation; fine-tunable |
| Stable Diffusion 3.5 | DiT architecture |
| CogVideoX-5B | Video generation |

**Core Concepts — Diffusion:**
- **Forward process:** Add Gaussian noise over T steps: `q(xₜ|xₜ₋₁) = N(xₜ; √(1-β)xₜ₋₁, βI)`
- **Reverse process:** Learn to denoise: `pθ(xₜ₋₁|xₜ) = N(xₜ₋₁; μθ(xₜ,t), Σθ(xₜ,t))`
- **DDPM objective:** Predict noise ε; MSE loss `||ε - εθ(xₜ, t)||²`
- **Latent diffusion:** VAE encodes image to latent; diffuse in latent (4–8× cheaper)
- **Flow Matching (FLUX):** Direct ODE path noise → data; more stable, faster convergence

**Fine-Tuning Methods:**
- **LoRA:** 100–1000 steps; best for style transfer
- **DreamBooth:** Full fine-tune 3–25 subject images; rare token `sks dog`; regularization images prevent forgetting
- **Textual Inversion:** Learn new token embedding; minimal data; limited flexibility
- **ControlNet:** Trainable encoder copy; needs paired (image, condition) data

```python
from diffusers import FluxPipeline
from peft import LoraConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=[
        "transformer_blocks.*.attn.to_q",
        "transformer_blocks.*.attn.to_k",
        "transformer_blocks.*.attn.to_v",
        "transformer_blocks.*.attn.to_out.0",
        "single_transformer_blocks.*.attn.to_q",
        "single_transformer_blocks.*.attn.to_k",
        "single_transformer_blocks.*.attn.to_v",
    ],
    bias="none",
)
# 200–1000 steps for style; 500–2000 for DreamBooth
# Requires >= 24 GB VRAM for FLUX BF16
```

---

### 0.6 Audio Models

**ASR (Automatic Speech Recognition):**

| Model | Notes |
|-------|-------|
| Whisper-large-v3 | OpenAI; best multilingual ASR; seq2seq |
| Distil-Whisper | 6× faster; knowledge distillation from v3 |
| SLAM-ASR variants | Frozen Whisper encoder + LLM + trainable projector |

**TTS (Text-to-Speech):**

| Model | Release | Notes |
|-------|---------|-------|
| **Qwen3-TTS** | Jan 2026 | 0.6B–1.7B; voice cloning with 3-sec audio; 10 languages; 97ms streaming latency; fine-tune via base models |
| Parler-TTS | 2024 | Decoder-only; description-conditioned voice cloning |
| CSM-1B (Sesame) | 2025 | Conversational; LoRA via Unsloth |
| Kokoro | 2024 | Lightweight; fast inference |
| F5-TTS | 2024 | Flow-matching TTS; fast fine-tuning |

**Qwen3-TTS Fine-Tuning Notes:**
- `Qwen3-TTS-12Hz-1.7B-Base` and `Qwen3-TTS-12Hz-0.6B-Base` are the fine-tuning targets
- Uses proprietary `Qwen3-TTS-Tokenizer-12Hz` for high-fidelity voice compression
- Single-speaker fine-tuning supported (Jan 2026); multi-speaker in development
- CUDA required (Jan 2026); MLX port in progress for Mac

**ASR Fine-Tuning (Whisper LoRA):**
```python
from transformers import WhisperForConditionalGeneration
from peft import get_peft_model, LoraConfig
import evaluate

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3", load_in_8bit=True, device_map="auto"
)
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none",
)
model = get_peft_model(model, lora_config)

wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids   = pred.predictions
    label_ids  = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str   = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str  = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}
```

---

### 0.7 Embedding Models

| Model | Dim | Notes |
|-------|-----|-------|
| BAAI/bge-m3 | 1024 | Multi-lingual, multi-granularity; best retrieval |
| intfloat/e5-mistral-7b | 4096 | Instruction-following embeddings |
| Alibaba-NLP/gte-Qwen2-7B | 3584 | Top MTEB 2026 |
| nomic-embed-text-v2-moe | 768 | MoE embedding model |
| Cohere embed-v4 | 1536 | Multimodal embeddings |

**Training Objectives:**
- **InfoNCE / MultipleNegativesRankingLoss:** Standard contrastive; in-batch negatives
- **Matryoshka Representation Learning (MRL):** Single pass → usable at 64/128/256/512/768/1024 dims
- **Hard negative mining:** BM25 or embedding similarity (not random) — critical for retrieval quality
- **GISTEmbedLoss:** Teacher-guided hard negative selection
- **RLVR for embeddings (frontier 2026):** Downstream task verifier scores embedding quality

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss

model      = SentenceTransformer("BAAI/bge-base-en-v1.5")
inner_loss = MultipleNegativesRankingLoss(model)
loss       = MatryoshkaLoss(
    model=model, loss=inner_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],
)
# Training data: {"anchor": ..., "positive": ..., "negative": ...}
# Hard negatives are critical — mine with BM25 or embedding similarity
```

---

## Part 1 — Theoretical Foundations

### 1.1 Attention Mechanisms

**Standard Multi-Head Attention (MHA):**
```
Attention(Q,K,V) = softmax(QK^T / √d_k) × V
Multi-head: Concat(head_1, ..., head_h) × W_O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```
- Memory complexity: O(n²d) — quadratic in sequence length

**Grouped Query Attention (GQA):**
- Multiple Q heads share same K,V heads
- e.g., Llama-3: 32 Q heads, 8 KV heads → 4× KV cache reduction
- Standard in every 2026 open LLM

**Multi-Query Attention (MQA):**
- All Q heads share single K,V head; max savings; quality trade-off

**Multi-Head Latent Attention (MLA) — DeepSeek:**
- `c^KV = W^{DKV} h` (compressed latent); decompress at attention time
- KV cache stores only compressed latent → ~50× smaller cache vs MHA
- In vLLM and SGLang natively

**DeepSeek V4 Hybrid Attention (CSA + HCA) — 2026 new:**
- CSA (Compressed Sparse Attention): sparse attention pattern + compression; efficient on long context
- HCA (Heavily Compressed Attention): aggressive compression for tail layers
- 27% of single-token inference FLOPs vs V3.2 at 1M-token context
- Makes 1M context genuinely viable without linear approximations

**Flash Attention 1 / 2 / 3:**
- FA1: Tiling; never materializes full n×n matrix; HBM reads O(n)
- FA2: Better parallelism; ~2× faster than FA1 on A100
- FA3: H100-optimized; warp-specialized; FP8; TMA async memory; ~1.5–2× faster than FA2; default vLLM ≥0.8

**Sparse / Linear Attention (Frontier):**
- Mamba (SSM-based): O(n) complexity
- RWKV: O(1) inference; parallelizable training
- Qwen3.5 Gated Delta Networks: hybrid SSM + Transformer; high throughput at inference

---

### 1.2 KV Cache — The Central Data Structure

**Memory Formula:**
```
KV Cache bytes = 2 × n_layers × n_kv_heads × d_head × dtype_bytes × seq_len

Llama-3-8B (FP16, 32 layers, 8 KV heads, 128 d_head):
= 2 × 32 × 8 × 128 × 2 × seq_len = 131,072 bytes/token ≈ 0.125 MB/token
At 128K context: ~16 GB KV cache (beyond 16 GB model weights)
At 1M context:  ~125 GB — exceeds GPU HBM

DeepSeek V4-Pro at 1M context: MLA + CSA/HCA reduces this dramatically
```

**KV Cache Management Strategies (2026 Standard):**

| Strategy | Mechanism | Status |
|----------|-----------|--------|
| **Paged KV (PagedAttention)** | Fixed-size blocks; dynamic allocation; non-contiguous | Universal standard |
| **Prefix Caching / APC** | Hash prefixes; reuse KV for identical prefixes | Standard (vLLM, SGLang) |
| **RadixAttention** | Trie-based prefix tree; automatic reuse | SGLang default |
| **KV Quantization (scalar)** | FP8 or INT8 per-value | Production-ready |
| **TurboQuant** | Vector quant ~3.5 bits; near-optimal; training-free | ICLR 2026 |
| **KVTC** | PCA + adaptive quant + entropy coding; 20× | ICLR 2026; calibration needed |
| **KV Offloading** | Spill to CPU/NVMe | Emerging (LMCache, HiCache) |
| **PD Separation** | Prefill/decode separate KV pools | Large-scale standard |
| **MLA / CSA / HCA** | Architectural KV compression | DeepSeek V3/V4 |

---

### 1.3 TurboQuant — Near-Optimal KV Cache Vector Quantization (ICLR 2026)

> **Paper:** "Online Vector Quantization with Near-optimal Distortion Rate"  
> **Authors:** Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research / NYU / Google DeepMind)  
> **arXiv:** 2504.19874 | **Venue:** ICLR 2026

**What TurboQuant achieves:**
- ~3.5 bits per value (3-bit primary + 1-bit error correction)
- ≥ 6× KV cache memory reduction on long-context benchmarks
- Up to 8× speedup on H100 attention logit computation (Q×K^T)
- **Zero measurable accuracy loss** across LongBench, RULER, NIAH, ZeroSCROLLS
- Training-free, data-oblivious, online — no calibration

**Provably near-optimal:** Within `sqrt(3π/2) ≈ 2.7×` of Shannon information-theoretic lower bound.

**The Two-Stage Algorithm:**
```
Stage 1: PolarQuant (MSE-optimal compression)
  1. Multiply input vector x by random orthogonal matrix R
     → coordinates concentrate to Gaussian N(0, 1/d)
  2. Pre-computed Lloyd-Max codebook for that distribution
     → NO per-block normalization constants needed
  Result: MSE-optimal quantization; zero normalization overhead

Stage 2: QJL Error Correction (unbiased inner products)
  Problem: MSE-optimal ≠ unbiased inner products (attention computes Q×K^T)
  Fix — 1-bit residual:
  1. Compute residual: err = x_original - x_quantized
  2. Apply Johnson-Lindenstrauss Transform to residual
  3. Take sign bit: +1 or -1 (1 bit)
  4. Eliminates systematic bias in inner product estimates

Total: (b-1) bits PolarQuant + 1 bit QJL = b bits
At b=3: provably near-optimal; zero attention bias
```

**OSS Implementations (May 2026):**

| Repo | Notes |
|------|-------|
| `0xSero/turboquant` | Triton kernels; vLLM monkey-patch; RTX 3090 + 5090 |
| `hackimov/turboquant-kv` | PyTorch; paged KV; GQA/MQA support |
| `Alberto-Codes/turboquant-vllm` | Drop-in vLLM plugin; 3.76× validated on 8 models |

> ⚠️ `torch.compile` in vLLM ≥0.17 bypasses Python monkey-patch hooks. Use `--enforce-eager` until upstream PR #21618 merges.

---

### 1.4 KVTC — KV Cache Transform Coding (ICLR 2026, NVIDIA)

> **Paper:** "KV Cache Transform Coding for Compact Storage in LLM Inference"  
> **arXiv:** 2511.01815 | **Venue:** ICLR 2026

**JPEG-style compression pipeline for KV caches — requires calibration, achieves 20× compression.**

```
Step 1: Calibration (one-time)
  Forward calibration set → collect KV statistics → fit PCA basis per layer

Step 2: Compression pipeline
  a) Remove RoPE positional bias before PCA
  b) PCA-based decorrelation: project into principal components
  c) Adaptive quantization: dynamic programming assigns bits per component
  d) Entropy coding: DEFLATE over quantized coefficients

Step 3: Storage — on-GPU (decompress on demand) or NVMe (serialized)
```

**TurboQuant vs KVTC:**

| Method | Compression | Accuracy | Calibration | Online |
|--------|-------------|----------|-------------|--------|
| FP8 KV | ~2× | near-lossless | None | Yes |
| TurboQuant | ~6× | zero loss | None | Yes |
| KVTC (default) | ~20× | <1% drop | Brief | No |
| KVTC (aggressive) | 40–64× | noticeable | Brief | No |

**When to use each:**
```
TurboQuant: no calibration; streaming; online; 6× enough; KV-as-throughput
KVTC:       maximum compression; multi-turn reuse; PD separation;
            stored KV on NVMe; calibration acceptable
```

---

### 1.5 The Full KV Cache Optimization Stack (2026)

```
Layer 1: PagedAttention / Blocked KV         — Required baseline
Layer 2: Prefix Caching (APC / RadixAttn)   — 85–95% savings on shared prefixes
Layer 3: Scalar KV Quant (FP8 / INT8)       — ~2× savings; production-ready
Layer 4a: TurboQuant                         — ~6×; training-free; long-context
Layer 4b: KVTC                               — ~20×; calibration; multi-turn
Layer 5: KV Offloading (CPU/NVMe)            — 1M+ context; LMCache, HiCache
Layer 6: PD Separation                       — Large-scale; KVTC reduces transfer cost
Layer 7: Architectural (MLA / CSA / HCA)     — DeepSeek V3/V4 only; built-in
```

---

### 1.6 Prefill vs Decode — The Two Phases

```
PREFILL PHASE                        DECODE PHASE
─────────────────────                ─────────────────────────
Full prompt → parallel               One token per autoregressive step
Matrix × Matrix (GEMM)               Matrix × Vector (GEMV)
Compute-bound                        Memory-bandwidth-bound
Metric: TTFT                         Metric: TPOT
GPU utilization: HIGH                GPU utilization: < 1% compute

Optimizations:                       Optimizations:
  Chunked prefill (default vLLM V1)    Speculative decoding (EAGLE3, DART)
  Flash Attention                      Continuous batching
  Batch prefill together               KV quantization (TurboQuant / KVTC)
```

**Chunked Prefill:** Splits large prompts into 512-token chunks interleaved with decode steps. Prevents head-of-line blocking. Default in vLLM V1.

---

### 1.7 Parallelism for Training and Inference

**Data Parallelism (DP):** Replicate model on each GPU; gradient all-reduce.

**FSDP (ZeRO-3 equivalent):**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

mp_policy = MixedPrecision(
    param_dtype  = torch.bfloat16,
    reduce_dtype = torch.float32,
    buffer_dtype = torch.bfloat16,
)
model = FSDP(
    model,
    auto_wrap_policy   = transformer_auto_wrap_policy,
    mixed_precision    = mp_policy,
    sharding_strategy  = ShardingStrategy.FULL_SHARD,
    device_id          = torch.cuda.current_device(),
)
```

**DeepSpeed ZeRO:**
- ZeRO-1: shard optimizer states
- ZeRO-2: + gradients
- ZeRO-3: + parameters (≡ FSDP)
- ZeRO-Offload: optimizer states → CPU
- ZeRO-Infinity: → NVMe

**Tensor Parallelism (TP):** Split layers across GPUs; requires NVLink; within-node (≤8 GPUs).

**Pipeline Parallelism (PP):** Split layers sequentially; bubble fraction = (p-1)/m.

**Expert Parallelism (EP):** MoE experts on different GPUs; all-to-all communication.
- **Hybrid-EP (NVIDIA 2026):** CUDA block-level overlap of all-to-all + compute; 514% throughput improvement over baseline EP.

---

### 1.8 Quantization Theory

**Weight Formats:**

| Format | Bits | Memory (8B) | Notes |
|--------|------|-------------|-------|
| BF16 | 16 | 16 GB | Training standard; H100 native |
| FP8 E4M3 | 8 | 8 GB | H100+; needs calibration |
| INT8 AWQ | 8 | 8 GB | Activation-aware; fast inference |
| INT8 GPTQ | 8 | 8 GB | Layer-wise calibration |
| NF4 (QLoRA) | 4 | 4 GB | Training only; not serving |
| INT4 AWQ | 4 | 4 GB | Consumer GPU serving |
| GGUF Q4_K_M | ~4.5 | ~4.5 GB | llama.cpp / Ollama format |
| FP4 NVFP4 | 4 | 4 GB | Blackwell (B100/B200); 2×FP8 speed |

> ⚠️ **Key distinction:** Weight quantization and KV cache quantization are fully orthogonal. FP8 weights + TurboQuant KV is a valid and common 2026 production config.

> ⚠️ **Qwen3.5 QLoRA warning:** QLoRA 4-bit NOT recommended for Qwen3.5 (MoE or dense) due to higher-than-normal BitsandBytes quantization errors. Use `bf16 LoRA` instead.

---

### 1.9 Optimizers

**AdamW (Standard):**
```
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
θ_t = θ_{t-1} - α(m̂_t/(√v̂_t+ε) + λθ_{t-1})
β₁=0.9, β₂=0.999, ε=1e-8, λ=0.01
```

**8-bit Adam (bitsandbytes):** Optimizer states in 8-bit; 75% memory savings.

**Paged Adam (QLoRA):** Optimizer states in CUDA unified memory; no OOM crashes.

**Muon (2025–2026):** Orthogonalized momentum via Newton–Schulz iteration; several 2026 pretraining runs.

**LR Schedule:** Warmup (100–500 steps) → Cosine decay. LoRA: 1e-4 to 5e-5. Full FT: 1e-5 to 2e-5.

---

### 1.10 Catastrophic Forgetting

| Strategy | When to Use |
|----------|-------------|
| PEFT (LoRA) — default | Base frozen; only adapters change |
| Replay buffers | Mix 10–30% general data in full fine-tune |
| Multi-task training | High-data regimes |
| Conservative LR | Full fine-tune; reduces forgetting rate |
| Layer-wise LR decay | Standard practice for full FT |
| Thinking mode preservation (Qwen3.5) | Mix ≥75% reasoning-style examples to preserve `<think>` capability |

---

## Part 2 — The Post-Training Stack

### 2.1 Overview

```
Base Pretrained Model
        ↓
[Stage 1] SFT — Supervised Fine-Tuning
        ↓
[Stage 2] Preference Optimization (DPO / SimPO / KTO / ORPO)
        ↓
[Stage 3] Reasoning RL (GRPO / DAPO / Dr.GRPO / GSPO / GRPO-λ / RLVR)
        ↓
Quantize + Export + Serve
```

---

### 2.2 Stage 1 — Supervised Fine-Tuning (SFT)

**Data Preparation:**
- `tokenizer.apply_chat_template()` — standardized across Llama, Qwen, Gemma, DeepSeek
- Multi-turn: `system → user → assistant → tool_call → tool_response → assistant`
- `DataCollatorForCompletionOnlyLM`: loss only on assistant turns
- Packing: concatenate short examples; 2–4× throughput improvement
- Quality >> quantity: 1K curated often beats 100K noisy

**VRAM Budget for SFT (8B model reference):**

| Config | Approx VRAM | Notes |
|--------|-------------|-------|
| Full FT, BF16, bs=1 | ~80 GB | A100/H100 required |
| Full FT + FSDP, 8×A100 | ~10 GB/GPU | Standard multi-GPU |
| LoRA r=64, BF16 | ~28 GB | Single A100 |
| QLoRA 4-bit r=32 | ~10 GB | RTX 3090/4090 |
| Unsloth QLoRA 4-bit | ~7–8 GB | Optimized kernels |
| Gemma 4 E2B (Unsloth QLoRA) | ~6 GB | < 5 min on RTX 3060 12 GB |
| Qwen3.5-2B (bf16 LoRA) | ~5 GB | No 4-bit for Qwen3.5 |

**SFT Frameworks (2026):**

| Framework | Best For | Key Feature |
|-----------|----------|-------------|
| **Unsloth** | Single GPU; consumer hardware | 2× faster; 50–80% VRAM reduction; MoE support |
| **Axolotl** | Production pipelines | YAML config; FSDP + Flash Attention + QLoRA |
| **LLaMA-Factory** | Rapid iteration | Web UI + CLI; broadest model support |
| **Tunix (Google)** | TPU clusters | JAX-native; Gemma/Llama/Qwen; GRPO |
| **TRL SFTTrainer** | HuggingFace native | Most flexible; programmatic |
| **Haven (YC 2025)** | Serverless | Hundreds of LoRA adapters; <1s cold start |

**LoRA Target Modules by Architecture:**
```python
# Llama / Mistral / Gemma 4 / DeepSeek V4-Flash
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Qwen3 / Qwen3.5 (dense)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Qwen3.5 MoE — same modules; Unsloth handles expert routing automatically

# BERT / RoBERTa (encoder-only)
target_modules = ["query", "value"]

# Whisper (ASR)
target_modules = ["q_proj", "v_proj"]  # encoder layers
```

**Thinking Mode Handling (Qwen3, Qwen3.5, Qwen3.6+, Gemma 4 reasoning variants):**
```python
# Qwen3.5 — enable thinking per turn
messages = [{"role": "user", "content": "/think Solve this math problem: ..."}]
# or add to system prompt: "enable_thinking: true"

# To fully disable thinking at inference
tokenizer.apply_chat_template(messages, enable_thinking=False)

# preserve_thinking for multi-turn agentic (Qwen3.6+):
# pass preserve_thinking=True in extra_body to retain CoT across turns
```

---

### 2.3 Stage 2 — Preference Optimization

| Method | Year | Reference Model | Reward Model | Key Formula |
|--------|------|----------------|--------------|-------------|
| PPO (RLHF) | 2022 | Yes | Yes | Complex actor-critic |
| DPO | 2023 | Yes (frozen) | No | Direct from pairs |
| IPO | 2023 | Yes | No | Better small-data |
| KTO | 2024 | Yes | No | Binary feedback |
| SimPO | 2024 | **No** | No | Avg log-prob reward |
| ORPO | 2024 | **No** | No | Single-stage SFT + align |

**DPO Deep Dive:**
```
L_DPO = -E [ log σ ( β × [
    (log π_θ(y_w|x) - log π_ref(y_w|x))
  - (log π_θ(y_l|x) - log π_ref(y_l|x))
]) ]
β = 0.1–0.2 (KL penalty)
```

**Best Practice:** Mix 80% SFT data + 20% preference data in DPO to prevent forgetting.

---

### 2.4 Stage 3 — Reasoning RL (GRPO family, 2026)

**GRPO — Group Relative Policy Optimization:**
```
For each prompt q:
  1. Sample G responses {o_1,...,o_G} (G = 8–64)
  2. Score: r_i = verifier(o_i, ground_truth)
  3. Advantage: A_i = (r_i - mean(r)) / std(r)
  4. Policy gradient update + KL regularization
No critic → 50% memory savings vs PPO
```

**DAPO (ByteDance/Tsinghua):** Clip-Higher + Dynamic Sampling + Token-level loss + Overlong Reward Shaping → 50 on AIME 2024 with Qwen2.5-32B.

**Dr.GRPO:** Remove std normalization + remove KL → better with sparse rewards.

**GSPO (Sequence-level):** Sequence-level importance ratios; improves MoE stability (Mixtral, DeepSeek).

**GRPO-λ:** Eligibility traces + TD credit assignment; +3–4.5 pts on math benchmarks vs GRPO.

**GRPO Variants Summary:**

| Method | Key Innovation | Best Use Case |
|--------|---------------|---------------|
| GRPO | Group-relative advantage | General RLVR |
| DAPO | Clip-higher + dynamic sampling | Long CoT reasoning |
| Dr.GRPO | Remove std-norm + KL | Sparse reward |
| **GSPO** | Sequence-level ratios | **MoE fine-tuning (DeepSeek V4, Qwen3.5 MoE)** |
| GRPO-λ | Eligibility traces | Math/code reasoning |

**Verifier Design:**
```python
# Math verifier
def math_verifier(response: str, ground_truth: str) -> float:
    extracted = extract_answer(response)  # parse \boxed{} or final line
    return 1.0 if extracted == ground_truth else 0.0

# Code verifier
def code_verifier(response: str, test_cases: list) -> float:
    code   = extract_code_block(response)
    passed = run_tests(code, test_cases)
    return len(passed) / len(test_cases)  # partial credit

# Format verifier
def format_verifier(response: str) -> float:
    has_think  = "" in response and "" in response
    has_answer = "" in response
    return 0.2 * has_think + 0.8 * has_answer

# GRPO on Qwen3.5 / Gemma 4 with Unsloth:
# Unsloth supports GRPO, GSPO, DAPO natively
# Gemma 4 E2B RL works on 9 GB; Qwen3.5-2B RL on ~8 GB
```

---

### 2.5 Agentic Fine-Tuning (2026 Frontier)

**ART — Agent Reinforcement Trainer:** GRPO backend on vLLM + Unsloth; multi-turn trajectories; RULER for LLM-as-judge reward.

**RULER:** LLM judge ranks G trajectories relatively. Absolute scores not needed for GRPO.

**ASTRA:** Synthesize tool environments → verifiable RL problems; SFT warm-start → online multi-turn RL.

**CM2:** Binary checklist decomposition; each criterion = classification with evidence grounding; LLM-simulated tool environment.

**Kimi K2.x PARL (deployed model insight):** Train orchestrator via RL to decompose + delegate to frozen sub-agents. Key lesson for agentic fine-tuning: decomposition ability > single-agent reasoning depth for multi-step tasks.

---

## Part 3 — Inference Engineering

### 3.1 Serving Engines (2026)

| Engine | Best Workload | Core Innovation | H100 Throughput |
|--------|---------------|-----------------|-----------------|
| **vLLM** | General; bursty | PagedAttention; broadest ecosystem | ~12,500 tok/s |
| **SGLang** | Shared-prefix; multi-turn; agentic | RadixAttention; EAGLE3/DART | ~16,200 tok/s |
| **TensorRT-LLM** | Max throughput; NVIDIA-only | FP8/FP4; engine compile; EP | ~2,780 out-tok/s at 100 req |
| **LMDeploy** | Consumer GPUs; INT4 | KV quantization; lightweight | ~16,200 tok/s (A100 INT4) |
| **llama.cpp** | Edge; CPU; no-GPU | GGUF; k-quants; portable | Device-dependent |
| **Ollama** | Developer local UX | Zero-config; GGUF wrapping; GUI | Device-dependent |
| **TGI** | ~~HF ecosystem~~ | **Maintenance mode Dec 2025** | — |

> ⚠️ **TGI is in maintenance mode as of December 2025.** New deployments: use vLLM or SGLang.

> ℹ️ **llama.cpp / Ollama scope:** These are local inference runtimes (CPU+GPU, consumer hardware, GGUF format). They are **not** fine-tuning tools and are not in the training pipeline. Their role in this roadmap is limited to the Edge/On-device serving config. For production serving at scale, use vLLM or SGLang.

**DeepSeek V4 Serving Notes:**
- Requires vLLM ≥ 0.7.0 or SGLang ≥ 0.4.4
- `--trust-remote-code` required (custom hybrid attention kernels)
- V4-Flash (284B/13B active): fits 2× A100 80 GB at Q4; 1× 96 GB GPU at Q4_K_M
- V4-Pro (1.6T): needs 4× H200 141 GB or 8× H100; BF16 = ~3.2 TB

**vLLM Core Concepts:** PagedAttention, continuous batching, APC, chunked prefill (default V1), FP8, `torch.compile`.

**SGLang Core Concepts:** RadixAttention (trie-based prefix), EAGLE3/DART integrated, Flashinfer backend.

---

### 3.2 Continuous Batching

Iteration-level scheduling: completed requests exit immediately, new ones enter. Near-zero GPU idle time. 3–10× throughput vs static batching.

---

### 3.3 Speculative Decoding (2026)

**Draft model proposes K tokens; target verifies all K in one pass. Mathematically lossless.**

| Method | Draft Source | Memory | Notes |
|--------|-------------|--------|-------|
| **EAGLE3** | Draft head on target hidden states | ~400M params | De-facto standard; 2–6× speedup |
| **DART** | Diffusion-inspired non-autoregressive | ~Qwen3-1.7B size | **6.8× lower draft latency vs EAGLE3 on H100** |
| **Medusa / MTP** | Tree-structured parallel heads | ~hundreds M | Multi-token prediction; DeepSeek style |
| **N-gram** | Statistical from KV cache | Zero | Repetitive outputs |

**DART (Jan 2026):** Non-autoregressive drafter generates all k draft tokens in parallel. 6.8× lower drafting latency vs EAGLE3; 53.3× vs standard speculative sampling (Qwen3-14B on H20). Best for k=8+ draft lengths.

> ⚠️ **Fine-tuned target models:** Retrain EAGLE3/DART draft head after fine-tuning — generic heads have lower acceptance rates.

**When speculative decoding hurts:** batch > 32; outputs < 100 tokens; acceptance rate < 50%.

---

### 3.4 Prefill/Decode Disaggregation (PD Separation)

```
Prefill instances (H100/A100):    Decode instances (A10G/L4):
  Compute-bound prefill             Memory-bandwidth-bound decode
  Build KV cache                    Receive compressed KV (KVTC)
  → Transfer KV (compressed)        Run continuous batching
```

KVTC-compressed KV dramatically reduces cross-node transfer cost. All major engines support PD separation natively in 2026 (vLLM, SGLang, TRT-LLM, Ray Serve LLM).

---

### 3.5 Quantization in Production (2026 Configs)

```
Interactive chat (latency-sensitive):
  Weights:    FP8 (H100) or INT4 AWQ (A100/consumer)
  KV:         TurboQuant 4-bit (6× savings, zero loss)
  Speculative: EAGLE3 or DART if batch < 8
  Engine:     SGLang (RadixAttention for multi-turn)

High-throughput batch inference:
  Weights:    FP8 or INT4 AWQ
  KV:         TurboQuant or KVTC 20× if max compression needed
  Speculative: off (high concurrency)
  Engine:     vLLM or TRT-LLM; chunked prefill

Edge / On-device:
  Weights:    GGUF Q4_K_M (llama.cpp / Ollama)
  KV:         standard
  Engine:     llama.cpp or Ollama

Multi-turn / Agentic (long history):
  Weights:    FP8 or INT4 AWQ
  KV:         KVTC (multi-turn KV reuse killer use case)
  Engine:     SGLang (RadixAttention) + KVTC + LMCache

DeepSeek V4-Flash (API or local):
  V4-Flash API: $0.14/1M input — cheapest frontier-tier
  Local:         Q4_K_M on 96 GB GPU (single node)
  --trust-remote-code required; vLLM ≥ 0.7.0
```

---

## Part 4 — Distributed Training (Production)

### 4.1 DeepSpeed ZeRO-3 + Offload (Axolotl YAML)

```yaml
deepspeed:
  zero_stage: 3
  offload_optimizer_device: cpu
  offload_param_device: cpu
  overlap_comm: true
  contiguous_gradients: true
  reduce_bucket_size: 5e8
  stage3_prefetch_bucket_size: 5e8
  stage3_param_persistence_threshold: 1e6
  bf16:
    enabled: true
  gradient_clipping: 1.0
```

### 4.2 Multi-GPU Axolotl Config (Production — Llama 4 Scout)

```yaml
base_model: meta-llama/Llama-4-Scout-17B-16E-Instruct
model_type: LlamaForCausalLM

datasets:
  - path: your_dataset
    type: sharegpt
    conversation: chatml

adapter: lora
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

sequence_len: 8192        # use 8K–32K during SFT; not full 10M ctx
micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 3
optimizer: adamw_8bit
lr_scheduler: cosine
learning_rate: 2e-4
warmup_ratio: 0.03

bf16: auto
tf32: true
flash_attention: true
gradient_checkpointing: true

fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer

output_dir: ./outputs/llama4-scout-run-001
```

---

## Part 5 — Specialized Fine-Tuning

### 5.1 VLM Fine-Tuning (Qwen2.5-VL)

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto"
)
lora_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
for param in model.model.visual.parameters():
    param.requires_grad = False
```

### 5.2 Diffusion Fine-Tuning (FLUX LoRA)

```python
from diffusers import FluxPipeline
from peft import LoraConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=[
        "transformer_blocks.*.attn.to_q",
        "transformer_blocks.*.attn.to_k",
        "transformer_blocks.*.attn.to_v",
        "single_transformer_blocks.*.attn.to_q",
        "single_transformer_blocks.*.attn.to_k",
        "single_transformer_blocks.*.attn.to_v",
    ],
    bias="none",
)
```

### 5.3 Audio Fine-Tuning (Whisper LoRA)

```python
from transformers import WhisperForConditionalGeneration
from peft import get_peft_model, LoraConfig

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3", load_in_8bit=True, device_map="auto"
)
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none",
)
model = get_peft_model(model, lora_config)
```

### 5.4 Embedding Fine-Tuning (Matryoshka)

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss

model      = SentenceTransformer("BAAI/bge-base-en-v1.5")
inner_loss = MultipleNegativesRankingLoss(model)
loss       = MatryoshkaLoss(
    model=model, loss=inner_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],
)
```

---

## Part 6 — MLOps / LLMOps for Production

### 6.1 Data Pipeline

```
1. Collection → provenance (source, license, date)
2. Cleaning  → PII removal (Presidio), toxicity filter, near-dedup (MinHash LSH)
3. Quality   → perplexity, reward model score, diversity, spot-check
4. Synthetic → 70/30 synthetic/real mix (2026 standard)
              → self-instruct, back-translation, SPIN
              → generate 5–10×, filter to best 20–30%
```

### 6.2 Experiment Tracking

```python
wandb.log({
    "train/loss"          : loss,
    "train/grad_norm"     : grad_norm,
    "train/lr"            : scheduler.get_last_lr()[0],
    "eval/loss"           : eval_loss,
    "eval/metric"         : eval_metric,
    "system/gpu_util"     : gpu_util,
    "system/gpu_memory_gb": gpu_mem / 1e9,
})
wandb.log_artifact(model_checkpoint_path, type="model")
```

### 6.3 Evaluation (2026 Benchmarks)

| Task | Metric | Threshold (competitive 2026) |
|------|--------|-------------------------------|
| Reasoning | AIME 2025/2026, AMC | Track raw score |
| Hard science | GPQA Diamond | > 75% competitive; top open > 84% |
| General knowledge | MMLU-Pro | > 65% |
| Math | MATH-500, GSM8K | MATH-500 > 75% |
| Code | LiveCodeBench v6, SWE-Bench | LiveCodeBench > 60% |
| Agentic coding | SWE-Bench Verified | > 55% competitive |
| Long context | RULER, NIAH | Task-dependent |
| MT | MT-Bench | > 8.0/10 |
| Embedding | MTEB Retrieval | > 55% nDCG@10 |

### 6.4 Deployment (vLLM with LoRA hot-swap)

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model                  = "meta-llama/Llama-3-8B-Instruct",
    enable_lora            = True,
    max_lora_rank          = 64,
    tensor_parallel_size   = 2,
    gpu_memory_utilization = 0.90,
    dtype                  = "bfloat16",
    enable_prefix_caching  = True,
    max_model_len          = 32768,
)
outputs = llm.generate(
    prompts,
    SamplingParams(temperature=0.7, max_tokens=512),
    lora_request=LoRARequest("adapter_v2", 1, "/path/to/lora"),
)
```

### 6.5 Cost Optimization

| Lever | Savings | Notes |
|-------|---------|-------|
| QLoRA vs full FT | 90% training cost | Default choice |
| bf16 LoRA (Qwen3.5 MoE) | vs QLoRA instability | Unsloth 12× MoE speedup |
| TurboQuant KV | 6× KV → more concurrency | Training-free |
| KVTC KV | 20× → dense multi-turn | Calibration required |
| INT4 AWQ weights | 4× → more concurrency | A100/consumer |
| Spot instances | 60–80% | Add checkpointing |
| Prefix caching | 85–95% on prefill | RAG/system-prompt-heavy |
| DeepSeek V4-Flash API | $0.14/1M in | Cheapest frontier-tier 2026 |

---

## Part 7 — Evolution Strategies (Gradient-Free Fine-Tuning)

```
1. Sample parameter perturbations: {ε_1, ..., ε_N}
2. Evaluate via inference + verifiable reward: {r_1, ..., r_N}
3. Estimate gradient: ∇ ≈ (1/Nσ²) Σ(ε_i × r_i)
4. Update: θ ← θ + α∇
No backpropagation. Works natively in quantized environments.
```

**QES (Quantized Evolution Strategies):** Fine-tune directly in NF4/INT4; sidesteps gradient instability through quantization. Competitive with GRPO on math reasoning. Emerging.

---

## Part 8 — Key Papers (Must Read)

| Paper | Year | Why Essential |
|-------|------|---------------|
| **LoRA** (Hu et al.) | 2022 | Foundation of all PEFT |
| **QLoRA** (Dettmers et al.) | 2023 | 4-bit fine-tuning; NF4; double quant |
| **FlashAttention** (Dao et al.) | 2022 | Efficient attention foundation |
| **FlashAttention-2** | 2023 | FA2 in every modern system |
| **PagedAttention / vLLM** (Kwon et al.) | 2023 | KV cache memory management |
| **DPO** (Rafailov et al.) | 2023 | Direct preference optimization |
| **DeepSeek-R1** | 2025 | GRPO; RLVR; emergent reasoning |
| **DAPO** (ByteDance) | 2025 | GRPO fixes for long CoT |
| **Dr.GRPO** | 2025 | Debiased advantages |
| **GSPO** | 2025 | Sequence-level; MoE stability |
| **GRPO-λ** | 2025 | Eligibility traces; +3–4.5 pts math |
| **SGLang / RadixAttention** | 2024 | Trie-based prefix caching |
| **DistServe / PD Disaggregation** | 2023 | Prefill/decode separation |
| **SPIN** (UCLA) | 2024 | Self-play FT without extra data |
| **Matryoshka Embeddings** | 2022 | Flexible embedding dimensions |
| **DeepSeek-V3 Technical Report** | 2024 | MoE; MLA; FP8 training at scale |
| **QJL** (Zandieh et al.) | AAAI 2025 | 1-bit JL transform; TurboQuant foundation |
| **PolarQuant** | AISTATS 2026 | Rotation-based KV quant; no normalization overhead |
| **TurboQuant** (Google Research) | ICLR 2026 | Near-optimal KV; 6×; zero loss |
| **KVTC** (NVIDIA) | ICLR 2026 | Transform coding KV; 20×; JPEG-style |
| **DART** (Alibaba) | Jan 2026 | Diffusion speculative decoding; 6.8× lower draft latency |
| **ASTRA** | 2026 | Multi-turn agentic RL pipeline |
| **Qwen3.5 Technical Report** | Feb 2026 | Native multimodal; Gated Delta + sparse MoE; 201 languages |
| **DeepSeek V4 Technical Report** | Apr 2026 | CSA + HCA; 1M context; 1.6T MoE |
| **Kimi K2 / PARL** | 2026 | Parallel-agent RL; orchestrator + sub-agent swarms |
| **Gemma 4 Technical Report** | Apr 2026 | Dense 31B; AIME 89.2%; GPQA 84.3%; Apache 2.0 |

---

## Part 9 — GitHub Repository Structure

```
llm-finetuning-engineering/
│
├── 01_theory/
│   ├── attention_mechanisms.md           # MHA, GQA, MQA, MLA, CSA/HCA, FlashAttn
│   ├── kv_cache_deep_dive.md
│   ├── turboquant_deep_dive.md           # PolarQuant + QJL
│   ├── kvtc_deep_dive.md
│   ├── prefill_vs_decode.md
│   ├── parallelism_taxonomy.md           # DP, FSDP, TP, PP, EP, Hybrid-EP
│   ├── quantization_survey.md
│   └── optimizer_theory.md
│
├── 02_peft_deep_dive/
│   ├── lora_math.py
│   ├── qlora_4bit.py
│   ├── dora_implementation.py
│   ├── loftq_initialization.py
│   ├── target_modules_guide.md           # includes Qwen3.5 MoE notes
│   └── multi_adapter_serving.py
│
├── 03_sft/
│   ├── axolotl_configs/
│   │   ├── qlora_llama3_8b.yaml
│   │   ├── qlora_gemma4_e4b.yaml         # NEW: Gemma 4 E4B config
│   │   ├── bf16_lora_qwen35_moe.yaml     # NEW: Qwen3.5 MoE (no 4-bit)
│   │   ├── fsdp_llama4_scout.yaml        # NEW: Llama 4 Scout
│   │   └── deepspeed_zero3.yaml
│   ├── data_formatting/
│   │   ├── chat_template_examples.py     # includes Qwen3.5 /think tokens
│   │   ├── thinking_mode_preservation.py # NEW: reasoning data mixing
│   │   ├── packing_implementation.py
│   │   └── completion_only_masking.py
│   ├── trl_sft_trainer.py
│   └── monitoring_wandb.py
│
├── 04_preference_optimization/
│   ├── dpo_training.py
│   ├── simpo_training.py
│   ├── kto_training.py
│   └── orpo_training.py
│
├── 05_reasoning_rl/
│   ├── grpo_from_scratch.py
│   ├── dapo_implementation.py
│   ├── gspo_implementation.py            # MoE stability; use for DeepSeek/Qwen3.5
│   ├── grpo_lambda.py
│   ├── dr_grpo.py
│   ├── verifier_design/
│   │   ├── math_verifier.py
│   │   ├── code_verifier.py
│   │   └── format_verifier.py
│   └── reward_hacking_mitigation.py
│
├── 06_agentic_rl/
│   ├── art_framework_demo.py
│   ├── multi_turn_grpo.py
│   ├── parl_orchestrator_pattern.py      # NEW: Kimi K2-style agent decomposition
│   └── astra_pipeline.py
│
├── 07_serving/
│   ├── vllm_production_config.yaml
│   ├── deepseek_v4_vllm_deploy.sh        # NEW: --trust-remote-code; version reqs
│   ├── sglang_radix_benchmark.py
│   ├── speculative_decoding_eagle3.py
│   ├── dart_speculative_decoding.py
│   ├── pd_disaggregation_setup.md
│   ├── kv_quantization/
│   │   ├── fp8_kv_vllm.py
│   │   ├── turboquant_integration.py
│   │   └── kvtc_pipeline.py
│   └── serve_benchmark_suite.ipynb
│
├── 08_specialized/
│   ├── vlm_finetuning/
│   │   ├── qwen2_5_vl_lora.py
│   │   └── gemma4_vision_unsloth.py      # NEW: FastVisionModel
│   ├── diffusion/
│   ├── audio/
│   │   ├── whisper_lora.py
│   │   └── qwen3_tts_finetune.py         # NEW: Qwen3-TTS base model FT
│   └── embeddings/
│
├── 09_distributed_training/
│   ├── fsdp_llama3.py
│   ├── deepspeed_zero3_config.json
│   ├── multi_node_launch.sh
│   └── moe_expert_parallelism/
│       └── hybrid_ep_megatron.md         # NVIDIA Megatron Hybrid-EP
│
├── 10_advanced/
│   ├── evolution_strategies/
│   ├── kv_compression/
│   │   ├── turboquant_from_scratch.py
│   │   ├── kvtc_from_scratch.py
│   │   └── kv_compression_benchmark.py  # FP8 vs TurboQuant vs KVTC
│   └── flash_attention_deep_dive/
│
├── 11_new_models_2026/                   # NEW SECTION
│   ├── gemma4_complete_guide.md          # E2B/E4B/26B-MoE/31B; Apache 2.0
│   ├── qwen35_family.md                  # Native multimodal; Gated Delta; 201 languages
│   ├── deepseek_v4_deployment.md         # Pro vs Flash; CSA/HCA; trust-remote-code
│   ├── kimi_k2x_agentic.md              # PARL; agent swarm; Modified MIT
│   ├── glm5_enterprise.md               # MIT; Ascend; zero NVIDIA; SWE-Bench
│   ├── mimo_v25_longcontext.md          # 1M ctx; 1T MoE; agentic
│   └── llama4_scout_maverick.md         # 10M ctx; multimodal; Meta custom license
│
├── capstone/
│   ├── README.md
│   ├── train_recipe.yaml
│   ├── grpo_verifier.py
│   ├── serve_config.py
│   └── benchmark_results.md
│
├── .env.example
├── requirements.txt
├── docker/
└── README.md
```

---

## Part 10 — Production Readiness Checklist

```
Training:
  ✅ Fine-tune 7B with QLoRA on single GPU (RTX 3090 / 4090 / A6000)
  ✅ Fine-tune Gemma 4 E4B with QLoRA on RTX 3060 12 GB
  ✅ Fine-tune Qwen3.5-2B with bf16 LoRA (NOT QLoRA — Qwen3.5 quirk)
  ✅ Fine-tune Qwen3.5 MoE with Unsloth bf16 LoRA (~74 GB for 35B-A3B)
  ✅ Fine-tune 7B with full LoRA on A100
  ✅ Fine-tune 70B with FSDP on multi-GPU (8× A100)
  ✅ Fine-tune Llama 4 Scout (Scout; not Maverick) on 8× H100
  ✅ Run GRPO with custom verifier (math and code)
  ✅ Run GSPO for MoE fine-tuning stability (DeepSeek V4-Flash or Qwen3.5 MoE)
  ✅ Run DPO/SimPO for preference optimization
  ✅ Fine-tune at least one VLM (Qwen2.5-VL, Gemma 4, or InternVL3)
  ✅ Fine-tune Whisper with domain adaptation
  ✅ Fine-tune Qwen3-TTS base model for voice cloning
  ✅ Train embedding model with Matryoshka loss
  ✅ Handle Qwen3.5 thinking mode preservation in SFT data (≥75% reasoning examples)

Serving:
  ✅ Deploy vLLM with APC and chunked prefill
  ✅ Configure SGLang for multi-turn workload with RadixAttention
  ✅ Deploy DeepSeek V4-Flash on vLLM (--trust-remote-code; vLLM ≥0.7.0)
  ✅ Measure and report TTFT, TPOT, throughput at target concurrency
  ✅ Implement EAGLE3 speculative decoding for low-latency
  ✅ Implement DART speculative decoding; compare vs EAGLE3
  ✅ Deploy FP8 weight-quantized model
  ✅ Implement TurboQuant KV compression; measure memory delta
  ✅ Implement KVTC for multi-turn workflow; measure TTFT vs prefix recompute
  ✅ Set up PD disaggregation for a long-output workload

Engineering:
  ✅ Automated eval pipeline (GPQA Diamond, LiveCodeBench, AIME; not just loss)
  ✅ W&B or MLflow tracking on every training run
  ✅ PII removal from training data before any run
  ✅ A/B test fine-tuned vs base model with statistical significance
  ✅ Checkpointing + recovery from failures
  ✅ KV cache memory budget analysis for target deployment
  ✅ Cost analysis: training GPU hours + serving cost per million tokens
  ✅ License audit for chosen model (MIT vs Apache 2.0 vs Meta Custom vs Modified MIT)

Deep Understanding:
  ✅ Can derive KV cache memory formula from first principles
  ✅ Can explain prefill vs decode bottleneck and what optimizes each
  ✅ Can explain PagedAttention block allocation with diagram
  ✅ Can write GRPO algorithm from scratch without reference
  ✅ Can explain why FSDP is equivalent to ZeRO-3
  ✅ Can explain why MLA reduces KV cache by ~50× in DeepSeek
  ✅ Can explain TurboQuant two-stage algorithm (PolarQuant + QJL)
  ✅ Can explain why MSE-optimal quantizers ≠ unbiased inner products
  ✅ Can explain KVTC pipeline (PCA + adaptive quant + DEFLATE)
  ✅ Can compare TurboQuant vs KVTC and select for a given workload
  ✅ Can explain why DART achieves lower draft latency than EAGLE3
  ✅ Can trace GRPO → DAPO → Dr.GRPO → GSPO → GRPO-λ lineage
  ✅ Can explain why Qwen3.5 MoE should NOT use QLoRA 4-bit
  ✅ Can explain DeepSeek V4 CSA + HCA and why it enables cheap 1M-context inference
  ✅ Can explain Kimi K2 PARL and why orchestrator-based RL outperforms single-agent RL
  ✅ Can choose between GLM-5.1 (MIT, Ascend) / Kimi K2.6 / DeepSeek V4 for a given constraint
  ✅ Can explain Gemma 4's advantage over Llama 4 Maverick on efficiency per active param
```

---

## Part 11 — New Model Decision Framework (Jan–May 2026)

```
Goal: Pick the right base model for fine-tuning.

"I need the best open-weight coding/agent model, open license"
  → DeepSeek V4-Flash (MIT; 13B active; cheapest API; fine-tune Flash not Pro)
  → Kimi K2.6 (Modified MIT; PARL swarm; best for multi-agent workflows)
  → GLM-5.1 (MIT; strong SWE-Bench; Huawei Ascend trained — no NVIDIA dependency)

"I need the best efficiency for consumer GPU fine-tuning"
  → Gemma 4 E4B (4.5B eff; Apache 2.0; QLoRA on RTX 3060 12 GB; < 5 min)
  → Gemma 4 26B-A4B (MoE; 4B active; exceptional quality/VRAM ratio)
  → Qwen3.5-2B (bf16 LoRA ~5 GB; thinking mode; 201 languages)

"I need native multimodal (text + image + audio)"
  → Qwen3.5 family (early fusion; all variants multimodal)
  → Gemma 4 E2B/E4B (audio support on edge models)
  → Llama 4 Scout/Maverick (open-weight multimodal; note: EU vision restriction)

"I need extreme long context (>256K tokens)"
  → Llama 4 Scout (10M context; open-weight)
  → DeepSeek V4-Pro / Flash (1M context; MIT; CSA/HCA makes it affordable)
  → MiMo-V2.5-Pro (1M context; open-source; agentic)

"I need enterprise deployment with MIT/Apache license"
  → GLM-5.1 (MIT; self-hostable; fine-tunable commercially)
  → DeepSeek V4 (MIT; best math/code at scale)
  → Gemma 4 (Apache 2.0; no MAU clause; Google tooling)
  → Avoid: Llama 4 (Meta Custom; 700M MAU clause)

"I need multi-turn RL / agentic fine-tuning"
  → Start with DeepSeek V4-Flash or Qwen3.5-35B-A3B as base
  → Use GSPO (not GRPO) for MoE stability
  → Use PARL pattern (orchestrator trains; sub-agents frozen) for complex tasks
  → Unsloth supports GRPO/GSPO on all 2026 models

"I need the best reasoning per active parameter"
  → Gemma 4 31B: GPQA 84.3%, AIME 89.2% from a 31B dense model
  → Qwen3.5-26B-A4B: 4B active params; surprisingly strong GPQA 82.3%
```

---

## The Three Rules (Updated 2026)

> **KV cache + GRPO are to inference/fine-tuning what REST + SQL are to web engineering — understand them at the mechanical level before touching production systems.**

> **In 2026, KV cache compression is no longer optional for long-context production. TurboQuant and KVTC together cover the full spectrum: zero-calibration near-optimal compression (TurboQuant) and maximum-ratio transform coding for stored/reusable KV (KVTC).**

> **The 2026 model landscape is Chinese-lab dominated at the frontier. Kimi K2.6, DeepSeek V4, GLM-5.1, and Qwen3.5 are the open-weight leaders. Gemma 4 is the best Western open-weight model and the top choice for consumer-hardware fine-tuning under Apache 2.0. Know the licenses: MIT (DeepSeek, GLM) ≠ Modified MIT (Kimi) ≠ Apache 2.0 (Gemma, Qwen3.5) ≠ Meta Custom (Llama 4). This matters in production.**

---

*Roadmap compiled May 2026.*

*Base (carried forward): DeepSeek-R1, DAPO, Dr.GRPO, GSPO, GRPO-λ, vLLM v0.19, SGLang v0.4, TensorRT-LLM v1.0, Axolotl, Unsloth, LLaMA-Factory, NVIDIA Megatron Core Hybrid-EP, SpecForge/EAGLE3, ASTRA, ART+RULER, Cognizant ES paper.*

*2026 additions:*
*— TurboQuant (arXiv 2504.19874, ICLR 2026), PolarQuant (arXiv 2502.02617, AISTATS 2026), QJL (AAAI 2025)*
*— KVTC (arXiv 2511.01815, ICLR 2026)*
*— DART speculative decoding (arXiv 2601.19278, Jan 2026)*
*— GRPO-λ (arXiv 2510.00194), GSPO*
*— Gemma 4 (Google, Apr 2, 2026): E2B/E4B/26B-MoE/31B; Apache 2.0*
*— Qwen3.5 (Alibaba, Feb 2026): native multimodal; Gated Delta + sparse MoE; 201 languages*
*— DeepSeek V4-Pro/Flash (DeepSeek, Apr 24, 2026): CSA + HCA; 1.6T/49B and 284B/13B; MIT*
*— Kimi K2.5/K2.6 (Moonshot AI, Jan + Apr 2026): PARL agent swarm; ~1T; Modified MIT*
*— GLM-5/5.1 (Z.AI/Zhipu, Feb + Apr 2026): 744-754B; MIT; Huawei Ascend; SWE-Bench Pro leader*
*— MiMo-V2.5-Pro/Flash (Xiaomi, Mar–Apr 2026): 1.02T/42B; 1M ctx; long-horizon agentic*
*— Llama 4 Scout + Maverick (Meta, Apr 5, 2026): 10M + 1M ctx; multimodal MoE; Meta Custom license*
*— NVIDIA Nemotron Ultra 253B (2025): fits 8×H100; FP8; 3M ctx; enterprise open license*