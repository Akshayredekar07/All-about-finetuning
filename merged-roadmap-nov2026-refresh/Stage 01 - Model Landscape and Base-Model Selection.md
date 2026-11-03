# **Stage 1 — Model Landscape and Base-Model Selection**

**Outcome:** you can take a task constraint, narrow it to one base model, and defend the choice on four axes — capability, license, memory footprint, and tooling support. Nothing is trained in this stage; everything downstream depends on it.

The distinction that decides this stage
**Released**, **open-weight**, and **fine-tunable in this roadmap** are three different sets. Mistral Large 4, Kimi K3, Muse Spark Open, Qwen3.8-2.4T and GLM-5.3 are all *released* and all *open-weight*, and none of them is fine-tunable on anything a learner or a small team owns. Kimi K3.1 Pro is *released* (API) and not yet *open-weight*. Qwen3-4B-Instruct-2507 and Gemma 4 E4B are the only models in the frontier-adjacent set that are fine-tunable on a consumer GPU. Every table below is labelled on both axes, because a roadmap that blurs them sends people to spend a week downloading 1 TB of weights they cannot train.

## **Refreshed landscape as of 3 November 2026**

The October–November 2026 release window moved the field more than the previous six months combined. Six families arrived or matured inside a single quarter, and two of them are directly fine-tunable in the sense this roadmap cares about.

| Model | Lab | Released | Total / active | Context | License | Fine-tunable here? | Notes |
|---|---|---|---|---|---|---|---|
| Mistral Large 4 | Mistral | 29–31 Oct 2026 | ~1T / 52B | 1M | Mistral Ecosystem Open-Weight | No — 8×B300 FP8 LoRA [67][68] | Natively multimodal (Pixtral-style vision encoder), 128 routed experts + 1 shared, 160+ languages, reasoning effort low/medium/high (default high, cannot be disabled), native function calling with strict JSON schema, embedded DSpark speculative draft head |
| Kimi K3 | Moonshot AI | 16 Jul 2026 | 2.8T / 104B | 1,048,576 | Kimi K3 License | No — vLLM recipe needs ≥8×GB300 [81] | 93 layers: 1 dense + 69 KDA + 24 Gated MLA; 896 routed experts (top-16) + 2 shared; latent MoE dim 3584; SiTU-GLU; vocab 160K; native text+image+video through a 401M-parameter MoonViT-V2 vision encoder; thinking always on, `reasoning_effort` low/high/max (default max); MXFP4 weights with MXFP8 activations, QAT from SFT |
| Kimi K3.1 Pro | Moonshot AI | API 23–24 Oct 2026 | 2.8T / 104B | 1M | not announced | **No — weights promised 4 Nov 2026** | The `moonshotai` Hub organisation has no K3.1 repository as of 3 Nov 2026 [129]. Plan against the API only |
| Qwen3.8-2.4T-A95B | Alibaba | 12 Aug 2026 | 2.4T / 95B | — | Qwen custom Max License | No — rack scale | Largest open-weight Qwen line |
| Qwen3.8-27B | Alibaba | 14 Aug 2026 | 27B dense | 262K native → 1M YaRN | Apache-2.0 | **Yes — QLoRA at 24 GB** [70] | Dense unified VLM; 64 layers = 16×(3×Gated DeltaNet→FFN + 1×Gated Attention→FFN), hidden 5120; native video, hour-scale; MTP-trained; thinking default ON with `enable_thinking=False` and `reasoning_effort` xhigh/medium/low |
| Qwen3.6-27B | Alibaba | 22 Apr 2026 | 27B dense | 262K | Apache-2.0 | **Yes — QLoRA at 24 GB** [71] | Current mid-size reasoning line; Qwen3.6-35B-A3B (16 Apr) is the MoE sibling |
| Qwen3.5 family | Alibaba | 2 Mar 2026 | 0.8B–397B | 262K | Apache-2.0 | Yes, **bf16 LoRA only** | 201 languages; Gated DeltaNet + sparse MoE hybrid; early-fusion multimodal. **Do not QLoRA at 4-bit** |
| Gemma 4 family | Google | 31 Mar / 2 Apr / 16 Apr 2026 | E2B 2.3B eff · E4B 4.5B eff · 12B 11.95B · 26B-A4B · 31B 30.7B | 128K / 256K | Apache-2.0 | **Yes — E2B ~8 GB, E4B ~10 GB, 31B QLoRA ~22 GB** [75][76] | E2B/E4B/12B take image **and audio**; 140+ languages; MTP gives 1.4–2.2× faster inference |
| GLM-5.3 | Z.ai | 9 Sep 2026 | 320B / 18B | — | MIT | No — rack scale | MIT-licensed frontier MoE; BF16 and Flash variants on the Hub [88] |
| MiMo-V2.6-Pro | Xiaomi | 22 Sep 2026 | 1.02T | 1M | MIT | No | Long-horizon agentic |
| Muse Spark Open | Meta | 15 Oct 2026 | 816B / 42B | 1M | Muse Model License | No — ≥700 GB NVFP4 / 2 TB BF16 [84][85] | Text + image + video + audio in, text out; 256 experts, top-8; vocab 202,048; `reasoning_effort` low/medium/high/xhigh (default xhigh); gated behind manual approval |
| Muse Glimmer 30B | Meta | Oct 2026 | 30B | — | Muse Model License | Gated; weights on the Hub [86][130] | The same-organisation sibling worth watching for a single-node agentic workflow |
| Beam 1.0 | deepset | 5–16 Oct 2026 | 501.3B / 23.4B | — | Apache-2.0 | No | Added to `transformers` v5.20.1 |
| LFM2.5-24B-A2B [125] | Liquid AI | 13–14 Oct 2026 | 24B / 2.45B | — | LFM Open License v1.0 | Gated | 2.45B active makes it the cheapest high-quality MoE per active parameter |
| Nemotron 3.5 family | NVIDIA | 15–28 Oct 2026 | Bolt 74B-A9B · Super VL 120B/12B · Embed 8B · Rerank 8B · ASR · Transcribe · Dory 1.0-4B | varies | NVIDIA Open | Bolt/Dory on multi-GPU only | The most complete enterprise family: text, vision, retrieval, speech [89][90] |
| LongCat-2.5 | Meituan | 2 Nov 2026 | — | — | — | No | Most recent release in the window |
| DeepSeek V4.1 Flash | DeepSeek | 2026 | 284B-class / 13B active | 1M | MIT | No | Native `transformers` support, image-text-to-text, **no Jinja chat template** — use the reference `encoding` implementation [83] |
| Qwen3-TTS 0.6B / 1.7B | Alibaba | 22 Jan 2026 | 0.6B / 1.7B | — | Apache-2.0 | **Yes** | 3-second voice cloning, 10 languages, ~97 ms streaming latency, trained on 5M+ hours [72][74] |
| EmbeddingGemma 2 | Google | 2026 (transformers v5.19) | — | 512/256/128 truncatable | Apache-2.0 | Yes | 768-dim cross-modal (text/image/audio/video) [41] |
| jina-reranker-v3.5 | jina.ai | 20 Jul 2026 | — | — | — | — | Current compact reranker |
| KURE-Reranker-nano | KURE | 9 Sep 2026 | — | — | — | — | Smallest reranker in the current MTEB space |

## **May 2026 leaderboard snapshot, retained with corrections**

The source document's May 2026 table is preserved because it is a useful record of how fast the frontier moved. Every row is now stale in at least one column, and the two rows that are still strategically useful are the DeepSeek and Gemma rows.

| Rank then | Model | Lab | Total / active | Context | Key strength | License | Status now |
|---|---|---|---|---|---|---|---|
| ~1 | Kimi K2.6 [82] | Moonshot AI | 1T / 32B | 262K | Coding agents; PARL swarm; GPQA ~90% | Modified MIT | Superseded by Kimi K3 (2.8T / 104B, 1M ctx) |
| ~2 | MiMo-V2.5-Pro | Xiaomi | 1.02T / 42B | 1M | Long-horizon agentic; engineering | Open | Superseded by MiMo-V2.6-Pro (22 Sep 2026) |
| ~3 | DeepSeek V4-Pro | DeepSeek | 1.6T / 49B | 1M | Math/coding; Hybrid Attention | MIT | Line continues as DeepSeek V4.1 Flash, MIT, native Transformers support |
| ~4 | GLM-5.1 | Z.ai | 754B / 40B | 200K | SWE-Bench Pro leader | MIT | Superseded by GLM-5.3 (320B / 18B) |
| ~5 | Qwen3.5-397B-A17B | Alibaba | 397B / 17B | 262K | GPQA 88.4%; multilingual; multimodal | Apache-2.0 | Extended by Qwen3.6 (Apr) and Qwen3.8 (Aug) |
| ~6 | Gemma 4 31B | Google | 31B dense | 256K | Best efficiency; AIME 89.2% | Apache-2.0 | **Still the current Western default** |
| ~7 | Llama 4 Scout | Meta | 109B / 17B | 10M | Extreme context; multimodal | Meta Custom | Still the longest open context; train at 8K–32K, not 10M |
| ~8 | Nemotron Ultra | NVIDIA | 253B dense | 3M | Enterprise; reasoning; efficient | Open | Superseded by Nemotron 3.5 family |
| ~9 | MiniMax M2.7 | MiniMax | ~230B MoE | 1M | Productivity; tool use | Open | Superseded by MiniMax M3.1 Flash (19 Oct 2026) |
| ~10 | DeepSeek V4-Flash | DeepSeek | 284B / 13B | 1M | Best price-per-effort | MIT | API price field has moved since May; re-check before costing |

The trend line the source noted still holds, with one correction: MIT/Apache-2.0 is *not* the universal norm at the frontier. The October–November wave introduced three custom licenses — Qwen's Max License for the 2.4T model, Mistral's Ecosystem Open-Weight License, and Meta's Muse Model License — each with a revenue-triggered clause. License auditing stopped being a formality in October 2026.

## **Model families, by type**

The source document organised the landscape by parameter class, which is still the right frame for "what can I actually train". Each subsection below keeps that structure and refreshes the numbers.

### **Small language models (under 4B effective parameters)**

| Model | Params | Release | Notes |
|---|---|---|---|
| Gemma 4 E2B | 2.3B effective (5.1B with embeddings) | Apr 2026 | Google; multimodal with audio; Apache-2.0; 35 layers, 512 sequence width, 128K context |
| Gemma 4 E4B | 4.5B effective (8B with embeddings) | Apr 2026 | Google; edge + multimodal; 42 layers, 512 SW, 256K context |
| Qwen3.5-0.8B / 2B | 0.8–2B | Mar 2026 | Alibaba; native multimodal; thinking mode |
| Qwen3-1.7B | 1.7B | 2025 | GRPO-friendly; hybrid reasoning |
| Phi-4-mini | 3.8B | 2025 | Microsoft; strong reasoning per parameter |
| SmolLM2 | 135M–1.7B | 2024 | Hugging Face; edge-optimised |
| Gemma-3-1B / 2B | 1–2B | 2025 | Google; CPU/mobile baseline |
| Llama-3.2-1B/3B | 1–3B | 2024 | Meta; instruction-tuned |

Single consumer GPU (8–16 GB VRAM) with QLoRA or bf16 LoRA. Gemma 4 E2B QLoRA runs in ~8 GB VRAM; Gemma 4 E4B QLoRA in ~10 GB. Qwen3.5-0.8B bf16 LoRA needs ~3 GB. GRPO on a small model is viable — Unsloth documents Gemma 4 E2B RL in the same low-VRAM band. Target for on-device, edge, and serverless deployment.

The Qwen3.5 family is **not recommended for QLoRA at 4-bit**, dense or MoE, because of higher-than-normal quantization error; use bf16 LoRA instead.

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "google/gemma-4-E4B",
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

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "Qwen/Qwen3.5-2B",
    max_seq_length= 2048,
    dtype         = torch.bfloat16,  # bf16; NOT load_in_4bit for Qwen3.5
    load_in_4bit  = False,
)

# Qwen3.5 uses transformers v5 — older versions will not work
# thinking mode: /think and /no_think tokens in user prompts
# preserve_thinking=True in API for multi-turn agentic workflows (Qwen3.6+)
```

### **Dense language models (7B–72B)**

| Model | Params | Release | Notes |
|---|---|---|---|
| Gemma 4 31B | 30.7B dense (60 layers) | Apr 2026 | 256K context; Apache-2.0; QLoRA ~22 GB [75][76] |
| Qwen3.8-27B | 27B dense | Aug 2026 | Unified VLM, 262K native / 1M YaRN, MTP, per-turn thinking control [70] |
| Qwen3.6-27B | 27B dense | Apr 2026 | Current mid-size reasoning line [71] |
| Qwen3.5-9B / 27B | 9–27B | Mar 2026 | Natively multimodal; thinking/non-thinking modes |
| Qwen3-7B / 14B / 32B / 72B | 7–72B | 2025 | Hybrid thinking mode; strong reasoning |
| Mistral Small 4 | ~22B | 2026 | Vision + reasoning |
| DeepSeek-R1 distills | 1.5B–70B | 2025 | RL-trained reasoning |
| Llama 4 Scout | 109B / 17B active | Apr 2025 | 10M context; multimodal; Meta Custom license |

Gemma 4 31B needs an A100-class card for bf16 LoRA. `transformers` v5 is required for the Qwen3.5 generation. Llama 4 Scout fine-tuning should focus on LoRA over the attention layers, and the 10M-token context is a serving feature, not a training one — train at 8K–32K.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-31b-it",
    dtype         = torch.bfloat16,
    device_map    = "auto",
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

### **Mixture-of-experts models**

| Model | Total params | Active params | Release | Notes |
|---|---|---|---|---|
| Qwen3.8-2.4T-A95B | 2.4T | 95B | Aug 2026 | Qwen custom Max License |
| Mistral Large 4 | ~1T | 52B | Oct 2026 | 128 routed experts + 1 shared; 1M ctx; multimodal; DSpark draft head [67] |
| Kimi K3 | 2.8T | 104B | Jul 2026 | KDA + MLA hybrid; 1M ctx; Kimi K3 License [81] |
| Muse Spark Open | 816B | 42B | Oct 2026 | Text+image+video+audio; 256 experts top-8; Muse Model License [84] |
| Beam 1.0 | 501.3B | 23.4B | Oct 2026 | Apache-2.0; in `transformers` v5.20.1 |
| GLM-5.3 | 320B | 18B | Sep 2026 | MIT [88] |
| Qwen3.6-35B-A3B | 35B | 3B | Apr 2026 | Apache-2.0 |
| LFM2.5-24B-A2B | 24B | 2.45B | Oct 2026 | LFM Open License v1.0 |
| Qwen3.5-397B-A17B | 397B | 17B | Mar 2026 | Native multimodal; Gated Delta + sparse MoE |
| Qwen3.5-122B-A10B | 122B | 10B | Mar 2026 | Strong mid-tier; 262K context |
| GLM-5.1 | 754B | 40B | Apr 2026 | Huawei Ascend 910B trained; zero NVIDIA dependency; MIT |
| MiMo-V2.5-Pro | ~1.02T | 42B | Apr 2026 | 1M ctx; agentic; long-horizon engineering |
| MiniMax M2.7 | ~230B | — | 2026 | 1M context; productivity |
| Llama 4 Maverick | ~400B | 17B | Apr 2025 | Multimodal; Meta Custom license |
| Qwen3-235B-A22B | 235B | 22B | 2025 | SOTA open MoE (pre-Qwen3.5) |
| DeepSeek-V3 | 671B | 37B | 2024 | MLA + FP8 training |
| Mixtral-8×7B | 47B | 12.9B | 2023 | Baseline MoE reference |

Architecture advances worth knowing before you fine-tune an MoE:

- **DeepSeek V4 hybrid attention** — Compressed Sparse Attention (CSA) plus Heavily Compressed Attention (HCA) reaches ~27% of single-token inference FLOPs versus V3.2 at 1M context, which is what makes 1M-token context affordable without linear approximations.
- **GLM-5.1** was the first top-tier open model trained with zero NVIDIA dependency (100K Huawei Ascend 910B chips). Hardware diversity is now a real selection criterion for a self-hosted shop.
- **Qwen3.5/3.8** use a hybrid Gated DeltaNet + attention stack; the Qwen3.8-27B dense card shows the shape clearly: 64 layers arranged as 16 × (3 × Gated DeltaNet→FFN + 1 × Gated Attention→FFN).
- **Kimi K3** alternates KDA (69 layers) and Gated MLA (24 layers) with one dense layer, 896 routed experts at top-16 plus 2 shared, latent MoE dimension 3584, SiTU-GLU activation, 160K vocab.

Fine-tuning practice for MoE, with the numbers corrected:

- **Qwen3.5 MoE (35B-A3B / 122B-A10B):** bf16 LoRA, not 4-bit — bitsandbytes cannot hold 4-bit `nn.Parameter` for MoE experts, so MoE QLoRA is blocked outright. `Qwen3.5-35B-A3B` bf16 LoRA needs ~74 GB VRAM [79].
- **DeepSeek V4 Flash, not Pro:** fine-tune Flash (284B / 13B active); Pro at 1.6T / 49B needs a rack. Flash now has native Transformers support, so the `--trust-remote-code` requirement from the May source is gone [83].
- **Llama 4 Maverick:** 8× H100 minimum even with QLoRA because of expert tensor size; most teams tune Scout and serve Maverick stock.
- **Expert collapse:** monitor the per-expert utilisation histogram during training and alert if any expert falls below ~1% utilisation.
- **GSPO over GRPO** for MoE RL, because sequence-level importance ratios stabilise sparse-expert training [94][96].

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
# Unsloth ships a dedicated MoE fast path; measure your own step time.
```

### **Vision-language models**

| Model | Notes |
|---|---|
| Qwen3.8-27B | Dense unified VLM, native text+image+video, 262K context, thinking controls [70] |
| Qwen3.5-VL / Omni | Early-fusion native multimodal (vision + audio), 262K context |
| Qwen2.5-VL-7B / 72B | 2025 SOTA VLM; dynamic resolution; video; still widely fine-tuned |
| Gemma 4 E4B / E2B / 12B | Text + image, with audio on E2B/E4B/12B [75] |
| InternVL3-8B / 78B | MPO training; strong multimodal reasoning |
| Llama 4 Scout / Maverick | Open-weight multimodal; Scout 10M ctx; Maverick video-capable |
| LLaVA-NeXT | Any-resolution; widely documented |
| PaliGemma 2 | Google; MLP adapter; compact and strong |
| DeepSeek-VL2 | Dynamic tiling; MoE backbone |
| UI-TARS-1.5 (ByteDance) | GUI agent; browser/computer/phone control |

Architecture: vision encoder → projection layer (linear / MLP / Q-Former) → LLM decoder.

Early fusion is the 2026 frontier: Qwen3.5 trains multimodal tokens jointly from pretraining instead of bolting on an adapter, reaching parity with dedicated VLMs on reasoning, coding, and agents. It is harder to fine-tune but a stronger base.

Fine-tuning strategies, in increasing cost:

1. **Projector only** — freeze vision encoder and LLM, train the projection layer. Fastest, least data.
2. **LLM LoRA + projector** — freeze vision encoder, LoRA the LLM, train the projector. The standard.
3. **Full fine-tune** — unfreeze everything, multi-GPU required, best quality.

**Image-LoRA** applies LoRA specifically to image-relevant attention weights and preserves text reasoning better than standard LoRA on VLMs. **GRPO on VLMs** with verifiable rewards (wildfire detection, driving analysis) has a published DGX Spark playbook from NVIDIA.

```python
from unsloth import FastVisionModel

# Gemma 4 E4B multimodal — vision fine-tuning
model, tokenizer = FastVisionModel.from_pretrained(
    "google/gemma-4-E4B-it",
    load_in_4bit = True,
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers   = False,  # freeze vision encoder by default
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules     = True,
    r = 16, lora_alpha = 16,
)
```

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    dtype=torch.bfloat16, device_map="auto"
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

### **Diffusion models**

| Model | Notes |
|---|---|
| FLUX.1-dev / schnell | Black Forest Labs; flow matching; strongest open T2I of the 2026 generation |
| FLUX.1-Kontext | Edit-in-context; instruction-following image editing |
| FLUX.3 Image Klein | Added in `diffusers` 0.42 (Oct 2026) |
| Wan2.1 | Video generation; fine-tunable |
| Stable Diffusion 3.5 | DiT architecture |
| CogVideoX-5B | Video generation |

Core concepts:

- **Forward process:** add Gaussian noise over T steps, `q(xₜ|xₜ₋₁) = N(xₜ; √(1-β)xₜ₋₁, βI)`.
- **Reverse process:** learn to denoise, `pθ(xₜ₋₁|xₜ) = N(xₜ₋₁; μθ(xₜ,t), Σθ(xₜ,t))`.
- **DDPM objective:** predict noise ε, MSE loss `||ε - εθ(xₜ, t)||²`.
- **Latent diffusion:** a VAE encodes the image to a latent, diffusion runs in latent space, 4–8× cheaper.
- **Flow matching (FLUX):** a direct ODE path from noise to data; more stable, faster convergence.

Fine-tuning methods:

- **LoRA:** 100–1000 steps; best for style transfer.
- **DreamBooth:** full fine-tune on 3–25 subject images with a rare token (`sks dog`); regularisation images prevent forgetting.
- **Textual inversion:** learn a new token embedding; minimal data, limited flexibility.
- **ControlNet:** trainable encoder copy; needs paired (image, condition) data.

```python
from diffusers import FluxPipeline
from peft import LoraConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", dtype=torch.bfloat16
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

### **Audio models**

ASR:

| Model | Notes |
|---|---|
| Whisper-large-v3 | OpenAI; strong multilingual ASR; seq2seq |
| Distil-Whisper | ~6× faster; distillation from v3 |
| Nemotron 3.5 ASR Streaming 0.6B | Streaming ASR; added to vLLM 0.32.0 |
| Nemotron 3.5 Transcribe | Transcription endpoint, 2.8% WER on NVIDIA's internal tracker |
| Voxtral Mini 3B | Mistral; Apache-2.0; beats Whisper large-v3 on internal benchmarks; ~9.5 GB bf16; llama.cpp native |
| Granite Speech 4.1 2B | IBM; 5.33% WER; Apache-2.0 |

TTS:

| Model | Release | Notes |
|---|---|---|
| Qwen3-TTS | 22 Jan 2026 | 0.6B–1.7B; 3-second voice cloning; 10 languages; ~97 ms streaming latency; Apache-2.0; trained on 5M+ hours [72][74] |
| Parler-TTS | 2024 | Decoder-only; description-conditioned voice cloning |
| CSM-1B (Sesame) | 2025 | Conversational; LoRA via Unsloth |
| Kokoro | 2024 | Lightweight; fast inference |
| F5-TTS | 2024 | Flow-matching TTS; fast fine-tuning |

Qwen3-TTS fine-tuning notes:

- `Qwen3-TTS-12Hz-1.7B-Base` and `Qwen3-TTS-12Hz-0.6B-Base` are the fine-tuning targets; the three variants are Base (clone + fine-tune target), CustomVoice (nine timbres plus instruction control), and VoiceDesign (1.7B only).
- The `Qwen3-TTS-Tokenizer-12Hz` is part of the **open** release under Apache-2.0, not proprietary.
- Single-speaker fine-tuning is supported; multi-speaker was still in development.
- CUDA required; an MLX port for Mac was in progress.
- The technical report's training recipe is DPO on speech preference pairs, then rule-based rewards with **GSPO** for capability and stability, then lightweight speaker fine-tuning [74].

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

### **Embedding models**

| Model | Dim | Notes |
|---|---|---|
| nvidia/Nemotron-3.5-Embed-8B-BF16 | 4,096 | 8.4B params; 32,768 max tokens; ViDoRe v3.1 61.17 nDCG@10, first among dense retrieval models on 20 Oct 2026; Sentence-Transformers compatible [89] |
| nvidia/Nemotron-3-Embed-8B-BF16 | 4,096 | RTEB-16 78.46, multilingual leader on 16 Jul 2026; ViDoRe-v3 text 60.60; MMTEB retrieval 75.45 |
| nvidia/Nemotron-3.5-Embed-1B | — | Smaller variant; NVFP4 build for Blackwell |
| EmbeddingGemma 2 | 768 | Cross-modal (text/image/audio/video); truncatable to 512/256/128 [41] |
| BAAI/bge-m3 | 1,024 | Multi-lingual, multi-granularity |
| intfloat/e5-mistral-7b | 4,096 | Instruction-following embeddings |
| Alibaba-NLP/gte-Qwen2-7B | 3,584 | Was "top MTEB 2026" in May; superseded |
| nomic-embed-text-v2-moe | 768 | MoE embedding model |
| Cohere embed-v4 | 1,536 | Multimodal embeddings |

Training objectives:

- **InfoNCE / MultipleNegativesRankingLoss** — standard contrastive loss with in-batch negatives.
- **Matryoshka Representation Learning (MRL)** — one training pass produces usable embeddings at 64/128/256/512/768/1024 dimensions.
- **Hard negative mining** — mine with BM25 or embedding similarity, never random; this is the single largest quality lever.
- **GISTEmbedLoss** — teacher-guided hard-negative selection.
- **RLVR for embeddings** — a downstream task verifier scores embedding quality, the 2026 frontier direction.

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

## **License audit for the candidate set**

License is a selection constraint, not a legal footnote. Three of the five October–November frontier releases carry revenue-triggered clauses, so this table decides real commercial outcomes.

| License family | Models in scope | Commercial use | Trigger to watch | Practical rule |
|---|---|---|---|---|
| Apache-2.0 | Qwen3.5/3.6/3.8-27B, Gemma 4, Beam 1.0, Kolibri-1, Nanbeige4.3-33B, Mage-VL-Cowork, Qwen3-TTS | Yes, including commercial and MaaS | None | Default choice. Use it unless there is a reason not to |
| MIT | DeepSeek V4.x, GLM-5.3, MiMo-V2.6-Pro | Yes | None | Equivalent in practice to Apache-2.0 for most uses; keep the NOTICE handling difference in mind |
| Kimi K3 License | Kimi K3 | Yes | >US$20M revenue requires negotiation with Moonshot | Treat as MIT-with-a-clause; read §3 before any product launch [81] |
| Mistral Ecosystem Open-Weight | Mistral Large 4 | Yes for weights and internal use | **MaaS requires a separate Mistral commercial license** | Free weights ≠ free service. If you sell inference built on it, you need a contract [67][68] |
| Muse Model License | Muse Spark Open, Muse Glimmer 30B | Yes, including fine-tuning, derivatives, distillation, sublicensing | MaaS with aggregate gross revenue >US$20M in any 12-month period requires notifying Meta and entering a separate written agreement | Gated download, full legal name, date of birth, organisation name required. Plan the commercial conversation early [84][85][87] |
| LFM Open License v1.0 | LFM2.5-24B-A2B, LFM3 [125] | Commercial use only below **US$10M annual revenue**; qualified non-profits exempt for non-commercial and research use; internal research, testing and evaluation permitted | The revenue threshold also applies per downstream customer | Free for learning and research; re-license before scaling [125] |
| Meta Custom (Llama 4) | Llama 4 Scout, Maverick | Yes with conditions | The 700M monthly-active-user clause | Avoid if you might reach that scale [85] |
| NVIDIA Open | Nemotron 3.5 family, Nemotron Ultra | Yes, including commercial | Standard NVIDIA Open Model License terms | The enterprise-friendly default for multi-model stacks |
| Qwen custom Max License | Qwen3.8-2.4T-A95B | Restricted relative to Apache-2.0 | Read before use; the 27B model is Apache-2.0 and is the one you will actually train | Keep the two Qwen3.8 releases separate in your model registry |
| Community License | Qwen3.8 Flash-Next family | Yes, with conditions | Qwen Community License 1.0 terms | Read the terms; not Apache-2.0 |

Three rules follow directly. First, **MIT ≠ Modified MIT ≠ Apache-2.0 ≠ "open"**: the source document's own closing rule called this out, and the October wave made it more important, not less. Second, **"open weights" does not mean "open source"** — Muse Spark Open permits modification and redistribution, but attaches a usage policy and a revenue trigger. Third, **a fine-tune inherits the base licence** unless the licence says otherwise; Muse Spark explicitly permits fine-tuning and derivatives, LFM does not grant commercial use above its threshold, and Kimi K3 requires negotiation above its own.

## **Memory and hardware matrix**

This is the table that decides the base model, because a model you cannot fit is not a candidate. Unsloth's published numbers are the reference; the column is what the configuration needs, not what the weights alone cost.

| Model | QLoRA (4-bit) | bf16 LoRA | Full fine-tune | Practical notes |
|---|---|---|---|---|
| Gemma 4 E2B | ~8 GB | 8–10 GB | not attempted | Smallest member that still takes image + audio |
| Gemma 4 E4B | ~10 GB | ~17 GB | not attempted | **Recommended** over E2B LoRA: bigger model, negligible quantisation difference [76] |
| Gemma 4 12B Unified | 7–8 GB (inference) | 13–14 GB (inference) | — | 25 GB for bf16 inference |
| Gemma 4 26B-A4B | blocked | >40 GB | — | 4-bit MoE is blocked by bitsandbytes; use bf16 LoRA |
| Gemma 4 31B | ~22 GB | ~56 GB (27B-class) | — | Fits a 24 GB card with QLoRA |
| Qwen3.5-0.8B | not recommended | ~3 GB | — | bf16 LoRA only |
| Qwen3.5-2B | not recommended | ~5 GB | — | bf16 LoRA only |
| Qwen3.5-4B | not recommended | ~10 GB | — | bf16 LoRA only |
| Qwen3.5-9B | not recommended | ~22 GB | — | bf16 LoRA only |
| Qwen3.5-27B | not recommended | ~56 GB | — | bf16 LoRA only |
| Qwen3.5-35B-A3B | blocked (MoE) | ~74 GB | — | bf16 LoRA; 8×80 GB or 2×140 GB |
| Qwen3.6-27B | ~24 GB | >36 GB | 4× QLoRA VRAM | Native vision, video, 262K context [71] |
| Qwen3.8-27B | ~24 GB | >36 GB | 4× QLoRA VRAM | QLoRA and RL both documented at 24 GB [78] |
| Qwen3.8-27B (NVFP4) | 40.2 GB peak (vs 72.9 GB bf16) | — | — | Unsloth dynamic NVFP4 quant; also 17 GB RAM via dynamic GGUFs [79] |
| Mistral Large 4 | — | 8×B300 FP8 | — | TRL ships a LoRA SFT example for it on a single 8×B300 node [6] |
| Kimi K3 | — | ≥8×GB300 | — | vLLM recipe requirement [81] |
| Muse Spark Open | — | 2 TB BF16 / 700 GB NVFP4 | — | Also optimised for two Mac Studio M5 Ultra via MLX [84] |
| FLUX.1-dev BF16 | — | ≥24 GB | — | Diffusion LoRA at r=16 |

Gemma 4 inference memory, for reference when you are serving rather than training: E2B 4 GB at 4-bit, 5–8 GB at 8-bit, 10 GB at bf16; E4B 5.5–6 GB / 9–12 GB / 16 GB; 12B Unified 7–8 GB / 13–14 GB / 25 GB. Unsloth's dynamic NVFP4 builds need 7 GB (E2B), 9 GB (E4B) and 11 GB (12B) and are 1.12–1.26× faster than bf16 on Blackwell [77]. Google's recommended sampling for Gemma 4 is `temperature=1.0`, `top_p=0.95`, `top_k=64`.

## **Base-model decision framework**

The source document's decision tree, rewritten against the November 2026 landscape. Each branch answers a real constraint, and each points at models that are actually trainable in this roadmap unless the answer says otherwise.

```
Goal: pick the right base model for fine-tuning.

"I need the best open-weight coding/agent model with an open license"
  → Qwen3.8-27B (Apache-2.0; dense; QLoRA at 24 GB)           ← trainable here
  → DeepSeek V4.1 Flash (MIT; 13B active)                      ← serve, not train
  → GLM-5.3 (MIT)                                             ← serve, not train
  → Kimi K3 (Kimi K3 License)                                  ← serve, not train

"I need the best efficiency for consumer-GPU fine-tuning"
  → Gemma 4 E4B (4.5B effective; Apache-2.0; QLoRA ~10 GB)      ← trainable here
  → Gemma 4 26B-A4B (bf16 LoRA, >40 GB)
  → Qwen3.5-2B (bf16 LoRA ~5 GB)

"I need native multimodal (text + image + audio)"
  → Qwen3.8-27B (text + image + video, 262K)                   ← trainable here
  → Gemma 4 E4B (text + image + audio)
  → Qwen3.5 family (early fusion, 201 languages)

"I need extreme long context (>256K tokens)"
  → Qwen3.8-27B (262K native, 1M via YaRN)                     ← trainable here
  → Llama 4 Scout (10M; train at 8K–32K)
  → DeepSeek V4 / Kimi K3 / MiMo-V2.6-Pro (1M)

"I need enterprise deployment with MIT or Apache licence"
  → Qwen3.8-27B (Apache-2.0)
  → Gemma 4 (Apache-2.0; no MAU clause; Google tooling)
  → GLM-5.3 / DeepSeek V4 (MIT)
  → Avoid: Mistral Large 4 as a service (MaaS needs a separate contract),
           Muse Spark Open as a service above US$20M revenue,
           LFM above US$10M revenue

"I need multi-turn RL or agentic fine-tuning"
  → Base: Qwen3.8-27B (24 GB RL) or Qwen3.6-27B
  → GSPO for MoE stability; GRPO-family for dense
  → Orchestrator/sub-agent decomposition for long-horizon tasks (Stage 9)

"I need the best reasoning per active parameter"
  → Gemma 4 31B (QLoRA 22 GB, Apache-2.0)                     ← trainable here
  → Qwen3.6-35B-A3B (3B active)
```

## **Model landscape pre-flight checklist**

- [ ] The chosen base is **open-weight**, not API-only, if the plan involves any adapter training.
- [ ] The licence has been read in full, and the revenue trigger (if any) is compatible with the intended product.
- [ ] The base is in the `transformers` version window that the serving engine accepts, or the adapter is exported separately.
- [ ] The target model is confirmed to support the modality you need (image, audio, video) — not assumed from the family name.
- [ ] VRAM for the chosen configuration is written down before the dataset is built.
- [ ] Thinking/reasoning mode behaviour is known for the base: on by default, togglable, or always on.
- [ ] The chat template is confirmed to exist. DeepSeek V4.1 Flash ships **no Jinja chat template** and needs the reference `encoding` implementation [83].
- [ ] Gated models have the access request submitted — approval for Muse Spark Open needs full legal name, date of birth, and organisation name [84].
- [ ] A dated snapshot of every leaderboard number used in a decision has been saved, because retrieval rankings move monthly [41].
