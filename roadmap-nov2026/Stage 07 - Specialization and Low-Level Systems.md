## **Stage 07: Specialization and Low-Level**

**Libraries:** `sentence-transformers`, `diffusers`, `timm`, `bitsandbytes`, `kernels`, `triton`, `torch`

**Outcome:** one specialization that matches the target role, plus the performance layer for when you hit a memory or speed wall.

### **1. Scope and Exit Criterion**

**Scope:** this stage has two halves that are usually confused. The **specializations** (embeddings, generative vision, classification vision) are career choices, so pick the one your target role needs. The **low-level layer** (quantization, kernels, Triton, CUDA) is a tool you reach for when a wall appears, not a curriculum to complete on its own.

**Exit criterion:** one specialization shipped end to end, and one kernel written and benchmarked against the library implementation it replaces.

### **2. Environment**

```bash
pip install -U sentence-transformers "sentence-transformers[train]"
pip install -U diffusers "diffusers[torch]"
pip install -U timm
pip install -U bitsandbytes kernels triton
```

### **3. Specialization A: Embeddings, Search, and Reranking**

**For:** RAG, semantic search, embeddings, similarity, clustering, reranking.

**Overview:** Sentence Transformers hosts Sentence-BERT and its variants, along with cross-encoders and sparse encoders.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["This is a cat", "This is a dog", "Paris is beautiful"]
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)   # (3, 384)

print(model.similarity(embeddings, embeddings))
sim = util.cos_sim(embeddings[0], embeddings[1])

query = model.encode("What is a pet?", normalize_embeddings=True)
hits = util.semantic_search(query, embeddings, top_k=2)
```

The same flow with a stronger model:

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
sentences = ["This is a cat", "This is a dog", "Paris is beautiful"]
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)

print(model.similarity(embeddings, embeddings))
sim = util.cos_sim(embeddings[0], embeddings[1])

query = model.encode("What is a pet?", normalize_embeddings=True)
hits = util.semantic_search(query, embeddings, top_k=2)
```

**Core concepts:**

- Sentence embeddings versus token embeddings
- Cosine similarity and dot score
- Bi-encoder versus cross-encoder
- Pooling strategies: mean, CLS, max
- Dense versus sparse encoders (`SparseEncoder`)
- `MultiVectorEncoder` (late interaction), and `merge()` now works across all four encoder types

**Popular models:**

- `all-MiniLM-L6-v2`: fast, small, general purpose
- `all-mpnet-base-v2`: higher quality general purpose
- `multi-qa-MiniLM`: optimized for QA search
- `paraphrase-multilingual`: many languages
- `BAAI/bge` series: strong retrieval models
- `intfloat/e5` series: strong retrieval models
- MTEB leaderboard: check it for the current best model for your task

**Leaderboard reading, November 2026:** MTEB v2 boards score 41 (English) and 131 (multilingual) tasks and rank by Borda count, so row order is not mean-score order. The leading open-weight general embedders are QZhou-Embedding (75.97) and Qwen3-Embedding-8B (75.23, Apache 2.0, 32K context). The multilingual leader is Microsoft's `harrier-oss-v1-27b` (MIT, 74.27, 5,376 dims, expensive to store). Prefer **RTEB** (held-out sets the maintainers evaluate) over MTEB-only claims, and check the live board before committing.

For reranking, the open-weight leaders include Qwen3-Reranker-4B (0.6367 retrieval average) and `bge-reranker-v2-m3`. NVIDIA's Nemotron-3.5-Rerank-8B landed on 20 Oct 2026. Reranking the top 100 lifted Recall@5 from 0.695 to 0.816 in one independent study, and reranking only the top 20 made it worse, so measure on your own queries.

**Model selection, November 2026:**

| **Use case** | **Pick** |
|---|---|
| Fast general purpose | `all-MiniLM-L6-v2` |
| Higher quality general | `all-mpnet-base-v2` |
| QA search | `multi-qa-MiniLM` |
| Strong retrieval, English | `Qwen3-Embedding-8B` (Apache 2.0, 32K context) or `Qwen3-Embedding-4B` |
| Multilingual | `microsoft/harrier-oss-v1-27b` (MIT, 74.27 on MTEB multilingual v2, expensive to store at 5,376 dims) |
| Reranking | `Qwen3-Reranker-4B` (0.6367 retrieval average), `bge-reranker-v2-m3`, `Nemotron-3.5-Rerank-8B` |

**Operations:**

- `encode()` with `normalize_embeddings`
- `model.similarity()` for similarity matrices
- `util.semantic_search()`, `util.cos_sim()`, `util.dot_score()`
- Paraphrase mining and clustering embeddings

**Cross-encoders:**

- `CrossEncoder` for reranking
- Bi-encoder retrieval followed by cross-encoder reranking for RAG
- Score normalization
- `CrossEncoderTrainer` for training rerankers

**Fine-tuning:**

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

**Losses:** `CosineSimilarityLoss` (regression pairs), `ContrastiveLoss` (positive/negative pairs), `TripletLoss` (anchor, positive, negative), `MultipleNegativesRankingLoss` (best default for retrieval), `MatryoshkaLoss` (flexible embedding sizes).

**Evaluators during training:** `EmbeddingSimilarityEvaluator`, `InformationRetrievalEvaluator`, `RerankingEvaluator`. Also hard negative mining and `push_to_hub()`.

**Import paths:** in recent releases, losses and evaluators live under `sentence_transformers.sentence_transformer.losses`, while top-level imports such as `SentenceTransformer` are unchanged. The old `model.fit()` with `InputExample` and a `DataLoader` is the deprecated route. A `train-sentence-transformers` skill is published in the Hub Skills registry, so a coding agent can run the training workflow end to end.

**Topic checklist:**

- Sentence embeddings versus token embeddings
- Cosine similarity and dot score
- Bi-encoder versus cross-encoder
- Pooling strategies: mean, CLS, max
- Dense versus sparse encoders (`SparseEncoder`)
- `MultiVectorEncoder` (late interaction) and `merge()` across all four encoder types
- `encode()` with `normalize_embeddings`
- `model.similarity()` similarity matrices
- `util.semantic_search()`, `util.cos_sim()`, `util.dot_score()`
- Paraphrase mining and clustering embeddings
- `CrossEncoder` for reranking; bi-encoder then cross-encoder for RAG
- `CrossEncoderTrainer` for training rerankers
- Losses: `CosineSimilarityLoss`, `ContrastiveLoss`, `TripletLoss`, `MultipleNegativesRankingLoss`, `MatryoshkaLoss`
- Evaluators: `EmbeddingSimilarityEvaluator`, `InformationRetrievalEvaluator`, `RerankingEvaluator`
- Hard negative mining
- `push_to_hub()` and the `train-sentence-transformers` Hub skill

**Install:**

```bash
pip install -U sentence-transformers
pip install -U "sentence-transformers[train]"
```

### **4. Retrieval Stack: Encoder, Reranker, and What to Fine-Tune**

**Overview:** the November 2026 retrieval landscape, and the practical split between the two model families.

| **Component** | **What it does** | **Current models** | **Fine-tune it?** |
|---|---|---|---|
| Bi-encoder (embedder) | Maps query and document into one vector each; scores by similarity | `nvidia/Nemotron-3.5-Embed-8B-BF16` (4,096-dim, 32,768 tokens), `nvidia/Nemotron-3-Embed-8B-BF16`, EmbeddingGemma 2 (768-dim cross-modal), `BAAI/bge-m3`, `intfloat/e5-mistral-7b` | Yes: contrastive, with hard negatives |
| Cross-encoder (reranker) | Scores query and document jointly, one pair at a time | `nvidia/Nemotron-3.5-Rerank-8B-BF16`, `jina-reranker-v3.5`, `KURE-Reranker-nano`, `KaLM-Reranker-V1-Small`, `Qwen3-VL-Reranker-8B` | Yes, but only with a judgement dataset |

Nemotron 3.5 Embed 8B leads the dense retrieval models on MTEB's ViDoRe v3.1 split with 61.17 nDCG@10 as of 20 October 2026, and its reranker reaches 60.49 on the combined image+text task against 59.17 for Qwen3-VL-Reranker-8B. Both are Sentence-Transformers compatible, so the training code here applies unchanged. Save the date alongside the number, because MTEB rankings move every month.

**Matryoshka training** is the technique that makes one embedder serve several vector dimensions:

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
# Hard negatives are critical: mine with BM25 or embedding similarity
```

**Two mistakes that ruin embedding fine-tuning:** training with random negatives, and evaluating only on the metric you trained against. Mine hard negatives with BM25 or embedding similarity, and evaluate on a separate retrieval benchmark with a held-out corpus.

**Reranker training:** this is a different shape. The training signal is a relevance judgement, not a contrastive pair, so the dataset is (query, positive document, judged-negative document) with graded labels, and the evaluation is nDCG@10 on a query set you did not train on.

### **5. Specialization B: Generative Vision and Audio**

**For:** image generation, video generation, audio synthesis.

**Overview:** Diffusers is the Transformers equivalent for diffusion models. Stable Diffusion, SDXL, FLUX, and ControlNet live here.

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
image = pipe("a photo of an astronaut riding a horse").images[0]
image.save("output.png")

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",       # gated: accept the license on the Hub first
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()           # fit on smaller GPUs

from diffusers import AutoPipelineForImage2Image
pipe = AutoPipelineForImage2Image.from_pretrained(...)
output = pipe(prompt="...", image=init_image, strength=0.75).images[0]

from diffusers import AutoPipelineForInpainting
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
```

**Model hosting note:** the original `runwayml/stable-diffusion-v1-5` repo is no longer the home of that model, so use the community-hosted `stable-diffusion-v1-5/stable-diffusion-v1-5`. A recent Diffusers release (27 Oct 2026) carries a security patch, which is a good reason to be on it rather than on an older line.

**Pipeline types:** text-to-image, image-to-image, inpainting, outpainting, text-to-video, image-to-video, text-to-audio, unconditional generation, and the auto pipelines `AutoPipelineForText2Image`, `AutoPipelineForImage2Image`, `AutoPipelineForInpainting`. Named classes worth knowing: `StableDiffusionPipeline`, `FluxPipeline`, the generic `DiffusionPipeline`, CogVideoX and AnimateDiff for video, and AudioLDM2 for text-to-audio.

**Core concepts:** forward noising and reverse denoising; denoising steps (more steps give better quality but are slower); guidance scale; negative prompts; seeds for reproducibility; latent diffusion in a compressed latent space.

**Schedulers:** DDPM, DDIM, DPMSolverMultistep, Euler, EulerAncestral, PNDM, UniPC, and swapping schedulers on an existing pipeline.

**Notable models and adapters:** SD 1.5, SDXL, SD3, FLUX.1; ControlNet (canny edges, depth, pose, segmentation, scribble); IP-Adapter for image prompts; LoRA for diffusion models; Textual Inversion.

**Memory and speed:** `enable_attention_slicing()`, VAE slicing and tiling, `enable_model_cpu_offload()`, xformers memory-efficient attention, `torch.compile()` for speed, float16 versus bfloat16.

**Topic checklist:**

- Pipeline types: text-to-image, image-to-image, inpainting, outpainting, text-to-video, image-to-video, text-to-audio, unconditional
- Auto pipelines: `AutoPipelineForText2Image`, `AutoPipelineForImage2Image`, `AutoPipelineForInpainting`
- Core concepts: forward noising, reverse denoising, denoising steps, guidance scale, negative prompts, seeds, latent diffusion
- Schedulers: DDPM, DDIM, DPMSolverMultistep, Euler, EulerAncestral, PNDM, UniPC, and swapping them on an existing pipeline
- Models: SD 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`, not the old `runwayml` repo), SDXL, SD3, FLUX.1 (gated, accept the license first)
- ControlNet: canny edges, depth, pose, segmentation, scribble
- IP-Adapter for image prompts
- LoRA for diffusion models; Textual Inversion
- Memory and speed: `enable_attention_slicing()`, VAE slicing and tiling, `enable_model_cpu_offload()`, xformers, `torch.compile()`, fp16 versus bf16

**Install:**

```bash
pip install -U diffusers
pip install -U "diffusers[torch]"
```

### **6. Specialization C: Vision Classification and Features**

**For:** image classification, detection backbones, feature extraction, transfer learning.

**Overview:** timm offers 1000+ pretrained vision models: ResNet, ViT, EfficientNet, Swin Transformer, ConvNeXt, and more.

```python
import timm

timm.list_models("resnet*")
timm.list_models("vit*", pretrained=True)

model = timm.create_model("resnet50", pretrained=True)
model = timm.create_model("vit_base_patch16_224", pretrained=True)

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=10)

model = timm.create_model("resnet50", pretrained=True, num_classes=0)
features = model(image_tensor)   # (batch, 2048) embeddings

backbone = timm.create_model("resnet50", pretrained=True,
                             features_only=True, out_indices=(1, 2, 3, 4))

data_config = timm.data.resolve_model_data_config(model)
transforms  = timm.data.create_transform(**data_config, is_training=False)
```

**Architecture families:** ResNet family (ResNet18, 34, 50, 101, 152), EfficientNet (B0 to B7), ViT, Swin, ConvNeXt, DenseNet, MobileNet, RegNet, MaxViT, DINOv2-based models, EfficientViT.

**Operations:** `timm.create_model()`, `pretrained=True`, `num_classes`, `num_classes=0` for feature extraction, `global_pool`, custom classifier heads, partial fine-tuning (freeze the trunk, train the head), `features_only=True` with `out_indices`.

**Preprocessing:** `resolve_model_data_config()` and `create_transform()` always give the right mean, std, and input size per model.

**Topic checklist:**

- `timm.list_models()` and filtering by architecture pattern and `pretrained=True`
- `timm.model_info()`
- Architecture families: ResNet, EfficientNet, ViT, Swin, ConvNeXt, DenseNet, MobileNet, RegNet, MaxViT, DINOv2-based models, EfficientViT
- `timm.create_model()`, `num_classes`, `num_classes=0` for feature extraction
- `global_pool` options and custom classifier heads
- Partial fine-tuning: freeze the trunk, train the head
- `resolve_model_data_config()` and `create_transform()` for correct normalization and input size
- `features_only=True` with `out_indices` for detection and segmentation backbones

**Install:**

```bash
pip install -U timm
```

### **7. Audio and Vision Specialisation**

**ASR LoRA:** the most reliable of the specialisations because the metric is unambiguous.

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

For current ASR baselines, compare against Nemotron 3.5 ASR Streaming (0.6B) and Nemotron 3.5 Transcribe, Voxtral Mini 3B (Apache-2.0, llama.cpp-native), and Granite Speech 4.1 2B (5.33% WER, Apache-2.0). If your Whisper fine-tune does not beat these on your own audio, the adapter is not earning its keep.

**TTS fine-tuning:** targets are `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and `Qwen/Qwen3-TTS-12Hz-0.6B-Base`. The `Qwen/Qwen3-TTS-Tokenizer-12Hz` ships openly under Apache-2.0 alongside them. The published training recipe is DPO on speech preference pairs, then rule-based rewards with GSPO for capability and stability, then a lightweight speaker fine-tune.

**Diffusion LoRA:** it targets the FLUX attention projections, trains in 200 to 1000 steps for style and 500 to 2000 for DreamBooth, and needs at least 24 GB VRAM at BF16. A recent Diffusers release added FLUX.3 Image Klein, Fibo 2.0, Kandinsky 6, SANA-WM, and RFD4-Proteina, and raised its Python floor to 3.11.

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
        "single_transformer_blocks.*.attn.to_q",
        "single_transformer_blocks.*.attn.to_k",
        "single_transformer_blocks.*.attn.to_v",
    ],
    bias="none",
)
```

**Vision-language LoRA:** it freezes the vision encoder by default and trains the LLM side. Enabling the projector is the middle option and full unfreezing is the expensive one. Unsloth's `FastVisionModel` path with `finetune_vision_layers=False` is the lowest-risk starting point, and `UnslothVisionDataCollator` is required for conversational vision datasets.

### **8. Specialisation Selection**

| **If your work is...** | **Learn** | **Skip** |
|---|---|---|
| Retrieval quality | Embedder training, hard negatives, Matryoshka | Diffusion, TTS |
| Document understanding | VLM LoRA, projector tuning, page-level OCR data | Whisper |
| Voice interfaces | Qwen3-TTS speaker fine-tuning, Whisper LoRA | Diffusion |
| Image generation | FLUX LoRA, DreamBooth, ControlNet | Whisper, TTS |
| Latency-critical serving | KV cache work (Stage 10) | Diffusion |
| Model quality at the frontier | No specialisation, Stage 05 covers it | Everything here |

### **9. The Low-Level Layer: bitsandbytes, kernels, Triton, CUDA**

**Overview:** learn this when you hit GPU memory or speed limitations.

**bitsandbytes:** see Stage 06 for the quantization and optimizer checklist.

**bitsandbytes topic checklist:**

- 8-bit quantization: `BitsAndBytesConfig(load_in_8bit=True)`
- 4-bit quantization: `BitsAndBytesConfig(load_in_4bit=True)`
- NF4 versus FP4, and when each wins
- Double quantization of the quantization constants
- Compute dtypes: float16 for older GPUs, bfloat16 for A100/H100 and newer, float32 for CPU
- `BitsAndBytesConfig` passed as `quantization_config` in Transformers
- `device_map="auto"` with quantized models
- `prepare_model_for_kbit_training()` before attaching PEFT adapters
- 8-bit optimizers: `bnb.optim.Adam8bit`, `bnb.optim.AdamW8bit`, paged `paged_adamw_8bit` / `paged_adamw_32bit`
- `optim="paged_adamw_8bit"` in `TrainingArguments`

```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)
```

**HuggingFace `kernels` library:**

```python
from kernels import get_kernel

kernel = get_kernel("kernels-community/flash-attn2")
# each kernel repo documents its own functions on its model card,
# for example flash_attn_func and flash_attn_varlen_func
# out = kernel.flash_attn_func(q, k, v, causal=True)
```

Kernels are built for specific torch and CUDA versions. If `get_kernel` fails, check the kernel repo's compatibility notes or pin the `kernels` package version. Transformers can also load community kernels directly, for example `attn_implementation="kernels-community/flash-attn2"`. To write and publish your own kernels, study the `kernel-builder` tooling in the Kernels docs.

**kernels topic checklist:**

- What a Hub-hosted kernel is, and why version compatibility matters
- `get_kernel()`, for example `kernels-community/flash-attn2`
- Reading each kernel repo's model card for its available functions
- Loading community kernels through `attn_implementation`
- Writing and publishing your own kernels with `kernel-builder`
- Pinning `kernels` when a build fails against your torch/CUDA versions

**Triton, Python-based GPU programming:**

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

**Triton checklist:**

- Why Triton: write GPU kernels in Python instead of C++
- `@triton.jit`, `tl.program_id()`, `tl.arange()`
- `tl.load()` and `tl.store()` with masks
- `BLOCK_SIZE` as `tl.constexpr`
- Launching with a grid function and `triton.cdiv()`
- `triton.autotune()` for automatic config search
- `triton.testing.do_bench()` to benchmark
- Fusion: combining multiple ops into one kernel
- Practice order: vector addition, softmax, matrix multiplication, then Flash Attention from scratch

**Raw CUDA with PyTorch:**

```python
from torch.utils.cpp_extension import load_inline

cuda_kernel = """
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""
```

To make this callable from Python you also need a small C++ wrapper that allocates the output tensor, computes the grid and block sizes, and launches the kernel, exposed with `load_inline(name=..., cpp_sources=..., cuda_sources=..., functions=[...])`.

**CUDA checklist:**

- `__global__`, `__device__`, `__host__`
- Launching kernels with grid and block dimensions
- `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- Shared memory (`__shared__`) and `__syncthreads()`
- Atomic operations
- cuBLAS and cuDNN (CNN primitives)
- PyTorch C++ extensions and pybind11

**GPU concepts:**

- Thread/block/grid hierarchy
- Global, shared, register, and constant memory
- Memory coalescing
- Occupancy
- Warps (32 threads)
- Bank conflicts in shared memory
- The roofline model for memory-bound versus compute-bound analysis

**Kernels worth knowing:** Flash Attention and Flash Attention 2 (memory-efficient attention), fused LayerNorm and RMSNorm, fused RoPE (Llama, Mistral, Qwen), Paged Attention (vLLM's serving innovation), fused cross-entropy (lower training memory), Triton matmul (can beat cuBLAS at specific shapes).

**Study order:**

```
1. CUDA by Example (book)
2. Triton tutorials: https://triton-lang.org/main/getting-started/tutorials
3. HuggingFace Kernels docs: https://huggingface.co/docs/kernels
4. PyTorch custom C++ and CUDA extension guide
5. NVIDIA CUDA Programming Guide
6. Flash Attention paper (Dao et al, 2022)
```

**Install:**

```bash
pip install -U kernels
pip install -U triton
pip install -U bitsandbytes
```

### **10. Quantization Theory**

**Weight formats and what each costs an 8B model:**

| **Format** | **Bits** | **Memory (8B)** | **Notes** |
|---|---|---|---|
| BF16 | 16 | 16 GB | Training standard; H100 native |
| FP8 E4M3 | 8 | 8 GB | H100 and newer; needs calibration |
| INT8 AWQ | 8 | 8 GB | Activation-aware; fast inference |
| INT8 GPTQ | 8 | 8 GB | Layer-wise calibration |
| NF4 (QLoRA) | 4 | 4 GB | Training only; not serving |
| INT4 AWQ | 4 | 4 GB | Consumer GPU serving |
| GGUF Q4_K_M | about 4.5 | about 4.5 GB | llama.cpp / Ollama format |
| FP4 NVFP4 | 4 | 4 GB | Blackwell (B100/B200); 2 times FP8 speed |

**Two orthogonal axes:** weight quantization and KV cache quantization are independent. FP8 weights with a 4-bit KV cache is a valid and common production configuration, and choosing one does not commit you to the other. The failure mode to avoid is picking a KV format because the weight format "felt compressed".

### **11. Labs**

**Lab 1, embedding and reranking pipeline:**

- Encode 10K documents with `Qwen3-Embedding-4B` and build a FAISS index.
- Rerank the top 100 with `Qwen3-Reranker-4B` and the top 20 with the same model.
- Measure Recall@5 for: retrieval only, rerank-top-20, rerank-top-100.
- Confirm the published finding on your own data: reranking too few candidates can hurt.

**Lab 2, fine-tune an embedder:**

- Fine-tune with `MultipleNegativesRankingLoss` on your domain pairs.
- Add hard negatives mined from the current model's top-k mistakes.
- Evaluate with `InformationRetrievalEvaluator` before and after.

**Lab 3, diffusion memory ladder:**

- Generate at batch 1 on a 16 GB card; record peak memory.
- Apply `enable_attention_slicing()`, then VAE tiling, then `enable_model_cpu_offload()`.
- Record the quality delta at each step and the throughput cost.

**Lab 4, timm transfer learning:**

- Train a head on a small classification dataset with `timm`.
- Compare `resolve_model_data_config()` against hardcoded ImageNet normalization and show the accuracy gap.
- Extract intermediate features with `features_only=True` and `out_indices`.

**Lab 5, write and benchmark a Triton kernel:**

- Write vector add, softmax, and a tiled matmul.
- Benchmark each against the PyTorch equivalent with `triton.testing.do_bench()`.
- Autotune the matmul and report where Triton beats cuBLAS.
- If you go all the way, implement Flash Attention and compare against the Hub kernel.

### **12. Pitfalls**

- **MTEB row order is Borda count, not mean score.** Comparing the wrong column changes your conclusion.
- **Embedding dimension drives storage cost.** A 5,376-dim model is a different order of magnitude in a vector database than a 1,024-dim one.
- **Reranking a short candidate list can reduce accuracy.** Always measure.
- **Diffusers still uses `torch_dtype`**, unlike recent Transformers releases, where the rename to `dtype` has already happened.
- **FLUX.1 is gated.** Accept the license on the Hub before the download works, and do not run it on a repo you have not cleared.
- **Kernel builds are tied to torch and CUDA versions.** A `get_kernel` failure is usually a version mismatch, not a broken repo.
- **`torch.compile` is not supported on ZeroGPU**; ahead-of-time compilation works with recent torch releases.
- **A Triton kernel that is slower on your shapes is not a win.** Benchmark before you optimize.

### **13. Acceptance Gate**

- One specialization shipped end to end with a public artifact.
- For embeddings: a retrieval eval showing a measured gain from the fine-tune.
- For diffusion: an image grid plus the memory/throughput ladder.
- For vision: a training script that uses `resolve_model_data_config()`.
- One Triton kernel benchmarked against the PyTorch equivalent, with numbers.
- A written note on which specialization you chose and which you skipped, and why.

### **14. Resources**

- sentence-transformers release notes: https://github.com/huggingface/sentence-transformers/releases/tag/v6.2.0
- MTEB leaderboard: https://leaderboard.mteb.org/models
- Diffusers release notes: https://github.com/huggingface/diffusers/releases/tag/v0.42.1
- timm on PyPI: https://pypi.org/project/timm/
- kernels release notes: https://github.com/huggingface/kernels/releases/tag/v0.18.1
- bitsandbytes on PyPI: https://pypi.org/project/bitsandbytes/
- Triton on PyPI: https://pypi.org/project/triton/
- torch on PyPI: https://pypi.org/project/torch/
- Hub Skills (includes `train-sentence-transformers`): https://github.com/huggingface/skills

**Next:** Stage 10: Production Serving.