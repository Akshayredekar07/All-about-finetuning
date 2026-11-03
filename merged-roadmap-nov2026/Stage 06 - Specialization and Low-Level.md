# **Stage 06 — Specialization and Low-Level**
**Outcome:** one specialization that matches the target role, plus the performance layer for when you hit a memory or speed wall.**Libraries:** `sentence-transformers` 6.2.0 · `diffusers` 0.42.1 · `timm` 1.0.30 · `bitsandbytes` 0.50.2 · `kernels` 0.18.1 · `triton` 3.9.0 · torch 2.15.0


---

## **Scope**

Stage 6 has two halves that are usually confused. The **specializations** (embeddings, generative vision, classification vision) are career choices — pick the one your target role needs. The **low-level layer** (quantization, kernels, Triton, CUDA) is a tool you reach for when a wall appears, not a curriculum to complete on its own.

**Exit criterion:** one specialization shipped end to end, and one kernel written and benchmarked against the library implementation it replaces.

## **Environment**

```bash
pip install -U sentence-transformers "sentence-transformers[train]"
pip install -U diffusers "diffusers[torch]"
pip install -U timm
pip install -U bitsandbytes kernels triton
```

---

## **Specialization A — Embeddings, search and reranking**

**For:** RAG, semantic search, similarity, clustering, reranking.

### **Topic checklist**

- [ ] Sentence embeddings versus token embeddings
- [ ] Cosine similarity and dot score
- [ ] Bi-encoder versus cross-encoder
- [ ] Pooling strategies: mean, CLS, max
- [ ] Dense versus sparse encoders (`SparseEncoder`, new in v5)
- [ ] `MultiVectorEncoder` (late interaction), new in v6, and `merge()` across all four encoder types
- [ ] `encode()` with `normalize_embeddings`
- [ ] `model.similarity()` similarity matrices
- [ ] `util.semantic_search()`, `util.cos_sim()`, `util.dot_score()`
- [ ] Paraphrase mining and clustering embeddings
- [ ] `CrossEncoder` for reranking; bi-encoder then cross-encoder for RAG
- [ ] `CrossEncoderTrainer` for training rerankers
- [ ] Losses: `CosineSimilarityLoss`, `ContrastiveLoss`, `TripletLoss`, `MultipleNegativesRankingLoss`, `MatryoshkaLoss`
- [ ] Evaluators: `EmbeddingSimilarityEvaluator`, `InformationRetrievalEvaluator`, `RerankingEvaluator`
- [ ] Hard negative mining
- [ ] `push_to_hub()` and the `train-sentence-transformers` Hub skill

### **Model selection, November 2026**

| Use case | Pick |
|---|---|
| Fast general purpose | `all-MiniLM-L6-v2` |
| Higher quality general | `all-mpnet-base-v2` |
| QA search | `multi-qa-MiniLM` |
| Strong retrieval, English | `Qwen3-Embedding-8B` (Apache 2.0, 32K context) or `Qwen3-Embedding-4B` |
| Multilingual | `microsoft/harrier-oss-v1-27b` (MIT, 74.27 on MTEB multilingual v2 — expensive to store at 5,376 dims) |
| Reranking | `Qwen3-Reranker-4B` (0.6367 retrieval average), `bge-reranker-v2-m3`, `Nemotron-3.5-Rerank-8B` |

MTEB v2 scores 41 (English) and 131 (multilingual) tasks and ranks by **Borda count**, so row order is not mean-score order — a frequent source of confusion. Prefer **RTEB** (the maintainers' held-out sets) over MTEB-only claims, and check the live board before committing to a checkpoint.

Reranking is the cheapest quality win in retrieval: top-100 reranking lifted Recall@5 from 0.695 to 0.816 in one independent study, while reranking only the top 20 made it worse. Measure on your own queries.

### **Code**

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
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss)
trainer.train()
model.save_pretrained("./my-embedding-model")
model.push_to_hub("username/my-embedding-model")
```

---

## **Specialization B — Generative vision and audio**

**For:** image generation, video generation, audio synthesis.

### **Topic checklist**

- [ ] Pipeline types: text-to-image, image-to-image, inpainting, outpainting, text-to-video, image-to-video, text-to-audio, unconditional
- [ ] Auto pipelines: `AutoPipelineForText2Image`, `AutoPipelineForImage2Image`, `AutoPipelineForInpainting`
- [ ] Core concepts: forward noising, reverse denoising, denoising steps, guidance scale, negative prompts, seeds, latent diffusion
- [ ] Schedulers: DDPM, DDIM, DPMSolverMultistep, Euler, EulerAncestral, PNDM, UniPC — and swapping them on an existing pipeline
- [ ] Models: SD 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`, not the old `runwayml` repo), SDXL, SD3, FLUX.1 (gated — accept the licence first)
- [ ] ControlNet: canny edges, depth, pose, segmentation, scribble
- [ ] IP-Adapter for image prompts
- [ ] LoRA for diffusion models; Textual Inversion
- [ ] Memory and speed: `enable_attention_slicing()`, VAE slicing and tiling, `enable_model_cpu_offload()`, xformers, `torch.compile()`, fp16 versus bf16

### **Code**

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
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

from diffusers import AutoPipelineForImage2Image
pipe = AutoPipelineForImage2Image.from_pretrained(...)
output = pipe(prompt="...", image=init_image, strength=0.75).images[0]
```

Diffusers 0.42.1 (27 October 2026) carries a security patch — a reason to be on it rather than on an older 0.4x line.

---

## **Specialization C — Vision classification and features**

**For:** image classification, detection backbones, feature extraction, transfer learning.

### **Topic checklist**

- [ ] `timm.list_models()` and filtering by architecture pattern and `pretrained=True`
- [ ] `timm.model_info()`
- [ ] Architecture families: ResNet, EfficientNet, ViT, Swin, ConvNeXt, DenseNet, MobileNet, RegNet, MaxViT, DINOv2-based models, EfficientViT
- [ ] `timm.create_model()`, `num_classes`, `num_classes=0` for feature extraction
- [ ] `global_pool` options and custom classifier heads
- [ ] Partial fine-tuning: freeze the trunk, train the head
- [ ] `resolve_model_data_config()` and `create_transform()` for correct normalization and input size
- [ ] `features_only=True` with `out_indices` for detection and segmentation backbones

### **Code**

```python
import timm

timm.list_models("resnet*")
timm.list_models("vit*", pretrained=True)

model = timm.create_model("resnet50", pretrained=True)
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=10)
model = timm.create_model("resnet50", pretrained=True, num_classes=0)
features = model(image_tensor)   # (batch, 2048)

backbone = timm.create_model("resnet50", pretrained=True,
                             features_only=True, out_indices=(1, 2, 3, 4))

data_config = timm.data.resolve_model_data_config(model)
transforms  = timm.data.create_transform(**data_config, is_training=False)
```

---

## **The low-level layer**

### **Bitsandbytes 0.50.2**

- [ ] 8-bit quantization — `BitsAndBytesConfig(load_in_8bit=True)`
- [ ] 4-bit quantization — `BitsAndBytesConfig(load_in_4bit=True)`
- [ ] NF4 versus FP4, and when each wins
- [ ] Double quantization of the quantization constants
- [ ] Compute dtypes: float16 for older GPUs, bfloat16 for A100/H100 and newer, float32 for CPU
- [ ] `BitsAndBytesConfig` passed as `quantization_config` in Transformers
- [ ] `device_map="auto"` with quantized models
- [ ] `prepare_model_for_kbit_training()` before attaching PEFT adapters
- [ ] 8-bit optimizers: `bnb.optim.Adam8bit`, `bnb.optim.AdamW8bit`, paged `paged_adamw_8bit` / `paged_adamw_32bit`
- [ ] `optim="paged_adamw_8bit"` in `TrainingArguments`

```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)
```

### **HuggingFace `kernels` 0.18.1**

- [ ] What a Hub-hosted kernel is, and why version compatibility matters
- [ ] `get_kernel()` — for example `kernels-community/flash-attn2`
- [ ] Reading each kernel repo's model card for its available functions
- [ ] Loading community kernels through `attn_implementation`
- [ ] Writing and publishing your own kernels with `kernel-builder`
- [ ] Pinning `kernels` when a build fails against your torch/CUDA versions

```python
from kernels import get_kernel

kernel = get_kernel("kernels-community/flash-attn2")
# **out = kernel.flash_attn_func(q, k, v, causal=True)**
```

### **Triton 3.9.0**

- [ ] Why Triton: write GPU kernels in Python instead of C++
- [ ] `@triton.jit`, `tl.program_id()`, `tl.arange()`
- [ ] `tl.load()` and `tl.store()` with masks
- [ ] `BLOCK_SIZE` as `tl.constexpr`
- [ ] Launching with a grid function and `triton.cdiv()`
- [ ] `triton.autotune()` for automatic config search
- [ ] `triton.testing.do_bench()` to benchmark
- [ ] Fusion: combining multiple ops into one kernel
- [ ] Practice order: vector addition, softmax, matrix multiplication, Flash Attention from scratch

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
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

### **Raw CUDA**

- [ ] `__global__`, `__device__`, `__host__`
- [ ] Launching kernels with grid and block dimensions
- [ ] `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- [ ] Shared memory (`__shared__`) and `__syncthreads()`
- [ ] Atomic operations
- [ ] cuBLAS and cuDNN
- [ ] PyTorch C++ extensions and pybind11
- [ ] Thread/block/grid hierarchy; global, shared, register, constant memory
- [ ] Memory coalescing, occupancy, warps, bank conflicts
- [ ] The roofline model: memory-bound versus compute-bound

```python
from torch.utils.cpp_extension import load_inline

cuda_kernel = """
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""
```

To make it callable from Python you also need a C++ wrapper that allocates the output tensor, computes the grid and block sizes, and launches the kernel, exposed with `load_inline(name=..., cpp_sources=..., cuda_sources=..., functions=[...])`.

### **Kernels worth knowing**

Flash Attention and Flash Attention 2; fused LayerNorm and RMSNorm; fused RoPE (Llama, Mistral, Qwen); Paged Attention (vLLM's serving innovation); fused cross-entropy (lower training memory); Triton matmul (can beat cuBLAS at specific shapes).

### **Study order**

```
1. CUDA by Example (book)
2. Triton tutorials: https://triton-lang.org/main/getting-started/tutorials
3. HuggingFace Kernels docs: https://huggingface.co/docs/kernels
4. PyTorch custom C++ and CUDA extension guide
5. NVIDIA CUDA Programming Guide
6. Flash Attention paper (Dao et al, 2022)
```

---

## **Labs**

### **Lab 6-1 — Embedding + reranking pipeline**
- Encode 10K documents with `Qwen3-Embedding-4B` and build a FAISS index.
- Rerank the top 100 with `Qwen3-Reranker-4B` and the top 20 with the same model.
- Measure Recall@5 for: retrieval only, rerank-top-20, rerank-top-100.
- Confirm the published finding on your own data — reranking too few candidates can hurt.

### **Lab 6-2 — Fine-tune an embedder**
- Fine-tune with `MultipleNegativesRankingLoss` on your domain pairs.
- Add hard negatives mined from the current model's top-k mistakes.
- Evaluate with `InformationRetrievalEvaluator` before and after.

### **Lab 6-3 — Diffusion memory ladder**
- Generate at batch 1 on a 16 GB card; record peak memory.
- Apply `enable_attention_slicing()`, then VAE tiling, then `enable_model_cpu_offload()`.
- Record the quality delta at each step and the throughput cost.

### **Lab 6-4 — timm transfer learning**
- Train a head on a small classification dataset with `timm`.
- Compare `resolve_model_data_config()` against hardcoded ImageNet normalization — show the accuracy gap.
- Extract intermediate features with `features_only=True` and `out_indices`.

### **Lab 6-5 — Write and benchmark a Triton kernel**
- Write vector add, softmax, and a tiled matmul.
- Benchmark each against the PyTorch equivalent with `triton.testing.do_bench()`.
- Autotune the matmul and report where Triton beats cuBLAS.
- If you go all the way, implement Flash Attention and compare against the Hub kernel.

---

## **Pitfalls**

- **MTEB row order is Borda count, not mean score.** Comparing the wrong column changes your conclusion.
- **Embedding dimension drives storage cost.** A 5,376-dim model is a different order of magnitude in a vector database than a 1,024-dim one.
- **Reranking a short candidate list can reduce accuracy.** Always measure.
- **Diffusers still uses `torch_dtype`**, unlike Transformers 5 — a rename that has not happened there yet.
- **FLUX.1 is gated.** Accept the licence on the Hub before the download works, and do not run it on a repo you have not cleared.
- **Kernel builds are tied to torch and CUDA versions.** A `get_kernel` failure is usually a version mismatch, not a broken repo.
- **`torch.compile` is not supported on ZeroGPU**; ahead-of-time compilation works with torch ≥2.8.
- **A Triton kernel that is slower on your shapes is not a win.** Benchmark before you optimize.

---

## **Acceptance gate**

- [ ] One specialization shipped end to end with a public artifact.
- [ ] For embeddings: a retrieval eval showing a measured gain from the fine-tune.
- [ ] For diffusion: an image grid plus the memory/throughput ladder.
- [ ] For vision: a training script that uses `resolve_model_data_config()`.
- [ ] One Triton kernel benchmarked against the PyTorch equivalent, with numbers.
- [ ] A written note on which specialization you chose and which you skipped, and why.

---

## **Resources**

- sentence-transformers v6.2.0 — https://github.com/huggingface/sentence-transformers/releases/tag/v6.2.0
- MTEB leaderboard — https://leaderboard.mteb.org/models
- Diffusers v0.42.1 — https://github.com/huggingface/diffusers/releases/tag/v0.42.1
- timm on PyPI — https://pypi.org/project/timm/
- kernels v0.18.1 — https://github.com/huggingface/kernels/releases/tag/v0.18.1
- bitsandbytes on PyPI — https://pypi.org/project/bitsandbytes/
- Triton on PyPI — https://pypi.org/project/triton/
- torch on PyPI — https://pypi.org/project/torch/
- Hub Skills (includes `train-sentence-transformers`) — https://github.com/huggingface/skills

**Next:** Stage 07 — Production Serving.
