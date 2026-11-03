# Stage 10 — Production Serving

**Weeks:** 29–30
**Stack:** vLLM 0.33.0 · SGLang 0.5.23 · TensorRT-LLM 1.2.1 · GPTQModel 7.8.0 · llm-compressor 0.15.0 · llama.cpp / GGUF · Ollama 0.19+ · MLX-LM 0.32.2 · HF Spaces / Jobs / ZeroGPU · Tinker 0.37.0
**Outcome:** a served model with a deployment decision you can defend, a quantization path that is still supported, and a benchmark table.

## The deployment decision

```
Trained LoRA adapter
   |
   +--> Multi-LoRA serve (keep adapters separate) --> one base, many tenants
   |
   +--> Merge into base --> quantize --> serve
            |
            +--> INT4 (GPTQModel / llm-compressor) --> datacenter GPU (vLLM / SGLang)
            +--> FP8 (llm-compressor)               --> H100 / H200 / B200
            +--> GGUF Q4_K_M (llama.cpp convert)   --> edge: laptop, Mac, phone
```

**Merge the adapter**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")
model = PeftModel.from_pretrained(base, "./outputs/llama3-alpaca-lora")
merged = model.merge_and_unload()
merged.save_pretrained("./outputs/llama3-alpaca-merged")
```

> [!warning] The source notes' AWQ code no longer works
> `from awq import AutoAWQForCausalLM` is dead: **AutoAWQ is officially deprecated** (last validated against torch 2.6 / transformers 4.51.3) and **AutoGPTQ is unmaintained**. The live path is **GPTQModel 7.8.0** for GPTQ/AWQ work, or **llm-compressor 0.15.0** — vLLM's own recommendation for AWQ and FP8 W8A8 quantization, since the AWQ functionality moved out of vLLM into llm-compressor [14][60].

**Quantize for a datacenter GPU**
```bash
# GPTQModel for GPTQ/AWQ
pip install -U gptqmodel

# llm-compressor for FP8 W8A8 and AWQ
pip install -U llmcompressor compressed-tensors
```

**Quantize for edge (GGUF)**
```bash
# llama.cpp's convert + quantize tooling
python convert_hf_to_gguf.py ./outputs/llama3-alpaca-merged --outfile model.gguf
./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
```
GGUF Q4_K_M retains roughly 92% of the quality of the unquantized model at a quarter of the weight size. Note that **GGUF support in vLLM is experimental** and requires the out-of-tree `vllm-gguf-plugin`; for a production GGUF path, run llama.cpp or Ollama.

**Serve — vLLM 0.33.0** [58]
```bash
# merged + quantized model
vllm serve ./outputs/llama3-alpaca-awq \
  --tensor-parallel-size 1 \
  --dtype fp16 \
  --max-model-len 4096

# multi-LoRA: keep the base and swap adapters
vllm serve meta-llama/Llama-3-8B-Instruct \
  --enable-lora \
  --lora-modules lora1=./outputs/lora1 lora2=./outputs/lora2 \
  --max-loras 8 \
  --max-lora-rank 16
# then call: {"model": "lora1", ...}
```

The LoRA knobs in vLLM, with their real defaults [22][23]:

| Flag | Default | Notes |
|---|---|---|
| `--enable-lora` | off | Enables adapter handling |
| `--max-loras` | **1** | Maximum LoRAs in a single batch — raise it or you serialize tenants |
| `--max-lora-rank` | **16** | Allowed values: 1, 8, 16, 32, 64, 128, 256, 320, 512 — it must cover your adapter's `r` |
| `--lora-dtype` | auto | Defaults to base model dtype |
| `--max-cpu-loras` | = `max_loras` | Host-side adapter cache; must be ≥ `max_loras` |
| `--fully-sharded-loras` | off | Shard all LoRA computation under tensor parallelism; faster at long context, high rank, or large TP |
| `--lora-target-modules` | all supported | Restrict which modules LoRA touches |
| `--specialize-active-lora` | off | Capture separate CUDA graphs per active-adapter count |
| `--default-mm-loras` | none | Per-modality default adapters for multimodal models |

**Serve — SGLang 0.5.23** [57]
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-8B-Instruct \
  --lora-paths ./outputs/lora1 ./outputs/lora2 \
  --max-loras-per-batch 8 \
  --max-loaded-loras 16 \
  --lora-backend csgmv
```
`--max-loras-per-batch` defaults to 8 including the base model; `--lora-backend csgmv` is the default backend and reports a 20–80% latency improvement over the alternative path for multi-adapter workloads.

**Why multi-LoRA is the default.** One base model plus many fine-tuned adapters equals one serving bill, with adapters hot-swapped per request. vLLM, SGLang, TGI, and TensorRT-LLM all support it [63], and the PEFT adapter format is portable across all three — which is why keeping the adapter separate from the merged weights is a deployment decision, not just a training detail.

**Local and edge runtimes**
- **llama.cpp** — GGUF, CPU/Metal/CUDA, the reference edge path.
- **Ollama 0.19+** — packaged local inference with an MLX backend on Apple Silicon.
- **MLX-LM 0.32.2** — LoRA, QLoRA, DoRA, and full fine-tuning on Apple Silicon, which makes a Mac Studio a legitimate fine-tuning box for a 70B QLoRA [44].
- **4-bit QAT checkpoints** — NVFP4 W4A16 builds for Hopper and Blackwell, and native Q4_0 GGUF builds for llama.cpp and Ollama, are published for Qwen3.5-4B, Qwen3.8-27B, and Gemma 4 E4B/12B; the Gemma E4B Q4_0 build reports 2× lower KL against the unquantized model than Google's own QAT, and the NVFP4 build reports +35% output throughput over BF16 on H100 [66].

## Compute options when you have no GPU

Cost and VRAM per option, for a 7B–13B QLoRA and for a served endpoint.

| Option | Typical cost | Shape |
|---|---|---|
| HF Spaces, T4 small | $0.40/h, 16 GB | Small QLoRA runs, demos [45][55] |
| HF Spaces, L4 | $0.80/h, 24 GB | 7B–13B QLoRA [45] |
| HF Spaces, A10G large | $1.50/h, 24 GB | Same, larger RAM [45] |
| HF Spaces, A100 large | $2.50/h, 80 GB | 13B–33B QLoRA, GRPO [45] |
| HF Spaces, RTX PRO 6000 | $2.75/h, 96 GB | Largest Spaces GPU [45][46] |
| HF GPU Instances, H100 | $4.50/h, 80 GB | Dedicated instances, not Spaces [46] |
| HF GPU Instances, H200 / B200 | $5.00/h / $9.25/h | FP8 training and serving [46] |
| HF ZeroGPU | Free tier 5 min/day; PRO 40 min/day; $1 per 10 min of credit | RTX Pro 6000 Blackwell, 48 GB (`large`) or 96 GB (`xlarge`); `xlarge` costs 2× quota [47][55] |
| HF Jobs | Per-minute billing, default 30-minute timeout | Batch training and eval [48] |
| Tinker | Hosted API | LoRA and RL without GPUs [36] |

Notes worth knowing: H100 is **no longer available in Spaces** (removed December 2025) but is available as a GPU Instance [46]. ZeroGPU does not support `torch.compile`, though ahead-of-time compilation works with torch ≥2.8. Free personal accounts can host two ZeroGPU Spaces; PRO ($9/month) raises that to ten and multiplies the daily quota by eight [45][47]. Spaces hardware can also be set programmatically through `huggingface_hub`.

## Evaluation gates for a served model

The source notes accepted "DPO improves MT-Bench by ≥5%" as a capstone criterion. MT-Bench is saturated and effectively unmaintained, and score differences of 0.1–0.2 on it are noise. Replace it with a panel that measures distinct failure modes:

| Gate | What it catches | Note |
|---|---|---|
| IFEval | Instruction-following regressions | Deterministic, so small deltas are meaningful |
| Arena-Hard-Auto v2 | Relative quality against strong baselines | Automated pairwise judging |
| AlpacaEval 2.0 | Length-controlled win rate | Controls the length bias |
| Domain held-out set | Your actual task | Never touched in training |
| Capability regression (MMLU, GSM8K, HumanEval) | Catastrophic forgetting | Report the delta, not the absolute |
| Format compliance | Reward hacking in GRPO | Should be ≥95% |

## Serving engines

| Engine | Best workload | Core idea | Status |
|---|---|---|---|
| **vLLM** | General; bursty traffic | PagedAttention; broadest ecosystem; TurboQuant, PD disaggregation, NIXL | **Default choice.** 0.33.0 (2 Nov 2026) [106] |
| **SGLang** | Shared-prefix, multi-turn, agentic | RadixAttention (trie prefix reuse); EAGLE3/DART integrated | **Default choice.** 0.5.23 [57] |
| **TensorRT-LLM** | Maximum throughput, NVIDIA-only | Engine compilation, FP8/FP4, expert parallelism | NVIDIA-only; measure before adopting |
| **LMDeploy** | Consumer GPUs; INT4 | KV quantization; lightweight | Niche, but strong on small GPUs |
| **llama.cpp** | Edge; CPU; no GPU | GGUF; k-quants; portable | The only sane CPU path |
| **Ollama** | Developer local UX | Zero-config GGUF wrapper with a GUI | A UI over llama.cpp, not a training tool |
| **TGI** | — | — | **Archived 21 Mar 2026.** Read-only; do not deploy anything new on it [115] |

llama.cpp and Ollama are local runtimes, not fine-tuning tools and not part of the training pipeline. Their role in this roadmap is the edge and on-device configuration only. For production serving at scale, use vLLM or SGLang.

**vLLM core concepts:** PagedAttention, continuous batching, automatic prefix caching, chunked prefill (default in V1), FP8, `torch.compile`, and the TurboQuant attention backend.

**SGLang core concepts:** RadixAttention (trie-based prefix caching), EAGLE3 and DART integrated, FlashInfer backend.

## The KV cache, measured

```
KV Cache bytes = 2 × n_layers × n_kv_heads × d_head × dtype_bytes × seq_len

Llama-3-8B (FP16, 32 layers, 8 KV heads, 128 d_head):
= 2 × 32 × 8 × 128 × 2 × seq_len = 131,072 bytes/token ≈ 0.125 MB/token
At 128K context: ~16 GB KV cache (beyond 16 GB model weights)
At 1M context:  ~125 GB — exceeds GPU HBM
```

Everything below exists because that formula, applied to a long-context request, exceeds the card. The ordering is deliberate: each layer buys more than the one above it, and the first three are effectively free.

| Strategy | Mechanism | Status |
|---|---|---|
| **Paged KV (PagedAttention)** | Fixed-size blocks, dynamic allocation, non-contiguous | Universal baseline |
| **Prefix caching / APC** | Hash prefixes; reuse KV for identical prefixes | Standard in vLLM and SGLang; RadixAttention is SGLang's default mechanism |
| **KV quantization (scalar)** | FP8 or INT8 per value | Production-ready, ~2× saving |
| **TurboQuant** | Vector quantization of KV, online and training-free | Shipped in vLLM since 0.20.0 [91][106] |
| **KVTC** | PCA decorrelation + adaptive quantization + entropy coding | ICLR 2026 research reference implementation [92] |
| **KV offloading** | Spill to CPU / NVMe | LMCache, HiCache |
| **PD separation** | Prefill and decode own separate KV pools | Large-scale standard [108] |
| **Architectural compression** | MLA, CSA, HCA | Built into DeepSeek V3/V4 and Qwen hybrid stacks |

### The full KV stack, in the order you should apply it

```
Layer 1: PagedAttention / blocked KV         — required baseline
Layer 2: Prefix caching (APC / RadixAttention) — 85–95% saving on shared prefixes
Layer 3: Scalar KV quant (FP8 / INT8)         — ~2× saving; production-ready
Layer 4a: TurboQuant                          — online, no calibration
Layer 4b: KVTC                                — maximum ratio, brief calibration
Layer 5: KV offloading (CPU/NVMe)             — 1M+ context; LMCache, HiCache
Layer 6: PD separation                        — large-scale; KVTC reduces transfer cost
Layer 7: Architectural (MLA / CSA / HCA)      — built into the model, not optional
```

### TurboQuant, as it actually ships

TurboQuant (arXiv 2504.19874) is an online vector quantisation method for the KV cache: roughly 3.5 bits per value, training-free, data-oblivious, and within `sqrt(3π/2) ≈ 2.7×` of the Shannon lower bound on distortion [91].

What the paper describes is a two-stage algorithm: **PolarQuant**, which multiplies the input by a random orthogonal matrix so coordinates concentrate to a Gaussian and a pre-computed Lloyd-Max codebook applies without per-block normalisation constants; and **QJL error correction**, a 1-bit Johnson-Lindenstrauss residual that removes the systematic bias MSE-optimal quantisers introduce into inner products.

What vLLM actually ships is a subset. TurboQuant was merged upstream through PR #38479 and shipped in **vLLM v0.20.0** as a v1 attention backend, using **Hadamard rotation plus scalar quantisation**, and it **deliberately omits QJL** — the upstream discussion concluded that QJL "hurts attention quality by amplifying variance through softmax" [106]. The shipped presets and their measured costs:

| Flag | KV compression | Perplexity cost |
|---|---|---|
| `--kv-cache-dtype turboquant_k8v4` | 2.6× | +1.17% |
| `--kv-cache-dtype turboquant_4bit_nc` | 3.8× | +2.71% |
| `--kv-cache-dtype turboquant_k3v4_nc` | ~3.5× | +10.63% |
| `--kv-cache-dtype turboquant_3bit_nc` | 4.9× | +20.59% |

This is the correction that matters most in this refresh. The source document's "≥6× reduction with zero measurable accuracy loss" describes the paper's best case under its own evaluation, not the configuration you can enable in vLLM today. The upstream presets trade 1–3% perplexity for a 2.6–3.8× memory win, and the 3-bit preset costs 20%. Run the perplexity check on your own model before enabling any of them. On Apple Silicon, vLLM-metal supports TurboQuant at 2.5–5× [106].

If you want the paper's exact algorithm rather than the shipped approximation, implement it yourself — the OSS implementations from the source document (`0xSero/turboquant`, `hackimov/turboquant-kv`, `Alberto-Codes/turboquant-vllm`) predate the upstream merge and still exist, but the source's advice to run with `--enforce-eager` until a patch lands is now obsolete: upstream ships a native backend rather than a monkey-patch, so `torch.compile` no longer bypasses it.

### KVTC, and when it beats TurboQuant

KVTC (arXiv 2511.01815, accepted ICLR 2026) applies JPEG-style transform coding to the KV cache:

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

The two methods are not competitors; they serve different failure modes.

| Method | Compression | Accuracy | Calibration | Online |
|---|---|---|---|---|
| FP8 KV | ~2× | near-lossless | None | Yes |
| TurboQuant (shipped presets) | 2.6–4.9× | small perplexity cost | None | Yes |
| KVTC (default) | up to 20× | preserves reasoning and long-context accuracy | Brief | No |
| KVTC (aggressive) | 40×+ | noticeable degradation | Brief | No |

```
TurboQuant: no calibration; streaming; online; enough when KV is a throughput problem
KVTC:       maximum compression; multi-turn reuse; PD separation;
            stored KV on NVMe; calibration acceptable
```

The KVTC paper reports up to 20× compression while preserving reasoning and long-context accuracy, with 40×+ in specific cases. It remains a research reference implementation rather than a first-class vLLM feature, so plan on engineering effort if you deploy it [92].

## Prefill and decode

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

**Chunked prefill** splits large prompts into 512-token chunks interleaved with decode steps, which prevents head-of-line blocking where one long prompt stalls every other request. It is the default in vLLM V1.

**Continuous batching** schedules at iteration level: a finished request exits immediately and a new one enters. Near-zero GPU idle time, and 3–10× the throughput of static batching.

## Speculative decoding

A draft model proposes K tokens; the target verifies all K in one pass. It is mathematically lossless — the output distribution is unchanged — so the only question is whether the acceptance rate is worth the draft cost.

| Method | Draft source | Memory | Notes |
|---|---|---|---|
| **EAGLE3** | Draft head on target hidden states | ~400M params | De-facto standard; 2–6× speedup |
| **DART** | Diffusion-inspired non-autoregressive drafter | ~Qwen3-1.7B size | **6.8× lower draft latency than EAGLE3 on H100**, 53.3× vs standard speculative sampling; best for k=8+ draft lengths [93] |
| **Medusa / MTP** | Tree-structured parallel heads | hundreds of M | Multi-token prediction; DeepSeek and Qwen style. Qwen3.8-27B was trained with MTP [70] |
| **DSpark** | Draft head embedded in the target model | 3 MLA layers, 3 tokens/forward, 2048-token sliding window, 4.2 GB BF16 | Ships inside Mistral Large 4 [67] |
| **N-gram** | Statistical from the KV cache | Zero | Repetitive outputs |

DART (arXiv 2601.19278, 27 January 2026) generates all K draft tokens in parallel instead of autoregressively, which is where the latency reduction comes from; the paper reports roughly 30% average end-to-end speedup over EAGLE3 and up to 65% on code workloads, with 2.03–3.44× end-to-end gains and Qwen3 1.7B–32B drafts [93].

Two operational rules. **Fine-tuned targets need their draft head retrained** — a generic EAGLE3 head has a materially lower acceptance rate against an adapter-tuned model. And speculative decoding **hurts** when the batch is larger than 32, when outputs are shorter than 100 tokens, or when the acceptance rate falls below 50%.

## Prefill/decode disaggregation

```
Prefill instances (H100/A100):    Decode instances (A10G/L4):
  Compute-bound prefill             Memory-bandwidth-bound decode
  Build KV cache                    Receive KV blocks over RDMA
  → Transfer KV                      Run continuous batching
```

The proxy ties the pair together. For each request it calls the prefill instance with `max_tokens=1` and `kv_transfer_params: {"do_remote_decode": true}`; the prefill instance computes the KV cache, holds the blocks and returns `kv_transfer_params` pointing at them; the proxy then forwards the original request to the decode instance, which pulls the blocks before generating.

vLLM ships nine connector types upstream, including **NIXL** (NVIDIA's transfer layer over UCX, the default for high-performance PD), **LMCache**, **Mooncake**, **AMD's MoRI-IO**, and a `MultiConnector` that chains them [108][109]. The `--kv-transfer-config` form looks like this:

```bash
--kv-transfer-config '{
  "kv_connector":"NixlConnector",
  "kv_role":"kv_both",
  "kv_buffer_device":"cuda",
  "kv_connector_extra_config":{"backends":["UCX","GDS"]}
}'
```

For multi-turn and agentic workloads, **bidirectional** KV transfer reuses blocks the decode node already holds instead of recomputing them, which the vLLM team reports cuts TTFT by up to 3.3× on long multi-turn prompts. Such a proxy needs a `conversation_id` in each request body to scope the cache across turns; without it the proxy cannot link turns and silently falls back to no-cache behaviour [111].

Two practical notes. A push-based NIXL connector now writes KV directly into decode memory over RDMA, taking the router relay and decode-side allocation off the TTFT critical path. And LMCache's technical report measures a 1.53–1.84× reduction in mean TTFT and 1.12–1.66× in mean ITL against vLLM's native PD disaggregation across four models, by copying each chunk of KV into a GPU buffer during chunked prefill and transferring it to a matching buffer on the decoder side [112][113].

The connector choice is a fabric decision as much as a software one: NIXL for single-cluster RDMA, LMCache for tiered HBM → DRAM → NVMe with a shared prefix index, Mooncake for a cluster-scale cache pool, MoRI-IO for AMD fabrics [114].

Measure the transfer before you benchmark anything else. If decode's `KV Transfer metrics` line shows an average transfer time in the hundreds of milliseconds, fix the fabric — the number lands on TTFT and caps throughput, and a misconfigured network usually slows collectives too.

## Quantization in production

```
Interactive chat (latency-sensitive):
  Weights:    FP8 (H100) or INT4 AWQ (A100/consumer)
  KV:         TurboQuant preset with the smallest perplexity cost
  Speculative: EAGLE3 or DART if batch < 8
  Engine:     SGLang (RadixAttention for multi-turn)

High-throughput batch inference:
  Weights:    FP8 or INT4 AWQ
  KV:         TurboQuant or KVTC if maximum compression is needed
  Speculative: off (high concurrency)
  Engine:     vLLM or TensorRT-LLM; chunked prefill

Edge / on-device:
  Weights:    GGUF Q4_K_M (llama.cpp / Ollama)
  KV:         standard
  Engine:     llama.cpp or Ollama

Multi-turn / agentic (long history):
  Weights:    FP8 or INT4 AWQ
  KV:         KVTC, the multi-turn KV reuse case
  Engine:     SGLang (RadixAttention) + LMCache for the offload tier
```

Mistral Large 4, as a concrete reference for a frontier open-weight model at serving time: GGUF builds run from UD-IQ1_S at 233 GB through UD-IQ1_M at 260 GB to BF16 at 2.1 TB, with FP8 and NVFP4 variants for Blackwell; vLLM is the recommended engine [67].

## Serving pre-flight checklist (November 2026)

- [ ] The engine is vLLM or SGLang. TGI is archived and is never the answer for a new deployment.
- [ ] `transformers` in the serving environment satisfies the engine's constraint (`>=5.16.1,<5.20.0` for vLLM 0.32.0+).
- [ ] Chunked prefill is on, and prefix caching is on for any RAG or system-prompt-heavy workload.
- [ ] KV cache memory is budgeted for the target context length using the formula, not estimated.
- [ ] A KV compression preset has been measured for perplexity drift on your own model, not assumed lossless.
- [ ] The KV transfer path has been timed, and the fabric has been verified.
- [ ] Speculative decoding is enabled only where the acceptance rate justifies it, and the draft head has been retrained after fine-tuning.
- [ ] Multi-turn workloads carry a `conversation_id` for the proxy.
- [ ] TTFT, TPOT and throughput are measured at the target concurrency and reported.
- [ ] The adapter used for training is the one being served — same hash, same base model revision.

## Serving pre-flight checklist

- [ ] Merged or multi-LoRA? Can you state the cost and latency trade-off?
- [ ] Does `--max-lora-rank` cover the rank of your adapter? Is `--max-loras` above 1?
- [ ] Which quantization format, and why? Does your serving engine support it natively?
- [ ] Is GGUF being served by vLLM (experimental, plugin needed) or by llama.cpp/Ollama (supported)?
- [ ] Have you measured TTFT p50/p95/p99, TPOT, and throughput at batch 1, 8, 32?
- [ ] Is there a quantization quality delta on your domain eval, and what is its budget?
- [ ] Are you serving behind a rate limiter, with a canary, and with observability?

---

---

## Worksheet additions

## Scope
A fine-tune that cannot be served is a notebook. Stage 7 covers the decisions that separate a demo from a service: merged or multi-LoRA, which quantization format, which engine, which hardware, and what the latency and throughput actually are.

**Exit criterion:** a served endpoint with a documented deployment decision, a benchmark table (TTFT, TPOT, throughput at batch 1/8/32), and a documented quality delta from quantization.

---

## The deployment decision
```
Trained LoRA adapter
   |
   +--> Multi-LoRA serve (keep adapters separate) --> one base, many tenants
   |
   +--> Merge into base --> quantize --> serve
            |
            +--> INT4 (GPTQModel / llm-compressor)  --> datacenter GPU (vLLM / SGLang)
            +--> FP8 (llm-compressor)                --> H100 / H200 / B200
            +--> GGUF Q4_K_M (llama.cpp convert)    --> edge: laptop, Mac, phone
```

| Decision | Choose when | Cost |
|---|---|---|
| Merge | One adapter, maximum throughput, simplest ops | Base and adapter versions must stay in lockstep |
| Multi-LoRA | Many adapters over one base, multi-tenant | Per-request overhead, `--max-loras` and `--max-lora-rank` must cover your adapters |
| Quantize | Memory-bound GPU, or cost per token matters | A quality delta you must measure and report |
| Skip quantization | Quality-critical and VRAM allows | 2–4× the VRAM of INT4 |

---

## Topic checklist — vLLM 0.33.0
- [ ] `--enable-lora` — enable adapter handling
- [ ] `--lora-modules` — `name=path` pairs or a JSON list
- [ ] `--max-loras` — maximum LoRAs in a batch; **default 1**
- [ ] `--max-lora-rank` — allowed values 1, 8, 16, 32, 64, 128, 256, 320, 512; **default 16**
- [ ] `--lora-dtype` — `auto` by default, otherwise the base model dtype
- [ ] `--max-cpu-loras` — host-side adapter cache, must be ≥ `--max-loras`
- [ ] `--fully-sharded-loras` — shard all LoRA computation under TP; faster at long context, high rank, or large TP
- [ ] `--lora-target-modules` — restrict which modules LoRA touches
- [ ] `--specialize-active-lora` — separate CUDA graphs per active-adapter count
- [ ] `--default-mm-loras` — per-modality default adapters for multimodal models
- [ ] `--tensor-parallel-size`, `--dtype`, `--max-model-len`
- [ ] TTFT and TPOT measurement, not just requests per second

## Topic checklist — SGLang 0.5.23
- [ ] `--lora-paths` — one or more adapter directories
- [ ] `--max-loras-per-batch` — default 8, including the base model
- [ ] `--max-loaded-loras` — how many stay resident
- [ ] `--lora-backend` — `csgmv` by default, reporting a 20–80% latency improvement on multi-adapter workloads
- [ ] Continuous batching and radix-prefix caching behavior under your traffic pattern

## Topic checklist — quantization
- [ ] Why AutoGPTQ is unmaintained and AutoAWQ is deprecated
- [ ] GPTQModel for GPTQ and AWQ work
- [ ] llm-compressor for AWQ and FP8 W8A8 — vLLM's own recommendation
- [ ] `compressed-tensors` as the checkpoint format llm-compressor produces
- [ ] GGUF via llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize`
- [ ] GGUF Q4_K_M retaining roughly 92% of unquantized quality at a quarter of the weight size
- [ ] 4-bit QAT checkpoints (NVFP4 W4A16 for Hopper and Blackwell, native Q4_0 GGUF for llama.cpp and Ollama)
- [ ] FP8 on Hopper and Blackwell
- [ ] Measuring the quality delta on your own domain eval

## Topic checklist — edge runtimes
- [ ] llama.cpp with GGUF (CPU, Metal, CUDA)
- [ ] Ollama 0.19+ with the MLX backend on Apple Silicon
- [ ] MLX-LM 0.32.2: LoRA, QLoRA, DoRA, and full fine-tuning on Apple Silicon
- [ ] Quantized 4-bit QAT GGUF builds for phone and laptop deployment

## Topic checklist — compute
- [ ] HF Spaces hardware tiers and hourly prices
- [ ] HF GPU Instances (H100, H200, B200) and the fact that H100 is no longer offered in Spaces
- [ ] HF ZeroGPU quotas, tiers, and the `large` / `xlarge` slice sizes
- [ ] HF Jobs: per-minute billing, default 30-minute timeout, retries, reruns, webhooks, storage buckets, sandboxes
- [ ] Tinker for hosted LoRA and RL

---

### Merge the adapter
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = PeftModel.from_pretrained(base, "./outputs/lora-adapter")
merged = model.merge_and_unload()
merged.save_pretrained("./outputs/merged-model")
```

### Quantize for a datacenter GPU
```bash
# INT4 (GPTQ) and AWQ
pip install -U gptqmodel

# FP8 W8A8 and AWQ, the vLLM-recommended path
pip install -U llmcompressor compressed-tensors
```

```python
# llm-compressor: the supported AWQ / FP8 path
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.quantization import QuantizationType

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(
    model="./outputs/merged-model",
    recipe=recipe,
    output_dir="./outputs/merged-awq",
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

### Quantize for edge
```bash
python convert_hf_to_gguf.py ./outputs/merged-model --outfile model.gguf
./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

### Serve — vLLM, merged
```bash
vllm serve ./outputs/merged-awq \
  --tensor-parallel-size 1 \
  --dtype fp16 \
  --max-model-len 4096
```

### Serve — vLLM, multi-LoRA
```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora \
  --lora-modules lora1=./outputs/lora1 lora2=./outputs/lora2 \
  --max-loras 8 \
  --max-lora-rank 16 \
  --max-cpu-loras 8
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "lora1", "messages": [{"role": "user", "content": "Summarize this ticket"}]}'
```

> [!warning] The default that will silently serialize you
> `--max-loras` defaults to **1**. With multiple adapters in a request stream and the default in place, requests queue instead of running concurrently. Set it to your real concurrency target, and set `--max-lora-rank` to at least the `r` of your adapter (allowed values 1, 8, 16, 32, 64, 128, 256, 320, 512). If your adapter was trained at `r=32` and the server is left at the default 16, the request fails or the adapter is silently downgraded — read the server log rather than assuming.

### Serve — SGLang
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --lora-paths ./outputs/lora1 ./outputs/lora2 \
  --max-loras-per-batch 8 \
  --max-loaded-loras 16 \
  --lora-backend csgmv
```

### Benchmark harness
```python
import time, statistics
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "lora1"

def measure(prompt, max_tokens=128, n=30):
    ttfts, tpots = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        stream = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, stream=True,
        )
        first, total, tokens = None, 0, 0
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                if first is None:
                    first = time.perf_counter()
                total += 1
                tokens += 1
        end = time.perf_counter()
        ttfts.append(first - t0)
        tpots.append((end - first) / max(tokens - 1, 1))
    return {
        "ttft_p50": statistics.median(ttfts),
        "ttft_p95": sorted(ttfts)[int(0.95 * len(ttfts))],
        "tpot_p50": statistics.median(tpots),
        "tpot_p95": sorted(tpots)[int(0.95 * len(tpots))],
    }

for batch_size in (1, 8, 32):
    print(batch_size, measure("Summarize this support ticket in one sentence."))
```

---

## Compute options
| Option | Cost | Shape |
|---|---|---|
| HF Spaces, T4 small | $0.40/h, 16 GB | Small QLoRA runs, demos |
| HF Spaces, T4 medium | $0.60/h, 16 GB | Same, more RAM and disk |
| HF Spaces, L4 | $0.80/h, 24 GB | 7B–13B QLoRA |
| HF Spaces, L40S | $1.80/h, 48 GB | 13B–33B QLoRA |
| HF Spaces, A10G small / large | $1.00 / $1.50 per hour, 24 GB | Fine-tuning with headroom |
| HF Spaces, A100 large | $2.50/h, 80 GB | 33B QLoRA, GRPO runs |
| HF Spaces, RTX PRO 6000 | $2.75/h, 96 GB | Largest Spaces GPU |
| HF GPU Instances, T4 | $0.50/h, 14 GB | Cheap dedicated inference |
| HF GPU Instances, H100 | $4.50/h, 80 GB | Dedicated training and serving |
| HF GPU Instances, H200 | $5.00/h, 141 GB | Long-context serving |
| HF GPU Instances, B200 | $9.25/h, 179 GB | FP8 training and serving |
| HF ZeroGPU | Free 5 min/day; PRO 40 min/day; $1 per 10 min of credit | RTX Pro 6000 Blackwell, 48 GB (`large`) or 96 GB (`xlarge`); `xlarge` costs 2× quota |
| HF Jobs | Per-minute billing, 30-minute default timeout | Batch training and evaluation |
| Tinker | Hosted API | LoRA and RL with no GPUs |

**Notes.** H100 is no longer available in Spaces (removed December 2025) but is available as a GPU Instance. ZeroGPU does not support `torch.compile`, though ahead-of-time compilation works with torch ≥2.8. Free personal accounts can host two ZeroGPU Spaces; PRO at $9/month raises that to ten and multiplies the daily quota by eight. Spaces hardware can also be set programmatically through `huggingface_hub`. Free Spaces sleep when idle; paid hardware bills per minute while running.

---

## Evaluation gates
MT-Bench is saturated and effectively unmaintained, and score differences of 0.1–0.2 on it are noise. Gate on a panel instead:

| Gate | What it catches | Why |
|---|---|---|
| IFEval | Instruction-following regressions | Deterministic, so small deltas are meaningful |
| Arena-Hard-Auto v2 | Relative quality against strong baselines | Automated pairwise judging |
| AlpacaEval 2.0 | Length-controlled win rate | Controls the length bias that fools other judges |
| Domain held-out set | Your actual task | Never touched during training |
| MMLU / GSM8K / HumanEval | Catastrophic forgetting | Report the delta, not the absolute |
| Format compliance | Reward hacking in GRPO | Should be at or above 95% |

---

### Lab 7-1 — Merged versus multi-LoRA (1–2 days)
- Serve the merged model and the multi-LoRA server on identical hardware.
- Measure TTFT p50/p95/p99, TPOT, throughput at batch 1/8/32, and VRAM.
- Write the decision rule for your workload: how many adapters, how much traffic, what latency budget.

### Lab 7-2 — Quantization sweep (1–2 days)
- Quantize the merged model to INT4, FP8 (if on Hopper/Blackwell), and GGUF Q4_K_M.
- Measure the domain-eval delta for each, and the throughput and memory changes.
- Report the Pareto frontier, not just the best point.

### Lab 7-3 — Adapter rank and batch limits (0.5 day)
- Train adapters at `r` = 8, 16, 32.
- Serve with `--max-lora-rank` set below the adapter rank and observe the failure.
- Find the `--max-loras` setting where latency starts to degrade under concurrent adapters.

### Lab 7-4 — Edge deployment (1 day)
- Convert to GGUF and serve with llama.cpp and with Ollama.
- Measure tokens/sec on a laptop and, if possible, on a phone.
- Test the MLX path on Apple Silicon if you have one.

### Lab 7-5 — Budget run on free compute (1–2 days)
- Run a training job on HF Jobs and an inference demo on ZeroGPU.
- Track wall time, quota consumption, and how long it takes to set up.
- Write down what you would and would not do on this tier.

---

## Pitfalls
- **`--max-loras` defaults to 1.** Multi-tenant serving without raising it serializes your requests.
- **`--max-lora-rank` defaults to 16.** An adapter trained at a higher rank will not serve until you raise it, and the failure message is not obvious.
- **AutoAWQ and AutoGPTQ are finished.** `from awq import AutoAWQForCausalLM` is the old path; use GPTQModel or llm-compressor.
- **GGUF in vLLM is experimental** and needs the out-of-tree `vllm-gguf-plugin`. For a supported GGUF path use llama.cpp or Ollama.
- **Quantization quality is workload-specific.** A 2% domain-eval loss may be fine for a chatbot and fatal for a classifier.
- **Distilled models quantize differently.** Re-tune the quantization threshold rather than assuming the teacher's setting carries over.
- **H100 is not a Spaces GPU any more.** Budget accordingly, or use GPU Instances.
- **ZeroGPU quotas are per day and reset on a rolling window.** A long job can exhaust the quota mid-run.
- **TTFT and TPOT are different problems.** Raising `max_num_seqs` helps throughput and can hurt TTFT; measure both.

---

## Acceptance gate
- [ ] A deployment decision (merged or multi-LoRA) written with its cost, latency, and memory trade-off.
- [ ] Benchmark table complete: TTFT p50/p95/p99, TPOT p50/p95/p99, throughput at batch 1/8/32, for base, FT-FP16, and FT-quantized.
- [ ] Quantization quality delta measured on the domain eval and stated as a budget.
- [ ] `--max-loras` and `--max-lora-rank` set deliberately and documented.
- [ ] One edge deployment working (GGUF on llama.cpp or Ollama).
- [ ] A fallback compute plan for when the preferred GPU tier is unavailable.

---

## Resources
- vLLM v0.33.0 — https://github.com/vllm-project/vllm/releases/tag/v0.33.0
- vLLM `serve` CLI (LoRA flags) — https://docs.vllm.ai/en/latest/cli/serve/
- vLLM engine arguments — https://docs.vllm.ai/en/latest/configuration/engine_args/
- SGLang v0.5.23 — https://github.com/sgl-project/sglang/releases/tag/v0.5.23
- TensorRT-LLM v1.2.1 — https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1
- GPTQModel on PyPI — https://pypi.org/project/gptqmodel/
- llm-compressor on PyPI — https://pypi.org/project/llmcompressor/
- HF Spaces overview and hardware — https://huggingface.co/docs/hub/main/spaces-overview
- HF GPU hardware and pricing — https://huggingface.co/docs/hub/spaces-gpus
- HF ZeroGPU — https://huggingface.co/docs/hub/spaces-zerogpu
- HF Jobs — https://huggingface.co/docs/hub/jobs
- HF pricing — https://huggingface.co/pricing
- mlx-lm on PyPI — https://pypi.org/project/mlx-lm/
- 4-bit QAT checkpoints (NVFP4, Q4_0 GGUF) — https://huggingface.co/collections/QUASAR-QAT/all-quasar-models

**Next:** Stage 11 — Capstones, Portfolio and JD Alignment.
