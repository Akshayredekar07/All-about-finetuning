# **Stage 07 — Production Serving**
**Outcome:** a served model with a deployment decision you can defend, a quantization path that is still supported, and a benchmark table.**Stack:** vLLM 0.33.0 · SGLang 0.5.23 · TensorRT-LLM 1.2.1 · GPTQModel 7.8.0 · llm-compressor 0.15.0 · llama.cpp / GGUF · Ollama 0.19+ · MLX-LM 0.32.2 · HF Spaces / Jobs / ZeroGPU · Tinker 0.37.0


---

## **Scope**

A fine-tune that cannot be served is a notebook. Stage 7 covers the decisions that separate a demo from a service: merged or multi-LoRA, which quantization format, which engine, which hardware, and what the latency and throughput actually are.

**Exit criterion:** a served endpoint with a documented deployment decision, a benchmark table (TTFT, TPOT, throughput at batch 1/8/32), and a documented quality delta from quantization.

---

## **The deployment decision**

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

## **Topic checklist — vLLM 0.33.0**

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

## **Topic checklist — SGLang 0.5.23**

- [ ] `--lora-paths` — one or more adapter directories
- [ ] `--max-loras-per-batch` — default 8, including the base model
- [ ] `--max-loaded-loras` — how many stay resident
- [ ] `--lora-backend` — `csgmv` by default, reporting a 20–80% latency improvement on multi-adapter workloads
- [ ] Continuous batching and radix-prefix caching behavior under your traffic pattern

## **Topic checklist — quantization**

- [ ] Why AutoGPTQ is unmaintained and AutoAWQ is deprecated
- [ ] GPTQModel for GPTQ and AWQ work
- [ ] llm-compressor for AWQ and FP8 W8A8 — vLLM's own recommendation
- [ ] `compressed-tensors` as the checkpoint format llm-compressor produces
- [ ] GGUF via llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize`
- [ ] GGUF Q4_K_M retaining roughly 92% of unquantized quality at a quarter of the weight size
- [ ] 4-bit QAT checkpoints (NVFP4 W4A16 for Hopper and Blackwell, native Q4_0 GGUF for llama.cpp and Ollama)
- [ ] FP8 on Hopper and Blackwell
- [ ] Measuring the quality delta on your own domain eval

## **Topic checklist — edge runtimes**

- [ ] llama.cpp with GGUF (CPU, Metal, CUDA)
- [ ] Ollama 0.19+ with the MLX backend on Apple Silicon
- [ ] MLX-LM 0.32.2: LoRA, QLoRA, DoRA, and full fine-tuning on Apple Silicon
- [ ] Quantized 4-bit QAT GGUF builds for phone and laptop deployment

## **Topic checklist — compute**

- [ ] HF Spaces hardware tiers and hourly prices
- [ ] HF GPU Instances (H100, H200, B200) and the fact that H100 is no longer offered in Spaces
- [ ] HF ZeroGPU quotas, tiers, and the `large` / `xlarge` slice sizes
- [ ] HF Jobs: per-minute billing, default 30-minute timeout, retries, reruns, webhooks, storage buckets, sandboxes
- [ ] Tinker for hosted LoRA and RL

---

## **Code**

### **Merge the adapter**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = PeftModel.from_pretrained(base, "./outputs/lora-adapter")
merged = model.merge_and_unload()
merged.save_pretrained("./outputs/merged-model")
```

### **Quantize for a datacenter GPU**

```bash
# **INT4 (GPTQ) and AWQ**
pip install -U gptqmodel

# **FP8 W8A8 and AWQ, the vLLM-recommended path**
pip install -U llmcompressor compressed-tensors
```

```python
# **llm-compressor: the supported AWQ / FP8 path**
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

### **Quantize for edge**

```bash
python convert_hf_to_gguf.py ./outputs/merged-model --outfile model.gguf
./llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
```

### **Serve — vLLM, merged**

```bash
vllm serve ./outputs/merged-awq \
  --tensor-parallel-size 1 \
  --dtype fp16 \
  --max-model-len 4096
```

### **Serve — vLLM, multi-LoRA**

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

The default that will silently serialize you
`--max-loras` defaults to **1**. With multiple adapters in a request stream and the default in place, requests queue instead of running concurrently. Set it to your real concurrency target, and set `--max-lora-rank` to at least the `r` of your adapter (allowed values 1, 8, 16, 32, 64, 128, 256, 320, 512). If your adapter was trained at `r=32` and the server is left at the default 16, the request fails or the adapter is silently downgraded — read the server log rather than assuming.

### **Serve — SGLang**

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --lora-paths ./outputs/lora1 ./outputs/lora2 \
  --max-loras-per-batch 8 \
  --max-loaded-loras 16 \
  --lora-backend csgmv
```

### **Benchmark harness**

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

## **Compute options**

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

## **Evaluation gates**

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

## **Labs**

### **Lab 7-1 — Merged versus multi-LoRA**
- Serve the merged model and the multi-LoRA server on identical hardware.
- Measure TTFT p50/p95/p99, TPOT, throughput at batch 1/8/32, and VRAM.
- Write the decision rule for your workload: how many adapters, how much traffic, what latency budget.

### **Lab 7-2 — Quantization sweep**
- Quantize the merged model to INT4, FP8 (if on Hopper/Blackwell), and GGUF Q4_K_M.
- Measure the domain-eval delta for each, and the throughput and memory changes.
- Report the Pareto frontier, not just the best point.

### **Lab 7-3 — Adapter rank and batch limits**
- Train adapters at `r` = 8, 16, 32.
- Serve with `--max-lora-rank` set below the adapter rank and observe the failure.
- Find the `--max-loras` setting where latency starts to degrade under concurrent adapters.

### **Lab 7-4 — Edge deployment**
- Convert to GGUF and serve with llama.cpp and with Ollama.
- Measure tokens/sec on a laptop and, if possible, on a phone.
- Test the MLX path on Apple Silicon if you have one.

### **Lab 7-5 — Budget run on free compute**
- Run a training job on HF Jobs and an inference demo on ZeroGPU.
- Track wall time, quota consumption, and how long it takes to set up.
- Write down what you would and would not do on this tier.

---

## **Pitfalls**

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

## **Acceptance gate**

- [ ] A deployment decision (merged or multi-LoRA) written with its cost, latency, and memory trade-off.
- [ ] Benchmark table complete: TTFT p50/p95/p99, TPOT p50/p95/p99, throughput at batch 1/8/32, for base, FT-FP16, and FT-quantized.
- [ ] Quantization quality delta measured on the domain eval and stated as a budget.
- [ ] `--max-loras` and `--max-lora-rank` set deliberately and documented.
- [ ] One edge deployment working (GGUF on llama.cpp or Ollama).
- [ ] A fallback compute plan for when the preferred GPU tier is unavailable.

---

## **Resources**

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

**Next:** Stage 08 — Capstones, Portfolio and JD Alignment.
