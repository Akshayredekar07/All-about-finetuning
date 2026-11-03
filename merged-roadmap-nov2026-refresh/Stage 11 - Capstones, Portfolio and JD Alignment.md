# **Stage 11 — Capstones, Portfolio and JD Alignment**

**Outcome:** four public artifacts with real numbers, a JD mapping that names your gaps, and an interview bank you can answer cold.

**The rule:** every skill in this roadmap should produce a public artifact. Every capstone must be public on GitHub with a clean README, real numbers, a stated baseline, reproduction instructions, and a benchmark rather than a claim.

## **Capstone 1 — Domain QLoRA fine-tune, quantize, serve (3 weeks)**

**Goal.** Take a base model, QLoRA fine-tune it on a real domain dataset, merge, quantize to INT4, serve with vLLM, and benchmark end to end.

**What you build**
- A domain dataset (1K–5K samples, public on the Hub with a data card).
- A QLoRA training script (Unsloth, or TRL + PEFT for the writeup).
- The merged FP16 model and the INT4 quantized model.
- A vLLM server serving the quantized model.
- A benchmark script measuring eval scores, TTFT, TPOT, and throughput.
- A comparison report: base model vs FT-FP16 vs FT-INT4.

**Stack.** Base model: `Qwen/Qwen3-4B-Instruct-2507` for a text-only path, `Qwen/Qwen3.8-27B` (QLoRA at 24 GB) for a multimodal path, or `Qwen/Qwen3.5-4B` / `google/gemma-4-E4B` (Apache 2.0, multimodal) when you want a vision-capable checkpoint — all three train on a single 24 GB GPU with QLoRA [21][64][65]. Training: Unsloth or TRL + PEFT. Quantization: GPTQModel or llm-compressor. Serving: vLLM 0.33.0. Quality evals: `lm-evaluation-harness` 0.4.13 plus your domain eval.

**Benchmarks to report**
- Training time and peak memory during training.
- Eval scores (MMLU, your domain eval) for base, FT-FP16, FT-INT4.
- TTFT p50, p95, p99 for each.
- TPOT p50, p95, p99 for each.
- Throughput at batch 1, 8, 32 for each.
- Model size on disk for each.

**Acceptance criteria**
- [ ] Public dataset on the Hub with a data card.
- [ ] Training script runs in under 4 hours on a single A100/H100 (or A10G/L4 for the small base).
- [ ] Quantization quality loss under 2% on the domain eval.
- [ ] Throughput improvement of INT4 over FP16 of at least 1.5×.
- [ ] Numbers reproducible from the README.

## **Capstone 2 — DPO for preference alignment (2–3 weeks)**

**Goal.** Take an SFT model (or Capstone 1's output), apply DPO on a preference dataset, and prove it improves alignment without losing capability.

**What you build**
- An SFT base model.
- A preference dataset (UltraFeedback subset or your own).
- A DPO training script (TRL `DPOTrainer`, TRL 1.17.1).
- Evaluation: IFEval, Arena-Hard-Auto v2, your domain eval, capability regression evals.
- A writeup comparing SFT-only versus SFT+DPO.

**Benchmarks to report**
- DPO training loss curve (chosen versus rejected reward margin).
- IFEval and domain eval score: SFT versus SFT+DPO.
- Capability regression: SFT versus SFT+DPO on MMLU, HumanEval.
- Qualitative examples: 5–10 prompts where DPO helped and 5 where it did not.
- Reference model comparison: DPO with a separate reference model versus LoRA-as-reference.

**Acceptance criteria**
- [ ] DPO improves the primary alignment gate (IFEval / AlpacaEval 2.0) by a margin larger than judge noise — report the delta and the judged sample size, not a percentage.
- [ ] Capability regression on MMLU under 1%.
- [ ] No catastrophic forgetting; capability evals still in the normal range.
- [ ] Cleanly visualized loss curves and eval bar charts.
- [ ] Reproducible from the README.

## **Capstone 3 — GRPO for reasoning improvement (3–4 weeks)**

**Goal.** Take an SFT base model, apply GRPO with a verifiable reward on a reasoning task (math or code), and prove improvement on the target task.

**What you build**
- An SFT base model small enough for one GPU (Qwen3.5-4B, Gemma 4 E2B, or Qwen3-4B-Instruct-2507).
- A reasoning task with a verifiable reward (GSM8K, MATH subset, HumanEval, MBPP).
- A GRPO training script (TRL `GRPOTrainer`).
- A programmatically verifiable reward function.
- Evaluation: held-out test set, capability regression, qualitative examples.

**The reward function**
```python
import re

def math_reward(completions, answer, **kwargs):
    """Reward: 1.0 if completion contains the correct answer, 0.0 otherwise."""
    rewards = []
    for completion, correct_answer in zip(completions, answer):
        match = re.search(r"\\boxed\{([^}]+)\}", completion)
        if match:
            try:
                rewards.append(1.0 if abs(float(match.group(1)) - float(correct_answer)) < 1e-3 else 0.0)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards
```

**Benchmarks to report**
- GRPO training curve (mean reward, group variance).
- Held-out test accuracy: SFT versus SFT+GRPO.
- Capability regression: SFT versus SFT+GRPO on MMLU and IFEval.
- Format compliance: does the model actually use `\boxed{}` after GRPO?
- Qualitative examples: 10 reasoning traces before and after.
- Compute cost: G=8 rollouts × 1000 steps = 8,000 generations; report GPU-hours.

**Acceptance criteria**
- [ ] GRPO improves held-out test accuracy by at least 10% over SFT-only.
- [ ] Format compliance rate at or above 95% after GRPO.
- [ ] Capability regression on MMLU under 3%.
- [ ] No reward hacking — the model is not gaming the format.
- [ ] Cleanly visualized curves and a sample of reasoning traces.
- [ ] Reproducible from the README.

## **Bonus capstone — Multi-LoRA serving (1 week)**

**Goal.** Train three LoRA adapters on three tasks, serve them with vLLM's multi-LoRA, hot-swap per request, benchmark the overhead.

**Artifacts:** three small adapters (for example "summarize", "translate to French", "extract JSON"); a vLLM server with `--enable-lora`; a routing client; a benchmark of hot-swap overhead versus single-adapter serving.

**Benchmarks:** latency overhead of multi-LoRA versus single-LoRA; memory usage of three adapters versus one; throughput at batch 1, 8, 32.

## **Capstone sequence**

Order the three capstones plus the bonus around the calendar below.

| # | Capstone | Stage | Target window |
|---|---|---|---|
| 1 | Domain QLoRA + quantize + serve | Stages 2, 3, 7 | Dec 2026 |
| 2 | DPO for preference alignment | Stage 5 | Jan 2027 |
| 3 | GRPO for reasoning | Stage 5 | Feb–Mar 2027 |
| Bonus | Multi-LoRA serving | Stage 10 | Any time |

All three should be public on GitHub before the target application window. Combined with the inference-side capstones from the sibling roadmap, that is eight to nine public artifacts.

## **JD alignment**

Which role asks for which skill, and which capstone answers it.

| JD source | Required skill | Covered by |
|---|---|---|
| Anthropic Research Engineer, Agents | Agent harnesses, rigorous quantitative evals for agentic tasks, automated model/prompt evaluation across the training and product lifecycle, optimized **data mixes** for agentic training, plus large-scale RL and multi-agent systems | Stages 2, 4, 8 — capstones 1, 2, 3 [49] |
| AI Engineer (General / Applied) | Fine-tuning, PEFT, model customization | Stage 4 — capstone 1 |
| Multimodal AI Engineer | Fine-tuning, vision-language tuning | Stage 7 — extend capstone 1 to a VLM |
| Inference Engineer | LoRA serving, multi-LoRA, deployment | Stage 10 — bonus capstone |

The Anthropic Research Engineer, Agents posting is the most useful target specification in the source material, because it shows what the market now values in a fine-tuning engineer: **harnesses, evals, and data mixes, not only weights**. The responsibilities read: ideate, develop, and compare agent harnesses (memory, context compression, communication architectures); design and implement rigorous quantitative benchmarks for large-scale agentic tasks; assist with automated evaluation of models and prompts across the training and product lifecycle; help create and optimize data mixes for model training that maximize performance on agentic tasks. Representative projects include designing a novel agent harness that outperforms existing agents, designing a novel eval for multi-agent interaction, building a scaled model-evaluation framework driven by model-based evaluation, and finetuning a model to maximize performance with a particular set of agent tools or harness [49]. Read that list as a specification: the missing pieces in a fine-tuning-only portfolio are **eval infrastructure and agent-harness design**, and Stage 11's eval panels plus the agent-trace data pipeline are where you close them.

**Role-by-role coverage**

*Role 1 — AI Engineer (General / Applied).* Fine-tuning is listed but not as an explicit top skill; the implicit expectation is "you can take a base model and adapt it to a domain." Capstone 1 covers this fully. Questions you can answer: How would you adapt a base model to a domain with no labeled data? LoRA versus QLoRA — when to use which? How do you evaluate a fine-tuned model? What is multi-LoRA serving?

*Role 3 — Multimodal AI Engineer.* Fine-tuning is in the tech stack. Extend Capstone 1 to a VLM with LLaVA-style data. Questions: How would you fine-tune a VLM on a custom image+text dataset? What is different about LoRA on a vision-language model?

*Role 4 — Inference Engineer.* LoRA serving, multi-LoRA, and adapter hot-swap are core. Bonus capstone plus Stage 10. Questions: How do you serve 50 fine-tuned adapters on one base model? What is the memory overhead of multi-LoRA serving?

*Role 5 — Harness / Agent Infrastructure Engineer.* "Fine-tuning for agents" sits in the what's-missing list. Capstones 1, 2, and 3 cover it. Questions: How would you fine-tune a model for agent behavior? What is the difference between SFT and DPO for agent alignment?

## **Interview question bank**

**Foundational**
- [ ] What is LoRA? Why is it preferred over full fine-tuning?
- [ ] What is the difference between LoRA, QLoRA, and DoRA?
- [ ] State the default LoRA config (r, α, target_modules, lr).
- [ ] How much memory does QLoRA save versus full FT on a 70B model?
- [ ] What is the DPO loss? Why is it "direct"?
- [ ] Compare DPO, IPO, KTO, SimPO, ORPO. When does each win?
- [ ] What is GRPO? Why does it beat PPO for reasoning?
- [ ] What is RLVR? Name two examples of verifiable rewards.
- [ ] State the consensus pipeline (SFT → DPO → GRPO).
- [ ] What is the LoRA adapter format? How does it interact with vLLM?

**Practical**
- [ ] How do you pick a fine-tuning framework for a project?
- [ ] For a 7B LoRA on a single RTX 4090, which framework? For 70B on 8×H100?
- [ ] Walk me through the LoRA → merge → quantize → serve pipeline.
- [ ] How do you serve multiple LoRA adapters on one base model?
- [ ] How do you evaluate a fine-tuned model?
- [ ] What is the most common cause of fine-tuning failure? (Hint: data, not algorithm.)
- [ ] How do you decontaminate training data against an eval set?
- [ ] What is the right chat template for Llama-3? For Qwen? For Mistral?
- [ ] How do you merge a LoRA adapter into the base model?
- [ ] How do you convert a merged model to GGUF for edge deployment?

**Depth**
- [ ] Derive the DPO loss from the Bradley-Terry preference model.
- [ ] Explain why GRPO does not need a critic network.
- [ ] What is auxiliary-loss-free load balancing in MoE training?
- [ ] How does Unsloth achieve a ~2× speedup on a single GPU?
- [ ] What is the difference between TRL's `SFTTrainer` and a vanilla training loop?
- [ ] How does FSDP2 differ from ZeRO-3?
- [ ] What is the difference between sequence packing and FlashAttention's causal masking?
- [ ] How would you fine-tune a VLM?
- [ ] What is the EU AI Act's main impact on fine-tuning practice, and which dates apply this quarter?
- [ ] How would you detect reward hacking in a GRPO training run?

**Serving and systems (added in the merge)**
- [ ] What are vLLM's `--max-loras` and `--max-lora-rank` defaults, and why do they matter?
- [ ] Which quantization path is supported today, and which two are deprecated?
- [ ] When is merged deployment better than multi-LoRA, and when is it worse?
- [ ] How do you choose a compute tier for a 7B QLoRA on a budget?

## **Interviewer question list**

- Does the team own model fine-tuning, or is it outsourced to a vendor?
- What is the team's preferred fine-tuning framework? (Surfaces tech-stack alignment.)
- Does the team do DPO/GRPO in-house, or rely on closed APIs?
- How are fine-tuned models served — multi-LoRA, merged, quantized?
- What is the eval pipeline for a new fine-tuned checkpoint?
- How does the team handle catastrophic forgetting and capability regression?
- What is the data governance process, and how is EU AI Act compliance handled?
- What is the agent harness, and how is agent behavior evaluated?
- How are data mixes constructed and tested?
- What is the compute budget and the latency budget for a new model release?

## **LLMOps pipeline**

### **Data pipeline**

```
1. Collection → provenance (source, license, date)
2. Cleaning  → PII removal (Presidio), toxicity filter, near-dedup (MinHash LSH)
3. Quality   → perplexity, reward model score, diversity, spot-check
4. Synthetic → 70/30 synthetic/real mix (2026 standard)
              → self-instruct, back-translation, SPIN
              → generate 5–10×, filter to the best 20–30%
```

The 70/30 synthetic/real ratio is a starting point, not a rule, and it is worth remembering that synthetic data generated by a model inherits that model's biases — which is why the filtering step, not the generation step, is where the quality is won.

### **Experiment tracking**

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

Log the **token count** of the training set and the exact base-model revision alongside the loss curve. When a run regresses three weeks later, those two fields are usually the answer.

### **Evaluation thresholds**

| Task | Metric | Competitive 2026 threshold |
|---|---|---|
| Reasoning | AIME 2025/2026, AMC | Track the raw score; compare against the base model's own score, not a leaderboard |
| Hard science | GPQA Diamond | >75% competitive; top open models >84% |
| General knowledge | MMLU-Pro | >65% |
| Math | MATH-500, GSM8K | MATH-500 >75% |
| Code | LiveCodeBench v6, SWE-Bench | LiveCodeBench >60% |
| Agentic coding | SWE-Bench Verified | >55% competitive |
| Long context | RULER, NIAH | Task-dependent; report the curve, not the peak |
| Embedding | MTEB retrieval | >55% nDCG@10 |
| Instruction following | IFEval | Use as a gate, not a goal |

The threshold column is a sanity floor. The number that actually decides whether your fine-tune worked is the **paired delta against the base model on your own held-out set**, with a confidence interval.

### **Deployment with LoRA hot-swap**

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model                  = "Qwen/Qwen3-4B-Instruct-2507",
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

`max_lora_rank` accepts exactly 1, 8, 16, 32, 64, 128, 256, 320 and 512, with 16 as the default, and `--max-loras` defaults to 1 — which means multi-adapter serving has to be turned on explicitly [22][23].

### **Cost levers**

| Lever | Saving | Notes |
|---|---|---|
| QLoRA vs full fine-tune | ~90% of training cost | The default choice |
| bf16 LoRA (Qwen3.5 MoE) | Avoids QLoRA instability | Unsloth MoE fast path; 4-bit is blocked for MoE |
| TurboQuant KV | 2.6–4.9× KV → more concurrency | Training-free; measure the perplexity cost |
| KVTC KV | up to 20× → dense multi-turn | Calibration required; research-grade integration |
| INT4 AWQ weights | 4× → more concurrency | A100 and consumer GPUs |
| Spot instances | 60–80% | Requires checkpointing and resume |
| Prefix caching | 85–95% on prefill | RAG and system-prompt-heavy workloads |
| Frontier API for evaluation | — | Use a hosted frontier model as the judge, not as the product |

## **Production readiness checklist**

### **Training**

- [ ] Fine-tune a 7B model with QLoRA on a single GPU (RTX 3090 / 4090 / A6000).
- [ ] Fine-tune Gemma 4 E4B with QLoRA on a 12 GB card (~10 GB VRAM).
- [ ] Fine-tune Qwen3.5-2B with bf16 LoRA, not QLoRA.
- [ ] Fine-tune a Qwen3.5 MoE model with Unsloth bf16 LoRA (~74 GB for 35B-A3B).
- [ ] Fine-tune a 7B model with full LoRA on an A100.
- [ ] Fine-tune a 70B model with FSDP on multi-GPU (8× A100).
- [ ] Run GRPO with a custom verifier (math and code).
- [ ] Run GSPO (`importance_sampling_level="sequence"`) for MoE training stability.
- [ ] Run DPO or SimPO for preference optimisation.
- [ ] Fine-tune at least one VLM (Qwen3.8-27B, Gemma 4, or Qwen2.5-VL).
- [ ] Fine-tune Whisper for domain adaptation.
- [ ] Fine-tune a Qwen3-TTS base model for voice cloning.
- [ ] Train an embedding model with Matryoshka loss.
- [ ] Handle thinking-mode preservation in SFT data (≥75% reasoning examples).

### **Serving**

- [ ] Deploy vLLM with prefix caching and chunked prefill.
- [ ] Configure SGLang for multi-turn workloads with RadixAttention.
- [ ] Measure and report TTFT, TPOT and throughput at the target concurrency.
- [ ] Implement EAGLE3 speculative decoding for low-latency serving.
- [ ] Implement DART speculative decoding and compare it against EAGLE3.
- [ ] Deploy an FP8 weight-quantised model.
- [ ] Enable a TurboQuant KV preset and measure the memory and perplexity delta.
- [ ] Evaluate KVTC for a multi-turn workflow and measure TTFT against prefix recompute.
- [ ] Set up PD disaggregation for a long-output workload, and time the KV transfer.

### **Engineering**

- [ ] Automated evaluation pipeline (GPQA Diamond, LiveCodeBench, AIME — not just loss).
- [ ] Experiment tracking on every training run.
- [ ] PII removal from training data before any run.
- [ ] A/B test the fine-tuned model against the base with statistical significance.
- [ ] Checkpointing, with a tested recovery path from failure.
- [ ] KV cache memory budget analysis for the target deployment.
- [ ] Cost analysis: training GPU hours plus serving cost per million tokens.
- [ ] Licence audit for every model in the stack (MIT vs Apache-2.0 vs Meta Custom vs Modified MIT vs the new ecosystem licences).

### **Understanding**

- [ ] Derive the KV cache memory formula from first principles.
- [ ] Explain the prefill versus decode bottleneck and what optimises each.
- [ ] Explain PagedAttention block allocation with a diagram.
- [ ] Write the GRPO algorithm from scratch without reference.
- [ ] Explain why FSDP is equivalent to ZeRO-3.
- [ ] Explain why MLA reduces the KV cache by roughly 50× in DeepSeek.
- [ ] Explain what the shipped TurboQuant backend does and why it omits QJL.
- [ ] Explain why MSE-optimal quantisers are not unbiased inner-product estimators.
- [ ] Explain the KVTC pipeline (PCA + adaptive quantisation + DEFLATE).
- [ ] Compare TurboQuant and KVTC and select one for a given workload.
- [ ] Explain why DART achieves lower draft latency than EAGLE3.
- [ ] Trace the GRPO → DAPO → Dr.GRPO → GSPO → GRPO-λ lineage.
- [ ] Explain why Qwen3.5 MoE should not use 4-bit QLoRA.
- [ ] Explain DeepSeek V4's CSA + HCA and why it makes 1M-context inference cheap.
- [ ] Explain the orchestrator pattern and why decomposition beats single-agent depth for long-horizon tasks.
- [ ] Choose between GLM-5.3, Kimi K3 and DeepSeek V4 for a given licence constraint.
- [ ] Explain Gemma 4's advantage over Llama 4 Maverick on efficiency per active parameter.

## **Job-market alignment, refreshed**

The target-role note referenced by the source documents pointed at Anthropic's Research Engineer, Agents posting. That posting is now **closed** — the job board entry reads "This job is no longer accepting applications" [127]. The current Anthropic research board lists roles including Pre-training, Post-Training Model Evaluations, and Staff Research Engineer, Multi-Agent Scaling [126].

The role that most closely matches the agentic work in this roadmap is now **Research Engineer, Universes**, which asks for people to build training environments for long-horizon agentic tasks — novel RL environments, simulation systems, large-scale ML infrastructure, and the evaluation harnesses to measure them. It explicitly names sandboxing, containerisation, VM infrastructure and distributed systems as deep expertise [128].

That is a meaningful change of signal, and it points the portfolio in a specific direction:

| Signal | Portfolio implication |
|---|---|
| The agents role closed; the environments role is live | A fine-tune alone is no longer the differentiating artifact |
| "Environments" is now a named discipline | Stage 9 — a tool environment, a verifier, a rubric, and a held-out environment split — is the strongest differentiator in this roadmap |
| Multi-agent scaling is a separate senior track | Orchestrator/sub-agent decomposition (Stage 9) is a differentiator beyond single-agent RL |
| Search and knowledge post-training is a live track | Retrieval is not a side skill: Nemotron 3.5 Embed/Rerank, contamination control, and citation-faithful evaluation are the substrate |
| Rigour language repeats ("hypothesis", "controlled experiment", "statistically significant") | Report deltas with intervals and ablations, not single benchmark scores |

The portfolio implication in one sentence: **build an environment and an evaluation, not just an adapter.** An adapter trained on a public dataset is reproducible in an afternoon; a released tool environment with a rubric, a held-out split and an ablation table is not, and it is what both the agents and environments roles are asking for.

### **Interview question bank, updated for the 2026 landscape**

- Explain the KV cache memory formula and how MLA changes it.
- What did TurboQuant actually ship in vLLM, and what did it leave out?
- Why does GSPO stabilise MoE training when GRPO does not?
- How would you design a verifier for a task with no programmatic ground truth?
- What is the difference between a reward model score and a checklist reward?
- How do you detect expert collapse in an MoE fine-tune?
- Which of your models are you allowed to serve commercially, and under what condition?
- How do you pick a KV compression preset without degrading the model?
- When does speculative decoding make things slower?
- How would you evaluate an agent that uses tools?

### **Interviewer question list additions (November 2026)**

Prepare to be asked: which model did you fine-tune and why that one; what was the baseline; how did you know the fine-tune helped; what broke; what would you do differently; how much did it cost to train and to serve.

## **The before-you-ship checklist**

- [ ] README has the problem, approach, results, and reproduction.
- [ ] Dockerfile builds clean.
- [ ] Reproduction works from a fresh clone.
- [ ] Numbers are real, not theoretical.
- [ ] Baseline comparison is included.
- [ ] Limitations are honest.
- [ ] Dataset is on the Hub with a data card.
- [ ] License is set (Apache 2.0 recommended).
- [ ] The repo is linked from your GitHub profile README.
- [ ] The repo is shared on LinkedIn or X with the key chart.
- [ ] The model card documents intended use, limitations, data provenance, and eval results.

## **Portfolio pre-flight**

- [ ] All capstones are public on GitHub with real numbers.
- [ ] Each README has: problem, approach, results, reproduction.
- [ ] GitHub profile README links the capstones.
- [ ] Resume updated with the new skills and capstones.
- [ ] The interview bank is drilled.
- [ ] The interviewer questions are ready.

---

---

## **Worksheet additions**

## **The rule**
Every skill in this roadmap should produce a public artifact. A skill you cannot show is a skill you cannot interview about.

Every capstone must be:

1. **Public on GitHub** with a clean README.
2. **Full of real numbers** — training loss, eval scores, throughput, latency, memory.
3. **Compared against a stated baseline** ("versus the base model with no fine-tuning").
4. **Reproducible** — Dockerfile, exact environment, exact commands.
5. **Benchmarked, not claimed.**

---

## **Capstone 1 — Domain QLoRA fine-tune, quantize, serve (3 weeks)**
**Goal.** Take a base model, QLoRA fine-tune it on a real domain dataset, merge, quantize to INT4, serve with vLLM, and benchmark end to end.

**What you build**

- A domain dataset (1K–5K samples, public on the Hub with a data card).
- A QLoRA training script (Unsloth, or TRL + PEFT if you want the writeup to teach the internals).
- The merged FP16 model.
- The INT4 quantized model.
- A vLLM server serving the quantized model.
- A benchmark script measuring eval scores, TTFT, TPOT, and throughput.
- A comparison report: base model vs FT-FP16 vs FT-INT4.

**Stack.** Base model: `Qwen/Qwen3-4B-Instruct-2507` for a text-only path, `Qwen/Qwen3.8-27B` (QLoRA at 24 GB) for a multimodal path, or `Qwen/Qwen3.5-4B` / `google/gemma-4-E4B` for a multimodal checkpoint. Training: Unsloth or TRL + PEFT. Quantization: GPTQModel or llm-compressor. Serving: vLLM 0.33.0. Quality evals: `lm-evaluation-harness` 0.4.13 plus your domain eval.

**Benchmarks to report**

- Training time and peak memory.
- Eval scores (MMLU, your domain eval) for base, FT-FP16, FT-INT4.
- TTFT p50, p95, p99 for each.
- TPOT p50, p95, p99 for each.
- Throughput at batch 1, 8, 32 for each.
- Model size on disk for each.

**Acceptance criteria**

- [ ] Public dataset on the Hub with a data card.
- [ ] Training script runs in under 4 hours on a single A100/H100 (an A10G or L4 works for the 4B base).
- [ ] Quantization quality loss under 2% on the domain eval.
- [ ] Throughput improvement of INT4 over FP16 of at least 1.5×.
- [ ] Numbers reproducible from the README.

**Why it matters.** It covers the full vertical — data → train → quantize → serve → benchmark — and demonstrates production thinking rather than training accuracy alone.

---

## **Capstone 2 — DPO for preference alignment (2–3 weeks)**
**Goal.** Take an SFT model (or Capstone 1's output), apply DPO on a preference dataset, and prove it improves alignment without losing capability.

**What you build**

- An SFT base model.
- A preference dataset (UltraFeedback subset or your own).
- A DPO training script (TRL `DPOTrainer`).
- Evaluation: IFEval, Arena-Hard-Auto v2, AlpacaEval 2.0, your domain eval, capability regression evals.
- A writeup comparing SFT-only versus SFT+DPO.

**Benchmarks to report**

- DPO training loss curve (chosen versus rejected reward margin).
- Primary alignment gate: SFT versus SFT+DPO.
- Domain eval score: SFT versus SFT+DPO.
- Capability regression: SFT versus SFT+DPO on MMLU, HumanEval.
- Qualitative examples: 5–10 prompts where DPO helped and 5 where it did not.
- Reference model comparison: separate reference model versus LoRA-as-reference.

**Acceptance criteria**

- [ ] DPO improves the primary alignment gate by a margin larger than judge noise — report the delta and the judged sample size, not a bare percentage.
- [ ] Capability regression on MMLU under 1%.
- [ ] No catastrophic forgetting; capability evals still in the normal range.
- [ ] Cleanly visualized loss curves and eval bar charts.
- [ ] Reproducible from the README.

The acceptance criterion changed, and here is why
The source notes required "DPO improves MT-Bench by ≥5%". MT-Bench is saturated and effectively unmaintained, and published score gaps below roughly 0.1–0.2 are judge noise, so a 5% target is either trivial or unreachable depending on the judge version. The gate is now IFEval plus Arena-Hard-Auto v2 plus AlpacaEval 2.0 plus your domain held-out set — with the sample size reported alongside the score so the reader can judge the margin.

**Why it matters.** Alignment is a distinct skill from SFT, DPO is the most common alignment interview topic, and this demonstrates the SFT → DPO pipeline production teams use.

---

## **Capstone 3 — GRPO for reasoning improvement (3–4 weeks)**
**Goal.** Take an SFT base model, apply GRPO with a verifiable reward on a reasoning task, and prove the model improves on the target task.

**What you build**

- An SFT base model small enough for one GPU (Qwen3.5-4B, Gemma 4 E2B, or Qwen3-4B-Instruct-2507).
- A reasoning task with a verifiable reward (GSM8K, MATH subset, HumanEval, MBPP).
- A GRPO training script (TRL `GRPOTrainer`).
- A programmatically verifiable reward function.
- Evaluation: held-out test set, capability regression, qualitative examples.

**The reward function**
```python
import re

def math_reward(completions, answer, **kwargs):
    """Reward: 1.0 if completion contains the correct answer, 0.0 otherwise."""
    rewards = []
    for completion, correct_answer in zip(completions, answer):
        match = re.search(r"\\boxed\{([^}]+)\}", completion)
        if match:
            try:
                rewards.append(1.0 if abs(float(match.group(1)) - float(correct_answer)) < 1e-3 else 0.0)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards
```

**Benchmarks to report**

- GRPO training curve (mean reward, group variance, completion length, KL).
- Held-out test accuracy: SFT versus SFT+GRPO.
- Capability regression: SFT versus SFT+GRPO on MMLU and IFEval.
- Format compliance: does the model actually use `\boxed{}` after GRPO?
- Qualitative examples: 10 reasoning traces before and after.
- Compute cost: G=8 rollouts × 1000 steps = 8,000 generations; report GPU-hours.

**Acceptance criteria**

- [ ] GRPO improves held-out test accuracy by at least 10% over SFT-only.
- [ ] Format compliance rate at or above 95% after GRPO.
- [ ] Capability regression on MMLU under 3%.
- [ ] No reward hacking — the model is not gaming the format.
- [ ] Cleanly visualized curves and a sample of reasoning traces.
- [ ] Reproducible from the README.

**Why it matters.** GRPO is the 2026 frontier for reasoning models, and the verifiable-reward pattern generalizes to any task with a programmatic check — code tests, format compliance, retrieval correctness.

---

## **Bonus capstone — Multi-LoRA serving (1 week)**
**Goal.** Train three LoRA adapters on three tasks, serve them with vLLM's multi-LoRA, hot-swap per request, and benchmark the overhead.

**Artifacts:** three small adapters (for example "summarize", "translate to French", "extract JSON"); a vLLM server with `--enable-lora`; a routing client; a benchmark of hot-swap overhead versus single-adapter serving.

**Benchmarks:** latency overhead of multi-LoRA versus single-LoRA; memory usage of three adapters versus one; throughput at batch 1, 8, 32; the `--max-loras` setting at which latency degrades.

---

## **Capstone sequence**
| # | Capstone | Stages | Target window |
|---|---|---|---|
| 1 | Domain QLoRA + quantize + serve | 2, 3, 7 | Dec 2026 |
| 2 | DPO for preference alignment | 4 | Jan 2027 |
| 3 | GRPO for reasoning | 4 | Feb–Mar 2027 |
| Bonus | Multi-LoRA serving | 7 | Any time |

All three public before the target application window. Combined with the sibling inference roadmap's capstones, that is eight to nine public artifacts.

---

## **JD alignment**
| JD source | Required skill | Covered by |
|---|---|---|
| Anthropic Research Engineer, Agents | Agent harnesses, rigorous quantitative evals for agentic tasks, automated model and prompt evaluation across the training and product lifecycle, optimized **data mixes** for agentic training, plus large-scale RL and multi-agent systems | Stages 2, 4, 8 — capstones 1, 2, 3 |
| AI Engineer (General / Applied) | Fine-tuning, PEFT, model customization | Stage 3 — capstone 1 |
| Multimodal AI Engineer | Fine-tuning, vision-language tuning | Stage 6 — extend capstone 1 to a VLM |
| Inference Engineer | LoRA serving, multi-LoRA, deployment | Stage 7 — bonus capstone |

### **Requirements named in the Anthropic posting**
The Research Engineer, Agents posting is the most useful target specification in the source material, because it shows what the market now values in a fine-tuning engineer: **harnesses, evals, and data mixes, not only weights**.

Responsibilities, as written: ideate, develop, and compare the performance of different agent harnesses (memory, context compression, communication architectures for agents); design and implement rigorous quantitative benchmarks for large-scale agentic tasks; assist with automated evaluation of Claude models and prompts across the training and product lifecycle; work with the product org on agent applications; help create and optimize data mixes for model training that maximize performance on agentic tasks. Strong candidates may also have experience with large-scale RL on language models and multi-agent systems.

Representative projects: design and build a novel agent harness that outperforms existing agents on coding or knowledge-work benchmarks; design agent affordances that unlock new capabilities; design a novel eval that measures how many agents interact in groups to solve problems; build a scaled model-evaluation framework driven by model-based evaluation; build prompting and model orchestration for a production application; **finetune a model to maximize its performance using a particular set of agent tools or harness**.

Read that list as a specification for what a fine-tuning-only portfolio is missing: **eval infrastructure and agent-harness design**. The merge closes part of that gap — Stage 2's agent-trace data pipeline, Stage 4's agent-behavior SFT, Stage 7's multi-adapter serving, and Stage 8's eval panels — but a capstone that ships an agent harness plus a benchmark is worth more than a fourth fine-tune.

### **Role-by-role coverage**
**Role 1 — AI Engineer (General / Applied).** Fine-tuning is listed but not as an explicit top skill; the implicit expectation is "you can take a base model and adapt it to a domain." Capstone 1 covers this fully. Questions you can answer: How would you adapt a base model to a domain with no labeled data? LoRA versus QLoRA — when to use which? How do you evaluate a fine-tuned model? What is multi-LoRA serving?

**Role 3 — Multimodal AI Engineer.** Fine-tuning is in the tech stack. Extend Capstone 1 to a VLM with LLaVA-style data. Questions: How would you fine-tune a VLM on a custom image+text dataset? What is different about LoRA on a vision-language model?

**Role 4 — Inference Engineer.** LoRA serving, multi-LoRA, and adapter hot-swap are core. Bonus capstone plus Stage 7. Questions: How do you serve 50 fine-tuned adapters on one base model? What is the memory overhead of multi-LoRA serving?

**Role 5 — Harness / Agent Infrastructure Engineer.** "Fine-tuning for agents" sits in the what's-missing list. Capstones 1, 2, and 3 cover it. Questions: How would you fine-tune a model for agent behavior? What is the difference between SFT and DPO for agent alignment?

---

### **Serving and systems**
- [ ] What are vLLM's `--max-loras` and `--max-lora-rank` defaults, and why do they matter?
- [ ] Which quantization path is supported today, and which two are deprecated?
- [ ] When is merged deployment better than multi-LoRA, and when is it worse?
- [ ] How do you choose a compute tier for a 7B QLoRA on a budget?
- [ ] What breaks if you serve a rank-32 adapter on a server left at the default `--max-lora-rank`?
- [ ] Why is GGUF in vLLM experimental, and what do you serve it with instead?

---

## **Interviewer question list**
- Does the team own model fine-tuning, or is it outsourced to a vendor?
- What is the team's preferred fine-tuning framework? (Surfaces tech-stack alignment.)
- Does the team do DPO/GRPO in-house, or rely on closed APIs?
- How are fine-tuned models served — multi-LoRA, merged, quantized?
- What is the eval pipeline for a new fine-tuned checkpoint?
- How does the team handle catastrophic forgetting and capability regression?
- What is the data governance process, and how is EU AI Act compliance handled?
- What is the agent harness, and how is agent behavior evaluated?
- How are data mixes constructed and tested?
- What is the compute budget and the latency budget for a new model release?

---

## **The before-you-ship checklist**
- [ ] README has the problem, approach, results, and reproduction.
- [ ] Dockerfile builds clean.
- [ ] Reproduction works from a fresh clone.
- [ ] Numbers are real, not theoretical.
- [ ] Baseline comparison is included.
- [ ] Limitations are honest.
- [ ] Dataset is on the Hub with a data card.
- [ ] License is set (Apache 2.0 recommended).
- [ ] The repo is linked from your GitHub profile README.
- [ ] The repo is shared with the key chart.
- [ ] The model card documents intended use, limitations, data provenance, and eval results.
- [ ] No secrets, tokens, or large weight files in the repository.

## **Calendar**
The original notes ran a 6–8 week fine-tuning plan alongside the inference roadmap at 25–30 hours per week. Re-anchored to November 2026, with the source notes' week 5 (the old Capstone 1 slot) now in the past:

| Week | Focus | Output |
|---|---|---|
| W1 (Nov 3–9) | Methods + LoRA from scratch in PyTorch | ~150-line LoRA implementation |
| W2 (Nov 10–16) | DPO + GRPO theory, TRL hands-on | First SFT + DPO runs on a 1B–4B model |
| W3 (Nov 17–23) | Data: build a domain dataset | 1K–5K sample curated dataset, public on the Hub |
| W4 (Nov 24–30) | Frameworks: Unsloth + Axolotl comparison | Same model trained with three frameworks, compared |
| W5 (Dec 1–7) | **Capstone 1** (QLoRA → quantize → vLLM) | **Public on GitHub with eval numbers** |
| W6 (Dec 8–14) | **Capstone 2** (DPO) + serving the merged model | **Public on GitHub + vLLM serve** |
| W7–8 (Dec 15–28) | **Capstone 3** (GRPO) | **Public on GitHub with reasoning traces** |

The capstone sequence, the acceptance criteria, and the interview bank are the deliverables that matter. The model weights are disposable; the writeup is not.

---

## **Resources**
- Anthropic Research Engineer, Agents — https://www.aijobs.com/jobs/396324642-research-engineer-agents
- lm-evaluation-harness v0.4.13 — https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13
- EU AI Act implementation timeline — https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act
- vLLM `serve` CLI — https://docs.vllm.ai/en/latest/cli/serve/
