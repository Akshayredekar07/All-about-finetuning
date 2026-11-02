---
title: "Capstones & JD Alignment — Public Proof, Mapped to Roles"
type: capstones
date_created: 2026-09-01
date_updated: 2026-09-01
tags:
  - fine-tuning
  - capstones
  - jd
  - alignment
  - public
  - github
  - portfolio
  - anthropic
  - ai-engineer
  - inference-engineer
status: active
related:
  - "[[inference-roadmap/16 - Capstone Projects]]"
  - "[[inference-roadmap/18 - Job Description Alignment]]"
  - "[[targets/anthropic-agents]]"
---

# 04 — Capstones & JD Alignment

> [!quote] The rule
> Every fine-tuning skill should produce a public artifact. The 3 capstones below, shipped with real numbers, cover **~90% of the fine-tuning requirements in the 2026 AI Engineer and Inference Engineer JDs.** Combined with the inference capstones in [[inference-roadmap/16 - Capstone Projects]], this is a complete portfolio for [[targets/anthropic-agents]].

> [!warning] The standard (same as the inference roadmap)
> Every capstone must:
> 1. Be **public on GitHub** with a clean README.
> 2. Include **real numbers** (training loss, eval scores, throughput, etc.).
> 3. Compare to a **stated baseline** (e.g., "vs base model with no fine-tuning").
> 4. Include **reproduction instructions** (Docker, exact env, exact commands).
> 5. Be **benchmarked, not just claimed.**

---

## Capstone 1 — Domain QLoRA fine-tune + AWQ serve (3 weeks)

### Goal
Take a base model (Llama-3.2-3B or similar), QLoRA fine-tune it on a real domain dataset, merge, quantize to AWQ INT4, serve with vLLM, benchmark end-to-end.

### What you build
- A domain dataset (1K-5K samples, public on HuggingFace).
- A QLoRA training script (Unsloth or TRL + PEFT).
- The merged FP16 model.
- The AWQ INT4 quantized model.
- A vLLM server with the AWQ model.
- A benchmark script that measures eval scores, TTFT, TPOT, throughput.
- A comparison report: base model vs FT-FP16 vs FT-AWQ-INT4.

### Stack
- Llama-3.2-3B-Instruct (base).
- Unsloth for training (or TRL+PEFT for the writeup).
- AWQ for quantization.
- vLLM for serving.
- `lm-evaluation-harness` for quality evals.

### Benchmarks to report
- **Training time** and **peak memory** during training.
- **Eval scores** (MMLU, your-domain eval) for: base, FT-FP16, FT-AWQ-INT4.
- **TTFT p50, p95, p99** for each.
- **TPOT p50, p95, p99** for each.
- **Throughput at batch 1, 8, 32** for each.
- **Model size on disk** for each.

### Acceptance criteria
- [ ] Public dataset on HuggingFace with a data card.
- [ ] Training script runs in < 4 hours on a single A100/H100.
- [ ] Quality loss from AWQ quantization is < 2% on the domain eval.
- [ ] Throughput improvement of AWQ over FP16: ≥ 1.5×.
- [ ] Numbers reproducible from the README.

### Why this capstone
- **Covers the full vertical:** data → train → quantize → serve → benchmark.
- **Demonstrates production thinking** (not just training accuracy).
- **Directly mappable to AI Engineer / Inference Engineer JD requirements** (see [[#JD alignment]] below).

---

## Capstone 2 — DPO for preference alignment (2-3 weeks)

### Goal
Take an SFT model (or use the Capstone 1 output), apply DPO on a preference dataset, prove it improves alignment without losing capability.

### What you build
- An SFT base model (or use Capstone 1's output).
- A preference dataset (UltraFeedback subset, or a custom one).
- A DPO training script (TRL `DPOTrainer`).
- Evaluation: MT-Bench, your-domain eval, capability regression evals.
- A writeup comparing SFT-only vs SFT+DPO.

### Stack
- Llama-3.2-3B or 8B (SFT base).
- TRL v0.15.0 (`DPOTrainer`).
- UltraFeedback or Argilla DPO mix.
- `lm-evaluation-harness` for capability evals.

### Benchmarks to report
- **DPO training loss curve** (chosen vs rejected reward margin).
- **MT-Bench score:** SFT vs SFT+DPO.
- **Domain eval score:** SFT vs SFT+DPO.
- **Capability regression:** SFT vs SFT+DPO on MMLU, HumanEval.
- **Qualitative examples:** show 5-10 prompts where DPO helped and 5 where it didn't.
- **Reference model comparison:** DPO with separate ref model vs LoRA-as-ref.

### Acceptance criteria
- [ ] DPO improves MT-Bench by ≥ 5% over SFT-only.
- [ ] Capability regression on MMLU < 1%.
- [ ] No catastrophic forgetting (capability evals still in normal range).
- [ ] Cleanly visualized loss curves and eval bar charts.
- [ ] Reproducible from the README.

### Why this capstone
- **Alignment is a distinct skill** from SFT. Most engineers conflate them.
- **DPO is the most common interview topic** for alignment work.
- **Demonstrates the SFT → DPO pipeline** that 2026 production teams use.

---

## Capstone 3 — GRPO for reasoning improvement (3-4 weeks)

### Goal
Take an SFT base model, apply GRPO with a verifiable reward on a reasoning task (math or code), prove the model improves on the target task.

### What you build
- An SFT base model (small enough for single-GPU: Llama-3.2-1B or Qwen2.5-1.5B).
- A reasoning task with a verifiable reward (GSM8K, MATH subset, HumanEval, MBPP).
- A GRPO training script (TRL `GRPOTrainer`).
- A reward function (programmatically verifiable).
- Evaluation: held-out test set, capability regression, qualitative examples.

### Stack
- Llama-3.2-1B or Qwen2.5-1.5B (Instruct).
- TRL v0.15.0 (`GRPOTrainer`).
- GSM8K or HumanEval (or your domain with a verifier).
- A single A100/H100 (or Unsloth for 4090).

### The reward function (example for GSM8K)

```python
import re

def math_reward(completions, answer, **kwargs):
    """Reward: 1.0 if completion contains the correct answer, 0.0 otherwise."""
    rewards = []
    for completion, correct_answer in zip(completions, answer):
        # Match \boxed{123} format
        match = re.search(r"\\boxed\{([^}]+)\}", completion)
        if match:
            try:
                if abs(float(match.group(1)) - float(correct_answer)) < 1e-3:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards
```

### Benchmarks to report
- **GRPO training curve** (mean reward, group variance).
- **Held-out test accuracy:** SFT vs SFT+GRPO.
- **Capability regression:** SFT vs SFT+GRPO on MMLU, MT-Bench.
- **Format compliance:** does the model actually use `\boxed{}` after GRPO?
- **Qualitative examples:** 10 reasoning traces before and after.
- **Compute cost:** G=8 rollouts × 1000 steps = 8000 generations. Report GPU-hours.

### Acceptance criteria
- [ ] GRPO improves held-out test accuracy by ≥ 10% over SFT-only.
- [ ] Format compliance rate ≥ 95% after GRPO.
- [ ] Capability regression on MMLU < 3%.
- [ ] No reward hacking (the model isn't gaming the format).
- [ ] Cleanly visualized curves and a sample of reasoning traces.
- [ ] Reproducible from the README.

### Why this capstone
- **GRPO is the 2026 frontier.** DeepSeek R1, o1, o3 — all GRPO-trained.
- **Reasoning is the most active research area** in 2026.
- **Demonstrates the most advanced alignment technique** — interview gold.
- **The verifiable reward pattern generalizes** to any task with a programmatic check (code tests, format compliance, retrieval correctness, etc.).

---

## Bonus capstone — Multi-LoRA serving (1 week)

### Goal
Train 3 LoRA adapters on 3 different tasks, serve them with vLLM's multi-LoRA, hot-swap between them per request, benchmark the overhead.

### What you build
- 3 small LoRA adapters (e.g., "summarize", "translate to French", "extract JSON").
- A vLLM server with `--enable-lora`.
- A client that routes requests to the right adapter.
- A benchmark of hot-swap overhead vs single-adapter serving.

### Benchmarks
- Latency overhead of multi-LoRA vs single-LoRA.
- Memory usage of 3 adapters vs 1.
- Throughput at batch 1, 8, 32.

### Why this capstone
- **Multi-LoRA serving is the 2026 production pattern.**
- **Demonstrates serving infra knowledge** (not just training).
- **Fast to ship** (1 week). Good for filling a gap in the portfolio.

---

## The 3-capstone plan (recommended)

| # | Capstone | Maps to file | When |
|---|---|---|---|
| 1 | Domain QLoRA + AWQ serve | [[fine-tuning-roadmap/01 - Methods Deep Dive]] (LoRA/QLoRA) + [[fine-tuning-roadmap/03 - Frameworks & Tools 2026]] (serving) | Oct 2026 |
| 2 | DPO for preference alignment | [[fine-tuning-roadmap/01 - Methods Deep Dive#Part 2 — Preference Optimization (DPO family)]] | Nov 2026 |
| 3 | GRPO for reasoning | [[fine-tuning-roadmap/01 - Methods Deep Dive#Part 3 — RL Fine-Tuning]] | Dec 2026 |
| Bonus | Multi-LoRA serving | [[fine-tuning-roadmap/03 - Frameworks & Tools 2026#Multi-LoRA serving (2026 essential)]] | Anytime |

> All three should be **public on GitHub by end of 2026.** Combined with the 5 inference capstones, you have **8-9 public artifacts** for the Anthropic application.

---

## JD alignment

### The 2026 fine-tuning skills that show up in JDs

| JD source | Required fine-tuning skill | Vault file + capstone |
|---|---|---|
| **Anthropic Research Engineer, Agents** | LoRA, QLoRA, DPO, large-scale RL (PPO, GRPO, RLVR) | [[inference-roadmap/18 - Job Description Alignment#Anthropic Research Engineer, Agents — full JD mapping]] + capstones 1, 2, 3 |
| **AI Engineer (General / Applied)** | Fine-tuning, PEFT, model customization | [[fine-tuning-roadmap/01 - Methods Deep Dive]] + capstone 1 |
| **Multimodal AI Engineer** | Fine-Tuning, vision-language model tuning | [[fine-tuning-roadmap/01 - Methods Deep Dive]] (extend to VLM) + capstone 1 |
| **Inference Engineer (general)** | LoRA serving, multi-LoRA, model deployment | [[fine-tuning-roadmap/03 - Frameworks & Tools 2026#Multi-LoRA serving]] + bonus capstone |

### Mapped to the role taxonomy ([[targets/ai-engineering-role-taxonomy]])

#### Role 1 — AI Engineer (General / Applied)
**Fine-tuning requirements:** Fine-tuning is listed but not in the explicit top skills. The implicit expectation is "you can take a base model and adapt it to a domain."
- **Coverage:** Capstone 1 covers this fully.
- **Interview questions you can answer:**
  - "How would you adapt Llama-3 to a domain with no labeled data?"
  - "LoRA vs QLoRA — when to use which?"
  - "How do you evaluate a fine-tuned model?"
  - "What is multi-LoRA serving?"

#### Role 3 — Multimodal AI Engineer
**Fine-tuning requirements:** Fine-Tuning is in the tech stack.
- **Coverage:** Capstone 1 (extending to VLM with LLaVA-style data) covers this.
- **Interview questions you can answer:**
  - "How would you fine-tune a VLM on a custom image+text dataset?"
  - "What is the difference between LoRA on text vs LoRA on vision-language models?"

#### Role 4 — Inference Engineer (LLM Infrastructure / Performance)
**Fine-tuning requirements:** Not in the explicit stack, but **LoRA serving, multi-LoRA, adapter hot-swap** are core.
- **Coverage:** Bonus capstone + [[fine-tuning-roadmap/03 - Frameworks & Tools 2026#Multi-LoRA serving]] covers this.
- **Interview questions you can answer:**
  - "How do you serve 50 fine-tuned adapters on one base model?"
  - "What's the memory overhead of multi-LoRA serving?"

#### Role 5 — Harness / Agent Infrastructure Engineer
**Fine-tuning requirements:** Not explicit, but "fine-tuning for agents" (LoRA, QLoRA, DPO) is in the "What's Missing" list in [[targets/anthropic-agents]].
- **Coverage:** Capstones 1 + 2 + 3 cover this fully.
- **Interview questions you can answer:**
  - "How would you fine-tune a model for agent behavior?"
  - "What's the difference between SFT and DPO for agent alignment?"

### The Anthropic target — what fine-tuning skills close the gap

From [[targets/anthropic-agents]] "What's Missing":

| Gap | Capstone that closes it |
|---|---|
| Fine-tuning for agents (LoRA, QLoRA, DPO) | **Capstone 1** (QLoRA) + **Capstone 2** (DPO) |
| Large-scale RL on LLMs (PPO, GRPO, RLVR) | **Capstone 3** (GRPO) |
| Benchmark design (evaluation systems) | Part of every capstone (eval is mandatory) |
| GitHub proof of from-scratch work | All 3 capstones public |

> The 3 capstones + 5-6 inference capstones from [[inference-roadmap/16 - Capstone Projects]] = **8-9 public artifacts** = a complete portfolio for the Anthropic application.

---

## The interview question bank (fine-tuning)

These are the questions you should be able to answer cold by interview time.

### Foundational
- [ ] What is LoRA? Why is it preferred over full fine-tuning?
- [ ] What's the difference between LoRA, QLoRA, and DoRA?
- [ ] State the 2026 default LoRA config (r, α, target_modules, lr).
- [ ] How much memory does QLoRA save vs full FT on a 70B model?
- [ ] What is the DPO loss? Why is it "direct"?
- [ ] Compare DPO, IPO, KTO, SimPO, ORPO. When does each win?
- [ ] What is GRPO? Why does it beat PPO for reasoning?
- [ ] What is RLVR? Name 2 examples of verifiable rewards.
- [ ] State the 2026 consensus pipeline (SFT → DPO → GRPO).
- [ ] What is the LoRA adapter format? How does it interact with vLLM?

### Practical
- [ ] How do you pick a fine-tuning framework for a project?
- [ ] For a 7B LoRA on a single RTX 4090, which framework? For 70B on 8×H100?
- [ ] Walk me through the LoRA → merge → quantize → serve pipeline.
- [ ] How do you serve multiple LoRA adapters on one base model?
- [ ] How do you evaluate a fine-tuned model?
- [ ] What's the most common cause of fine-tuning failure? (Hint: data, not algorithm.)
- [ ] How do you decontaminate training data against an eval set?
- [ ] What's the right chat template for Llama-3? For Qwen? For Mistral?
- [ ] How do you merge a LoRA adapter into the base model?
- [ ] How do you convert a merged model to GGUF for edge deployment?

### Depth
- [ ] Derive the DPO loss from the Bradley-Terry preference model.
- [ ] Explain why GRPO doesn't need a critic network.
- [ ] What is auxiliary-loss-free load balancing in MoE training?
- [ ] How does Unsloth achieve 2× speedup on a single GPU?
- [ ] What's the difference between TRL's `SFTTrainer` and a vanilla training loop?
- [ ] How does FSDP2 differ from ZeRO-3?
- [ ] What's the difference between sequence packing and FlashAttention's causal masking?
- [ ] How would you fine-tune a VLM? (LLaVA-style.)
- [ ] What is the EU AI Act's main impact on fine-tuning practice?
- [ ] How would you detect reward hacking in a GRPO training run?

---

## The "before you ship" checklist (per capstone)

- [ ] README has the problem, approach, results, reproduction.
- [ ] Dockerfile builds clean.
- [ ] Reproduction works from a fresh clone.
- [ ] Numbers are real (not theoretical).
- [ ] Baseline comparison is included.
- [ ] Limitations are honest.
- [ ] Dataset is on HuggingFace Hub with a data card.
- [ ] License is set (Apache 2.0 recommended).
- [ ] The repo is linked from your main GitHub profile README.
- [ ] The repo is shared on LinkedIn / Twitter with the key chart.

---

## The 6-week plan (parallel with the inference roadmap)

| Week | Focus | Output |
|---|---|---|
| **W1 (Oct 6-12)** | File 01 (Methods) + LoRA from scratch in PyTorch | ~150-line LoRA implementation |
| **W2 (Oct 13-19)** | DPO + GRPO theory + TRL hands-on | First SFT + DPO runs on Llama-3.2-1B |
| **W3 (Oct 20-26)** | File 02 (Data) + build a domain dataset | 1K-5K sample curated dataset, public on HF |
| **W4 (Oct 27 - Nov 2)** | File 03 (Frameworks) + Unsloth + Axolotl comparison | Train same model with 3 frameworks, compare |
| **W5 (Nov 3-9)** | **Capstone 1** (QLoRA + AWQ + vLLM) | **Public on GitHub with eval numbers** |
| **W6 (Nov 10-16)** | **Capstone 2** (DPO) + serving the merged model | **Public on GitHub + vLLM serve** |
| **W7-8 (Nov 17 - Dec 1)** | **Capstone 3** (GRPO) | **Public on GitHub with reasoning traces** |

> The capstones are the **proof.** The interviews ask about them. The GitHub link is in the application.

---

## What interviewers actually look at

From [[targets/ai-engineering-role-taxonomy]] and the 2026 JDs:

> "The most important thing the candidate can demonstrate is that they have **shipped under performance budgets, cost budgets, and safety requirements**."

For fine-tuning specifically:
- **Can they pick the right algorithm for the data?** (SFT vs DPO vs GRPO.)
- **Can they pick the right framework for the hardware?** (Unsloth vs Axolotl vs TRL.)
- **Can they curate a dataset?** (Data > algorithm, every time.)
- **Can they evaluate properly?** (Decontamination, capability regression, hard negatives.)
- **Can they serve the result?** (Merge, quantize, multi-LoRA.)
- **Do they know when not to fine-tune?** (Prompting first, RAG second, fine-tuning third.)

---

## The "what to ask the interviewer" list (fine-tuning flavor)

- Does the team own model fine-tuning, or is it outsourced to a vendor?
- What's the team's preferred fine-tuning framework? (Surfaces tech stack alignment.)
- Does the team do DPO/GRPO in-house, or rely on closed APIs?
- How are fine-tuned models served? (Multi-LoRA? Merged? Quantized?)
- What's the eval pipeline for a new fine-tuned checkpoint?
- How does the team handle catastrophic forgetting / capability regression?
- What's the data governance process? (EU AI Act compliance.)

---

## Pre-flight checklist (before sending the application)

- [ ] All 3 (or 4 with bonus) fine-tuning capstones are public on GitHub with real numbers.
- [ ] Each capstone README has: problem, approach, results, reproduction.
- [ ] GitHub profile README links to the capstones (inference + fine-tuning).
- [ ] Resume is updated with the new skills and capstones.
- [ ] The 30-question fine-tuning interview bank is drilled.
- [ ] The 10 questions to ask the interviewer are ready.

---

**End of the fine-tuning roadmap.** The full picture:
- **Inference roadmap:** 18 files in [[inference-roadmap/00 - Overview & Index]].
- **Fine-tuning roadmap:** 4 files in this vault (you are here).
- **Total:** 22 interlinked Obsidian files, ~7,500 lines, 8-9 public capstones planned.

Go ship the capstones. The Anthropic target is Q1-Q2 2027. You have the curriculum. The rest is execution.
