---
title: Data & Datasets — The Boring, Critical Part
type: data
date_created: 2026-09-01
date_updated: 2026-09-01
tags: [fine-tuning, data, datasets, alpaca, sharegpt, ultrafeedback, synthetic, decontamination, licensing]
status: active
time_budget: 1-2 weeks
prereq_for: "[[fine-tuning-roadmap/03 - Frameworks & Tools 2026]]"
related: "[[fine-tuning-roadmap/01 - Methods Deep Dive]]", "[[inference-roadmap/14 - Production & Reliability#Eval pipeline]]
---

# 02 — Data & Datasets

> [!quote] The hard truth
> A bad dataset will defeat a perfect training loop every time. The single biggest mistake in fine-tuning is not picking the wrong algorithm — it's picking bad data. The 2026 consensus is unambiguous: **data > algorithm, every time.**

> [!warning] What this file covers
> The five things that actually move metrics:
> 1. **Diversity over volume** — 5K diverse > 50K near-duplicates.
> 2. **Hard negatives** for preference data.
> 3. **Decontamination** — strip eval-set leakage.
> 4. **Format consistency** — tokenize early, catch chat-template mismatches.
> 5. **PII and licensing** — EU AI Act territory.

---

## TL;DR — the data menu

| Data shape | Used for | Format | Example datasets |
|---|---|---|---|
| **Instruction-completion pairs** | SFT | Alpaca, ShareGPT, OpenAI ChatML | Alpaca, Dolly, OpenHermes, Tulu |
| **Multi-turn chat** | SFT (chat) | ShareGPT, ChatML | ShareGPT, UltraChat |
| **Tool-use / function-calling** | SFT (agents) | OpenAI tools format | ToolBench, Glaive |
| **Preference pairs (chosen, rejected)** | DPO, IPO, SimPO | `{prompt, chosen, rejected}` | UltraFeedback, HH-RLHF, Argilla DPO mix |
| **Binary good/bad** | KTO | `{prompt, completion, label: bool}` | Real user feedback |
| **Verifiable-reward tasks** | GRPO, RLVR | `{prompt, verifiable_check_fn}` | MATH, GSM8K, HumanEval, MBPP |
| **Distillation** | Teacher → student | Same as SFT, plus teacher logits | Teacher model outputs |

---

## Part 1 — Data formats

### Alpaca format (instruction-completion)

```json
{
  "instruction": "Explain speculative decoding in 2 sentences.",
  "input": "",  // optional context
  "output": "Speculative decoding uses a small draft model to propose several tokens..."
}
```

The original Stanford Alpaca format. Simple, widely supported. Good for single-turn SFT.

### ShareGPT format (multi-turn chat)

```json
{
  "conversations": [
    {"from": "human", "value": "What's the weather in Paris?"},
    {"from": "gpt", "value": "I don't have real-time data, but..."},
    {"from": "human", "value": "OK then what's the capital of France?"},
    {"from": "gpt", "value": "Paris."}
  ]
}
```

Multi-turn. Each model's chat template maps the `from` field to the role tokens. **Make sure your template matches the target model's chat template** (e.g., Llama-3 uses `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...`).

### OpenAI ChatML (the de facto standard)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "Paris."}
  ]
}
```

Used by OpenAI, Mistral, Qwen, Yi, and most 2024-2026 models. **The safest default for new projects.** Most frameworks auto-convert.

### OpenAI tool-calling format

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18, \"condition\": \"cloudy\"}"},
    {"role": "assistant", "content": "It's 18°C and cloudy in Paris."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {...}}}
  ]
}
```

The standard for agent / function-calling fine-tunes. The model learns to call tools, receive results, and continue. Datasets: ToolBench, Glaive-function-calling, Salesforce/xlam-function-calling-60k.

### DPO preference format

```json
{
  "prompt": "Explain speculative decoding.",
  "chosen": [
    {"role": "user", "content": "Explain speculative decoding."},
    {"role": "assistant", "content": "Speculative decoding uses a small draft model..."}
  ],
  "rejected": [
    {"role": "user", "content": "Explain speculative decoding."},
    {"role": "assistant", "content": "Speculative decoding is a technique where you guess..."}
  ]
}
```

The "chosen" is what the model should prefer. The "rejected" is what it should de-prefer. The prompt is the input.

> [!tip] The hard-negative trick
> **The most useful preference pairs are where chosen and rejected are ALMOST equally good.** If chosen is obviously better than rejected, the model learns nothing. If they're close, the model has to actually learn the preference.

### GRPO / RLVR format

```json
{
  "prompt": "What is 47 * 83?",
  "verifier": "lambda completion: int(eval(completion.strip())) == 47 * 83",
  "reward": 1.0
}
```

Or simpler: the prompt is the question, the reward function is a programmatic check (math correctness, code passes tests, format matches regex).

---

## Part 2 — Public datasets (2026 cheat sheet)

### SFT
| Dataset | Size | Domain | Notes |
|---|---|---|---|
| **Alpaca** | 52K | General instruction | The original. Self-instruct from GPT-3.5. |
| **Dolly** | 15K | Human-written | High quality but small. |
| **OpenHermes-2.5** | 1M | Mixed | The 2024 workhorse. From GPT-4 outputs. |
| **Tulu-3-SFT-mixture** | 940K | Mixed (incl. code, math, safety) | Allen AI's curated mix. **2025-2026 standard.** |
| **UltraChat** | 1.5M | Multi-turn | Multi-turn, broad coverage. |
| **SmolLM-Corpus** | 2T tokens | Web | Pretraining data, not SFT. |

### DPO / Preference
| Dataset | Size | Notes |
|---|---|---|
| **UltraFeedback** | 64K | The 2024 DPO workhorse. GPT-4 ranked responses. |
| **HH-RLHF** | 170K | Anthropic's helpful + harmless. |
| **Argilla DPO mix** | ~50K | Curated mix from Argilla. |
| **Tulu-3-Preference-mixture** | ~300K | Allen AI's 2025 preference mix. **Strong 2026 default.** |

### Tool use / function calling
| Dataset | Notes |
|---|---|
| **ToolBench** | Multi-step tool use. |
| **Glaive-function-calling-v2** | 113K samples. Single + multi-turn. |
| **xLAM-function-calling-60k** | Salesforce. **2025-2026 standard for function-calling SFT.** |
| **ToolACE** | Multi-turn, complex tool use. |

### Reasoning / RLVR
| Dataset | Domain | Verifier |
|---|---|---|
| **MATH** | Math olympiad | `\\boxed{...}` parsing |
| **GSM8K** | Grade school math | Numeric answer |
| **AIME 1983-2024** | Competition math | Integer answer |
| **HumanEval / MBPP** | Code | `exec` test cases |
| **LiveCodeBench** | Code (contamination-free) | `exec` test cases |
| **MMLU-Pro** | Multi-domain QA | Multiple choice |
| **ARC-AGI** | Abstract reasoning | Grid match |

> [!warning] Decontamination
> **MMLU and similar evals are commonly present in SFT training data.** If you train on a contaminated dataset and evaluate on MMLU, your numbers are lies. Use the **lm-evaluation-harness decontaminator** (`--decontamination_ngram_size=8` or similar) to strip eval-set n-grams from training data. **Always.**

### Multimodal (for completeness)
| Dataset | Notes |
|---|---|
| **LLaVA-Instruct** | 150K. The 2024 visual instruction workhorse. |
| **ShareGPT4V** | 1.2M. Higher quality captions. |
| **ALLaVA-4V** | 1.4M. 2025. |
| **Cambrian-10M** | 10M. The 2025 SOTA visual instruction mix. |

---

## Part 3 — Data quality — the things that actually move metrics

> [!example] The five rules
> 1. **Diversity over volume.** 5K diverse > 50K near-duplicates. A model trained on near-duplicates overfits and generalizes poorly.
> 2. **Hard negatives.** For preference data, pairs where chosen and rejected are almost equally good teach more than obvious wins.
> 3. **Decontamination.** Strip eval-set leakage from training data. Always.
> 4. **Format consistency.** Tokenize early to catch chat-template mismatches before you waste 10 GPU-hours.
> 5. **PII and licensing.** Document provenance. EU AI Act applies.

### How to assess data quality
- **Read 100 random examples.** If the quality is bad in 100, it's bad in 1M.
- **Check the token-length distribution.** Long-tail tokens indicate bloat. Trim or filter.
- **Check the source distribution.** If 80% of your data comes from one source, you have a single-source bias.
- **Check for near-duplicates.** Use MinHash or sentence embeddings to find and dedupe.
- **Check for eval contamination.** N-gram overlap with MMLU, HumanEval, etc.

### The decontamination toolchain

```python
# Using lm-evaluation-harness decontaminator
from lm_eval.decontamination.decontaminate import decontaminate

clean_dataset = decontaminate(
    train_dataset=my_dataset,
    eval_datasets=["mmlu", "humaneval", "gsm8k"],
    ngram_size=8,
)

# Or use the standalone dedup tool
# pip install datasketch
from datasketch import MinHash, MinHashLSH

def minhash_dedup(texts, threshold=0.8):
    """Remove near-duplicates using MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    sigs = {}
    keep = []
    for i, text in enumerate(texts):
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode())
        if i == 0:
            lsh.insert(i, m)
            sigs[i] = m
            keep.append(i)
        else:
            result = lsh.query(m)
            if not result:
                lsh.insert(i, m)
                sigs[i] = m
                keep.append(i)
    return keep
```

### The format consistency check

```python
# Tokenize a sample BEFORE training to catch template mismatches
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

sample = {
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
}

# Apply chat template
formatted = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
print(formatted)
# <|begin_of_text|><|start_header_id|>user<|end_header_id|>
#
# Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# Hi there!<|eot_id|>

# Tokenize
tokens = tokenizer(formatted, return_tensors="pt")
print(f"Token count: {tokens.input_ids.shape[1]}")
# If this is way off from your expected sequence length, your template is wrong.
```

---

## Part 4 — Synthetic data

### When to use
- You don't have enough real data.
- You need data for a specific edge case that's rare in real data.
- You need preference pairs at scale (use a strong LLM to generate candidates, a stronger LLM to rank them).

### The standard pattern (2026)
1. **Generate candidates** with a strong model (GPT-4, Claude Opus 4, DeepSeek V3/R1).
2. **Filter** by quality (length, format, contains the right concepts, passes basic checks).
3. **Rank** for preference data (GPT-4-as-judge, Claude-as-judge, or LLM-as-judge with rubric).
4. **Decontaminate** against your eval set.
5. **Train** on the result.

### The tools
- **Distilabel** (Argilla) — pipeline for synthetic data generation + filtering + ranking.
- **LangSmith** — LangChain's tracing + eval framework.
- **PromptTools** — for prompt A/B testing.
- **Auto-DPO data** (HuggingFace) — automatic preference pair generation.

### The gotchas
- **Model collapse.** Training on synthetic data from the same model degrades over generations. **Always include real data in the mix.**
- **Style homogenization.** Synthetic data tends to have a "GPT-4 voice" that's detectable. Mix sources.
- **Hallucinations.** Synthetic data inherits the generator's hallucinations. Verify a sample manually.
- **Licensing.** If you generate with a closed model, the output's licensing is murky. **Open-source generators (Llama, DeepSeek) are safer.**

---

## Part 5 — Data licensing and the EU AI Act (2026 reality)

> [!warning] The compliance reality
> As of 2025-2026, the EU AI Act applies to most production LLM deployments. The relevant obligations for fine-tuning:
> - **Document training data provenance.** Where did it come from? Under what license? Was consent given?
> - **Model card** documenting intended use, limitations, training data summary.
> - **Risk assessment** if the use case touches Annex III (HR, education, critical infrastructure, law enforcement, etc.).
> - **Logging** of significant model updates and eval results.

### The license landscape
| Source | License | Commercial fine-tune OK? |
|---|---|---|
| **Alpaca** | CC BY-NC 4.0 | No (NC = non-commercial) |
| **Dolly** | CC BY-SA 3.0 | Yes (with share-alike) |
| **OpenHermes-2.5** | Apache 2.0 (mostly) | Yes (check per-source) |
| **Tulu-3** | ODC-BY or AI2 license | Mostly yes, check |
| **UltraFeedback** | MIT | Yes |
| **HH-RLHF** | MIT | Yes |
| **xLAM** | CC BY-NC 4.0 | No (NC) |
| **HumanEval** | MIT | Yes (for evaluation; don't train on it) |
| **Wikipedia** | CC BY-SA | Yes (with share-alike) |
| **Common Crawl** | Varies | **Filtering required** |
| **The Stack / CodeParrot** | Various permissive | Yes (check per-file) |

### PII and consent
- **Strip PII** before training. Use Presidio, PII-Codes, or scrub with a strong LLM.
- **For user-generated data:** consent is required under GDPR. Document the consent flow.
- **For synthetic data:** less PII risk, but verify the generator didn't memorize training data with PII.

---

## Part 6 — Building a domain dataset (the practical guide)

### Step 1: Define the task
- Be specific. "Customer support chatbot" is too vague. "Customer support chatbot for SaaS billing questions, with structured JSON output and citations" is actionable.

### Step 2: Source candidates
- **Real data:** past conversations, support tickets, internal docs (with consent and PII scrubbing).
- **Synthetic data:** generate with a strong LLM using a detailed rubric.
- **Public data:** relevant HuggingFace datasets, with license check.

### Step 3: Filter and clean
- Deduplicate (MinHash).
- Filter by quality (length, format, contains required concepts).
- Decontaminate against eval sets.

### Step 4: Format
- Convert to your target format (Alpaca, ShareGPT, ChatML, DPO).
- Apply chat template and verify tokenization.

### Step 5: Split
- **Train:** 80-90%.
- **Validation:** 5-10% (for early stopping, hyperparameter selection).
- **Test (held-out):** 5-10% (for final evaluation; **never touched during training**).

### Step 6: Version and document
- Tag with version (e.g., `billing-support-v1.0`).
- Document: source, license, size, date, quality checks performed.
- Store in a versioned location (HuggingFace Dataset, S3, etc.).

---

## Hands-on labs

### Lab 1: Inspect a public dataset (0.5 day)
- Pick `OpenHermes-2.5` or `Tulu-3-SFT-mixture`.
- Read 100 random examples.
- Compute length distribution.
- Check for duplicates with MinHash.
- Check for MMLU contamination with the decontaminator.

### Lab 2: Build a small domain dataset (1-2 days)
- Pick a domain (e.g., "explain ML papers").
- Generate 1K synthetic samples with a strong LLM (DeepSeek V3 or Claude).
- Filter by quality.
- Format as ChatML.
- Verify chat template tokenization.
- Push to HuggingFace Hub.

### Lab 3: Build a DPO dataset (1-2 days)
- Take the SFT dataset from Lab 2.
- Generate 2-3 candidate completions per prompt with a strong LLM.
- Rank with LLM-as-judge (e.g., Claude-as-judge with a rubric).
- Format as DPO `{prompt, chosen, rejected}`.
- Verify hard-negative distribution (not all obvious wins).

### Lab 4: Decontamination pipeline (1 day)
- Take a known contaminated dataset (e.g., one of the Orca-style mixes).
- Run the lm-evaluation-harness decontaminator against MMLU, HumanEval, GSM8K.
- Measure how much was stripped.
- Document the result.

### Lab 5: Format consistency check (0.5 day)
- Take a dataset in Alpaca format.
- Convert to ShareGPT, OpenAI ChatML, Llama-3 chat template.
- Compare token counts.
- Verify the model's chat template handles each format correctly.

---

## Pre-flight checklist

- [ ] Name the 3 standard SFT formats and when to use each.
- [ ] What is the DPO preference format? What's the "hard negative" rule?
- [ ] What is decontamination and why is it non-negotiable?
- [ ] State the 5 rules that actually move fine-tuning metrics.
- [ ] What license is Alpaca under? Can you fine-tune commercially on it?
- [ ] What is the EU AI Act's main impact on fine-tuning practice?
- [ ] For a domain dataset, what's the right train/val/test split?
- [ ] What is MinHash used for in the data prep pipeline?
- [ ] What is model collapse? How do you prevent it with synthetic data?
- [ ] How do you assess data quality before training?

**Next:** [[fine-tuning-roadmap/03 - Frameworks & Tools 2026]] — Unsloth, Axolotl, LLaMA-Factory, TRL, PEFT, TorchTune. The 2026 stack.
