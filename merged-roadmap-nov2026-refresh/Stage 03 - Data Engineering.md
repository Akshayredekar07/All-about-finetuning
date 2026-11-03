# **Stage 03 — Data Engineering**

**Libraries:** `datasets` 5.1.0 · `datasketch` · `transformers` tokenizers · `presidio` (PII) · `distilabel` (synthetic)
**Outcome:** a dataset you can defend — sourced, licensed, deduplicated, decontaminated, format-checked, versioned, and documented.

**Why this stage exists at all:** the sources' own verdict was "a bad dataset will defeat a perfect training loop every time" and "data > algorithm, every time". Every later stage (training, alignment, evaluation, compliance) consumes this stage's output, so it sits before training rather than beside it.

## **Data shapes**

Match the data shape to the method before you choose a method.

| Data shape | Used for | Format | Example datasets |
|---|---|---|---|
| Instruction-completion pairs | SFT | Alpaca, ShareGPT, ChatML | Alpaca, Dolly, OpenHermes, Tulu |
| Multi-turn chat | SFT (chat) | ShareGPT, ChatML | ShareGPT, UltraChat |
| Tool-use / function-calling | SFT (agents) | OpenAI tools format | ToolBench, Glaive, xLAM |
| Preference pairs (chosen, rejected) | DPO, IPO, SimPO | `{prompt, chosen, rejected}` | UltraFeedback, HH-RLHF, Argilla DPO mix |
| Binary good/bad | KTO | `{prompt, completion, label: bool}` | Real user feedback |
| Verifiable-reward tasks | GRPO, RLVR | `{prompt, verifiable_check_fn}` | MATH, GSM8K, HumanEval, MBPP |
| Distillation | Teacher → student | Same as SFT, plus teacher logits | Teacher model outputs |

## **Format specifications**

The shape determines the columns, and the columns determine which trainer can consume your data without conversion.

**Alpaca (instruction-completion)**
```json
{
  "instruction": "Explain speculative decoding in 2 sentences.",
  "input": "",
  "output": "Speculative decoding uses a small draft model to propose several tokens..."
}
```

**ShareGPT (multi-turn chat)**
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
Each model's chat template maps `from` to role tokens. Make sure your template matches the target model's (for example Llama-3 uses `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...`).

**OpenAI ChatML (the de facto standard)**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "Paris."}
  ]
}
```
Used by OpenAI, Mistral, Qwen, Yi, and most 2024–2026 models. The safest default for new projects; most frameworks auto-convert.

**OpenAI tool-calling**
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
    {"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {}}}
  ]
}
```
The standard for agent / function-calling fine-tunes. The model learns to call tools, receive results, and continue.

**DPO preference format**
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

The hard-negative rule
The most useful preference pairs are where chosen and rejected are almost equally good. If chosen is obviously better, the model learns nothing. If they are close, the model has to learn the preference.

**GRPO / RLVR format**
```json
{
  "prompt": "What is 47 * 83?",
  "verifier": "lambda completion: int(eval(completion.strip())) == 47 * 83",
  "reward": 1.0
}
```

## **Public dataset catalog**

SFT
| Dataset | Size | Domain | Notes |
|---|---|---|---|
| Alpaca | 52K | General instruction | The original. Self-instruct from GPT-3.5. |
| Dolly | 15K | Human-written | High quality but small. |
| OpenHermes-2.5 | 1M | Mixed | The 2024 workhorse. From GPT-4 outputs. |
| Tulu-3-SFT-mixture | 940K | Mixed (code, math, safety) | Allen AI's curated mix. 2025–2026 standard. |
| UltraChat | 1.5M | Multi-turn | Multi-turn, broad coverage. |
| SmolLM-Corpus | 2T tokens | Web | Pretraining data, not SFT. |

DPO / Preference
| Dataset | Size | Notes |
|---|---|---|
| UltraFeedback | 64K | The DPO workhorse. GPT-4 ranked responses. |
| HH-RLHF | 170K | Anthropic's helpful + harmless. |
| Argilla DPO mix | ~50K | Curated mix from Argilla. |
| Tulu-3-Preference-mixture | ~300K | Allen AI's preference mix. Strong 2026 default. |

Tool use / function calling
| Dataset | Notes |
|---|---|
| ToolBench | Multi-step tool use. |
| Glaive-function-calling-v2 | 113K samples. Single + multi-turn. |
| xLAM-function-calling-60k | Salesforce. 2025–2026 standard for function-calling SFT. |
| ToolACE | Multi-turn, complex tool use. |

Reasoning / RLVR
| Dataset | Domain | Verifier |
|---|---|---|
| MATH | Math olympiad | `\boxed{...}` parsing |
| GSM8K | Grade school math | Numeric answer |
| AIME 1983–2024 | Competition math | Integer answer |
| HumanEval / MBPP | Code | `exec` test cases |
| LiveCodeBench | Code (contamination-free) | `exec` test cases |
| MMLU-Pro | Multi-domain QA | Multiple choice |
| ARC-AGI | Abstract reasoning | Grid match |

Multimodal
| Dataset | Notes |
|---|---|
| LLaVA-Instruct | 150K. The 2024 visual instruction workhorse. |
| ShareGPT4V | 1.2M. Higher quality captions. |
| ALLaVA-4V | 1.4M. 2025. |
| Cambrian-10M | 10M. The 2025 state-of-the-art visual instruction mix. |

## **The five rules**

1. **Diversity over volume.** 5K diverse beats 50K near-duplicates.
2. **Hard negatives.** Preference pairs where chosen and rejected are almost equally good teach more than obvious wins.
3. **Decontamination.** Strip eval-set leakage from training data. Always.
4. **Format consistency.** Tokenize early to catch chat-template mismatches before you waste ten GPU-hours.
5. **PII and licensing.** Document provenance. The EU AI Act applies.

## **Quality assessment**

- Read 100 random examples. If the quality is bad in 100, it is bad in 1M.
- Check the token-length distribution. Long-tail tokens indicate bloat; trim or filter.
- Check the source distribution. If 80% of your data comes from one source, you have single-source bias.
- Check for near-duplicates with MinHash or sentence embeddings.
- Check for eval contamination (n-gram overlap with MMLU, HumanEval, and so on).

```python
# pip install datasketch
from datasketch import MinHash, MinHashLSH

def minhash_dedup(texts, threshold=0.8):
    """Remove near-duplicates using MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    keep = []
    for i, text in enumerate(texts):
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode())
        if i == 0:
            lsh.insert(i, m)
            keep.append(i)
        else:
            if not lsh.query(m):
                lsh.insert(i, m)
                keep.append(i)
    return keep
```

For benchmark contamination specifically, use the harness's own n-gram machinery (`--decontamination_ngrams_path`) rather than a hand-rolled overlap check, so the measurement matches how the evaluation will be scored [53].

## **Format consistency check**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

sample = {
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
}

formatted = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
print(formatted)
# <|begin_of_text|><|start_header_id|>user<|end_header_id|>
#
# Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# Hi there!<|eot_id|>

tokens = tokenizer(formatted, return_tensors="pt")
print(f"Token count: {tokens.input_ids.shape[1]}")
```

## **Synthetic data**

**When to use:** you do not have enough real data; you need a rare edge case; you need preference pairs at scale (generate candidates with a strong model, rank with a stronger one).

**The standard pattern**
1. Generate candidates with a strong model.
2. Filter by quality (length, format, required concepts, basic checks).
3. Rank for preference data with an LLM judge and an explicit rubric.
4. Decontaminate against your eval set.
5. Train on the result.

**Tools:** Distilabel (Argilla) for generation + filtering + ranking — note the PyPI release is 1.5.3 from Jan 2025, so pin it and read the source before relying on new features [29]; LangSmith (LangChain's tracing and eval framework) for tracing and eval; PromptTools for prompt A/B testing; HuggingFace auto-DPO data generation.

**Gotchas**
- Model collapse: training on synthetic data from the same model degrades over generations. Always mix in real data.
- Style homogenization: synthetic data carries a detectable generator voice. Mix sources.
- Hallucinations: synthetic data inherits the generator's errors. Manually verify a sample.
- Licensing: outputs from a closed model have murky terms. Open-weight generators are safer.

## **Licensing and the EU AI Act**

The Act applies progressively, and the fine-tuning-relevant dates are now concrete [18][19][20]:

| Date | What applies |
|---|---|
| 2 Feb 2025 | Prohibitions, definitions, AI-literacy obligations |
| 2 Aug 2025 | Governance and general-purpose AI model obligations |
| **2 Aug 2026** | Main rules: Article 50 transparency, innovation measures, **enforcement begins** for prohibitions, GPAI, transparency, AI literacy |
| **2 Dec 2026** | New prohibitions (non-consensual sexual deepfakes, CSAM) + Article 50(2) transition deadline for systems already on the market |
| 2 Dec 2027 | High-risk obligations for Annex III use cases (education, employment, law enforcement, migration…) |
| 2 Aug 2028 | High-risk obligations for AI embedded in regulated products (Annex I) |

The Digital Omnibus on AI entered into force 27 July 2026 and is what moved the high-risk dates. Systems placed on the market before 2 Aug 2026 come into scope if they are **significantly modified** — which a fine-tune of your own model can be. Document that assessment rather than assuming you are out of scope.

**Fine-tuning obligations in practice**
- Document training data provenance: where it came from, under what license, whether consent was obtained.
- Model card: intended use, limitations, training data summary.
- Risk assessment if the use case touches Annex III.
- Logging of significant model updates and eval results.
- The GPAI downstream-modifier threshold (a model whose training compute is more than one third of the original provider's, roughly 3·10²¹ FLOP, or 3·10²⁴ for systemic risk) is far above a 7B LoRA, but documenting the computation is cheap and becomes relevant if you scale.

**License landscape**

| Source | License | Commercial fine-tune OK? |
|---|---|---|
| Alpaca | CC BY-NC 4.0 | No (NC = non-commercial) |
| Dolly | CC BY-SA 3.0 | Yes (with share-alike) |
| OpenHermes-2.5 | Apache 2.0 (mostly) | Yes (check per-source) |
| Tulu-3 | ODC-BY under the AI2 license, **but the SFT mixture embeds NC subsets** (No Robots, NuminaMath-TIR cc-by-nc-4.0) | Not cleanly — audit the mixture, or use Tulu-3 subsets you have cleared |
| UltraFeedback | MIT | Yes |
| HH-RLHF | MIT | Yes |
| xLAM | CC BY-NC 4.0 | No (NC) |
| HumanEval | MIT | Yes, for evaluation only — never train on it |
| Wikipedia | CC BY-SA | Yes (with share-alike) |
| Common Crawl | Varies | Filtering required |
| The Stack / CodeParrot | Various permissive | Yes (check per-file) |

**PII and consent**
- Strip PII before training. Use Presidio, PII-Codes, or scrub with a strong LLM.
- For user-generated data, GDPR consent is required. Document the consent flow.
- For synthetic data there is less PII risk, but verify the generator did not memorize PII from its training data.

## **Domain dataset construction**

**Step 1 — Define the task.** "Customer support chatbot" is too vague. "SaaS billing questions, structured JSON output, with citations" is actionable.

**Step 2 — Source candidates.** Real data (past conversations, support tickets, internal docs with consent and PII scrubbing), synthetic data (generate with a strong LLM using a detailed rubric), public data (relevant Hub datasets with a license check).

**Step 3 — Filter and clean.** Deduplicate with MinHash, filter by quality, decontaminate against eval sets.

**Step 4 — Format.** Convert to the target format, apply the chat template, verify tokenization.

**Step 5 — Split.** Train 80–90%, validation 5–10% for early stopping and hyperparameter selection, test 5–10% held out and never touched during training.

**Step 6 — Version and document.** Tag it (`billing-support-v1.0`), document source, license, size, date, and quality checks performed, and store it in a versioned location (Hub dataset, S3).

## **Data labs**

**Lab 1 — Inspect a public dataset.** Take `OpenHermes-2.5` or `Tulu-3-SFT-mixture`; read 100 random examples; compute the length distribution; check for duplicates with MinHash; check for MMLU contamination.

**Lab 2 — Build a small domain dataset.** Pick a domain (for example "explain ML papers"); generate 1K synthetic samples with a strong model; filter by quality; format as ChatML; verify chat-template tokenization; push to the Hub with a data card.

**Lab 3 — Build a DPO dataset.** Reuse Lab 2's SFT data; generate 2–3 candidate completions per prompt; rank with an LLM judge and a rubric; format as `{prompt, chosen, rejected}`; verify the hard-negative distribution is not all obvious wins.

**Lab 4 — Decontamination pipeline.** Take a known-contaminated dataset; run the harness decontamination flow against MMLU, HumanEval, and GSM8K using the n-gram file path; measure how much was stripped; document the result.

**Lab 5 — Format consistency check.** Convert an Alpaca-format dataset to ShareGPT, ChatML, and the Llama-3 chat template; compare token counts; verify the model's chat template handles each format correctly.

## **Data pre-flight checklist**

- [ ] Name the three standard SFT formats and when to use each.
- [ ] What is the DPO preference format? What is the hard-negative rule?
- [ ] What is decontamination and why is it non-negotiable?
- [ ] State the five rules that actually move fine-tuning metrics.
- [ ] What license is Alpaca under? Can you fine-tune commercially on it?
- [ ] What is the EU AI Act's main impact on fine-tuning practice, and which dates apply this quarter?
- [ ] For a domain dataset, what is the right train/val/test split?
- [ ] What is MinHash used for in the data prep pipeline?
- [ ] What is model collapse? How do you prevent it with synthetic data?
- [ ] How do you assess data quality before training?

---

---

## **Worksheet additions**

## **Scope**
Stage 2 sits before training on purpose. A bad dataset defeats a perfect training loop every time; the dominant failure mode of a fine-tune is data, not algorithm. Every later stage consumes this stage's output: Stage 3 trains on it, Stage 4 aligns on it, Stage 8 benchmarks against it and publishes it.

**Exit criterion:** a public domain dataset on the Hub with a data card, a license table, a decontamination report, and a held-out split that no training run ever touched.

## **The five rules**
1. **Diversity over volume.** 5K diverse examples beat 50K near-duplicates.
2. **Hard negatives.** Preference pairs where chosen and rejected are almost equally good teach more than obvious wins.
3. **Decontamination.** Strip eval-set leakage from training data. Always.
4. **Format consistency.** Tokenize early to catch chat-template mismatches before you waste ten GPU-hours.
5. **PII and licensing.** Document provenance; the EU AI Act applies from 2 August 2026.

---

## **Data shape table**
| Data shape | Used for | Columns | Example datasets |
|---|---|---|---|
| Instruction-completion | SFT | `instruction`, `input`, `output` | Alpaca, Dolly, OpenHermes, Tulu |
| Multi-turn chat | SFT (chat) | `conversations` or `messages` | ShareGPT, UltraChat |
| Tool-use / function-calling | SFT (agents) | `messages` + `tools` | ToolBench, Glaive, xLAM |
| Preference pairs | DPO, IPO, SimPO | `prompt`, `chosen`, `rejected` | UltraFeedback, HH-RLHF, Argilla DPO mix |
| Binary feedback | KTO | `prompt`, `completion`, `desirable` | Real user feedback |
| Verifiable-reward tasks | GRPO, RLVR | `prompt` + verifier function | MATH, GSM8K, HumanEval, MBPP |
| Distillation | Teacher → student | SFT fields + teacher logits | Teacher model outputs |
| Agent traces | Agent SFT | `messages` with tool calls | datasets 5.x trace loaders |

---

### **Format design**
- [ ] Choosing one canonical format for the project and converting everything into it
- [ ] Alpaca: `instruction` / `input` / `output`
- [ ] ShareGPT: `conversations` with `from` / `value`
- [ ] ChatML: `messages` with `role` / `content`
- [ ] OpenAI tool-calling: `messages` with `tool_calls` + `tools`
- [ ] DPO: `prompt` / `chosen` / `rejected` as message lists
- [ ] GRPO / RLVR: `prompt` plus a programmatic verifier
- [ ] Matching the target model's chat template exactly, including special tokens
- [ ] Deciding whether multi-turn conversations keep the prompt in every turn or only the first

### **Catalog selection**
- [ ] SFT: Alpaca (52K), Dolly (15K), OpenHermes-2.5 (1M), Tulu-3-SFT-mixture (940K), UltraChat (1.5M)
- [ ] Preference: UltraFeedback (64K), HH-RLHF (170K), Argilla DPO mix (~50K), Tulu-3-Preference-mixture (~300K)
- [ ] Tool use: ToolBench, Glaive-function-calling-v2 (113K), xLAM-function-calling-60k, ToolACE
- [ ] Reasoning: MATH, GSM8K, AIME 1983–2024, HumanEval, MBPP, LiveCodeBench, MMLU-Pro, ARC-AGI
- [ ] Multimodal: LLaVA-Instruct (150K), ShareGPT4V (1.2M), ALLaVA-4V (1.4M), Cambrian-10M
- [ ] Pretraining corpora are not SFT data (SmolLM-Corpus, C4) — know the difference

### **Quality assessment**
- [ ] Read 100 random examples before anything else
- [ ] Token-length distribution, and a decision about the long tail
- [ ] Source distribution — 80% from one source is single-source bias
- [ ] Near-duplicate detection with MinHash / LSH
- [ ] Exact-duplicate detection with a hash pass
- [ ] Eval-set contamination (n-gram overlap with MMLU, HumanEval, GSM8K)
- [ ] Label sanity checks: per-class counts, class balance, annotation agreement on a sample
- [ ] Refusal and safety coverage for a public-facing model

### **Decontamination**
- [ ] Generating n-gram files for your training set
- [ ] Running the harness with `--decontamination_ngrams_path`
- [ ] Reading the `*_decontaminate` metrics, not just the clean metrics
- [ ] Recording how many documents were stripped and which benchmark triggered it

### **Synthetic data**
- [ ] Generate → filter → rank → decontaminate → train
- [ ] Rubric-driven generation rather than free-form prompting
- [ ] Real-data mixing ratio to prevent model collapse
- [ ] Manual spot-check of a sample for hallucinations
- [ ] Licensing posture of the generator's output

### **Compliance**
- [ ] Provenance table: source, license, date, consent basis
- [ ] PII stripping (Presidio, or an LLM scrub with a sample audit)
- [ ] GDPR consent trail for user-generated data
- [ ] Model card sections for intended use and limitations
- [ ] Logging of significant updates and eval results

---

### **Canonical ChatML conversion**
```python
import json
from datasets import Dataset

def to_chatml(records):
    """records: list of {instruction, input, output} -> list of {messages}."""
    out = []
    for r in records:
        user = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        out.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": r["output"]},
        ]})
    return out

ds = Dataset.from_list(to_chatml(json.load(open("raw.jsonl"))))
```

### **Near-duplicate removal with MinHash**
```python
# pip install datasketch
from datasketch import MinHash, MinHashLSH

def minhash_dedup(texts, threshold=0.8, num_perm=128):
    """Keep one representative per near-duplicate cluster."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = []
    for i, text in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for word in text.split():
            m.update(word.encode())
        if i == 0:
            lsh.insert(i, m)
            keep.append(i)
        elif not lsh.query(m):
            lsh.insert(i, m)
            keep.append(i)
    return keep

texts = ds["text"]
keep  = minhash_dedup(texts)
clean = ds.select(keep)
print(f"{len(ds)} -> {len(clean)} after dedup")
```

### **Exact-duplicate pass**
```python
import hashlib

def dedup_exact(ds, column="text"):
    seen, keep = set(), []
    for i, row in enumerate(ds):
        h = hashlib.sha256(row[column].strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return ds.select(keep)
```

### **Format consistency check before training**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

sample = {"messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]}

formatted = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
print(formatted)
tokens = tokenizer(formatted, return_tensors="pt")
print(f"Token count: {tokens.input_ids.shape[1]}")

# Run this over 500 random rows and look at the distribution, not one row.
# A bimodal distribution means two chat templates are mixed in the dataset.
```

### **Length distribution and source balance**
```python
import numpy as np
from collections import Counter

lens = [len(tokenizer(text).input_ids) for text in ds["text"][:5000]]
print("p50", np.percentile(lens, 50), "p95", np.percentile(lens, 95), "max", max(lens))
print("share over 4096 tokens:", np.mean(np.array(lens) > 4096))

print(Counter(ds["source"]).most_common(10))
```

### **PII scrubbing**
```python
# pip install presidio-analyzer presidio-anonymizer
from presidio_anonymizer import AnonymizerEngine

engine = AnonymizerEngine()
result = engine.anonymize(text=text, language="en")
print(result.text)
```

### **Synthetic preference generation with a rubric**
```python
# pip install distilabel
# Distilabel's latest PyPI release is 1.5.3 (Jan 2025) — pin it and read the
# source for the current API before wiring it into a pipeline.
from distilabel.pipeline import Pipeline
from distilabel.models import InferenceAPIModel, TextGeneration

# build: prompts -> N candidates -> judge (rubric) -> best/worst pairs
```

### **Domain dataset construction, end to end**
```python
from datasets import Dataset, load_dataset

# 1. Define: "SaaS billing questions, structured JSON output, with citations"
# 2. Source: real tickets (consent + PII scrub) + synthetic (rubric) + public (licensed)
# 3. Clean: dedup -> quality filter -> decontaminate
# 4. Format: convert to ChatML and verify tokenization
# 5. Split: 85 / 7.5 / 7.5
train, rest = ds.train_test_split(test_size=0.15, seed=42)
val, test   = rest.train_test_split(test_size=0.5, seed=42)

# 6. Version and document
ds = DatasetDict({"train": train, "validation": val, "test": test})
ds.push_to_hub("username/billing-support-v1.0")
```

```python
from datasets import DatasetDict
DatasetDict({"train": train, "validation": val, "test": test}).push_to_hub(
    "username/billing-support-v1.0")
```

---

## **Licensing and the EU AI Act**
| Date | What applies |
|---|---|
| 2 Feb 2025 | Prohibitions, definitions, AI literacy |
| 2 Aug 2025 | Governance and GPAI obligations |
| **2 Aug 2026** | Main rules: Article 50 transparency, innovation measures, **enforcement starts** for prohibitions, GPAI, transparency, AI literacy |
| **2 Dec 2026** | New prohibitions (non-consensual sexual deepfakes, CSAM) + Article 50(2) transition deadline for systems already on the market |
| 2 Dec 2027 | Annex III high-risk obligations (education, employment, law enforcement, migration) |
| 2 Aug 2028 | Annex I high-risk obligations (AI embedded in regulated products) |

The Digital Omnibus on AI entered into force 27 July 2026 and is what moved the high-risk dates. Systems placed on the market before 2 August 2026 enter scope when they are **significantly modified** — a fine-tune of your own model can qualify, so write the assessment down rather than assuming you are exempt.

**License landscape**

| Source | License | Commercial fine-tune |
|---|---|---|
| Alpaca | CC BY-NC 4.0 | No |
| Dolly | CC BY-SA 3.0 | Yes, share-alike |
| OpenHermes-2.5 | Apache 2.0 (mostly) | Yes, check per source |
| Tulu-3 SFT mixture | ODC-BY, but embeds NC subsets (No Robots, NuminaMath-TIR cc-by-nc-4.0) | Not cleanly — audit the mixture |
| UltraFeedback | MIT | Yes |
| HH-RLHF | MIT | Yes |
| xLAM | CC BY-NC 4.0 | No |
| HumanEval | MIT | Evaluation only, never train on it |
| Wikipedia | CC BY-SA | Yes, share-alike |
| Common Crawl | Varies | Filtering required |
| The Stack / CodeParrot | Various permissive | Yes, check per file |

**Fine-tuning obligations in practice:** document data provenance, publish a model card with intended use and limitations, run a risk assessment if the use case touches Annex III, and log significant model updates and eval results. The GPAI downstream-modifier threshold (more than one third of the original provider's training compute, roughly 3·10²¹ FLOP; 3·10²⁴ for systemic risk) sits far above a 7B LoRA, but documenting the computation is cheap and matters if you scale.

---

### **Lab 2-1 — Inspect a public dataset**
- Take `OpenHermes-2.5` or `Tulu-3-SFT-mixture`.
- Read 100 random examples.
- Compute the length distribution.
- Check for duplicates with MinHash.
- Check MMLU contamination with the harness decontaminator.

### **Lab 2-2 — Build a small domain dataset**
- Pick a domain (for example "explain ML papers").
- Generate 1K synthetic samples with a strong model and a rubric.
- Filter by quality.
- Format as ChatML.
- Verify chat-template tokenization across 500 rows.
- Push to the Hub with a data card.

### **Lab 2-3 — Build a DPO dataset**
- Reuse Lab 2-2's prompts.
- Generate 2–3 candidate completions per prompt.
- Rank with an LLM judge and an explicit rubric.
- Format as `{prompt, chosen, rejected}`.
- Measure the hard-negative rate: what share of pairs are "close calls" rather than obvious wins.

### **Lab 2-4 — Decontamination pipeline**
- Take a known-contaminated dataset.
- Generate n-gram files and run the harness against MMLU, HumanEval, and GSM8K.
- Measure how much was stripped, per benchmark.
- Document the result in the data card.

### **Lab 2-5 — Format consistency check**
- Convert an Alpaca-format dataset to ShareGPT, ChatML, and the target model's chat template.
- Compare token counts and length distributions.
- Verify the chat template handles each format correctly, including multi-turn.

### **Lab 2-6 — Compliance pass**
- Build the provenance table for your dataset: source, license, date, consent basis.
- Run PII scrubbing and manually audit 100 outputs.
- Write the data card: intended use, known limits, collection method, license, eval contamination status.
- Assess whether your use case touches Annex III and record the conclusion.

---

## **Pitfalls**
- **"5K diverse" beats "50K near-duplicates"** — synthetic generation without dedup is the single most common way people waste a training budget.
- **A chat-template mismatch is invisible until you tokenize.** Always render 500 rows, not one.
- **Decontamination is not optional.** MMLU and similar evals appear inside common SFT mixtures; if you train on them, your eval numbers are not measurements.
- **The fake `decontaminate()` API does not exist.** Use the harness's n-gram file flow.
- **Tulu-3 is not automatically clean.** The SFT mixture is ODC-BY but embeds non-commercial subsets.
- **Synthetic data collapses without real data in the mix**, and inherits the generator's hallucinations.
- **PII in a public dataset is a legal event, not a quality metric.** Scrub before pushing, not after.
- **A test split you have looked at is a validation split.** Reserve it, write it down, and do not open it until the final eval.

---

## **Acceptance gate**
- [ ] The five rules can each be explained in one sentence with an example from your own data.
- [ ] A domain dataset of 1K–5K examples is public on the Hub with a data card.
- [ ] Deduplication report: how many rows removed by exact and near-duplicate passes.
- [ ] Decontamination report: which benchmarks, how many documents stripped.
- [ ] Token-length distribution plotted; an explicit trimming or filtering decision recorded.
- [ ] Provenance and license table attached to the data card.
- [ ] PII scrub audited on 100 examples.
- [ ] Train / validation / test split fixed at 85/7.5/7.5 with a recorded seed.

---

## **Resources**
- datasets on PyPI — https://pypi.org/project/datasets/
- EU AI Act implementation timeline — https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act
- Navigating the AI Act — https://digital-strategy.ec.europa.eu/en/faqs/navigating-ai-act
- lm-eval decontamination guide — https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md
- distilabel on PyPI — https://pypi.org/project/distilabel/

**Next:** Stage 04 — Training Core.
