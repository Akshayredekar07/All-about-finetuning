## **Stage 03: Data Engineering**

**Libraries:** `datasets`, `datasketch`, `transformers` tokenizers, `presidio` (PII), `distilabel` (synthetic data)

**Outcome:** a dataset you can defend: sourced, licensed, deduplicated, decontaminated, format-checked, versioned, and documented.

### **1. Why This Stage Exists**

### **1.1 Purpose**

The sources' own verdict was "a bad dataset will defeat a perfect training loop every time" and "data > algorithm, every time". The dominant failure mode of a fine-tune is data, not algorithm. Every later stage (training, alignment, evaluation, compliance) consumes this stage's output, so it sits before training rather than beside it. Training uses the data, alignment uses it, and benchmarking and publishing are done against it.

### **1.2 Exit Criterion**

A public domain dataset on the Hub with a data card, a license table, a decontamination report, and a held-out split that no training run ever touched.

### **2. Data Shapes**

Match the data shape to the method before you choose a method.

| **Data shape** | **Used for** | **Format** | **Columns** | **Example datasets** |
|---|---|---|---|---|
| Instruction-completion pairs | SFT | Alpaca, ShareGPT, ChatML | `instruction`, `input`, `output` | Alpaca, Dolly, OpenHermes, Tulu |
| Multi-turn chat | SFT (chat) | ShareGPT, ChatML | `conversations` or `messages` | ShareGPT, UltraChat |
| Tool-use / function-calling | SFT (agents) | OpenAI tools format | `messages` + `tools` | ToolBench, Glaive, xLAM |
| Preference pairs (chosen, rejected) | DPO, IPO, SimPO | `{prompt, chosen, rejected}` | `prompt`, `chosen`, `rejected` | UltraFeedback, HH-RLHF, Argilla DPO mix |
| Binary good/bad | KTO | `{prompt, completion, label: bool}` | `prompt`, `completion`, `desirable` (the boolean label) | Real user feedback |
| Verifiable-reward tasks | GRPO, RLVR | `{prompt, verifiable_check_fn}` | `prompt` + verifier function | MATH, GSM8K, HumanEval, MBPP |
| Distillation | Teacher to student | Same as SFT, plus teacher logits | SFT fields + teacher logits | Teacher model outputs |
| Agent traces | Agent SFT | `messages` with tool calls | `messages` with tool calls | `datasets` trace loaders |

### **3. Format Specifications**

The shape determines the columns, and the columns determine which trainer can consume your data without conversion.

### **3.1 Alpaca (instruction-completion)**

```json
{
  "instruction": "Explain speculative decoding in 2 sentences.",
  "input": "",
  "output": "Speculative decoding uses a small draft model to propose several tokens..."
}
```

### **3.2 ShareGPT (multi-turn chat)**

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

### **3.3 OpenAI ChatML (the de facto standard)**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "Paris."}
  ]
}
```

Used by OpenAI, Mistral, Qwen, Yi, and most 2024 to 2026 models. It is the safest default for new projects, and most frameworks auto-convert.

### **3.4 OpenAI Tool-Calling**

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

This is the standard for agent and function-calling fine-tunes. The model learns to call tools, receive results, and continue.

### **3.5 DPO Preference Format**

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

**The hard-negative rule:** the most useful preference pairs are those where chosen and rejected are almost equally good. If chosen is obviously better, the model learns nothing. If they are close, the model has to learn the preference.

### **3.6 GRPO / RLVR Format**

```json
{
  "prompt": "What is 47 * 83?",
  "verifier": "lambda completion: int(eval(completion.strip())) == 47 * 83",
  "reward": 1.0
}
```

### **3.7 Format Design Checklist**

- Choose one canonical format for the project and convert everything into it
- Alpaca: `instruction` / `input` / `output`
- ShareGPT: `conversations` with `from` / `value`
- ChatML: `messages` with `role` / `content`
- OpenAI tool-calling: `messages` with `tool_calls` + `tools`
- DPO: `prompt` / `chosen` / `rejected` as message lists
- GRPO / RLVR: `prompt` plus a programmatic verifier
- Match the target model's chat template exactly, including special tokens
- Decide whether multi-turn conversations keep the prompt in every turn or only the first

### **3.8 Canonical ChatML Conversion**

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

### **4. Public Dataset Catalog**

### **4.1 SFT**

| **Dataset** | **Size** | **Domain** | **Notes** |
|---|---|---|---|
| Alpaca | 52K | General instruction | The original. Self-instruct from GPT-3.5. |
| Dolly | 15K | Human-written | High quality but small. |
| OpenHermes-2.5 | 1M | Mixed | Widely used in 2024. From GPT-4 outputs. |
| Tulu-3-SFT-mixture | 940K | Mixed (code, math, safety) | Allen AI's curated mix. Standard for 2025 to 2026. |
| UltraChat | 1.5M | Multi-turn | Multi-turn, broad coverage. |
| SmolLM-Corpus | 2T tokens | Web | Pretraining data, not SFT. |

### **4.2 DPO / Preference**

| **Dataset** | **Size** | **Notes** |
|---|---|---|
| UltraFeedback | 64K | The DPO workhorse. GPT-4 ranked responses. |
| HH-RLHF | 170K | Anthropic's helpful + harmless. |
| Argilla DPO mix | ~50K | Curated mix from Argilla. |
| Tulu-3-Preference-mixture | ~300K | Allen AI's preference mix. Strong 2026 default. |

### **4.3 Tool Use / Function Calling**

| **Dataset** | **Notes** |
|---|---|
| ToolBench | Multi-step tool use. |
| Glaive-function-calling-v2 | 113K samples. Single + multi-turn. |
| xLAM-function-calling-60k | Salesforce. Standard for 2025 to 2026 function-calling SFT. |
| ToolACE | Multi-turn, complex tool use. |

### **4.4 Reasoning / RLVR**

| **Dataset** | **Domain** | **Verifier** |
|---|---|---|
| MATH | Math olympiad | `\boxed{...}` parsing |
| GSM8K | Grade school math | Numeric answer |
| AIME 1983 to 2024 | Competition math | Integer answer |
| HumanEval / MBPP | Code | `exec` test cases |
| LiveCodeBench | Code (contamination-free) | `exec` test cases |
| MMLU-Pro | Multi-domain QA | Multiple choice |
| ARC-AGI | Abstract reasoning | Grid match |

### **4.5 Multimodal**

| **Dataset** | **Notes** |
|---|---|
| LLaVA-Instruct | 150K. Widely used visual instruction set in 2024. |
| ShareGPT4V | 1.2M. Higher quality captions. |
| ALLaVA-4V | 1.4M. 2025. |
| Cambrian-10M | 10M. State-of-the-art visual instruction mix in 2025. |

### **4.6 Catalog Selection Checklist**

- SFT: Alpaca (52K), Dolly (15K), OpenHermes-2.5 (1M), Tulu-3-SFT-mixture (940K), UltraChat (1.5M)
- Preference: UltraFeedback (64K), HH-RLHF (170K), Argilla DPO mix (~50K), Tulu-3-Preference-mixture (~300K)
- Tool use: ToolBench, Glaive-function-calling-v2 (113K), xLAM-function-calling-60k, ToolACE
- Reasoning: MATH, GSM8K, AIME 1983 to 2024, HumanEval, MBPP, LiveCodeBench, MMLU-Pro, ARC-AGI
- Multimodal: LLaVA-Instruct (150K), ShareGPT4V (1.2M), ALLaVA-4V (1.4M), Cambrian-10M
- Pretraining corpora are not SFT data (SmolLM-Corpus, C4). Know the difference

### **5. The Five Rules**

- **Diversity over volume.** 5K diverse examples beat 50K near-duplicates.
- **Hard negatives.** Preference pairs where chosen and rejected are almost equally good teach more than obvious wins.
- **Decontamination.** Strip eval-set leakage from training data. Always.
- **Format consistency.** Tokenize early to catch chat-template mismatches before you waste ten GPU-hours.
- **PII and licensing.** Document provenance. The EU AI Act applies from 2 August 2026.

### **6. Quality Assessment**

### **6.1 What to Check**

- Read 100 random examples before anything else. If the quality is bad in 100, it is bad in 1M.
- Check the token-length distribution. Long-tail tokens indicate bloat, so trim or filter, and record a decision about the long tail.
- Check the source distribution. If 80% of your data comes from one source, you have single-source bias.
- Check for near-duplicates with MinHash / LSH or sentence embeddings.
- Check for exact duplicates with a hash pass.
- Check for eval contamination (n-gram overlap with MMLU, HumanEval, GSM8K).
- Run label sanity checks: per-class counts, class balance, annotation agreement on a sample.
- Check refusal and safety coverage for a public-facing model.

### **6.2 Near-Duplicate Removal with MinHash**

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

### **6.3 Exact-Duplicate Pass**

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

### **6.4 Length Distribution and Source Balance**

```python
import numpy as np
from collections import Counter

lens = [len(tokenizer(text).input_ids) for text in ds["text"][:5000]]
print("p50", np.percentile(lens, 50), "p95", np.percentile(lens, 95), "max", max(lens))
print("share over 4096 tokens:", np.mean(np.array(lens) > 4096))

print(Counter(ds["source"]).most_common(10))
```

### **7. Format Consistency Check**

### **7.1 Llama-3 Example**

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

### **7.2 Qwen Example Before Training**

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

### **8. Decontamination**

For benchmark contamination, use the harness's own n-gram machinery (`--decontamination_ngrams_path`) rather than a hand-rolled overlap check, so the measurement matches how the evaluation will be scored.

- Generate n-gram files for your training set
- Run the harness with `--decontamination_ngrams_path`
- Read the `*_decontaminate` metrics, not just the clean metrics
- Record how many documents were stripped and which benchmark triggered it
- Note that a `decontaminate()` API of the kind sometimes shown in tutorials does not exist. Use the harness's n-gram file flow.

### **9. Synthetic Data**

### **9.1 When to Use**

You do not have enough real data, you need a rare edge case, or you need preference pairs at scale (generate candidates with a strong model, rank with a stronger one).

### **9.2 The Standard Pattern**

- Generate candidates with a strong model (rubric-driven generation rather than free-form prompting).
- Filter by quality (length, format, required concepts, basic checks).
- Rank for preference data with an LLM judge and an explicit rubric.
- Decontaminate against your eval set.
- Train on the result.

### **9.3 Tools**

Distilabel (Argilla) for generation, filtering, and ranking. Its latest PyPI release dates from January 2025, so pin your installed version and read the source before relying on newer features. LangSmith (LangChain's tracing and eval framework) for tracing and eval. PromptTools for prompt A/B testing. HuggingFace auto-DPO data generation.

### **9.4 Synthetic Preference Generation with a Rubric**

```python
# pip install distilabel
# Pin the distilabel version and read the source for the current API
# before wiring it into a pipeline.
from distilabel.pipeline import Pipeline
from distilabel.models import InferenceAPIModel, TextGeneration

# build: prompts -> N candidates -> judge (rubric) -> best/worst pairs
```

### **9.5 Gotchas**

- **Model collapse:** training on synthetic data from the same model degrades over generations. Always mix in real data, and set a real-data mixing ratio.
- **Style homogenization:** synthetic data carries a detectable generator voice. Mix sources.
- **Hallucinations:** synthetic data inherits the generator's errors. Manually verify a sample.
- **Licensing:** outputs from a closed model have murky terms. Open-weight generators are safer. Record the licensing posture of the generator's output.

### **10. Licensing and the EU AI Act**

### **10.1 Timeline**

The Act applies progressively, and the fine-tuning-relevant dates are now concrete.

| **Date** | **What applies** |
|---|---|
| 2 Feb 2025 | Prohibitions, definitions, AI-literacy obligations |
| 2 Aug 2025 | Governance and general-purpose AI (GPAI) model obligations |
| **2 Aug 2026** | Main rules: Article 50 transparency, innovation measures, **enforcement begins** for prohibitions, GPAI, transparency, AI literacy |
| **2 Dec 2026** | New prohibitions (non-consensual sexual deepfakes, CSAM) and Article 50(2) transition deadline for systems already on the market |
| 2 Dec 2027 | High-risk obligations for Annex III use cases (education, employment, law enforcement, migration, and others) |
| 2 Aug 2028 | High-risk obligations for AI embedded in regulated products (Annex I) |

The Digital Omnibus on AI entered into force on 27 July 2026 and is what moved the high-risk dates. Systems placed on the market before 2 August 2026 come into scope if they are **significantly modified**, which a fine-tune of your own model can be. Document that assessment rather than assuming you are out of scope.

### **10.2 Fine-Tuning Obligations in Practice**

- Document training data provenance: where it came from, under what license, whether consent was obtained. Keep a provenance table with source, license, date, and consent basis.
- Publish a model card with intended use, limitations, and a training data summary.
- Run a risk assessment if the use case touches Annex III.
- Log significant model updates and eval results.
- The GPAI downstream-modifier threshold is a model whose training compute is more than one third of the original provider's compute, roughly $3 \times 10^{21}$ FLOP, or $3 \times 10^{24}$ FLOP for systemic risk. This is far above a 7B LoRA, but documenting the computation is cheap and becomes relevant if you scale.

$$\boxed{3 \times 10^{21}\ \text{FLOP (GPAI)}} \qquad \boxed{3 \times 10^{24}\ \text{FLOP (systemic risk)}}$$

### **10.3 License Landscape**

| **Source** | **License** | **Commercial fine-tune OK?** |
|---|---|---|
| Alpaca | CC BY-NC 4.0 | No (NC means non-commercial) |
| Dolly | CC BY-SA 3.0 | Yes, with share-alike |
| OpenHermes-2.5 | Apache 2.0 (mostly) | Yes, check per source |
| Tulu-3 SFT mixture | ODC-BY under the AI2 license, but the mixture embeds NC subsets (No Robots, NuminaMath-TIR cc-by-nc-4.0) | Not cleanly. Audit the mixture, or use Tulu-3 subsets you have cleared |
| UltraFeedback | MIT | Yes |
| HH-RLHF | MIT | Yes |
| xLAM | CC BY-NC 4.0 | No (NC) |
| HumanEval | MIT | Evaluation only, never train on it |
| Wikipedia | CC BY-SA | Yes, with share-alike |
| Common Crawl | Varies | Filtering required |
| The Stack / CodeParrot | Various permissive | Yes, check per file |

### **11. PII and Consent**

- Strip PII before training. Use Presidio, PII-Codes, or scrub with a strong LLM (with a sample audit).
- For user-generated data, GDPR consent is required. Document the consent flow and keep a consent trail.
- For synthetic data there is less PII risk, but verify the generator did not memorize PII from its training data.

```python
# pip install presidio-analyzer presidio-anonymizer
from presidio_anonymizer import AnonymizerEngine

engine = AnonymizerEngine()
result = engine.anonymize(text=text, language="en")
print(result.text)
```

### **12. Domain Dataset Construction**

### **12.1 Steps**

- **Step 1, define the task.** "Customer support chatbot" is too vague. "SaaS billing questions, structured JSON output, with citations" is actionable.
- **Step 2, source candidates.** Real data (past conversations, support tickets, internal docs with consent and PII scrubbing), synthetic data (generate with a strong LLM using a detailed rubric), public data (relevant Hub datasets with a license check).
- **Step 3, filter and clean.** Deduplicate with MinHash, filter by quality, decontaminate against eval sets.
- **Step 4, format.** Convert to the target format, apply the chat template, verify tokenization.
- **Step 5, split.** Train 80 to 90%, validation 5 to 10% for early stopping and hyperparameter selection, test 5 to 10% held out and never touched during training. The recommended fixed split is:

$$\boxed{85\% \,/\, 7.5\% \,/\, 7.5\%}$$

- **Step 6, version and document.** Tag it (`billing-support-v1.0`), document source, license, size, date, and quality checks performed, and store it in a versioned location (Hub dataset, S3).

### **12.2 End-to-End Code**

```python
from datasets import Dataset, DatasetDict, load_dataset

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

### **13. Labs**

### **13.1 Lab 1: Inspect a Public Dataset**

- Take `OpenHermes-2.5` or `Tulu-3-SFT-mixture`.
- Read 100 random examples.
- Compute the length distribution.
- Check for duplicates with MinHash.
- Check MMLU contamination with the harness decontaminator.

### **13.2 Lab 2: Build a Small Domain Dataset**

- Pick a domain (for example "explain ML papers").
- Generate 1K synthetic samples with a strong model and a rubric.
- Filter by quality.
- Format as ChatML.
- Verify chat-template tokenization across 500 rows.
- Push to the Hub with a data card.

### **13.3 Lab 3: Build a DPO Dataset**

- Reuse Lab 2's SFT data and prompts.
- Generate 2 to 3 candidate completions per prompt.
- Rank with an LLM judge and an explicit rubric.
- Format as `{prompt, chosen, rejected}`.
- Measure the hard-negative rate: what share of pairs are close calls rather than obvious wins. Verify the distribution is not all obvious wins.

### **13.4 Lab 4: Decontamination Pipeline**

- Take a known-contaminated dataset.
- Generate n-gram files and run the harness decontamination flow against MMLU, HumanEval, and GSM8K using the n-gram file path.
- Measure how much was stripped, per benchmark.
- Document the result in the data card.

### **13.5 Lab 5: Format Consistency Check**

- Convert an Alpaca-format dataset to ShareGPT, ChatML, and the target model's chat template (for example the Llama-3 chat template).
- Compare token counts and length distributions.
- Verify the chat template handles each format correctly, including multi-turn.

### **13.6 Lab 6: Compliance Pass**

- Build the provenance table for your dataset: source, license, date, consent basis.
- Run PII scrubbing and manually audit 100 outputs.
- Write the data card: intended use, known limits, collection method, license, eval contamination status.
- Assess whether your use case touches Annex III and record the conclusion.

### **14. Pitfalls**

- **"5K diverse" beats "50K near-duplicates."** Synthetic generation without dedup is the single most common way people waste a training budget.
- **A chat-template mismatch is invisible until you tokenize.** Always render 500 rows, not one.
- **Decontamination is not optional.** MMLU and similar evals appear inside common SFT mixtures. If you train on them, your eval numbers are not measurements.
- **The fake `decontaminate()` API does not exist.** Use the harness's n-gram file flow.
- **Tulu-3 is not automatically clean.** The SFT mixture is ODC-BY but embeds non-commercial subsets.
- **Synthetic data collapses without real data in the mix**, and inherits the generator's hallucinations.
- **PII in a public dataset is a legal event, not a quality metric.** Scrub before pushing, not after.
- **A test split you have looked at is a validation split.** Reserve it, write it down, and do not open it until the final eval.

### **15. Data Pre-Flight Checklist**

- Name the three standard SFT formats and when to use each.
- What is the DPO preference format? What is the hard-negative rule?
- What is decontamination and why is it non-negotiable?
- State the five rules that actually move fine-tuning metrics.
- What license is Alpaca under? Can you fine-tune commercially on it?
- What is the EU AI Act's main impact on fine-tuning practice, and which dates apply this quarter?
- For a domain dataset, what is the right train/val/test split?
- What is MinHash used for in the data prep pipeline?
- What is model collapse? How do you prevent it with synthetic data?
- How do you assess data quality before training?

### **16. Acceptance Gate**

- The five rules can each be explained in one sentence with an example from your own data.
- A domain dataset of 1K to 5K examples is public on the Hub with a data card.
- Deduplication report: how many rows were removed by the exact and near-duplicate passes.
- Decontamination report: which benchmarks, how many documents stripped.
- Token-length distribution plotted, with an explicit trimming or filtering decision recorded.
- Provenance and license table attached to the data card.
- PII scrub audited on 100 examples.
- Train / validation / test split fixed at 85/7.5/7.5 with a recorded seed.

### **17. Resources**

- datasets on PyPI: https://pypi.org/project/datasets/
- EU AI Act implementation timeline: https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act
- Navigating the AI Act: https://digital-strategy.ec.europa.eu/en/faqs/navigating-ai-act
- lm-eval decontamination guide: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md
- distilabel on PyPI: https://pypi.org/project/distilabel/

**Next:** Stage 04: Training Core.