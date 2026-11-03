# **Stage 02 — Data Engineering**
**Outcome:** a dataset you can defend — sourced, licensed, deduplicated, decontaminated, format-checked, versioned, and documented.**Libraries:** `datasets` 5.1.0 · `datasketch` · `transformers` tokenizers · `presidio` (PII) · `distilabel` (synthetic)


---

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

## **Topic checklist**

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

## **Code**

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
# **pip install datasketch**
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

# **Run this over 500 random rows and look at the distribution, not one row.**
# **A bimodal distribution means two chat templates are mixed in the dataset.**
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
# **pip install presidio-analyzer presidio-anonymizer**
from presidio_anonymizer import AnonymizerEngine

engine = AnonymizerEngine()
result = engine.anonymize(text=text, language="en")
print(result.text)
```

### **Synthetic preference generation with a rubric**

```python
# **pip install distilabel**
# **Distilabel's latest PyPI release is 1.5.3 (Jan 2025) — pin it and read the**
# **source for the current API before wiring it into a pipeline.**
from distilabel.pipeline import Pipeline
from distilabel.models import InferenceAPIModel, TextGeneration

# **build: prompts -> N candidates -> judge (rubric) -> best/worst pairs**
```

### **Domain dataset construction, end to end**

```python
from datasets import Dataset, load_dataset

# **1. Define: "SaaS billing questions, structured JSON output, with citations"**
# **2. Source: real tickets (consent + PII scrub) + synthetic (rubric) + public (licensed)**
# **3. Clean: dedup -> quality filter -> decontaminate**
# **4. Format: convert to ChatML and verify tokenization**
# **5. Split: 85 / 7.5 / 7.5**
train, rest = ds.train_test_split(test_size=0.15, seed=42)
val, test   = rest.train_test_split(test_size=0.5, seed=42)

# **6. Version and document**
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

## **Labs**

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

**Next:** Stage 03 — Training Core.
