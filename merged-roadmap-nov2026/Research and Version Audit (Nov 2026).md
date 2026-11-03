# **Research and Version Audit (November 2026)**

**Audit date:** 3 November 2026
**Scope:** every library pin, breaking change, deprecation, and factual claim in the merged roadmap, verified against first-party registries and official documentation.

This file exists because the seven source documents were written against September 2026 pins, and roughly a dozen of their claims are now wrong in ways that produce silently broken code rather than obvious errors. Each correction below names the source claim, the current fact, and the evidence.

---

## **Method**

Three evidence tiers, in descending priority:

1. **Package registries** — version and release date read directly from the PyPI JSON API (`https://pypi.org/pypi/<package>/json`), including `requires_python` and dependency bounds.
2. **Project release pages** — GitHub releases API for the same repositories, used for tag names, dates, and URLs.
3. **Official documentation and advisories** — Hub docs, framework docs, the European Commission's AI Act Service Desk, and NVD / the GitHub Advisory Database for the LlamaFactory CVE.

Where only secondary sources were available, the claim is marked as such in the audit tables. Search-result snippets were not treated as verification on their own; pages were read.

---

## **Version ledger**

| Package | Version | Released | Evidence |
|---|---|---|---|
| `transformers` | 5.22.1 | 2026-10-30 | https://github.com/huggingface/transformers/releases/tag/v5.22.1 |
| `huggingface_hub` | 2.4.0 | 2026-10-29 | https://pypi.org/project/huggingface_hub/ |
| `datasets` | 5.1.0 | 2026-10-05 | https://github.com/huggingface/datasets/releases/tag/5.1.0 |
| `trl` | 1.17.1 | 2026-10-30 | https://github.com/huggingface/trl/releases/tag/v1.17.1 |
| `peft` | 0.21.2 | 2026-10-01 | https://github.com/huggingface/peft/releases/tag/v0.21.2 |
| `sentence-transformers` | 6.2.0 | 2026-10-21 | https://github.com/huggingface/sentence-transformers/releases/tag/v6.2.0 |
| `accelerate` | 1.15.0 | 2026-09-09 | https://pypi.org/project/accelerate/ |
| `tokenizers` | 0.23.3 | 2026-10-09 | https://pypi.org/project/tokenizers/ |
| `safetensors` | 0.8.0 | 2026-06-09 | https://pypi.org/project/safetensors/ |
| `evaluate` | 0.4.6 | 2025-09-18 | https://github.com/huggingface/evaluate/releases/tag/v0.4.6 |
| `lighteval` | 0.13.0 | 2025-11-24 | https://github.com/huggingface/lighteval/releases/tag/v0.13.0 |
| `lm-eval` | 0.4.13 | 2026-08-31 | https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13 |
| `kernels` | 0.18.1 | 2026-11-02 | https://github.com/huggingface/kernels/releases/tag/v0.18.1 |
| `bitsandbytes` | 0.50.2 | 2026-08-27 | https://pypi.org/project/bitsandbytes/ |
| `diffusers` | 0.42.1 | 2026-10-27 | https://github.com/huggingface/diffusers/releases/tag/v0.42.1 |
| `timm` | 1.0.30 | 2026-09-22 | https://pypi.org/project/timm/ |
| `torch` | 2.15.0 | 2026-10-28 | https://pypi.org/project/torch/ |
| `axolotl` | 0.21.1 | 2026-10-27 | https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.21.1 |
| `llamafactory` | 0.9.5 | 2026-05-30 | https://github.com/hiyouga/LlamaFactory/releases/tag/v0.9.5 |
| `unsloth` | 2026.10.11 / v0.1.913-beta | 2026-10-30 | https://github.com/unslothai/unsloth/releases/tag/v0.1.913-beta |
| `vllm` | 0.33.0 (v0.33.0rc3) | 2026-11-02 | https://github.com/vllm-project/vllm/releases/tag/v0.33.0 |
| `sglang` | 0.5.23 | 2026-11-02 | https://github.com/sgl-project/sglang/releases/tag/v0.5.23 |
| `torchtitan` | 0.4.0 | 2026-10-28 | https://github.com/pytorch/torchtitan/releases/tag/v0.4.0 |
| `torchtune` | 0.6.1 | **2025-04-07** | https://github.com/meta-pytorch/torchtune/releases/tag/v0.6.1 |
| `verl` | 0.9.1 | 2026-09-20 | https://github.com/verl-project/verl/releases/tag/v0.9.1 |
| `openrlhf` | 0.11.2 | 2026-09-14 | https://github.com/OpenRLHF/OpenRLHF/releases/tag/v0.11.2 |
| `skyrl` | 0.4.0 | 2026-10-01 | https://pypi.org/project/skyrl/ |
| `tinker` | 0.37.0 | 2026-11-02 | https://pypi.org/project/tinker/ |
| `gptqmodel` | 7.8.0 | 2026-11-02 | https://pypi.org/project/gptqmodel/ |
| `llmcompressor` | 0.15.0 | 2026-10-28 | https://pypi.org/project/llmcompressor/ |
| `compressed-tensors` | 0.20.1 | 2026-10-28 | https://pypi.org/project/compressed-tensors/ |
| `agentlightning` | 1.0.2 | 2026-09-29 | https://pypi.org/project/agentlightning/ |
| `distilabel` | 1.5.3 | **2025-01-28** | https://pypi.org/project/distilabel/ |
| `mlx-lm` | 0.32.2 | 2026-10-16 | https://pypi.org/project/mlx-lm/ |
| `triton` | 3.9.0 | 2026-10-27 | https://pypi.org/project/triton/ |
| `flash-attn` | 2.8.3.post2 | 2026-10-14 | https://pypi.org/project/flash-attn/ |
| `tensorrt-llm` | 1.2.1 | 2026-04-20 | https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1 |

### **Python version floor**

| Package | `requires_python` | Source |
|---|---|---|
| `transformers` 5.22.1 | **>=3.11** | PyPI metadata |
| `trl` 1.17.1 | **>=3.11** | PyPI metadata |
| `datasets` 5.1.0 | >=3.10 | PyPI metadata |
| `huggingface_hub` 2.4.0 | >=3.10 | PyPI metadata |
| `peft` 0.21.2 | >=3.10 | PyPI metadata |
| `sentence-transformers` 6.2.0 | >=3.10 | PyPI metadata |
| `accelerate` 1.15.0 | >=3.10 | PyPI metadata |

The source documents said "Python 3.9 or newer" for the Hub client. The real floor for the full stack is **Python 3.11**, because `transformers` 5.22.1 and `trl` 1.17.1 both require it.

### **Dependency bounds that constrain installs**

- `transformers` 5.22.1 requires `huggingface-hub<3.0,>=1.31.0`, `accelerate>=1.1.0` (the `torch` extra), `torch>=2.6` (the `torch` extra), `numpy>=1.17`.
- `datasets` 5.1.0 requires `huggingface-hub<3.0,>=1.31.0`; the audio extra needs `torchcodec>=0.6.0`; the `teich==0.1.5` dependency backs the agent-trace loader.
- `trl` 1.17.1 requires `transformers>=4.56.2`, `datasets>=4.7.0`, `accelerate>=1.11.0`, `torch>=2.6.0`, `numpy>=1.26.0`; extras `peft` (peft>=0.17.0), `kernels` (`transformers[kernels]`), `harbor` (harbor>=0.13.0, Python >=3.12), `vlm` (torchvision).
- `sentence-transformers` 6.2.0 requires `transformers>=5.0.0,<6.0.0`, `accelerate>=1.3.0` for the `train` extra, `numpy>=1.24.0`.

---

## **Correction log**

| # | Source claim | Corrected fact | Impact if unfixed | Evidence |
|---|---|---|---|---|
| 1 | `huggingface_hub` 1.x; Python 3.9+ | 2.4.0; Python >=3.10, and >=3.11 for `transformers`/`trl` | `huggingface-cli`, `hf repo`, `hf repo-files delete`, `hf upload-large-folder`, `hf jobs ps`, `list_models`, `upload_large_folder`, `duplicate_space`, `repo_type_and_id_from_hf_id` all fail or are renamed | https://pypi.org/project/huggingface_hub/ · https://huggingface.co/docs/hub/en/guides/cli |
| 2 | `datasets` 4.x | 5.1.0; streaming shuffle defaults now `buffer_size=1000`, `max_buffer_input_shards=10`; agent-trace loading via `teich` | Silent loss-curve differences after upgrade; missing agent-trace path | https://github.com/huggingface/datasets/releases/tag/5.1.0 · https://pypi.org/project/datasets/ |
| 3 | TRL "v0.15.0 (Mar 2026)"; PPO in the experimental tier | TRL 1.17.1; **PPO removed** in 1.13.0 (`PPOTrainer`, `PPOConfig`, `modeling_value_head.py`); `trl.losses` removed in 1.14.0; KTO stable since 1.8; Distillation stable since 1.10 | `from trl import PPOTrainer` raises ImportError; experimental trainers have been deleted in minor releases (six in 1.14, TPO in 1.17) | https://github.com/huggingface/trl/releases/tag/v1.17.1 |
| 4 | "Axolotl v0.29.0 (Feb 2026)" | Axolotl 0.21.1 (27 Oct 2026); no 0.29.0 line exists | Pip install failure | https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.21.1 |
| 5 | "Six-way framework race" including TorchTune | TorchTune's last release was 0.6.1 (Apr 2025) and the project is unmaintained; the field is five-way plus TorchTitan / verl / OpenRLHF / SkyRL / Tinker | Building new work on a dead path | https://github.com/meta-pytorch/torchtune/releases/tag/v0.6.1 · https://github.com/pytorch/torchtitan/releases/tag/v0.4.0 |
| 6 | `from awq import AutoAWQForCausalLM` for AWQ quantization | AutoAWQ is deprecated and AutoGPTQ is unmaintained; the supported paths are GPTQModel 7.8.0 and llm-compressor 0.15.0 (vLLM's AWQ / FP8 W8A8 recommendation) | Installation and runtime failure on the documented capstone path | https://pypi.org/project/gptqmodel/ · https://pypi.org/project/llmcompressor/ |
| 7 | LlamaFactory as a "GUI era" tool with no caveat | 0.9.5 carries **CVE-2026-58116** (CVSS-B 9.3, critical RCE via the WebUI model path, `trust_remote_code=True` hardcoded, no patched version listed); an earlier `vhead_file` RCE was fixed in 0.9.4; the vendor list also carries an SSRF guard-bypass issue (CVE-2026-85673) | **Remote code execution** against any host exposing the WebUI | https://nvd.nist.gov/vuln/detail/cve-2026-58116 · https://github.com/advisories/GHSA-mwc7-mf87-v3mf |
| 8 | "DPO improves MT-Bench by >=5%" as an acceptance criterion | MT-Bench is saturated and effectively unmaintained; gaps below ~0.1–0.2 are judge noise. Gate on IFEval + Arena-Hard-Auto v2 + AlpacaEval 2.0 + a domain held-out set | A criterion that is either trivial or unreachable depending on the judge version | https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13 |
| 9 | `from lm_eval.decontamination.decontaminate import decontaminate(train_dataset=..., eval_datasets=[...], ngram_size=8)` | That API does not exist. The harness detects contamination over **precomputed n-gram files** via `--decontamination_ngrams_path`, reports `*_decontaminate` metrics, and does not decontaminate new tasks by default unless `should_decontaminate` / `doc_to_decontamination_query` are overridden; the default basis is 13-grams | Lab 4 in the source data stage cannot run | https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md |
| 10 | "EU AI Act now applies" (undated) | Progressive application: 2 Feb 2025 prohibitions/AI literacy; 2 Aug 2025 governance and GPAI; **2 Aug 2026** main rules, Article 50, enforcement begins; **2 Dec 2026** new prohibitions plus the Article 50(2) transition; 2 Dec 2027 Annex III; 2 Aug 2028 Annex I. The Digital Omnibus entered into force 27 Jul 2026. Systems on the market before 2 Aug 2026 enter scope if **significantly modified** | Compliance work scheduled against the wrong dates | https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act · https://digital-strategy.ec.europa.eu/en/faqs/navigating-ai-act |
| 11 | Capstone bases Llama-3.2-3B/1B, Qwen2.5-1.5B | `Qwen/Qwen3-4B-Instruct-2507` (text-only), `Qwen/Qwen3.5-4B` and `google/gemma-4-E4B` (multimodal, Apache 2.0) are the current picks | Training a stale checkpoint family, and missing the modality split | https://huggingface.co/Qwen/Qwen3.5-4B · https://huggingface.co/google/gemma-4-E4B |
| 12 | vLLM `--max-lora-rank` "choices 1–512" | Exact choices: 1, 8, 16, 32, 64, 128, 256, 320, 512. `--max-loras` default **1**; `--max-lora-rank` default **16**; `--max-cpu-loras` defaults to `--max-loras` | An adapter trained at r=32 will not serve on a default server; multi-tenant serving silently serializes | https://docs.vllm.ai/en/latest/cli/serve/ |
| 13 | "Tulu-3: ODC-BY, mostly yes, check" | The Tulu-3 SFT mixture is ODC-BY but **embeds non-commercial subsets** (No Robots; NuminaMath-TIR cc-by-nc-4.0), so it is not commercially clean as a whole | Shipping a model trained on NC data into a commercial product | Dataset license metadata, checked during the catalog review |
| 14 | "Llama-3 chat template" as the general reference | Each target model has its own template (Qwen, Mistral, Gemma 4, Llama 4). Always render and inspect before training | Silent template mismatch; wasted GPU-hours | https://huggingface.co/Qwen/Qwen3.5-4B |
| 15 | "MT-Bench, AlpacaEval" as the DPO evaluation set | AlpacaEval 2.0 is length-controlled; Arena-Hard-Auto v2 is the current automated pairwise judge; IFEval is deterministic | Confounded win rates driven by output length | https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13 |

---

## **Deprecation and removal watchlist**

| Item | Status | Action |
|---|---|---|
| TRL `PPOTrainer`, `PPOConfig`, `modeling_value_head.py` | **Removed** in 1.13.0 | Use verl or OpenRLHF for PPO |
| TRL `trl.losses` | **Removed** in 1.14.0 | Use the public trainer APIs |
| TRL experimental trainers | Deleted in minor releases without deprecation (six in 1.14; TPO in 1.17) | Pin `trl`; read release notes before upgrading |
| TRL `use_liger_kernel` in trainers | Deprecated; removal scheduled for 2.0.0 | Migrate to `trl.kernels` or the Liger integration directly |
| `trl.kernels` fused Triton logprob + entropy | **Default** | Raises DPO's max sequence length from 10,240 to 59,392 on a 79 GiB GPU |
| `huggingface-cli` | Removed in hub v2 | `hf` |
| `hf repo`, `hf repo-files delete`, `hf upload-large-folder`, `hf jobs ps` | Renamed in hub v2 | `hf repos`, `hf repos delete-files`, `hf upload`, `hf jobs --status` |
| `Repository`, `HfFolder` | Removed in hub v1 | `HfApi` |
| `hf_transfer` | Removed | `hf_xet` (automatic) |
| `encode_plus()`, `batch_encode_plus()` | Removed in transformers v5 | Call the tokenizer directly |
| `torch_dtype` in transformers | Renamed | `dtype` (note: Diffusers still uses `torch_dtype`) |
| `Trainer(tokenizer=...)` | Renamed | `processing_class=...` |
| `DataCollatorForCompletionOnlyLM` | Removed in TRL v1 | `completion_only_loss` / `assistant_only_loss` |
| `SFTConfig(max_seq_length=...)` | Renamed | `max_length` |
| `max_prompt_length`, `max_completion_length` in DPO | Removed | `max_length` only; filter overlong rows first |
| `sentence_transformers.model.fit()` | Deprecated | `SentenceTransformerTrainer` |
| `TRL vllm-serve` | Superseded in practice by `trl vllm-serve` with the vLLM 0.33 API | Verify flags against your vLLM version |
| AutoGPTQ | Unmaintained | GPTQModel |
| AutoAWQ | Deprecated | GPTQModel or llm-compressor |
| TorchTune | Unmaintained since Apr 2025 | TorchTitan |
| `evaluate` | Maintained but legacy; no release since Sep 2025 | Keep for classic metrics; use `lm-eval` for benchmarks |
| `lighteval` | No release since Nov 2025 | Pin; read source for new entry points |
| `distilabel` | Last release Jan 2025 | Pin; read source before extending |

---

## **Current landscape notes (November 2026)**

- **Embedding leaderboards.** MTEB v2 boards score 41 (English) and 131 (multilingual) tasks and rank by **Borda count**, so row order is not mean-score order. Leading open-weight general embedders on MTEB(eng, v2) include QZhou-Embedding (75.97) and Qwen3-Embedding-8B (75.23, Apache 2.0, 32K context); MTEB(multilingual, v2) is led by Microsoft's `harrier-oss-v1-27b` (MIT, 74.27, 5,376 dims). Prefer **RTEB** — the maintainers' held-out sets — over MTEB-only claims. Open rerankers include Qwen3-Reranker-4B, `bge-reranker-v2-m3`, and NVIDIA's Nemotron-3.5-Rerank-8B (released 2026-10-20). https://leaderboard.mteb.org/models
- **Base model families in the current line.** Qwen3.5 (0.8B, 2B, 4B, 27B, 35B-A3B, 122B-A10B, 397B-A17B, Feb 2026), Qwen3.6 (27B, 35B-A3B, Apr 2026), Qwen3.8-27B, and Gemma 4 (E2B, E4B, 12B Unified, 26B A4B, 31B; Apache 2.0; 128K–256K context).
- **4-bit QAT checkpoints.** NVFP4 W4A16 builds for Hopper and Blackwell, and native Q4_0 GGUF builds for llama.cpp and Ollama, are published for Qwen3.5-4B, Qwen3.8-27B, and Gemma 4 E4B/12B. The Gemma 4 E4B Q4_0 build reports 2× lower KL against the unquantized model than Google's own QAT, and the NVFP4 build reports +35% output throughput over BF16 on H100. https://huggingface.co/collections/QUASAR-QAT/all-quasar-models
- **Compute.** HF Spaces GPU tiers run from $0.40/h (T4 small) to $2.75/h (RTX PRO 6000, 96 GB); A100 large is $2.50/h. H100 was **removed from Spaces in December 2025** and is available only as a GPU Instance ($4.50/h), alongside H200 ($5.00/h) and B200 ($9.25/h). ZeroGPU gives a free tier of 5 minutes/day, PRO 40 minutes/day, Team 40, Enterprise 60, with extra time at $1 per 10 minutes; the `large` slice is 48 GB and `xlarge` is 96 GB at twice the quota cost. PRO is $9/month and multiplies ZeroGPU quota by eight. https://huggingface.co/pricing · https://huggingface.co/docs/hub/spaces-zerogpu
- **Agent Skills.** The Hub ships a curated skill set (`hf-cli`, `huggingface-datasets`, `huggingface-llm-trainer`, `huggingface-community-evals`, `huggingface-vision-trainer`, `huggingface-trackio`, `gradio`, `transformers-js`) following the Agent Skills spec, installed with `hf skills add <name> [--global]` and refreshed with `hf skills update -g`. The `hf-cli` skill is generated from the locally installed CLI, so it tracks your version. https://github.com/huggingface/skills · https://huggingface.co/docs/hub/en/agents-skills
- **Target-role signal.** Anthropic's Research Engineer, Agents posting asks for agent harness design, rigorous quantitative evals for agentic tasks, automated model and prompt evaluation across the training and product lifecycle, and optimized data mixes for agentic training — with large-scale RL and multi-agent systems as extras, and "finetune a model to maximize its performance using a particular set of agent tools or harness" as a representative project. A fine-tuning-only portfolio is missing eval infrastructure and harness design, not more fine-tunes. https://www.aijobs.com/jobs/396324642-research-engineer-agents

---

## **Open questions and what was deliberately not asserted**

- **Reported but unconfirmed:** secondary sources describe a major hosted fine-tuning platform winding down to new users and a hosted RL provider retiring several model families. Neither claim was confirmed against a first-party announcement, so neither appears in the roadmap. If either is true, it changes the "no GPU" compute table (Stage 7) and nothing else.
- **MT-Bench maintenance status:** treated as saturated and effectively unmaintained, based on its absence from current release activity and the noise floor of published comparisons. The roadmap therefore never gates on it, which is safe under either reading.
- **Judge-version sensitivity:** Arena-Hard-Auto v2 and AlpacaEval 2.0 scores move with the judge model. Every capstone in this roadmap reports the judge version and the sample size alongside the score; without those two numbers a preference score is not reproducible.
- **Unstable quantization thresholds:** the source material's claim that distilled models have different optimal quantization thresholds is carried forward as guidance to re-measure, not as a number.
- **Absent vault notes.** `inference-roadmap/*` (18 files), `progress/phase1-hf-finetuning`, and `targets/ai-engineering-role-taxonomy` were referenced by the sources but not provided. They remain pointers; no content was invented for them.

---

## **Source coverage matrix**

Every source document mapped to where its content lives in the merged set. "Master" means the canonical merged roadmap; "Stage" means the expanded stage file.

| Source | Lines | Content carried | Where it lives |
|---|---|---|---|
| S1 `First.md` | 1,638 | Pipeline map; version/breaking-changes table; install blocks; Phase 0 hub and datasets walkthrough with code; Phase 1 transformers, tokenizers, evaluate; Phase 2 accelerate, peft, trl, safetensors; Phase 3 sentence-transformers, diffusers, timm; Phase 4 bitsandbytes, kernels/CUDA/Triton; full connection map; week-by-week schedule; GitHub repo structure; the One Rule; Phase 0 DONE / Phase 1 NEXT status | Master Stages 0–8, Schedule, Repo structure, One Rule; expanded in Stages 00–08 |
| S2 `Second.md` | 1,053 | Full topic checklists for every library; the ~40 items absent from S1: `to_polars()`, `to_iterable_dataset()`, `ds.data`, `skip()`/`take()`, split-specific push, private versus public datasets, `squad`, adapter combination (`add`/`cat`/`svd`/`linear`), `trl vllm-serve`, `SparseEncoder`, auto pipelines, `AutoModelForInstanceSegmentation`, `AutoModelForImageToImage`, Falcon, Pegasus, `TextIteratorStreamer`, community kernels via `attn_implementation`, and the full pipeline task inventory including `table-question-answering`, `mask-generation`, `image-to-image`, `image-feature-extraction`, `keypoint-matching`, `video-classification`, `zero-shot-audio-classification`, `text-to-speech`, `any-to-any` | Master Stage checklists (preserved verbatim in substance); expanded in Stages 00–07 |
| S3 `00 - Overview & Index.md` | 179 | Five-file structure; the fine-tuning decision tree; the three practitioner rules; the fine-tuning-to-inference bridge diagram; the 2026 changes list; the relationship table to existing notes; the 6-week plan; the quick start; related-note pointers | Master spine and method-selection gate; Stage 08 calendar; the notes-referenced section |
| S4 `01 - Methods Deep Dive.md` | 361 | SFT scope; full fine-tuning memory arithmetic; the LoRA equation and `LoRALinear` implementation; the default-LoRA table; memory savings; QLoRA config; DoRA; GaLore; the LoRA-variant table; the DPO derivation and loss; IPO/KTO/SimPO/ORPO/iterative DPO; the DPO decision table; PPO/GRPO/RLVR; DAPO, Dr.GRPO, CISPO, GFPO, VIGOR; the consensus pipeline; distillation; the mental model; the pre-flight checklist | Master Stages 3–4; expanded in Stages 03–04 |
| S5 `02 - Data & Datasets.md` | 418 | The data-shape table; the five formats with JSON; the dataset catalogs; the five rules; quality assessment; the decontamination and MinHash code; the format-consistency check; the synthetic-data pattern and gotchas; the license landscape; EU AI Act obligations; the six-step domain dataset build; five labs; the pre-flight checklist | Master Stage 2; expanded in Stage 02 |
| S6 `03 - Frameworks & Tools 2026.md` | 606 | The framework race; the decision table; the landscape table; Unsloth, Axolotl, LlamaFactory, TRL, PEFT, and TorchTune sections with code; bitsandbytes; distributed training and the scale matrix; the merge/quantize/serve pipeline; the AWQ code; multi-LoRA serving; the hands-on decision flow; six labs; the pre-flight checklist | Master Stages 5 and 7; expanded in Stages 05 and 07 |
| S7 `04 - Capstones & JD Alignment.md` | 399 | The capstone standard; Capstones 1–3 and the bonus, with goals, artifacts, stacks, benchmarks, and acceptance criteria; the reward function; the capstone sequence; JD alignment and the role taxonomy mapping; the 30-question interview bank; the 10 interviewer questions; the before-you-ship checklist; the 6–8 week plan; the "what interviewers look at" section | Master Stage 8; expanded in Stage 08 |

**Preservation check.** Every checklist item, code block, lab, schedule line, table, and callout in all seven sources appears in the merged master. Where a source claim was factually wrong, the item is retained and corrected, with the correction recorded above rather than deleted.

---

## **Re-verification cadence**

This ledger is accurate as of **3 November 2026**. Six things in it will age quickly and are worth re-checking monthly:

1. `trl` — experimental trainers can vanish in a minor release.
2. `vllm` and `sglang` — both released within days of this audit.
3. `axolotl` and `unsloth` — both move weekly.
4. `huggingface_hub` — CLI and Python renames continue between major lines.
5. LlamaFactory — watch for a release that patches CVE-2026-58116.
6. The EU AI Act timeline — the Digital Omnibus amendments are still landing.

---

## **Reference list**

1. transformers v5.22.1 — https://github.com/huggingface/transformers/releases/tag/v5.22.1
2. transformers on PyPI — https://pypi.org/project/transformers/
3. huggingface_hub on PyPI — https://pypi.org/project/huggingface_hub/
4. Datasets 5.1.0 — https://github.com/huggingface/datasets/releases/tag/5.1.0
5. datasets on PyPI — https://pypi.org/project/datasets/
6. TRL v1.17.1 — https://github.com/huggingface/trl/releases/tag/v1.17.1
7. trl on PyPI — https://pypi.org/project/trl/
8. Axolotl v0.21.1 — https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.21.1
9. PEFT v0.21.2 — https://github.com/huggingface/peft/releases/tag/v0.21.2
10. TorchTune v0.6.1 — https://github.com/meta-pytorch/torchtune/releases/tag/v0.6.1
11. TorchTitan v0.4.0 — https://github.com/pytorch/torchtitan/releases/tag/v0.4.0
12. verl v0.9.1 — https://github.com/verl-project/verl/releases/tag/v0.9.1
13. OpenRLHF v0.11.2 — https://github.com/OpenRLHF/OpenRLHF/releases/tag/v0.11.2
14. GPTQModel on PyPI — https://pypi.org/project/gptqmodel/
15. NVD CVE-2026-58116 — https://nvd.nist.gov/vuln/detail/cve-2026-58116
16. GitHub advisory GHSA-mwc7-mf87-v3mf — https://github.com/advisories/GHSA-mwc7-mf87-v3mf
17. LlamaFactory v0.9.5 — https://github.com/hiyouga/LlamaFactory/releases/tag/v0.9.5
18. EU AI Act implementation timeline — https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act
19. EU AI Act timeline FAQ — https://ai-act-service-desk.ec.europa.eu/en/ai-act/faq/when-does-ai-act-go-effect-what-its-timeline-implementation
20. Navigating the AI Act — https://digital-strategy.ec.europa.eu/en/faqs/navigating-ai-act
21. Qwen3.5-4B model card — https://huggingface.co/Qwen/Qwen3.5-4B
22. vLLM `serve` CLI — https://docs.vllm.ai/en/latest/cli/serve/
23. vLLM engine arguments — https://docs.vllm.ai/en/latest/configuration/engine_args/
24. Hub CLI guide — https://huggingface.co/docs/hub/en/guides/cli
25. huggingface/skills — https://github.com/huggingface/skills
26. Hub Agent Skills — https://huggingface.co/docs/hub/en/agents-skills
27. CLI for AI Agents — https://huggingface.co/docs/hub/agents-cli.md
28. tokenizers on PyPI — https://pypi.org/project/tokenizers/
29. distilabel on PyPI — https://pypi.org/project/distilabel/
30. accelerate on PyPI — https://pypi.org/project/accelerate/
31. safetensors on PyPI — https://pypi.org/project/safetensors/
32. DPO, Rafailov et al. — https://arxiv.org/abs/2305.18290
33. GRPO, Shao et al. — https://arxiv.org/abs/2402.03300
34. PPO, Schulman et al. — https://arxiv.org/abs/1707.06347
35. SkyRL on PyPI — https://pypi.org/project/skyrl/
36. Tinker on PyPI — https://pypi.org/project/tinker/
37. AgentLightning on PyPI — https://pypi.org/project/agentlightning/
38. LlamaFactory CVE list — https://opencve.alliance.unm.edu/cve/?vendor=hiyouga&product=llama-factory
39. bitsandbytes on PyPI — https://pypi.org/project/bitsandbytes/
40. sentence-transformers v6.2.0 — https://github.com/huggingface/sentence-transformers/releases/tag/v6.2.0
41. MTEB leaderboard — https://leaderboard.mteb.org/models
42. Diffusers v0.42.1 — https://github.com/huggingface/diffusers/releases/tag/v0.42.1
43. timm on PyPI — https://pypi.org/project/timm/
44. mlx-lm on PyPI — https://pypi.org/project/mlx-lm/
45. HF Spaces overview — https://huggingface.co/docs/hub/main/spaces-overview
46. HF GPU hardware — https://huggingface.co/docs/hub/spaces-gpus
47. HF ZeroGPU — https://huggingface.co/docs/hub/spaces-zerogpu
48. HF Jobs — https://huggingface.co/docs/hub/jobs
49. Anthropic Research Engineer, Agents — https://www.aijobs.com/jobs/396324642-research-engineer-agents
50. Evaluate v0.4.6 — https://github.com/huggingface/evaluate/releases/tag/v0.4.6
51. LightEval v0.13.0 — https://github.com/huggingface/lighteval/releases/tag/v0.13.0
52. lm-evaluation-harness v0.4.13 — https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13
53. lm-eval decontamination guide — https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/decontamination.md
54. lm-eval CLI interface — https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
55. HF pricing — https://huggingface.co/pricing
56. kernels v0.18.1 — https://github.com/huggingface/kernels/releases/tag/v0.18.1
57. SGLang v0.5.23 — https://github.com/sgl-project/sglang/releases/tag/v0.5.23
58. vLLM v0.33.0 — https://github.com/vllm-project/vllm/releases/tag/v0.33.0
59. Unsloth v0.1.913-beta — https://github.com/unslothai/unsloth/releases/tag/v0.1.913-beta
60. llm-compressor on PyPI — https://pypi.org/project/llmcompressor/
61. torch on PyPI — https://pypi.org/project/torch/
62. Triton on PyPI — https://pypi.org/project/triton/
63. TensorRT-LLM v1.2.1 — https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1
64. Gemma 4 E4B model card — https://huggingface.co/google/gemma-4-E4B
65. Qwen3-4B-Instruct-2507 model card — https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
66. QUASAR 4-bit QAT checkpoints — https://huggingface.co/collections/QUASAR-QAT/all-quasar-models
