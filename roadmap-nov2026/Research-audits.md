## **Research and Version Audit Nov2026**

This audit verifies the factual claims in the master roadmap that depend on library releases, model releases, licence terms, benchmark numbers, and technical papers.

### **0.1 Verification Scope**

| Item | Value |
|---|---|
| Verification date | 3 November 2026 |
| Sources merged | 8 documents, 7,061 source lines |
| Public surfaces checked | Hugging Face Hub API and model cards, GitHub releases for vLLM, SGLang, TRL, Unsloth, Axolotl, DeepEP, ART, ASTRA, CM2, and TGI, PyPI metadata for 28 packages, arXiv abstract pages, vendor documentation, and vendor blogs |
| Deliverables audited | 1 master file, 12 stage files, this audit, and 1 archive |
| Reference list | 130 entries, numbered 1 to 130, each cited at least once, with no duplicate URLs |
| Unresolved items | 4, each carried forward with a stated reason |

### **0.2 Method**

1. **Inventory.** Every model, library, paper, licence, and configuration claim in the eight source documents was recorded with its asserted value and date.
2. **Verification.** Each claim was checked against a primary source: the model card for parameters, context, licence, and chat-template behaviour; release notes or package metadata for versions and Python requirements; the paper abstract page for paper claims; and repository state for archival and dependency conflicts.
3. **Application.** Verified values were written into the master, and every difference from the original source was recorded as a numbered correction with evidence.

Where a source claimed a number that no primary source supports, the number was removed. Where a number was supported only under different conditions, such as a different hardware target or preset, those conditions were recorded alongside the number.

### **0.3 Source Coverage**

| Source | Lines | Tables | Code blocks | Checklists | Mapped to |
|---|---|---|---|---|---|
| `First.md`, core stack prose | 1,638 | 14 | 63 | 9 | Stages 0 to 11, schedule, repository structure, one rule, key papers |
| `Second.md`, core stack checklist | 1,053 | 2 | 18 | More than 40 | Stage checklists, grouped by topic |
| `00 - Overview & Index.md` | 179 | 1 | 1 | 6 | Stage spine, method-selection gate, weekly schedule |
| `01 - Methods Deep Dive.md` | 361 | 5 | 14 | 12 | Stages 4 and 5 |
| `02 - Data & Datasets.md` | 418 | 3 | 9 | 8 | Stage 3 |
| `03 - Frameworks & Tools 2026.md` | 606 | 9 | 21 | 15 | Stages 6 and 10 |
| `04 - Capstones & JD Alignment.md` | 399 | 4 | 5 | 11 | Stage 11 |
| `LLM Fine-Tuning & Inference Engineering Roadmap (2026)` | 1,499 | 27 | 34 | 62 | Stages 1, 2, 4, 5, 7, 8, 9, 10, and 11, key papers, repository structure |

After the rebuild, 231 named identifiers from the 2026 roadmap source and 79 identifiers from the other seven sources were checked against the canonical set. All were present. The two initially missing identifiers, `Falcon TII` and `vllm_lora`, were restored.

### **0.4 Library and Version Findings**

| Claim | Result | Evidence |
|---|---|---|
| Core Hugging Face training packages | Confirmed against PyPI metadata on 3 November 2026 | `huggingface_hub`, `datasets`, `transformers`, `trl`, `peft`, and `accelerate` |
| Tokenization, storage, kernels, vision, and embedding packages | Confirmed against PyPI metadata on 3 November 2026 | `tokenizers`, `safetensors`, `kernels`, `diffusers`, `timm`, and `sentence-transformers` |
| Fine-tuning and RL packages | Confirmed against PyPI metadata on 3 November 2026 | `unsloth`, `axolotl`, `llamafactory`, `torchtitan`, `verl`, `openrlhf`, and `skyrl` |
| Inference and quantization packages | Confirmed against PyPI metadata and GitHub releases on 3 November 2026 | `vllm`, `sglang`, `gptqmodel`, `llmcompressor`, `torch`, and `triton` |
| Axolotl Python requirement | New finding: Axolotl requires Python 3.12 or newer | PyPI metadata |
| vLLM Transformers requirement | New finding: recent vLLM releases constrain the compatible Transformers range | vLLM release notes |

The Axolotl requirement breaks a shared Python 3.11 base environment. The vLLM Transformers constraint creates a conflict with the newer training-side Transformers pin.

### **0.5 Model Landscape**

| Model family | Verified state, 3 November 2026 |
|---|---|
| **Mistral Large 4** | Released in late October 2026. Approximately 1 trillion total parameters with 52 billion active parameters. Supports 1 million context. Uses 128 routed experts plus 1 shared expert. Natively multimodal. Reasoning effort defaults to high and cannot be disabled. Includes an embedded DSpark drafter. Uses the Mistral Ecosystem Open-Weight Licence. Managed API use requires a separate commercial agreement. API pricing is $0.70 per million input tokens, $0.07 per million cached input tokens, and $2.10 per million output tokens. A LoRA supervised fine-tuning example ships with TRL for one 8-way B300 node. |
| **Kimi K3** | Released 16 July 2026. 2.8 trillion total parameters with 104 billion active parameters. Context length is 1,048,576 tokens. Has 93 layers: 1 dense layer, 69 KDA layers, and 24 Gated MLA layers. Uses 896 routed experts with top-16 routing plus 2 shared experts. Uses SiTU-GLU and a 160,000-token vocabulary. Includes the MoonViT-V2 401 million parameter vision encoder. Thinking is always enabled, with reasoning effort settings of low, high, and max, defaulting to max. Uses MXFP4 weights and MXFP8 activations. The Kimi K3 Licence has a revenue trigger above US$20 million. |
| **Kimi K3.1 Pro** | API only. The Moonshot AI Hub organisation lists K2.5, K2.6, K2.7-Code, K3, and Kimi-Linear, but no K3.1 repository. Weights were promised for 4 November 2026. |
| **Qwen3.8-27B** | Dense 27 billion parameter unified vision-language model. Has 64 layers arranged as 16 repeated blocks, each containing three Gated DeltaNet and feed-forward pairs plus one Gated Attention and feed-forward pair. Hidden size is 5,120. Native context is 262,144 tokens, extendable to 1 million with YaRN. Supports native image and hour-scale video input. Trained with multi-token prediction. Apache-2.0 licensed. Thinking is enabled by default. Supports `enable_thinking=False`, reasoning effort levels of xhigh, medium, and low, and `preserve_thinking=True` by default. Thinking sampling uses temperature 1.0, top-p 0.95, and top-k 20. Non-thinking sampling uses temperature 0.7, top-p 0.80, and presence penalty 1.5. |
| **Qwen3.6-27B** | Dense 27 billion parameter model, Apache-2.0 licensed, released in April 2026. Fine-tunable with Unsloth using roughly 18 to 23 GB of RAM depending on the build. |
| **Qwen3.5 family** | Released 2 March 2026. Supports 201 languages. Uses Gated DeltaNet and sparse mixture-of-experts components. Four-bit QLoRA is not recommended for either dense or MoE variants. BF16 LoRA memory footprints are approximately 3, 5, 10, 22, 56, and 74 GB for the 0.8B, 2B, 4B, 9B, 27B, and 35B-A3B variants respectively. |
| **Gemma 4** | E2B has 2.3 billion effective parameters, 5.1 billion including embeddings, 35 layers, 512 sliding-window size, and 128K context. E4B has 4.5 billion effective parameters, 8 billion total parameters, 42 layers, 512 sliding-window size. The 12B model has 11.95 billion parameters, 48 layers, 1,024 sliding-window size, and 256K context. Also includes 26B-A4B and 31B variants, with the 31B model at 30.7 billion parameters, 60 layers, and 256K context. Supports text and image, with audio support on E2B, E4B, and 12B. Supports more than 140 languages. Released across late March and April 2026. Apache-2.0 licensed. Multi-token prediction provides a 1.4 to 2.2 times inference speed-up. |
| **Gemma 4 fine-tuning footprint** | E2B QLoRA needs about 8 GB. E4B QLoRA needs about 10 GB. E4B LoRA needs about 17 GB. The 31B QLoRA needs about 22 GB. The 26B-A4B LoRA needs more than 40 GB. E4B QLoRA is recommended over E2B LoRA. Four-bit MoE is blocked because bitsandbytes does not provide a four-bit `nn.Parameter` implementation for MoE experts. |
| **DeepSeek V4.1 Flash** | MIT licensed. Supports 1 million context and image-text-to-text tasks. Has native Transformers support. Does not include a Jinja chat template, so the reference `encoding` implementation should be used. Includes `deepseek-recipe` Rust libraries. |
| **Muse Spark Open** | Released 15 October 2026. 816 billion total parameters with 42 billion active parameters. Supports 1 million context. Uses 256 experts with top-8 routing. Has 62 layers, hidden size 8,192, and vocabulary size 202,048. Accepts text, image, video, and audio input. Reasoning effort levels are low, medium, high, and xhigh, with xhigh as the default. Requires at least 700 GB for NVFP4 or 2 TB for BF16. Access is gated and requires manual approval, legal name, date of birth, and organisation details. The Muse Model License requires notification to Meta and a separate written agreement for managed API use above US$20 million aggregate gross revenue in any 12-month period. |
| **GLM-5.3** | 320 billion total parameters with 18 billion active parameters. MIT licensed. Released 9 September 2026. BF16 and Flash variants are available on the Hub. |
| **LFM2.5-24B-A2B** | 24 billion total parameters with 2.45 billion active parameters. The LFM Open License permits commercial use only below US$10 million annual revenue. Qualified non-profits are exempt for non-commercial and research use. Internal research, testing, and evaluation are permitted. The revenue threshold also applies per downstream customer. |
| **Qwen3-TTS** | Released 22 January 2026. Available in 0.6B and 1.7B sizes using a 12 Hz codec. Supports three-second voice cloning, 10 languages, and approximately 97 ms streaming latency. Trained on more than 5 million hours of audio. Apache-2.0 licensed. Variants include Base, CustomVoice, and VoiceDesign. The training recipe uses DPO on speech preference pairs, followed by rule-based rewards with GSPO, followed by speaker fine-tuning. The `Qwen3-TTS-Tokenizer-12Hz` tokenizer is part of the open release. |
| **Nemotron 3.5 family** | Includes Bolt 74B-A9B, Super VL 120B and 12B, Embed 8B and 1B with NVFP4 support for Blackwell, Rerank 8B, ASR Streaming 0.6B, Transcribe, and Dory 1.0-4B. Released in October 2026. |
| **Beam 1.0** | 501.3 billion total parameters with 23.4 billion active parameters. Apache-2.0 licensed. Released between 5 and 16 October 2026. Added to Transformers. |
| **MiniMax M3.1 Flash** | Released 19 October 2026. |
| **LongCat-2.5** | Released 2 November 2026. |

### **0.6 Technique Findings**

| Claim | Result | Evidence |
|---|---|---|
| TurboQuant equals PolarQuant plus QJL, with at least 6 times compression and no measurable accuracy loss | Rejected as stated | Upstream vLLM merged TurboQuant and shipped it as a v1 attention backend using Hadamard rotation and scalar quantization. It omits QJL because QJL can hurt attention quality by amplifying variance through softmax. Shipped presets include `turboquant_k8v4` at 2.6 times compression with a 1.17 percent perplexity increase, `turboquant_4bit_nc` at 3.8 times with a 2.71 percent increase, `turboquant_k3v4_nc` at about 3.5 times with a 10.63 percent increase, and `turboquant_3bit_nc` at 4.9 times with a 20.59 percent increase. |
| KVTC achieves about 20 times compression with less than 1 percent drop and needs calibration | Confirmed with scope | The paper reports up to 20 times compression while preserving reasoning and long-context accuracy, with more than 40 times in specific cases. It uses brief per-layer PCA calibration and requires no model changes. It is not a first-class vLLM feature. |
| Hybrid-EP gives a 514 percent throughput improvement | Rejected | The NVIDIA technical blog reports a 14 percent improvement for DeepSeek-V3, from 829 to 943 TFLOPs per GPU, an 8 percent improvement for Megatron-FSDP, and improvements of 5.5 percent in BF16 and 9.9 percent in MXFP8 on Qwen3-235B. DeepEP documents the Hybrid-EP backend as TMA-based and using fine-grained communication and compute overlap. |
| TGI has been in maintenance mode since December 2025 | Understated | The TGI repository is archived. Its last push was 21 March 2026. |
| DART gives 6.8 times lower draft latency than EAGLE3 | Confirmed | The paper reports 6.8 times lower draft latency than EAGLE3 and 53.3 times lower than standard speculative sampling on Qwen3-14B with H20 hardware. End-to-end speed-up is 2.03 to 3.44 times, with about 30 percent average improvement and up to 65 percent on code workloads. Draft models range from Qwen3 1.7B to 32B. |
| GRPO-lambda improves math benchmarks by 3 to 4.5 points | Replaced | The paper describes critic-free temporal-difference learning with an eligibility-trace reformulation and claims 30 to 40 percent better performance than GRPO on math benchmarks. |
| GSPO uses `loss_type='gspo'` | Wrong API | In TRL, GSPO is configured using `importance_sampling_level="sequence"` with `loss_type="grpo"`, `beta=0.0`, `epsilon=3e-4`, `epsilon_high=4e-4`, and `steps_per_generation=4`. The supported `loss_type` values include `grpo`, `dr_grpo`, `dapo`, `bnpo`, `cispo`, `sapo`, `luspo`, and `vespo`. TRL warns against pairing sequence-level importance sampling with `bnpo`, `dr_grpo`, `dapo`, or `cispo`. |
| DeepSeek V4 requires `--trust-remote-code` | Outdated | DeepSeek V4.1 Flash ships native Transformers support. |
| Qwen3.5 on vLLM requires a minimum vLLM release | Outdated | Current vLLM releases introduce a Transformers upper-bound constraint. |
| Gemma 4 26B-A4B fits in about 12 GB with Unsloth QLoRA | Wrong | Unsloth reports that 26B-A4B LoRA needs more than 40 GB. Four-bit MoE is blocked because bitsandbytes lacks the required four-bit MoE expert parameter support. |
| Unsloth MoE gives about 12 times faster training and more than 35 percent less VRAM | Replaced | Current Unsloth documentation reports about 1.5 times faster training and 50 to 60 percent less VRAM than FlashAttention 2 setups for Gemma 4 and Qwen3.8, plus a dedicated MoE fast path. |
| Gemma 4 E4B QLoRA trains in under 5 minutes on an RTX 3060 12 GB | Unverified timing | Unsloth publishes VRAM figures, about 10 GB for E4B and 8 GB for E2B, but does not publish a training-time figure for E4B. |
| Engine throughput table with vLLM at about 12,500 tokens per second, SGLang at about 16,200, and LMDeploy at about 16,200 | Unusable | The claim lacks model, hardware, precision, and concurrency details. A defensible reported figure is 25,000 total tokens per second per GPU for Qwen3.5-397B-A17B-NVFP4 on GB200 NVL72 disaggregated serving. |
| Qwen3-TTS-Tokenizer-12Hz is proprietary | Wrong | The tokenizer is published openly on the Hub alongside the Base models under Apache-2.0. |
| TGI health checks pass for current models | Wrong | Models released after the TGI repository archive date fail TGI health checks. |

### **0.7 Retrieval and Benchmark Findings**

| Claim | Verified value |
|---|---|
| `Alibaba-NLP/gte-Qwen2-7B` is a top MTEB model | Stale. The MTEB model space lists `nvidia/Nemotron-3.5-Embed-8B-BF16`, an 8.4 billion parameter model with 4,096 dimensions, 32,768 maximum tokens, and Sentence-Transformers compatibility, at 61.17 nDCG@10 on ViDoRe v3.1. It was first among dense retrieval models as of 20 October 2026. `nvidia/Nemotron-3-Embed-8B-BF16` leads RTEB-16 at 78.46 as of 16 July 2026. Nemotron 3.5 Rerank 8B reaches 60.49 on the combined image and text task, compared with 59.17 for Qwen3-VL-Reranker-8B. |
| Ranking stability | Retrieval rankings move monthly. Every benchmark number in the master carries its date, and the Stage 1 pre-flight requires saving a dated snapshot before making a decision. |

### **0.8 Licence Audit**

| Licence | Verified terms |
|---|---|
| Apache-2.0 | Unconditional commercial use. Applies to Qwen3.8-27B and Gemma 4. |
| MIT | Unconditional commercial use. Applies to DeepSeek V4.1 and GLM-5.3. |
| Kimi K3 Licence | MIT-style licence with a revenue trigger above US$20 million. |
| Mistral Ecosystem Open-Weight Licence | Modified MIT licence. Managed API use requires a separate Mistral commercial licence. |
| Muse Model License | Permits use, modification, fine-tuning, derivatives, distribution, and sublicensing. Managed API use above US$20 million aggregate gross revenue in any 12-month period requires notification to Meta and a separate written agreement. The usage policy is incorporated into the licence. |
| LFM Open License v1.0 | Commercial use is allowed only below US$10 million annual revenue. Qualified non-profits are exempt for non-commercial and research use. Internal research, testing, and evaluation are permitted. The threshold also applies per downstream customer. |
| Meta Custom, Llama 4 | Contains a 700 million monthly active user clause. |

### **0.9 Job-Market Signals**

| Item | Verified state, 3 November 2026 |
|---|---|
| Anthropic Research Engineer, Agents | Closed. The board entry states that the role is no longer accepting applications. |
| Anthropic current research board | Lists Pre-training, Post-Training Model Evaluations, and Staff Research Engineer, Multi-Agent Scaling. |
| Anthropic Research Engineer, Universes | Live. The role involves building training environments for long-horizon agentic tasks. It names reinforcement-learning environment design, simulation systems, large-scale ML infrastructure, sandboxing, containerization, VM infrastructure, and distributed systems. |

### **0.10 Version Verification Commands**

The following commands were used to verify package versions and Python requirements on 3 November 2026.

```bash
for p in vllm trl unsloth sglang transformers datasets peft accelerate tokenizers \
        safetensors diffusers timm sentence-transformers evaluate lighteval lm-eval \
        kernels bitsandbytes axolotl llamafactory GPTQModel llmcompressor \
        verl openrlhf skyrl torch triton torchtitan; do
  curl -s "https://pypi.org/pypi/$p/json" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['info']['name'],d['info']['version'],d['info']['requires_python'])"
done
```

The following commands check engine release history and repository state.

```bash
# Engine release history and dependency constraints
curl -s https://api.github.com/repos/vllm-project/vllm/releases
curl -s https://api.github.com/repos/sgl-project/sglang/releases

# Repository archival state
curl -s https://api.github.com/repos/huggingface/text-generation-inference \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('archived=',d['archived'],'pushed=',d['pushed_at'])"

# Model existence, licence, and access gating
curl -s "https://huggingface.co/api/models?author=moonshotai"
curl -s https://huggingface.co/api/models/meta-models/Muse-Spark-Open-816B-A42B-NVFP4
```

### **0.11 Coverage Checks**

| Check | Method | Result |
|---|---|---|
| Code fence balance | Counted triple backticks in every file | 278 in the master, an even number. All 13 files are balanced. |
| Reference integrity | Parsed the reference list and compared it with every citation marker in the body | 130 entries, numbered 1 to 130 continuously, all cited, with no duplicate URLs. `[0]` is a Python index, not a citation. |
| Heading uniqueness | Collected every heading at every level | No duplicate headings in the master. |
| Stage order | Parsed `# Stage N` headings | 12 stages, numbered 0 to 11, contiguous and in dependency order. |
| Stage-file superset | Extracted every backticked code token from each master stage and checked it in the corresponding stage file | No missing tokens across stages 0 to 10. Apparent misses in stage 11 are shared back matter that lives only in the master. |
| Source identifier coverage | Checked 231 identifiers from the 2026 roadmap source and 79 identifiers from the other sources | All present. |
| Stale model pick scan | Searched for superseded model IDs | Remaining occurrences are historical or contrast references inside the May 2026 snapshot table and correction ledger, each explicitly labelled as superseded. |

### **0.12 Unresolved Items**

Four claims could not be confirmed against a primary source during the verification window. They are treated as pointers rather than established facts.

1. **Haven, YC 2025, serverless LoRA serving.** The claim of hundreds of LoRA adapters with sub-second cold start was not confirmed by a primary source. It remains in the Stage 4 framework table as a pointer and is excluded from the version ledger.
2. **Tunix, Google, for TPU clusters.** Described as JAX-native with GRPO support, but not re-verified against a primary source. It is retained as a pointer.
3. **Kimi K3.1 Pro BrowseComp improvement from 60.6 to 78.4.** Attributed to an orchestrator and sub-agent pattern. The result is vendor-reported, with no independent reproduction found. The master states it as vendor-reported.
4. **Muse Glimmer 30B specification.** The repository is confirmed on the Hub under `meta-models`, but a primary model card with parameters and licence terms was not opened. The master lists it with the Hub reference and no capability claims.

### **0.13 Pointer Notes**

The source files reference notes that were not included in this merge. They remain pointers, and no content was invented for them.

- `progress/phase1-hf-finetuning`
- `inference-roadmap/*`, 18 files
- `targets/ai-engineering-role-taxonomy`
- `targets/anthropic-agents`

### **0.14 Reference Coverage**

The master carries the full 130-entry reference list.

| Reference range | Source type | Used for |
|---|---|---|
| 1 to 66 | Library, framework, and policy citations inherited from the first merge | Core stack, tooling, training, and governance references |
| 67 to 88 | Model cards, licences, and vendor pages for Mistral Large 4, Qwen3.6 and Qwen3.8, Qwen3-TTS, Gemma 4, Kimi K3, DeepSeek V4.1, Muse Spark, and GLM-5.3 | Stage 1 model landscape, licence matrix, and serving notes |
| 89 to 90 | Nemotron 3.5 Embed and Rerank model cards | Stage 1 retrieval landscape and Stage 7 retrieval stack |
| 91 to 96 | TurboQuant, KVTC, DART, GSPO, GRPO-lambda papers, and the Qwen GSPO note | Stages 5 and 10 |
| 97 to 101 | ASTRA, CM2, and ART papers and repositories | Stage 9 |
| 102 to 105 | TRL configuration, trainer, and paper index | Stages 4, 5, and 9 |
| 106 to 114 | vLLM releases and disaggregation documentation, LMCache report and guide, distributed-inference patterns | Stage 10 |
| 115 to 118 | TGI repository state, NVIDIA Hybrid-EP blog, DeepEP, and Megatron Core MoE guide | Stages 8 and 10 |
| 119 to 124 | PyPI pages for vLLM, Unsloth, Axolotl, SGLang, sentence-transformers, and diffusers | Version ledger |
| 125 | LFM2.5-24B-A2B model card and LFM Open License | Stage 1 licence matrix |
| 126 to 128 | Anthropic careers board and Research Engineer postings | Stage 11 job-market alignment |
| 129 to 130 | Hub models API for the `moonshotai` and `meta-models` organisations | Stage 1 release-status checks |