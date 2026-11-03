# Research and Version Audit (November 2026) — Model Refresh

Companion audit for **00 - Master Roadmap (Merged and Model-Refreshed, Nov 2026)**. Every factual claim in the master that depends on a library release, a model release, a licence term, a benchmark number, or a technical paper is listed here with its source and the date it was verified.

| | |
|---|---|
| Verification window | 3 November 2026 (one session, same day) |
| Sources merged | 8 documents, 7,061 source lines |
| Public surfaces checked | HuggingFace Hub API and model cards · GitHub releases API for vLLM, SGLang, TRL, Unsloth, Axolotl, DeepEP, ART, ASTRA, CM2, TGI · PyPI JSON API for 28 packages · arXiv abstract pages · vendor documentation and blogs |
| Deliverables audited | 1 master + 12 stage files + this audit + 1 archive |
| Reference list | 130 entries, continuous 1–130, every entry cited at least once, no duplicate URLs |
| Unresolved items | 4 (listed in the final section, each carried with a stated reason) |

## Method

Three passes, in this order.

1. **Inventory.** Every model, library, paper, licence and configuration claim in all eight source documents was extracted into a ledger with its asserted value and its asserted date.
2. **Verification.** Each ledger row was checked against a primary surface: the model's own card for parameters, context, licence and chat-template behaviour; the package's release notes or PyPI metadata for versions and Python floors; the paper's own abstract page for claims attributed to it; the repository state for archival and pin conflicts.
3. **Application.** Verified values were written into the master in place, and every deviation from a source assertion was recorded as a numbered correction with its evidence.

Where a source asserted a number that no primary surface supports, the number was removed rather than softened, and the removal is recorded. Where a source asserted a number that a primary surface supports but with different conditions (a different hardware target, a different preset), the conditions were attached rather than the number being dropped.

## Source coverage matrix

The eight sources and where their content lives in the canonical set.

| Source | Lines | Tables | Code blocks | Checklists | Landed in |
|---|---|---|---|---|---|
| S1 `First.md` — core stack prose | 1,638 | 14 | 63 | 9 | Stages 0–11, Schedule, Repo structure, One Rule, Key papers |
| S2 `Second.md` — core stack checklist | 1,053 | 2 | 18 | 40+ | Stage checklists (per-topic blocks) |
| S3 `00 - Overview & Index.md` | 179 | 1 | 1 | 6 | Stage spine, Method-selection gate, Weekly schedule |
| S4 `01 - Methods Deep Dive.md` | 361 | 5 | 14 | 12 | Stages 4, 5 |
| S5 `02 - Data & Datasets.md` | 418 | 3 | 9 | 8 | Stage 3 |
| S6 `03 - Frameworks & Tools 2026.md` | 606 | 9 | 21 | 15 | Stages 6, 10 |
| S7 `04 - Capstones & JD Alignment.md` | 399 | 4 | 5 | 11 | Stage 11 |
| S8 `LLM Fine-Tuning & Inference Engineering Roadmap (2026)` | 1,499 | 27 | 34 | 62 | Stages 1, 2, 4, 5, 7, 8, 9, 10, 11; Key papers; Repo structure |

Coverage was re-verified mechanically after the rebuild: 231 named identifiers from S8 (architectures, algorithms, flags, function names, model families, benchmarks, tools) and 79 identifiers from S1–S7 were checked against the canonical set. All are present. The only two initially absent — `Falcon TII` and `vllm_lora` — were restored by the rebuild.

## Correction ledger

The master's corrections table carries the 37 corrections in roadmap order. This section records the evidence behind each group of them.

### Group A — library and version pins (corrections 1–12, 36)

| Assertion under review | Verdict | Evidence |
|---|---|---|
| `huggingface_hub` 2.4.0, `datasets` 5.1.0, `transformers` 5.22.1, `trl` 1.17.1, `peft` 0.21.2, `accelerate` 1.15.0 | Confirmed | PyPI JSON API, 3 November 2026 — upload dates 30 Oct (transformers, trl), 1 Oct (peft), 9 Sep (accelerate), 5 Oct (datasets) |
| `tokenizers` 0.23.3, `safetensors` 0.8.0, `kernels` 0.18.1, `diffusers` 0.42.1, `timm` 1.0.30, `sentence-transformers` 6.2.0 | Confirmed | PyPI JSON API, 3 November 2026 |
| `unsloth` 2026.10.11, `axolotl` 0.21.1, `llamafactory` 0.9.5, `torchtitan` 0.4.0, `verl` 0.9.1, `openrlhf` 0.11.2, `skyrl` 0.4.0 | Confirmed | PyPI JSON API, 3 November 2026 |
| `vllm` 0.33.0, `sglang` 0.5.23, `gptqmodel` 7.8.0, `llmcompressor` 0.15.0, `torch` 2.15.0, `triton` 3.9.0 | Confirmed | PyPI JSON API; vLLM GitHub releases show v0.33.0rc3 tagged 2 November 2026 and v0.32.0 on 18 October 2026 |
| Axolotl requires Python ≥3.12 | New finding | PyPI metadata `requires_python >=3.12` — breaks the shared 3.11 base environment |
| vLLM 0.33.0 requires `transformers>=5.16.1,<5.20.0` | New finding | vLLM v0.32.0 release notes; creates a hard conflict with the training-side `transformers 5.22.1` pin |

### Group B — model landscape (corrections 13–17, 25, 28–33, 37)

| Model family | Verified state, 3 November 2026 | Source |
|---|---|---|
| Mistral Large 4 | Released 29–31 Oct 2026; ~1T total / 52B active; 1M context; 128 routed experts + 1 shared; natively multimodal; reasoning effort default high and non-disableable; embedded DSpark drafter; Mistral Ecosystem Open-Weight Licence, MaaS needs a separate commercial agreement; API $0.70/M in, $0.07/M cached, $2.10/M out; LoRA SFT example ships in TRL 1.17.0 for one 8×B300 node | Model card [67], licence [68], pricing [69], TRL release notes [6] |
| Kimi K3 | Released 16 Jul 2026; 2.8T / 104B active; 1,048,576 context; 93 layers (1 dense + 69 KDA + 24 Gated MLA); 896 routed experts top-16 + 2 shared; SiTU-GLU; 160K vocab; MoonViT-V2 401M vision encoder; thinking always on, `reasoning_effort` low/high/max (default max); MXFP4 weights / MXFP8 activations; Kimi K3 Licence with a >US$20M-revenue trigger | Model card [81] |
| Kimi K3.1 Pro | **API only.** The `moonshotai` Hub organisation lists K2.5, K2.6, K2.7-Code, K3 and Kimi-Linear, and no K3.1 repository. Weights promised 4 November 2026 | Hub models API [129] |
| Qwen3.8-27B | Dense 27B unified VLM; 64 layers = 16×(3×Gated DeltaNet→FFN + 1×Gated Attention→FFN); hidden 5120; 262,144 native context → 1M via YaRN; native image and hour-scale video; trained with MTP; Apache-2.0; thinking default ON, `enable_thinking=False`, `reasoning_effort` xhigh/medium/low, `preserve_thinking=True` default; thinking sampling temp 1.0/top_p 0.95/top_k 20, non-thinking temp 0.7/top_p 0.80/presence_penalty 1.5 | Model card [70] |
| Qwen3.6-27B | Dense 27B, Apache-2.0, released April 2026; fine-tunable in Unsloth at 18–23 GB RAM depending on the build | Model card [71], Unsloth changelog [79] |
| Qwen3.5 family | Released 2 Mar 2026; 201 languages; Gated DeltaNet + sparse MoE; **4-bit QLoRA not recommended** for dense or MoE; bf16 LoRA footprints 3/5/10/22/56/74 GB for 0.8B/2B/4B/9B/27B/35B-A3B | Unsloth docs [79][80] |
| Gemma 4 | E2B 2.3B effective (5.1B with embeddings, 35 layers, 512 SW, 128K ctx); E4B 4.5B effective (8B, 42 layers, 512 SW); 12B 11.95B (48 layers, 1024 SW, 256K ctx); 26B-A4B; 31B 30.7B (60 layers, 256K ctx); text + image, plus audio on E2B/E4B/12B; 140+ languages; releases 31 Mar / 2 Apr / 16 Apr 2026; Apache-2.0; MTP 1.4–2.2× inference speed-up | Google model card [75] |
| Gemma 4 fine-tuning footprint | E2B QLoRA ~8 GB; E4B QLoRA ~10 GB; E4B LoRA ~17 GB; 31B QLoRA ~22 GB; 26B-A4B LoRA >40 GB; E4B QLoRA recommended over E2B LoRA; 4-bit MoE blocked by bitsandbytes | Unsloth Gemma 4 guide [76] |
| DeepSeek V4.1 Flash | MIT; 1M context; image-text-to-text; native `transformers` support; **no Jinja chat template** (reference `encoding` implementation); `deepseek-recipe` Rust libraries | Model card [83] |
| Muse Spark Open | Released 15 Oct 2026; 816B / 42B active; 1M context; 256 experts top-8; 62 layers; hidden 8192; vocab 202,048; text+image+video+audio in; `reasoning_effort` low/medium/high/xhigh, default xhigh; ≥700 GB NVFP4 or 2 TB BF16; gated (manual approval, legal name + date of birth + organisation required); Muse Model License §3: MaaS above US$20M aggregate gross revenue in 12 months requires notifying Meta and a separate written agreement | Model card [84], licence [85], AWS announcement [87], Hub API [130] |
| GLM-5.3 | 320B total / 18B active; MIT; released 9 Sep 2026; BF16 and Flash variants on the Hub | Model card [88] |
| LFM2.5-24B-A2B | 24B / 2.45B active; LFM Open License v1.0: commercial use only below US$10M annual revenue, qualified non-profits exempt for non-commercial and research use, internal research/testing/evaluation permitted, threshold also applies per downstream customer | Model card [125] |
| Qwen3-TTS | Released 22 Jan 2026; 0.6B and 1.7B on a 12Hz codec; 3-second voice cloning; 10 languages; ~97 ms streaming latency; 5M+ hours of training; Apache-2.0; variants Base, CustomVoice, VoiceDesign; recipe = DPO on speech preference pairs → rule-based rewards + GSPO → speaker fine-tune; the `Qwen3-TTS-Tokenizer-12Hz` is part of the **open** release | Model cards [72][73], technical report [74] |
| Nemotron 3.5 family | Bolt 74B-A9B; Super VL 120B/12B; Embed 8B and 1B (+NVFP4 for Blackwell); Rerank 8B; ASR Streaming 0.6B; Transcribe; Dory 1.0-4B; October 2026 | Model cards [89][90] |
| Beam 1.0 | 501.3B / 23.4B; Apache-2.0; released 5–16 Oct 2026; added to `transformers` v5.20.1 | Release notes, model card |
| MiniMax M3.1 Flash | Released 19 Oct 2026 | Release notes |
| LongCat-2.5 | Released 2 Nov 2026 | Release notes |

### Group C — technique claims (corrections 18–27)

| Assertion under review | Verdict | Evidence |
|---|---|---|
| "TurboQuant = PolarQuant + QJL, ≥6× compression, zero measurable accuracy loss" | **Rejected as stated** | Upstream vLLM merged TurboQuant through PR #38479, shipped in v0.20.0 as a v1 attention backend using Hadamard rotation + scalar quantisation, and **omits QJL** because QJL "hurts attention quality by amplifying variance through softmax". Shipped presets: `turboquant_k8v4` 2.6× / +1.17% PPL, `turboquant_4bit_nc` 3.8× / +2.71%, `turboquant_k3v4_nc` ~3.5× / +10.63%, `turboquant_3bit_nc` 4.9× / +20.59% [91][106] |
| "KVTC: ~20×, NVIDIA, <1% drop, needs calibration" | Confirmed with scope | arXiv 2511.01815, accepted ICLR 2026: up to 20× preserving reasoning and long-context accuracy, 40×+ in specific cases, brief per-layer PCA calibration, no model changes. Not a first-class vLLM feature [92] |
| "Hybrid-EP: 514% throughput improvement" | **Rejected** | NVIDIA technical blog, 2 Feb 2026: +14% DeepSeek-V3 (829 → 943 TFLOPs/GPU), +8% Megatron-FSDP, +5.5% BF16 and +9.9% MXFP8 on Qwen3-235B [116]. DeepEP README documents the Hybrid-EP backend, TMA-based, fine-grained comm–compute overlap [117] |
| "TGI in maintenance mode since December 2025" | **Understated** | GitHub API: `archived = true`, last push 2026-03-21 — the repository was archived on 21 March 2026 [115] |
| "DART: 6.8× lower draft latency vs EAGLE3" | Confirmed | arXiv 2601.19278: 6.8× vs EAGLE3 and 53.3× vs standard speculative sampling on Qwen3-14B/H20; 2.03–3.44× end-to-end; ~30% average and up to 65% on code workloads; drafts from Qwen3 1.7B–32B [93] |
| "GRPO-λ: +3–4.5 points on math benchmarks" | **Replaced** | arXiv 2510.00194: critic-free TD learning with eligibility-trace reformulation, claiming 30–40% better than GRPO on math benchmarks [95] |
| "GSPO = `loss_type='gspo'`" | **Wrong API** | TRL 1.17.1 `GRPOConfig.loss_type` accepts `grpo`, `dr_grpo`, `dapo`, `bnpo`, `cispo`, `sapo`, `luspo`, `vespo`. GSPO is `importance_sampling_level="sequence"` with `loss_type="grpo"`, `beta=0.0`, `epsilon=3e-4`, `epsilon_high=4e-4`, `steps_per_generation=4` [102][104]. TRL warns against pairing sequence-level sampling with `bnpo`/`dr_grpo`/`dapo`/`cispo` [103] |
| "DeepSeek V4 needs `--trust-remote-code`" | Outdated | `deepseek-ai/DeepSeek-V4.1-Flash-1017` ships native `transformers` support [83] |
| "Qwen3.5 on vLLM needs ≥0.17.0" | Outdated | vLLM is at 0.33.0; 0.32.0 introduced the `transformers<5.20.0` upper bound [106][107] |
| "Gemma 4 26B-A4B fits ~12 GB with Unsloth QLoRA" | **Wrong** | Unsloth: 26B-A4B LoRA needs >40 GB, and 4-bit MoE is blocked because bitsandbytes has no 4-bit `nn.Parameter` for MoE experts [76] |
| "Unsloth MoE: ~12× faster, >35% less VRAM" | **Replaced** | Current Unsloth docs: ~1.5× faster and ~50–60% less VRAM than FA2 setups for Gemma 4 and Qwen3.8, plus a dedicated MoE fast path [76][78] |
| "Gemma 4 E4B QLoRA <5 minutes on an RTX 3060 12 GB" | **Unverified timing** | Unsloth publishes the VRAM figures (~10 GB E4B, ~8 GB E2B) but no training-time figure for E4B [76] |
| Engine throughput table (vLLM ~12,500 tok/s, SGLang ~16,200 tok/s, LMDeploy ~16,200 tok/s) | **Unusable** | No model, hardware, precision or concurrency attached. vLLM's own reporting — 25K total TPS/GPU on Qwen3.5-397B-A17B-NVFP4 with GB200 NVL72 disaggregated serving — is the shape of a defensible claim [110] |
| "Qwen3-TTS-Tokenizer-12Hz is proprietary" | **Wrong** | The tokenizer is published openly on the Hub alongside the Base models under Apache-2.0 [73] |
| "TGI health checks pass for current models" | **Wrong** | Models released after the 21 March 2026 archive fail TGI health checks [115] |

### Group D — retrieval and benchmark numbers (correction 29)

| Claim | Verified value | Source |
|---|---|---|
| "Alibaba-NLP/gte-Qwen2-7B — Top MTEB 2026" | Stale. MTEB's model space lists `nvidia/Nemotron-3.5-Embed-8B-BF16` (8.4B params, 4,096 dims, 32,768 max tokens, Sentence-Transformers compatible) at ViDoRe v3.1 61.17 nDCG@10, first among dense retrieval models as of 20 Oct 2026, and `nvidia/Nemotron-3-Embed-8B-BF16` at RTEB-16 78.46, multilingual leader as of 16 Jul 2026. Nemotron 3.5 Rerank 8B reaches 60.49 on the combined image+text task against 59.17 for Qwen3-VL-Reranker-8B | [41][89][90] |
| Ranking stability | Rankings move monthly. Every number in the master carries its date, and the Stage 1 pre-flight requires saving a dated snapshot before a decision | [41] |

### Group E — licence audit (correction 31)

| Licence | Verified terms | Source |
|---|---|---|
| Apache-2.0 | Unconditional commercial use | Qwen3.8-27B card [70], Gemma 4 card [75] |
| MIT | Unconditional commercial use | DeepSeek V4.1 card [83], GLM-5.3 card [88] |
| Kimi K3 Licence | MIT-style; revenue trigger above US$20M | Model card [81] |
| Mistral Ecosystem Open-Weight | Modified MIT; **MaaS requires a separate Mistral commercial licence** | Licence file [68] |
| Muse Model License | Permits use, modification, fine-tuning, derivatives, distribution and sublicensing; MaaS above US$20M aggregate gross revenue in any 12-month period requires notifying Meta and a separate written agreement; usage policy incorporated | Licence [85] |
| LFM Open License v1.0 | Commercial use only below US$10M annual revenue; qualified non-profits exempt for non-commercial and research use; internal research, testing and evaluation permitted; threshold also applies per downstream customer | [125] |
| Meta Custom (Llama 4) | 700M monthly-active-user clause | [85] |

### Group F — job-market signal (Stage 11)

| Item | Verified state, 3 November 2026 | Source |
|---|---|---|
| Anthropic "Research Engineer, Agents" | **Closed** — the board entry reads "This job is no longer accepting applications" | [127] |
| Anthropic current research board | Lists Pre-training, Post-Training Model Evaluations, and Staff Research Engineer, Multi-Agent Scaling | [126] |
| Anthropic "Research Engineer, Universes" | Live; builds training environments for long-horizon agentic tasks; names RL environment design, simulation systems, large-scale ML infrastructure, sandboxing, containerisation, VM infrastructure and distributed systems | [128] |

## Version ledger verification commands

Every version in the master's ledger can be re-checked with these, and each was run on 3 November 2026:

```bash
for p in vllm trl unsloth sglang transformers datasets peft accelerate tokenizers \
         safetensors diffusers timm sentence-transformers evaluate lighteval lm-eval \
         kernels bitsandbytes axolotl llamafactory GPTQModel llmcompressor \
         verl openrlhf skyrl torch triton torchtitan; do
  curl -s "https://pypi.org/pypi/$p/json" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['info']['name'],d['info']['version'],d['info']['requires_python'])"
done
```

```bash
# engine release history and pin constraints
curl -s https://api.github.com/repos/vllm-project/vllm/releases
curl -s https://api.github.com/repos/sgl-project/sglang/releases

# archival state
curl -s https://api.github.com/repos/huggingface/text-generation-inference \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('archived=',d['archived'],'pushed=',d['pushed_at'])"

# model existence, licence and gating
curl -s "https://huggingface.co/api/models?author=moonshotai"
curl -s https://huggingface.co/api/models/meta-models/Muse-Spark-Open-816B-A42B-NVFP4
```

## Coverage checks run against the rebuilt set

| Check | Method | Result |
|---|---|---|
| Code fence balance | Count of ``` per file | 278 in the master (even), even in all 13 files |
| Reference integrity | Parse the reference list, compare with every `[n]` marker in the body | 130 entries, 1–130 continuous, all cited, no duplicate URLs; `[0]` is the `ds[0]` Python index, not a citation |
| Heading uniqueness | Collect every `#`/`##`/`###` heading | No duplicate headings in the master |
| Stage order | Parse `# Stage N` headings | 12 stages, 0–11, contiguous, in dependency order |
| Stage-file superset | For each master stage section, extract every backticked code token and check it appears in the corresponding stage file | 0 missing across stages 0–10; stage 11's apparent misses are the shared back matter, which lives in the master only |
| Source identifier coverage | 231 identifiers from S8 and 79 from S1–S7 checked against the canonical set | All present |
| Stale model pick scan | Search for superseded model IDs | Remaining occurrences are historical or contrast references inside the May 2026 snapshot table and the correction ledger, each explicitly labelled as superseded |

## Items carried forward unresolved

Four assertions could not be confirmed against a primary surface and are therefore marked as pointers rather than facts in the master.

1. **Haven (YC 2025) serverless LoRA serving** — claimed "hundreds of LoRA adapters, <1s cold start". No primary source confirmed in this window. Kept in the Stage 4 framework table as a pointer, excluded from the version ledger.
2. **Tunix (Google) for TPU clusters** — described as JAX-native with GRPO support. Not re-verified against a primary source. Same treatment.
3. **Kimi K3.1 Pro BrowseComp 60.6 → 78.4** — attributed to the orchestrator/sub-agent pattern. Vendor-reported; no independent reproduction found. Stated as vendor-reported in the master.
4. **Muse Glimmer 30B specification** — the repository is confirmed on the Hub under `meta-models`, but a primary model card with parameters and licence terms was not opened. The master lists it with the Hub reference and no capability claims.

## Pointer notes not present in this merge

The source files cross-link to notes that were not provided. They remain pointers; nothing was invented for them.

- `progress/phase1-hf-finetuning`
- `inference-roadmap/*` (18 files)
- `targets/ai-engineering-role-taxonomy`
- `targets/anthropic-agents`

## Reference list

The master carries the full 130-entry reference list. Entries 1–66 are the library, framework and policy citations inherited from the first merge; entries 67–130 are the model, licence, paper, engine and job-market citations added by this refresh.

| # | Source | Used for |
|---|---|---|
| 67–88 | Model cards, licences and vendor pages for Mistral Large 4, Qwen3.6/3.8, Qwen3-TTS, Gemma 4, Kimi K3, DeepSeek V4.1, Muse Spark, GLM-5.3 | Stage 1 landscape, license matrix, serving notes |
| 89–90 | Nemotron 3.5 Embed and Rerank cards | Stage 1 retrieval landscape, Stage 7 retrieval stack |
| 91–96 | TurboQuant, KVTC, DART, GSPO, GRPO-λ papers and the Qwen GSPO note | Stages 5, 10 |
| 97–101 | ASTRA, CM2 and ART papers and repositories | Stage 9 |
| 102–105 | TRL v1.17.1 configuration, trainer and paper index | Stages 4, 5, 9 |
| 106–114 | vLLM releases and disaggregation documentation, LMCache report and guide, distributed-inference patterns | Stage 10 |
| 115–118 | TGI repository state, NVIDIA Hybrid-EP blog, DeepEP, Megatron Core MoE guide | Stages 8, 10 |
| 119–124 | PyPI pages for vLLM, Unsloth, Axolotl, SGLang, sentence-transformers, diffusers | Version ledger |
| 125 | LFM2.5-24B-A2B model card and LFM Open License | Stage 1 license matrix |
| 126–128 | Anthropic careers board and the two Research Engineer postings | Stage 11 job-market alignment |
| 129–130 | Hub models API for the `moonshotai` and `meta-models` organisations | Stage 1 release-status checks |
