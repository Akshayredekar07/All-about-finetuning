# **All About Fine-Tuning**

<p align="center">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="20%" alt="Hugging Face Logo"/>
</p>

<p align="center">
  Structured fine-tuning journey: Platform and Data Foundations, SFT and Alignment, Agentic RL and Production Serving
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-In%20Progress-FFB300?style=flat-square"/>
  <img src="https://img.shields.io/badge/Focus-Fine--Tuning%20and%20Agentic%20RL-6A1B9A?style=flat-square"/>
</p>

A personal, structured fine-tuning repo, built from first principles, not tutorials. One roadmap runs from Hub basics to fine-tuning, alignment, agentic RL, and production serving, organized as twelve stages (00 to 11). Every stage has one outcome, a concrete build target, and pre-flight questions. No skipping to RL before SFT is solid, and no fine-tuning before prompting and RAG have been ruled out.

Fine-tuning is my route into **agentic post-training and production LLM serving**: models that use tools over multiple turns, trained against verifiable rewards, and served under a real latency and cost budget. The goal is public artifacts with real numbers and a stated baseline, not notebooks.

Full roadmap: [`roadmap-nov2026/complete-readmap.md`](roadmap-nov2026/complete-readmap.md)

---

### Before You Fine-Tune

**Method gate:** try prompting first, then RAG, then fine-tuning, in that order. Fine-tuning changes behavior, format, and style. It is a poor way to add facts.

| Stage | Pick it when | Skip it when |
|---|---|---|
| Prompting | The base model can already do the task with examples in context | Output format or behavior is consistently wrong |
| RAG | The model lacks facts that live in documents | The problem is style, format, or tool behavior |
| SFT and PEFT | You need a stable format, style, or domain behavior | The data is small, noisy, or contaminated |
| Preference tuning and RL | SFT works but preferences or reasoning need to improve | There is no SFT base or no reliable reward |

---

### Stage 00: Platform Foundation

**Build target:** Authenticate against the Hub, move models and datasets in both directions, and load, transform, and republish a dataset without keeping a local copy.

| Topic | Key Concepts |
|---|---|
| Hub Client | `login()`, `whoami()`, `HF_TOKEN`, fine-grained tokens, `hf auth login` |
| Download | `hf_hub_download()`, `snapshot_download()`, revisions, `allow_patterns`, gated access, offline mode |
| Upload | `upload_file()`, `upload_folder()`, `create_commit()`, `create_repo()`, Spaces runtime control |
| Error Handling | `HfHubHTTPError` and subclasses, `GatedRepoError` |
| Datasets | Apache Arrow memory mapping, `load_dataset` from every source, `map` with `batched=True`, `train_test_split` |
| Streaming | Buffered shuffle, `buffer_size` for reproducibility, `take`, `filter` |
| Combine and Export | `concatenate_datasets`, `interleave_datasets`, `push_to_hub`, dataset cards |
| Agent Traces | Tool-call conversations loaded into a normalized `messages` schema |

---

### Stage 01: Model Landscape and Selection

**Build target:** A written decision record that narrows a task constraint to one base model and defends it on four axes: capability, license, memory footprint, and tooling support. Nothing is trained.

| Topic | Key Concepts |
|---|---|
| Three Different Sets | Released, open-weight, and fine-tunable are not the same set. Label every model on both axes |
| Working Shortlist | Small dense and edge models as the realistic target, chosen for experimentability, not leaderboard rank |
| Architecture Facts | Hybrid linear attention models need bf16 LoRA, MoE experts block 4-bit training, chat template presence, thinking mode behavior |
| License Audit | Apache-2.0 and MIT as defaults, revenue-triggered clauses, a fine-tune inherits the base license |
| VRAM Planning | Memory matrix for QLoRA and bf16 LoRA per model, weights memory as parameters times bytes |
| Decision Framework | Match the base model to the hardware, modality, context length, and deployment constraint |

---

### Stage 02: Core ML Stack

**Build target:** Fine-tune a classifier with the built-in `Trainer` and measure it with a metric that matches the task, plus one benchmark run through `lm-evaluation-harness`.

| Topic | Key Concepts |
|---|---|
| Pipelines | Task families, removed pipelines and their replacements, chat models through `text-generation` |
| Auto Classes | `AutoTokenizer`, `AutoModelFor...` families, `AutoProcessor`, `AutoConfig` |
| Trainer | `TrainingArguments`, `processing_class`, data collators, `compute_metrics`, bf16 vs fp16 |
| Generation | Greedy, beam, sampling, temperature, top-k, top-p, repetition penalty, streaming |
| Tokenization | BPE, WordPiece, SentencePiece, `attention_mask`, offset mapping, overflow and stride, chat templates, training your own tokenizer |
| Evaluation | `evaluate`, `lighteval`, `lm-evaluation-harness`, n-gram decontamination |
| Attention | Scaled dot-product, multi-head, GQA, MQA, MLA, FlashAttention, KV cache as an architectural property |

---

### Stage 03: Data Engineering

**Build target:** A domain dataset (1K to 5K samples) that is cleaned, formatted, split, versioned, and published to the Hub with a data card. A trainer must consume it without conversion.

| Topic | Key Concepts |
|---|---|
| Data Shapes | Instruction and completion, multi-turn chat, tool use, preference pairs, binary labels, verifiable reward, distillation |
| Formats | ChatML, Alpaca, ShareGPT, OpenAI tool calling, `{prompt, chosen, rejected}` |
| Quality | Read 100 random rows, length distribution, source bias, MinHash near-dedup |
| Decontamination | Harness n-gram files, `--decontamination_ngrams_path`, never train on eval sets |
| Synthetic Data | Generate, filter, rank with a rubric judge, mix with real data, model collapse |
| Licensing and Compliance | Per-source license audit, EU AI Act timeline, PII removal, model cards |
| Dataset Construction | Define the task, source, filter, format, split 80/10/10, version and document |

---

### Stage 04: SFT and PEFT

**Build target:** LoRA from scratch in about 150 lines of PyTorch, then a QLoRA SFT run on a single GPU that saves, merges, and resumes from a checkpoint another tool can load.

| Topic | Key Concepts |
|---|---|
| LoRA | Low-rank update $W + \frac{\alpha}{r}BA$, $B$ initialized to zero, multiply $B(Ax)$ never $BA$ |
| Default Config | Rank 16, alpha 32, all linear layers, dropout 0.05, learning rate $10^{-4}$ to $5 \times 10^{-4}$ |
| LoRA Family | DoRA, rsLoRA, GaLore, VeRA, AdaLoRA, IA3, LoftQ, switch only on a measured problem |
| QLoRA | NF4, double quantization, bf16 compute, paged 8-bit optimizers, `prepare_model_for_kbit_training` |
| SFTTrainer | `max_length`, `packing`, `completion_only_loss`, `assistant_only_loss`, stable vs experimental trainers |
| Optimizers | AdamW, 8-bit and paged variants, warmup then cosine decay |
| Thinking Mode | `enable_thinking`, at least 75 percent reasoning examples to preserve reasoning behavior |
| Forgetting | PEFT by default, replay buffer, conservative learning rate |
| Checkpoints | `merge_and_unload`, merging a QLoRA adapter on a 16-bit base, safetensors, Accelerate |

---

### Stage 05: Alignment and Post-Training

**Build target:** Take an SFT checkpoint through a DPO run and a GRPO run with a programmatic verifier, and show the delta against the SFT baseline on a held-out set.

| Topic | Key Concepts |
|---|---|
| Post-Training Stack | SFT, then preference optimization, then reasoning RL, then agentic RL |
| Preference Optimization | Bradley-Terry, why $Z(x)$ cancels in DPO, DPO loss from first principles, reference model handling |
| DPO Family | IPO, KTO, SimPO, ORPO, iterative DPO, chosen by data shape |
| Reward Models | `RewardTrainer`, chosen and rejected format |
| GRPO and RLVR | Group-relative advantage with no critic, zero-variance groups, clipped objective with KL |
| GRPO Family | DAPO, Dr.GRPO, GSPO, GRPO-lambda, `importance_sampling_level` and `loss_type` in TRL |
| Verifier Design | Math, code with partial credit, format rewards, reward hacking mitigation |
| Distillation | Teacher to student with KL plus cross-entropy, recalibration, re-quantization |
| Gradient-Free | Evolution strategies on quantized models |

---

### Stage 06: Frameworks at Scale

**Build target:** Train the same model on the same data with Unsloth, TRL and PEFT, and Axolotl using matched hyperparameters. Compare time, peak memory, final loss, and eval scores.

| Topic | Key Concepts |
|---|---|
| Layer Cake | PEFT at the bottom, TRL on top, Axolotl and LlamaFactory wrapping both, TorchTune is a dead end |
| Unsloth | Single-GPU speed, long context, `FastLanguageModel`, bf16 LoRA constraints for Qwen3.5 |
| Axolotl | One YAML for the whole run, FSDP, DeepSpeed, TP, CP, EP, kept in its own environment |
| LlamaFactory | Broadest model coverage, CLI only on localhost, never the web UI |
| TRL and PEFT | Reference implementations, `environment_factory`, the portable adapter format |
| RL Engines | TorchTitan, verl, OpenRLHF, SkyRL, Tinker, AgentLightning |
| Scale Strategy | Single GPU, FSDP2, ZeRO-3, TP with CP, Megatron-LM, EP with TP for MoE |

---

### Stage 07: Specializations (Optional Fork)

**Build target:** One specialization that matches the target role, with its own primary metric, plus a performance layer for when memory or speed becomes the wall.

| Topic | Key Concepts |
|---|---|
| Embeddings and Reranking | Bi-encoder vs cross-encoder, `MultipleNegativesRankingLoss`, Matryoshka loss, hard negatives, nDCG@10 |
| Vision-Language Models | Vision encoder, projector, LLM decoder, projector-only vs LLM LoRA vs full fine-tune |
| Diffusion | Latent diffusion, flow matching, LoRA, DreamBooth, textual inversion, ControlNet |
| Audio | Whisper LoRA, word error rate, TTS speaker fine-tuning |
| GPU Kernels | Triton basics, memory coalescing, occupancy, roofline model, FlashAttention, Hugging Face `kernels` |
| Quantization Theory | BF16, FP8, INT8, NF4, INT4 AWQ, GGUF Q4_K_M, NVFP4, weights vs KV cache quantization |

---

### Stage 08: Distributed Training at Scale (Depth Track)

**Build target:** Put a model that does not fit on one GPU onto a node, choose the parallelism strategy on evidence, and read a training log well enough to name the bottleneck.

| Topic | Key Concepts |
|---|---|
| Parallelism Taxonomy | DDP, FSDP and ZeRO-3, tensor, pipeline, and expert parallelism, and which resource each one saves |
| Pipeline Bubble | Bubble fraction $\frac{p-1}{m}$ and how micro-batches shrink it |
| Sharded Data Parallelism | ZeRO stages 1 to 3, offload, FSDP `FULL_SHARD`, mixed precision policy, wrap class |
| Expert Parallelism | All-to-all dispatch and combine, DeepEP, Hybrid-EP, keeping EP and TP inside NVLink |
| MoE Health | Expert utilization histogram, alert below about 1 percent |
| Diagnosis | Compute-bound vs communication-bound vs data-bound log signatures, `nsys`, `NCCL_DEBUG` |
| Multi-Node Launch | Rendezvous endpoint, shared filesystem, `/dev/shm`, hangs with no error |

---

### Stage 09: Agentic Fine-Tuning and RL

**Build target:** A tool environment with a deterministic verifier or a checklist rubric, a held-out environment split, and an ablation table showing the trained agent is better, not just more verbose.

| Topic | Key Concepts |
|---|---|
| Why a Separate Stage | Trajectories, outcome rewards, rollouts against an environment, task-completion evaluation |
| ART and RULER | GRPO over multi-turn rollouts, trainer and inference server split, ranking trajectories instead of scoring them |
| ASTRA | Synthesized tool environments, schema-level dependency graphs, F1-style trajectory reward that penalizes flailing |
| CM2 | Checklist rewards, binary criteria with evidence, sparse reward with dense evaluation |
| Orchestrator Pattern | Trained orchestrator, frozen sub-agents, delegation granularity as a hyperparameter |
| TRL Environments | `GRPOTrainer` with `environment_factory`, vLLM server mode |
| Agent Evaluation | tau-squared-Bench, BFCL, ToolSandbox, SWE-Bench Verified, held-out environments, tool calls per solved task, three runs minimum |

---

### Stage 10: Inference Serving

**Build target:** A served endpoint with a documented deployment decision (merged or multi-LoRA), a quantization quality delta, and a benchmark table for TTFT, TPOT, and throughput at batch 1, 8, and 32.

| Topic | Key Concepts |
|---|---|
| Deployment Decision | Merge vs multi-LoRA, INT4, FP8, GGUF Q4_K_M, cost of each |
| Quantization Tooling | GPTQModel, llm-compressor, `compressed-tensors`, AutoAWQ deprecated, AutoGPTQ unmaintained |
| vLLM and SGLang | `--max-loras` defaults to 1, `--max-lora-rank` defaults to 16, RadixAttention, `--lora-backend` |
| Engines | vLLM, SGLang, TensorRT-LLM, LMDeploy, llama.cpp, Ollama, TGI archived |
| KV Cache | Memory formula, paged KV, prefix caching, FP8 KV, TurboQuant, KVTC, offloading, MLA |
| Prefill and Decode | Compute-bound vs memory-bound, chunked prefill, continuous batching |
| Speculative Decoding | EAGLE3, DART, MTP, retraining the draft head after fine-tuning, when it hurts |
| PD Disaggregation | KV transfer connectors, NIXL, LMCache, Mooncake, timing the transfer first |
| Edge | llama.cpp, Ollama with MLX, MLX-LM, 4-bit QAT checkpoints |
| Compute Without a GPU | HF Spaces, GPU Instances, ZeroGPU, Jobs, Tinker |

---

### Stage 11: Capstones, Portfolio, and JD Alignment

**Build target:** Public artifacts on GitHub, each with a clean README, real numbers, a stated baseline, reproduction instructions, and a benchmark instead of a claim.

| Capstone | What it proves | Target window |
|---|---|---|
| Domain QLoRA, quantize, serve | The full vertical: data, train, quantize, serve, benchmark | Dec 2026 |
| DPO for preference alignment | Alignment improves without capability loss, measured beyond judge noise | Jan 2027 |
| GRPO for reasoning | A verifiable reward improves held-out accuracy by at least 10 percent | Feb to Mar 2027 |
| Agentic environment | A tool environment, verifier, rubric, held-out split, and ablation table | After Stage 09 |
| Multi-LoRA serving | Three adapters hot-swapped per request with measured overhead | Any time |

| Topic | Key Concepts |
|---|---|
| JD Alignment | Map each target role to the stage and capstone that covers it, name the gaps |
| Interview Bank | Foundational, practical, depth, and serving questions, answered cold |
| LLMOps Pipeline | Provenance, PII removal, dedup, experiment tracking with token count and base revision, hot-swap deployment |
| Cost Levers | QLoRA, bf16 LoRA, KV compression, INT4 weights, spot instances, prefix caching |
| Ship Checklist | Dockerfile, fresh-clone reproduction, honest limitations, model card, no secrets in the repo |

---

## Repository Structure

```
All-about-finetuning/
├── README.md
├── roadmap-nov2026/              # complete roadmap and per-stage notes
├── pyproject.toml
├── envs/                         # one file per isolated environment
├── common/                       # code imported by two or more stages
│   ├── evals/
│   ├── data/
│   ├── verifiers/
│   └── bench/
├── docs/
│   ├── decisions/                # dated notes: why this model or framework
│   ├── licenses.md
│   └── model_matrix.md
├── 00_platform/
├── 01_model_selection/
├── 02_core_stack/
├── 03_data_engineering/
├── 04_sft_peft/
├── 05_alignment/
├── 06_frameworks/
├── 07_specializations/
├── 08_distributed_training/
├── 09_agentic_rl/
├── 10_inference_serving/
└── 11_capstones/
    ├── 01_domain_qlora/
    ├── 02_dpo_alignment/
    ├── 03_grpo_reasoning/
    ├── 04_agentic_environment/
    └── 05_multi_lora_serving/
```

**Template inside every stage and capstone** (created on demand, names fixed):

```
NN_stage_name/
├── README.md      # outcome and pre-flight questions
├── notes/
├── labs/          # runnable scripts
├── configs/       # yaml, model names allowed in filenames here
└── results/       # metrics, plots, tables
```

Capstones add a `Dockerfile` and a `src/` directory.

**Isolated environments:** the training and serving stacks pin conflicting dependency ranges, so they never share an environment. The adapter file is the only artifact that crosses the boundary.

| Environment | Purpose | Notes |
|---|---|---|
| `train` | TRL, PEFT, Unsloth | Newest dependency range |
| `serve` | vLLM, SGLang | Serving engines cap the allowed `transformers` range |
| `axolotl` | Config-driven multi-GPU training | Needs a newer Python than the rest of the stack |
| `edge` | llama.cpp, MLX, Ollama | GGUF path, no Python training stack needed |

```
uv venv envs/.train && source envs/.train/bin/activate
uv pip install -U huggingface_hub datasets transformers accelerate evaluate
uv pip install -U peft trl safetensors tokenizers
```

---

## Framework Decision Table

| Situation | Framework | Why |
|---|---|---|
| Single consumer GPU | Unsloth | Fastest, lowest memory, longest context |
| Multi-GPU, complex parallelism | Axolotl | FSDP, DeepSpeed, TP, CP, EP composable |
| Rapid model-by-model iteration | LlamaFactory CLI | Broadest model coverage |
| RL and preference research | TRL | Reference implementation |
| Custom research, full control | TRL and PEFT | Lowest abstraction |
| Distributed RL | verl, OpenRLHF, SkyRL, Tinker | Engines built for rollouts |

---

## Resources

| Resource | Type |
|---|---|
| [Hugging Face Hub docs](https://huggingface.co/docs/hub) | Official, start here |
| [Transformers docs](https://huggingface.co/docs/transformers) | Pipelines, Auto classes, Trainer |
| [Datasets docs](https://huggingface.co/docs/datasets) | Arrow, streaming, `map` |
| [PEFT docs](https://huggingface.co/docs/peft) | LoRA, QLoRA, adapter format |
| [TRL docs](https://huggingface.co/docs/trl) | SFT, DPO, GRPO, environments |
| [Unsloth docs](https://docs.unsloth.ai) | Single-GPU fine-tuning |
| [Axolotl docs](https://docs.axolotl.ai) | Config-driven multi-GPU training |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Standard benchmark harness |
| [vLLM docs](https://docs.vllm.ai) | Serving, LoRA flags, KV transfer |
| [vLLM `serve` CLI](https://docs.vllm.ai/en/latest/cli/serve/) | Engine and LoRA arguments |
| [LoRA paper](https://arxiv.org/abs/2106.09685) | Low-rank adaptation |
| [QLoRA paper](https://arxiv.org/abs/2305.14314) | 4-bit NF4 fine-tuning |
| [DPO paper](https://arxiv.org/abs/2305.18290) | Direct preference optimization |
| [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) | Where GRPO was introduced |
| [TurboQuant paper](https://arxiv.org/abs/2504.19874) | Online KV cache vector quantization |
| [KVTC paper](https://arxiv.org/abs/2511.01815) | Transform coding for the KV cache |

---

## Principles

**Do not fine-tune first.** Prompting, then RAG, then fine-tuning.

**SFT first, RL later.** No RL algorithm works well without a seeded SFT base.

**Data beats algorithm.** The dominant failure mode of a fine-tune is bad data, not a bad optimizer. Decontaminate, always.

**Released is not the same as fine-tunable.** An API model, an open-weight model, and a model you can adapter-tune on your own GPU are three different sets.

**Measure, do not claim.** Report the delta against a stated baseline on a held-out set, with the sample size. A single benchmark score is not evidence.

**KV cache and GRPO come first.** They are to inference and fine-tuning what SQL and transactions are to web engineering. Understand them mechanically before touching production.

**Libraries change fast.** Check the installed version before debugging and never trust an old tutorial. Use `uv pip install`, not plain `pip`.

**Code on GitHub, weights on the Hub.** Never commit `.bin`, `.safetensors`, or `.pt` files, and never commit `HF_TOKEN`.

---

<p align="center">
  Built by <a href="https://github.com/Akshayredekar07">@Akshayredekar07</a>, AI Engineer learning fine-tuning for production LLM systems
</p>