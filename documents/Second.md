# **HuggingFace Learning**

```
Hub → Datasets → Transformers → Fine-tuning → Optimization → Deployment
```

This file is the complete topic checklist. Use it to tick off every concept as you learn it. Each item is something you should be able to explain and write code for.

## **Version Notes — What Changed in the Current Releases**

Old tutorials break on current versions. This checklist is written against these lines:

- **transformers 5.x** — PyTorch only (TensorFlow and Flax removed). The summarization, translation, text2text-generation and image-to-text pipelines were removed. `torch_dtype` became `dtype`. `Trainer(tokenizer=...)` became `processing_class=...`. `encode_plus()` removed. `TRANSFORMERS_CACHE` replaced by `HF_HOME`
- **huggingface_hub 1.x** — `huggingface-cli` removed in favor of `hf`. httpx replaced requests. `hf_xet` is the default transfer backend and `hf_transfer` is removed. `Repository` and `HfFolder` removed
- **datasets 4.x** — loading scripts and `trust_remote_code` removed. `ds["col"]` returns a lazy `Column` object. `Sequence` replaced by `List`. Audio and video decode through torchcodec
- **trl 1.x** — stable trainers are SFT, DPO, Reward, RLOO and GRPO. Most other methods live under `trl.experimental`. `max_seq_length` became `max_length`
- **sentence-transformers 5.x** — training through `SentenceTransformerTrainer`. `model.fit()` is deprecated. `SparseEncoder` added in v5. Internal module paths moved in v5.4
- **evaluate 0.4.x** — works for classic metrics. LightEval is recommended for LLM evaluation
- **kernels 0.x** — fast moving. Kernel builds depend on your torch and CUDA versions

Check your environment any time:

```bash
pip list | grep -i -E "transformers|datasets|huggingface|trl|peft|accelerate|sentence|evaluate|kernels"
```

## **PHASE 0 — Platform Foundation**

### **1. HuggingFace Hub + huggingface_hub library**

**Authentication**
- `login()` — programmatic
- `notebook_login()` — Jupyter widget
- `interpreter_login()` — force terminal prompt
- Environment variable `HF_TOKEN`
- CLI: `hf auth login`, `hf auth whoami`
- Switching between multiple saved tokens
- `logout()` and `whoami()`
- Fine-grained tokens versus read and write tokens

**Downloading**
- `hf_hub_download()` — single file
- `snapshot_download()` — entire repo
- Downloading specific revisions and branches
- Filtering with `allow_patterns` and `ignore_patterns`
- Gated model access and `GatedRepoError` handling
- Offline mode with cached files (`HF_HUB_OFFLINE=1`)
- Cache location via `HF_HOME`
- Fast transfers through `hf_xet` (automatic)
- CLI: `hf download`

**Uploading**
- `upload_file()` — single file
- `upload_folder()` — entire directory
- `CommitOperationAdd` — atomic multi-file commits in one commit
- `create_commit()` — full commit control
- Ignoring files during upload
- CLI: `hf upload`

**Repository Management**
- `create_repo()` — model, dataset, space
- `delete_repo()`
- `list_models()`, `list_datasets()`, `list_spaces()`
- `model_info()`, `dataset_info()`
- `repo_exists()`

**Spaces Management (API only)**
- `get_space_runtime()`
- `pause_space()`, `restart_space()`
- Space hardware management

**Error Handling in v1**
- Catch `HfHubHTTPError` and its subclasses (`GatedRepoError`, `RepositoryNotFoundError`)
- Do not catch `requests` exceptions, the library no longer uses requests

**Install**
```bash
pip install -U huggingface_hub
```

### **2. Datasets**

**Loading — All Sources**
- `load_dataset()` from the HuggingFace Hub
- Loading with configs and subsets
- Loading specific splits
- Loading split slices — first 10%, row ranges, percentages
- Loading from local CSV
- Loading from local JSON and JSONL
- Loading from Parquet
- Loading from text files
- Loading from multiple files mapped to splits
- `Dataset.from_dict()`
- `Dataset.from_pandas()`
- `Dataset.from_list()`
- `Dataset.from_generator()`
- `load_from_disk()`
- Why dataset scripts no longer load (datasets 4.x) and how to find or make a Parquet version

**Inspecting**
- `features`, `column_names`, `shape`, `num_rows`, `num_columns`
- Accessing rows by index
- Accessing rows by slice
- Accessing columns by name (lazy `Column` object in 4.x, wrap with `list()` to materialize)
- Accessing a specific row and column value
- Iterating over rows

**Row Operations**
- `sort()` — ascending and descending
- `shuffle()` — with seed for reproducibility
- `select()` — pick rows by index list
- `filter()` — keep rows matching a condition
- `filter()` with `with_indices=True`
- `skip()` and `take()`
- `train_test_split()` — by proportion or absolute count
- `train_test_split()` stratified by column
- `shard()` — divide into N equal chunks

**Column Operations**
- `rename_column()` and `rename_columns()`
- `remove_columns()`
- `add_column()`
- `cast_column()` — change dtype
- `flatten()` — unnest nested fields
- `select_columns()`
- Features schema with `Value`, `ClassLabel`, `List`, `Image`, `Audio`
- `align_labels_with_mapping()`

**Transformation — map()**
- `map()` row by row
- `map()` with `batched=True` for speed
- `map()` with `num_proc` for multiprocessing
- `map()` with `remove_columns`
- `map()` that adds new columns
- `map()` that generates multiple rows from one
- `with_transform()` — on-the-fly transformation
- `with_format()` — non-destructive format change

**Combining**
- `concatenate_datasets()` — stack vertically
- `interleave_datasets()` — alternate rows
- `interleave_datasets()` with probabilities and seed

**Export and Conversion**
- `to_pandas()`
- `to_dict()`
- `to_list()`
- `to_csv()`
- `to_json()`
- `to_parquet()`
- `to_polars()`
- `to_iterable_dataset()`
- Accessing the underlying Arrow table through `ds.data`

**Format — Framework Integration**
- `set_format("torch")` for PyTorch DataLoader
- `set_format("numpy")`
- `set_format("pandas")`
- `reset_format()`

**Saving and Loading**
- `save_to_disk()` — local Arrow format
- `load_from_disk()`
- `DatasetDict` save and load

**Streaming — Large Datasets**
- Loading with `streaming=True`
- Iterating without downloading the whole dataset
- `map()`, `filter()`, `shuffle(buffer_size=...)` on streams
- `take()` and `skip()` on streams
- `interleave_datasets()` with streams
- Building a regular `Dataset` from a stream with `from_generator()`

**Pushing to Hub**
- `push_to_hub()` — single Dataset
- `push_to_hub()` with a specific split
- `DatasetDict.push_to_hub()` — all splits at once
- `push_to_hub(num_proc=...)` — faster uploads
- Private versus public datasets
- Dataset cards and README metadata

**Install**
```bash
pip install -U datasets
```

## **PHASE 1 — Core ML**

Status: NEXT

### **3. Transformers**

**Pipeline API — Tasks**

The task list below was taken from the Transformers 5 supported tasks. Confirm yours with:

```python
from transformers.pipelines import get_supported_tasks
print(get_supported_tasks())
```

Text tasks
- text-classification — sentiment, topic, intent (`sentiment-analysis` alias)
- token-classification — NER, POS tagging (`ner` alias)
- question-answering — extractive QA
- table-question-answering
- text-generation — open-ended generation and chat
- fill-mask — masked language modeling
- zero-shot-classification — no training needed
- feature-extraction — get embeddings from any model

Removed in Transformers 5 (use `generate()` or a chat model instead)
- summarization
- translation
- text2text-generation
- conversational (removed earlier)

Vision tasks
- image-classification
- object-detection
- image-segmentation
- mask-generation
- zero-shot-image-classification
- zero-shot-object-detection
- depth-estimation
- image-to-image
- image-feature-extraction
- keypoint-matching
- video-classification

Audio tasks
- automatic-speech-recognition — Whisper and others
- audio-classification
- zero-shot-audio-classification
- text-to-audio
- text-to-speech

Multimodal tasks
- image-text-to-text — vision-language chat, captioning (replaces image-to-text)
- visual-question-answering
- document-question-answering
- any-to-any

**Replacing the Removed Pipelines**
- Chat model through the `text-generation` pipeline with a messages list
- Seq2seq models (T5, BART, Marian, NLLB) through `AutoModelForSeq2SeqLM` and `generate()`
- Staying on Transformers 4 in a separate environment for old tutorials: `pip install -U "transformers<5"`

**AutoTokenizer**
- `from_pretrained()` — load any tokenizer
- Encoding a single sentence
- Encoding sentence pairs
- Batch encoding
- Calling the tokenizer directly (`encode_plus()` is removed)
- Padding strategies — `longest`, `max_length`, `do_not_pad`
- Truncation strategies — `longest_first`, `only_first`, `only_second`
- `return_tensors` — `pt`, `np`
- Special tokens — CLS, SEP, PAD, UNK, MASK, BOS, EOS
- `offset_mapping` — map tokens back to characters
- `word_ids()` — map tokens back to words
- Decoding — ids back to text, `skip_special_tokens`
- Chat templates — `apply_chat_template()`
- Saving and loading custom tokenizers
- Adding new special tokens and resizing model embeddings
- Fast tokenizers (Rust) as the default

**AutoModel — Task-Specific Classes**

Natural Language Understanding
- `AutoModelForSequenceClassification` — text classification, sentiment
- `AutoModelForMultipleChoice` — multiple choice QA
- `AutoModelForTokenClassification` — NER, POS
- `AutoModelForQuestionAnswering` — extractive span QA

Natural Language Generation
- `AutoModelForCausalLM` — GPT-style decoder models
- `AutoModelForMaskedLM` — BERT-style masked prediction
- `AutoModelForSeq2SeqLM` — encoder-decoder, T5, BART
- `AutoModelForSpeechSeq2Seq` — Whisper-style ASR

Vision
- `AutoModelForImageClassification`
- `AutoModelForObjectDetection`
- `AutoModelForSemanticSegmentation`
- `AutoModelForInstanceSegmentation`
- `AutoModelForDepthEstimation`
- `AutoModelForImageToImage`

Vision + Language (Multimodal)
- `AutoModelForVisualQuestionAnswering`
- `AutoModelForDocumentQuestionAnswering`
- `AutoModelForImageTextToText` — LLaVA, Qwen-VL, PaliGemma, SmolVLM style models

Audio
- `AutoModelForAudioClassification`
- `AutoModelForCTC` — connectionist temporal classification ASR
- `AutoModelForSpeechSeq2Seq` — Whisper

Base and Utility
- `AutoModel` — raw hidden states, embeddings
- `AutoConfig` — model architecture config
- `AutoImageProcessor` — image preprocessing
- `AutoFeatureExtractor` — audio preprocessing
- `AutoProcessor` — combined tokenizer and image or audio processor for multimodal models

**Popular Model Architectures to Know**

Encoder-only (understanding tasks)
- BERT — base for most classification tasks
- RoBERTa — improved BERT
- DistilBERT — smaller, faster BERT
- ALBERT — parameter-efficient BERT
- DeBERTa — strong on NLU benchmarks
- XLM-RoBERTa — multilingual

Decoder-only (generation tasks)
- GPT-2 — classic open generation
- Llama 3 family — Meta open LLMs
- Mistral — efficient 7B model
- Qwen family (Qwen2.5, Qwen3) — Alibaba open LLMs
- Phi family — Microsoft small models
- Gemma family — Google open LLMs
- Falcon — TII open LLMs

Encoder-decoder (seq2seq tasks)
- T5 and Flan-T5 — text-to-text framework
- BART — denoising, summarization
- mT5 — multilingual T5
- Pegasus — specialized for summarization
- MarianMT — translation

Vision
- ViT — Vision Transformer base
- DeiT — data-efficient ViT
- Swin Transformer — hierarchical vision
- DINOv2 — self-supervised vision
- CLIP and SigLIP — vision-language alignment

Multimodal
- LLaVA — visual instruction tuning
- Idefics
- Qwen-VL
- PaliGemma — Google vision-language
- SmolVLM — small vision-language model

Audio
- Whisper — ASR across many languages
- Wav2Vec 2.0 — speech representation
- HuBERT — self-supervised speech

**Trainer**
- `TrainingArguments` — all training hyperparameters
- `Trainer` — standard supervised training loop
- `processing_class` argument (replaces `tokenizer`)
- `Seq2SeqTrainer` — for seq2seq models with `generate()`
- `TrainerCallback`
- `EarlyStoppingCallback`
- `compute_metrics` — custom evaluation function
- Mixed precision: `bf16=True` on newer GPUs, `fp16=True` on older GPUs
- Data collators
  - `DataCollatorWithPadding` — for classification
  - `DataCollatorForSeq2Seq` — for summarization, translation
  - `DataCollatorForLanguageModeling` — for causal and masked LM
  - `DataCollatorForTokenClassification` — for NER

**Model Configuration and Loading**
- `AutoConfig` — load model architecture config
- `from_pretrained()` — loading weights from Hub or disk
- `from_pretrained()` with `device_map="auto"`
- `from_pretrained()` with `dtype="auto"` (v5 name, was `torch_dtype`)
- `save_pretrained()` — save to disk
- `push_to_hub()` — push model to Hub
- Model cards and README
- Loading community kernels through `attn_implementation`

**Generation — Text and Sequence**
- `model.generate()` — main generation method
- `max_new_tokens` (preferred over `max_length`)
- Greedy decoding
- Beam search — `num_beams`
- Sampling — `do_sample=True`
- Temperature — creativity control
- Top-k sampling
- Top-p nucleus sampling
- Repetition penalty
- Length penalty
- Constrained generation
- Streaming generation token by token (`TextStreamer`, `TextIteratorStreamer`)
- `GenerationConfig` — reusable generation settings

**Install**
```bash
pip install -U transformers torch
```

### **4. Tokenizers**

**Tokenization Algorithms**
- BPE — Byte-Pair Encoding, used by GPT-style models
- Byte-level BPE — used by GPT-2, RoBERTa
- WordPiece — used by BERT
- SentencePiece — used by T5 and LLaMA 1 and 2
- Unigram — used by some multilingual models
- Character-level tokenization

**Core Concepts**
- Vocabulary and `vocab_size`
- Subword tokenization — why words get split
- Special tokens and their roles
- Token IDs and `input_ids`
- `attention_mask` — what it means and why it matters
- `token_type_ids` — segment IDs for sentence pairs
- `offset_mapping` — token to character alignment
- `word_ids()` — token to word alignment
- Overflow tokens and stride for long documents
- Chat templates for instruction-tuned models

**Operations**
- Encoding single text
- Encoding text pairs
- Batch encoding
- Padding — to longest, to `max_length`
- Truncation — strategies for long text
- Decoding ids back to text
- `skip_special_tokens` in decoding
- Adding new tokens to the vocabulary
- Training a tokenizer from scratch (`Tokenizer`, `BpeTrainer`, `Whitespace` pre-tokenizer)
- Wrapping a trained tokenizer with `PreTrainedTokenizerFast`
- Saving and loading custom tokenizers

**Fast Tokenizers**
- Rust-based implementation and the speed difference
- Unified fast implementation in Transformers 5
- Parallelism in fast tokenizers

**Install**
```bash
pip install -U tokenizers
```

### **5. Evaluate**

**Text Metrics**
- `accuracy` — classification
- `f1` — binary and multiclass, macro, micro, weighted
- `precision` and `recall`
- `matthews_correlation` — MCC
- `exact_match` — QA
- `bleu` — machine translation
- `rouge` — rouge1, rouge2, rougeL for summarization
- `meteor` — translation metric
- `ter` — translation edit rate
- `sacrebleu` — standardized BLEU

**Semantic Metrics**
- `bertscore` — semantic similarity using BERT embeddings
- `bleurt` — learned evaluation metric

**Code and Structured**
- `code_eval` — for code generation (executes code, use only in a sandbox)
- `squad` — exact match and F1 for QA

**Usage Patterns**
- `evaluate.load()` — load a single metric
- `evaluate.combine()` — multiple metrics at once
- `metric.compute()` — evaluate predictions
- `metric.add_batch()` — streaming evaluation
- `compute_metrics` function for Trainer integration
- `evaluate.list_evaluation_modules()` — discover metrics
- Creating custom metrics

**LLM Evaluation**
- LightEval — the library Hugging Face recommends for benchmarking LLMs
- Learn it after your first fine-tunes

**Install**
```bash
pip install -U evaluate
pip install scikit-learn rouge_score sacrebleu
```

## **PHASE 2 — Training and Fine-tuning**

### **6. Accelerate**

**Core Concepts**
- `Accelerator` object — central controller
- `accelerator.prepare()` — wraps model, optimizer, dataloader
- `accelerator.backward()` — replaces `loss.backward()`
- Mixed precision — fp16, bf16
- Gradient accumulation
- Gradient clipping (`accelerator.clip_grad_norm_`)

**Training Setups**
- Single GPU training
- Multi-GPU training — DDP (Distributed Data Parallel)
- Multi-node training
- TPU training
- CPU-only training
- DeepSpeed integration — ZeRO stages 1, 2, 3
- FSDP — Fully Sharded Data Parallel

**State Management**
- `accelerator.save_state()` — full training checkpoint
- `accelerator.load_state()` — resume training
- `accelerator.wait_for_everyone()` — sync across processes
- `accelerator.gather()` — collect tensors from all GPUs
- `accelerator.is_main_process` — run code only on rank 0

**CLI**
- `accelerate config` — interactive setup wizard
- `accelerate launch` — run training script
- `accelerate test` — verify setup

**Utilities**
- `find_executable_batch_size` — automatic batch size finder
- `notebook_launcher` — run distributed training from a notebook
- `PartialState` — lightweight distributed state

**Install**
```bash
pip install -U accelerate
```

### **7. PEFT — Parameter Efficient Fine-Tuning**

**LoRA — Low Rank Adaptation**
- Concept — train only rank decomposition matrices
- `r` — rank, controls how many parameters are trained
- `lora_alpha` — scaling factor
- `lora_dropout`
- `target_modules` — which layers to apply LoRA to
- Choosing `target_modules` for different architectures
- `target_modules="all-linear"` shortcut
- `LoraConfig`
- `get_peft_model()`
- `print_trainable_parameters()`
- Saving LoRA adapters separately from the base model
- Loading adapters with `PeftModel.from_pretrained()`
- `merge_and_unload()` — merge adapter into base model

**QLoRA — Quantized LoRA**
- 4-bit quantization with NF4 dtype
- Double quantization
- `BitsAndBytesConfig` integration
- `prepare_model_for_kbit_training()`
- Fine-tuning 7B models on a single consumer GPU
- Merging a QLoRA adapter: reload the base in 16-bit, attach the adapter, then merge

**Other PEFT Methods**
- Prefix Tuning — prepend learnable tokens
- Prompt Tuning — learn soft prompt embeddings
- P-Tuning — learnable continuous prompts
- IA3 — rescaling activations
- AdaLoRA — adaptive rank allocation
- LoftQ — LoRA with quantization-aware initialization

**Task Types in PEFT**
- `TaskType.SEQ_CLS` — sequence classification
- `TaskType.TOKEN_CLS` — token classification
- `TaskType.CAUSAL_LM` — language modeling
- `TaskType.SEQ_2_SEQ_LM` — seq2seq
- `TaskType.QUESTION_ANS` — question answering

**Multi-adapter Usage**
- Loading multiple adapters
- Switching between adapters
- Combining adapters — add, cat, svd, linear

**Install**
```bash
pip install -U peft
```

### **8. TRL — Transformer Reinforcement Learning**

**Stable and Experimental Tiers (TRL v1)**
- Stable (semver guarantees): SFT, DPO, Reward modeling, RLOO, GRPO and their close variants
- Experimental (`trl.experimental`, APIs can change): KTO, ORPO, CPO, IPO, GKD, PPO and 75+ other methods
- Methods get promoted from experimental to stable over time, so check the docs
- Import difference: `from trl import SFTTrainer` versus `from trl.experimental.orpo import ORPOTrainer`

**Supervised Fine-Tuning**
- `SFTTrainer` — easiest way to fine-tune any LLM
- `SFTConfig` — training arguments for SFT
- Passing a model id string directly to the trainer
- Dataset formats — plain text, prompt and completion, conversational messages
- `formatting_func` — structure data as prompt and response
- Chat templates — `apply_chat_template()`
- `packing` — combine short examples for efficiency
- `max_length` control (was `max_seq_length`)
- `completion_only_loss` — train only on the completion
- `assistant_only_loss` — train only on assistant turns (needs a chat template with generation markers)
- Integration with PEFT and LoRA
- Memory options — `gradient_checkpointing`, `use_liger_kernel`, `activation_offloading`

**Preference Learning**
- `DPOTrainer` — Direct Preference Optimization (stable)
- `DPOConfig` — `beta`, loss type
- Dataset format — `prompt`, `chosen`, `rejected` columns
- Truncation with `max_length` only (`max_prompt_length` and `max_completion_length` removed)
- Reference model created automatically, or derived from the adapter with PEFT
- Experimental: IPO (Identity Preference Optimization)
- Experimental: KTO (Kahneman-Tversky Optimization, binary feedback)
- Experimental: CPO (Contrastive Preference Optimization)
- Experimental: ORPO

**Reinforcement Learning**
- `RewardTrainer` and `RewardConfig` — train a reward model from preferences (stable)
- `GRPOTrainer` and `GRPOConfig` — Group Relative Policy Optimization (stable)
- Reward functions: plain Python functions, reward models, or a list of both
- GRPO is the approach behind DeepSeek-R1 style reasoning training
- `RLOOTrainer` — REINFORCE Leave-One-Out (stable)
- `PPOTrainer` and `PPOConfig` — older approach with KL penalty and clip range, now in the experimental area

**Utilities**
- `trl vllm-serve` — fast generation during online RL
- Frozen reference model for KL divergence
- Training logs and warnings for collapsed reward signals

**Install**
```bash
pip install -U trl
```

### **9. Safetensors**

**Core Concepts**
- Why safetensors exists — no arbitrary code execution unlike pickle
- Format structure — JSON header plus raw tensor data
- Zero-copy memory mapping
- Framework support — PyTorch, NumPy, JAX

**Operations**
- `save_file()` — save tensors to disk
- `load_file()` — load all tensors
- `safe_open()` — inspect and load selectively without a full load
- Getting tensor names without loading weights
- Loading a single specific tensor
- Metadata in safetensors files
- Sharded checkpoints and `model.safetensors.index.json`

**Integration with Transformers**
- `save_pretrained()` uses safetensors by default
- `from_pretrained()` loads safetensors by default
- Converting an old `.bin` checkpoint to `.safetensors` by loading and saving again

**Install**
```bash
pip install -U safetensors
```

## **PHASE 3 — Specialized**

Pick based on your goal.

### **10. Sentence Transformers**

For: semantic search, RAG, embeddings, similarity, clustering, reranking

**Core Concepts**
- Sentence embeddings versus token embeddings
- Cosine similarity for semantic comparison
- Bi-encoder versus cross-encoder architecture
- Pooling strategies — mean, CLS, max
- Dense versus sparse embedding models (`SparseEncoder`, new in v5)

**Popular Models**
- `all-MiniLM-L6-v2` — fast, small, general purpose
- `all-mpnet-base-v2` — higher quality general purpose
- `multi-qa-MiniLM` — optimized for QA search
- `paraphrase-multilingual` — many languages
- `BAAI/bge` series — strong retrieval models
- `intfloat/e5` series — strong retrieval models
- MTEB leaderboard — check it for the current best model for your task

**Operations**
- `encode()` — single or batch encoding
- `encode()` with `normalize_embeddings`
- `model.similarity()` for similarity matrices
- Semantic search with `util.semantic_search()`
- Cosine similarity with `util.cos_sim()`
- Dot score with `util.dot_score()`
- Paraphrase mining
- Clustering embeddings

**Cross Encoders**
- `CrossEncoder` for reranking
- Bi-encoder plus cross-encoder pipeline for RAG
- Score normalization
- `CrossEncoderTrainer` for training rerankers (v4 and later)

**Fine-tuning (Trainer-based API)**
- `SentenceTransformerTrainer` and `SentenceTransformerTrainingArguments`
- Datasets with columns such as `anchor`, `positive`, `negative`
- Training losses
  - `CosineSimilarityLoss` — regression pairs
  - `ContrastiveLoss` — positive and negative pairs
  - `TripletLoss` — anchor, positive, negative
  - `MultipleNegativesRankingLoss` — best default for retrieval
  - `MatryoshkaLoss` — flexible embedding sizes
- Hard negative mining
- Evaluators during training
  - `EmbeddingSimilarityEvaluator`
  - `InformationRetrievalEvaluator`
  - `RerankingEvaluator`
- `push_to_hub()` for sharing models
- Import paths: from v5.4 losses and evaluators live under `sentence_transformers.sentence_transformer`, top-level imports unchanged
- The old `model.fit()` with `InputExample` is the deprecated route

**Install**
```bash
pip install -U sentence-transformers
pip install -U "sentence-transformers[train]"
```

### **11. Diffusers**

For: image generation, video generation, audio synthesis

**Pipeline Types**
- Text to Image — `StableDiffusionPipeline`, `FluxPipeline`, or the generic `DiffusionPipeline`
- Image to Image — style transfer
- Inpainting — fill masked regions
- Outpainting — extend image beyond borders
- Text to Video — CogVideoX, AnimateDiff
- Image to Video
- Text to Audio — AudioLDM2
- Unconditional generation
- Auto pipelines: `AutoPipelineForText2Image`, `AutoPipelineForImage2Image`, `AutoPipelineForInpainting`

**Core Concepts**
- Diffusion process — forward noise, reverse denoise
- Denoising steps — more steps, better quality, slower
- Guidance scale — how closely to follow the prompt
- Negative prompts — what to avoid
- Seed for reproducibility
- Latent diffusion — work in compressed latent space

**Schedulers**
- DDPM — original
- DDIM — deterministic, fewer steps
- DPMSolverMultistep — fast, high quality
- Euler and EulerAncestral
- PNDM
- UniPC
- Choosing and swapping schedulers

**Notable Models**
- Stable Diffusion 1.5 — base, widely supported (repo `stable-diffusion-v1-5/stable-diffusion-v1-5`, the old `runwayml` repo is no longer the home)
- Stable Diffusion XL (SDXL) — higher resolution
- Stable Diffusion 3 (SD3)
- FLUX.1 — black-forest-labs, gated model: accept the license on the Hub first
- ControlNet — guided generation with conditions
  - Canny edges
  - Depth maps
  - Human pose
  - Segmentation maps
  - Scribble
- IP-Adapter — image prompt adapter
- LoRA for diffusion models
- Textual Inversion — learn new concepts

**Memory and Speed Optimization**
- `enable_attention_slicing()`
- VAE slicing and tiling
- `enable_model_cpu_offload()`
- xformers memory efficient attention
- `torch.compile()` for speed
- float16 versus bfloat16

**Install**
```bash
pip install -U diffusers
pip install -U "diffusers[torch]"
```

### **12. timm — PyTorch Image Models**

For: image classification, feature extraction, transfer learning

**Model Discovery**
- `timm.list_models()` — all available
- Filtering by architecture pattern
- Filtering with `pretrained=True`
- `timm.model_info()` — details about a model

**Model Architectures**
- ResNet family — ResNet18, 34, 50, 101, 152
- EfficientNet family — B0 through B7
- Vision Transformer (ViT) family
- Swin Transformer — hierarchical patches
- ConvNeXt — modern CNN baseline
- DenseNet family
- MobileNet family — edge devices
- RegNet family
- MaxViT — hybrid CNN and attention
- DINOv2 — self-supervised features
- EfficientViT — efficient vision transformer

**Operations**
- `timm.create_model()` — create any model
- `pretrained=True` — load ImageNet weights
- `num_classes` — set output classes
- `num_classes=0` — feature extractor mode, no head
- `global_pool` — how to pool spatial features
- Custom classifier head replacement
- Partial fine-tuning — freeze trunk, train head

**Data Preprocessing**
- `resolve_model_data_config()` — correct config for each model
- `create_transform()` — correct transforms for each model
- Correct mean and std normalization per model
- Input size requirements per architecture

**Feature Extraction**
- `features_only=True` — intermediate feature maps
- `out_indices` — which stages to extract
- Using timm as a backbone for detection and segmentation

**Install**
```bash
pip install -U timm
```

## **PHASE 4 — Low Level and Advanced**

### **13. Bitsandbytes**

Learn when you hit GPU memory limits during fine-tuning or inference.

**Quantization Types**
- 8-bit quantization — `BitsAndBytesConfig(load_in_8bit=True)`
- 4-bit quantization — `BitsAndBytesConfig(load_in_4bit=True)`
- NF4 — NormalFloat4, best quality for 4-bit
- FP4 — Float4, alternative to NF4
- Double quantization — quantize the quantization constants

**Compute dtypes**
- float16 for older GPUs
- bfloat16 for A100, H100, and newer GPUs
- float32 for CPU

**Integration Points**
- `BitsAndBytesConfig` passed as `quantization_config` in transformers
- `device_map="auto"` with quantized models
- `prepare_model_for_kbit_training()` before applying PEFT
- QLoRA — quantization plus LoRA combined

**8-bit Optimizers**
- `bnb.optim.Adam8bit`
- `bnb.optim.AdamW8bit`
- Paged versions — `paged_adamw_8bit`, `paged_adamw_32bit`
- Memory savings from quantized optimizer states
- Setting `optim="paged_adamw_8bit"` in `TrainingArguments`

**Install**
```bash
pip install -U bitsandbytes
```

### **14. Kernels + CUDA + Triton**

Advanced. For maximum performance and understanding model internals.

**HuggingFace Kernels Library**
- Concept — optimized compute kernels hosted on the Hub
- `get_kernel()` — load a kernel from the Hub, for example `kernels-community/flash-attn2`
- Reading each kernel repo's model card for its available functions
- Using community kernels in model forward passes
- Loading kernels through `attn_implementation` in Transformers
- Writing and publishing your own kernels with `kernel-builder`
- Kernel versioning and compatibility with your torch and CUDA versions

**Key GPU Concepts to Understand**
- What a GPU kernel is — a parallel function across thousands of threads
- Thread, block, grid hierarchy in CUDA
- Memory types — global, shared, registers, constant
- Memory coalescing — why access patterns matter
- Occupancy — how many threads run simultaneously
- Warp — 32 threads that execute together
- Bank conflicts in shared memory
- Roofline model — memory bound versus compute bound analysis

**Notable Kernels and Why They Matter**
- Flash Attention — memory-efficient attention, used in modern LLMs
- Flash Attention 2 — faster, better parallelism
- Paged Attention — vLLM's key innovation for serving
- Fused LayerNorm and RMSNorm — combine operations, fewer memory reads
- Rotary Embeddings (RoPE) — fused kernel for Llama, Mistral, Qwen
- Fused cross-entropy — lower memory during training
- Triton matmul — can beat cuBLAS in specific shapes

**Triton — Python-Based GPU Programming**
- Why Triton — write GPU kernels in Python, not C++
- `tl.program_id()` — which block am I running as
- `tl.arange()` — indices within a block
- `tl.load()` and `tl.store()` — memory access with masks
- `BLOCK_SIZE` as `tl.constexpr` — compile-time constant
- `@triton.jit` decorator
- Launching with a grid function and `triton.cdiv()`
- `triton.autotune()` — automatic config search
- `triton.testing.do_bench()` — benchmark your kernel
- Fusion — combining multiple ops into one kernel
- What to build as practice
  - Vector addition kernel — the hello world equivalent
  - Softmax kernel
  - Matrix multiplication kernel
  - Flash Attention from scratch

**Raw CUDA Path**
- CUDA C++ basics — `__global__`, `__device__`, `__host__`
- Launching kernels — grid and block dimensions
- `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- Shared memory usage — `__shared__`
- Synchronization — `__syncthreads()`
- Atomic operations
- cuBLAS — highly optimized matrix ops
- cuDNN — deep learning primitives
- Writing PyTorch C++ and CUDA extensions (`load_inline` needs a C++ wrapper that launches the kernel)
- Binding CUDA to Python with pybind11

**Learning Resources for This Path**
- CUDA by Example — start here
- Triton tutorials on triton-lang.org
- HuggingFace Kernels docs on huggingface.co/docs/kernels
- Flash Attention paper — Dao et al, 2022
- CUDA Programming Guide — NVIDIA official docs
- PyTorch custom C++ and CUDA extension tutorial

**Install**
```bash
pip install -U kernels
pip install -U triton
```

## **Full Connection Map**

```
FOUNDATION
  huggingface_hub     auth, download, upload, repos
  datasets            load, process, map, filter, push_to_hub
        |
        v
CORE ML
  transformers
    - pipeline API, current task list
    - AutoTokenizer
    - AutoModel and all task-specific variants
    - Trainer and TrainingArguments
    - generate() and GenerationConfig
  tokenizers          how text becomes numbers
  evaluate            measure model quality (LightEval for LLMs)
        |
        v
TRAINING
  accelerate          multi-GPU, mixed precision, DeepSpeed, FSDP
  peft                LoRA, QLoRA, adapters
  trl                 SFT, DPO, GRPO, RLOO, reward models
  safetensors         save and load weights safely
        |
        v
SPECIALIZED
  sentence-transformers   embeddings, search, RAG, reranking
  diffusers               image and video generation
  timm                    vision models and features
        |
        v
LOW LEVEL
  bitsandbytes        quantization for big models
  kernels             optimized GPU ops from the Hub
  triton              custom GPU kernels in Python
  CUDA                raw GPU programming in C++
        |
        v
SHARE
  huggingface_hub     push model, model card, versioned releases
```

## **Week by Week Schedule**

```
Weeks 01-02   huggingface_hub                      DONE
Weeks 03-04   datasets, all operations             DONE
Weeks 05-06   transformers: pipeline API
Weeks 07-08   transformers: AutoModel, Trainer, generate()
Week  09      tokenizers
Week  10      evaluate + first full fine-tune
Week  11      accelerate
Weeks 12-13   peft + LoRA + QLoRA
Week  14      trl: SFTTrainer
Week  15      trl: DPO, GRPO, reward models
Week  16      safetensors
Week  17      pick one: sentence-transformers or diffusers or timm
Week  18      bitsandbytes
Weeks 19+     kernels, triton, CUDA
```

## **GitHub Repo Structure**

```
huggingface-learning/
├── 01_hub/
├── 02_datasets/
├── 03_transformers/
│   ├── pipeline/
│   ├── automodel/
│   │   ├── text/
│   │   ├── vision/
│   │   ├── audio/
│   │   └── multimodal/
│   ├── tokenizer/
│   ├── generation/
│   └── trainer/
├── 04_tokenizers/
├── 05_evaluate/
├── 06_accelerate/
├── 07_peft/
│   ├── lora/
│   └── qlora/
├── 08_trl/
│   ├── sft/
│   ├── dpo/
│   └── grpo/
├── 09_safetensors/
├── 10_specialized/
│   ├── sentence_transformers/
│   ├── diffusers/
│   └── timm/
├── 11_advanced/
│   ├── bitsandbytes/
│   ├── kernels/
│   ├── triton/
│   └── cuda/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## **The One Rule**

**Code lives on GitHub. Model weights live on HuggingFace Hub.**

Never push `.bin`, `.safetensors`, or `.pt` files to GitHub.
Use `model.push_to_hub()` to put weights on HF Hub instead.

Never commit your `HF_TOKEN`. Keep it in `.env` and list `.env` in `.gitignore`.

*Last updated: November 2026*
*Covers: huggingface_hub, datasets, transformers, tokenizers, evaluate,*
*accelerate, peft, trl, safetensors, sentence-transformers, diffusers,*
*timm, bitsandbytes, kernels, triton, CUDA*