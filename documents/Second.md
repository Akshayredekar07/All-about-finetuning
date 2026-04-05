# **HuggingFace Learning**

```
Hub → Datasets → Transformers → Fine-tuning → Optimization → Deployment → Agents
```

## **PHASE 0 — Platform Foundation**

### 1. HuggingFace Hub + huggingface_hub library

**Authentication**
- login() — programmatic
- notebook_login() — Jupyter widget
- interpreter_login() — force terminal prompt
- Environment variable HF_TOKEN
- CLI: hf auth login
- Switching between multiple saved tokens
- logout() and whoami()

**Downloading**
- hf_hub_download() — single file
- snapshot_download() — entire repo
- Downloading specific revisions and branches
- Downloading with filters and ignore_patterns
- Gated model access and GatedRepoError handling
- Offline mode with cached files
- CLI: hf download

**Uploading**
- upload_file() — single file
- upload_folder() — entire directory
- CommitOperationAdd — atomic multi-file commits in one commit
- create_commit() — full commit control
- Ignoring files during upload

**Repository Management**
- create_repo() — model, dataset, space
- delete_repo()
- list_models(), list_datasets(), list_spaces()
- model_info(), dataset_info()
- repo_exists()

**Spaces Management**
- Creating spaces with different SDKs
- get_space_runtime()
- pause_space(), restart_space()
- Space hardware management



### 2. Datasets

**Loading — All Sources**
- load_dataset() from HuggingFace Hub
- Loading with configs and subsets
- Loading specific splits
- Loading split slices — first 10%, row ranges, percentages
- Loading from local CSV
- Loading from local JSON / JSONL
- Loading from Parquet
- Loading from text files
- Loading from multiple files mapped to splits
- Dataset.from_dict()
- Dataset.from_pandas()
- Dataset.from_list()
- Dataset.from_generator()
- load_from_disk()

**Inspecting**
- features, column_names, shape, num_rows, num_columns
- Accessing rows by index
- Accessing rows by slice
- Accessing columns by name
- Accessing specific row + column value
- Iterating over rows

**Row Operations**
- sort() — ascending and descending
- shuffle() — with seed for reproducibility
- select() — pick rows by index list
- filter() — keep rows matching a condition
- filter() with indices using with_indices=True
- skip() and take() for IterableDataset
- train_test_split() — by proportion or absolute count
- train_test_split() stratified by column
- shard() — divide into N equal chunks

**Column Operations**
- rename_column()
- rename_columns() — multiple at once
- remove_columns()
- add_column()
- cast_column() — change dtype
- flatten() — unnest nested fields
- select_columns() — keep only specific columns
- Features schema definition with Value, ClassLabel, Sequence
- align_labels_with_mapping()

**Transformation — map()**
- map() row by row
- map() batched=True for speed
- map() with num_proc for multiprocessing
- map() with remove_columns
- map() that adds new columns
- map() that generates multiple rows from one
- with_transform() — on-the-fly transformation
- with_format() — non-destructive format change

**Combining**
- concatenate_datasets() — stack vertically
- interleave_datasets() — alternate rows
- interleave_datasets() with probabilities and weights

**Export and Conversion**
- to_pandas()
- to_dict()
- to_list()
- to_csv()
- to_json()
- to_parquet()
- to_arrow()

**Format — Framework Integration**
- set_format("torch") for PyTorch DataLoader
- set_format("tensorflow") for Keras
- set_format("numpy")
- set_format("pandas")
- reset_format()

**Saving and Loading**
- save_to_disk() — local Arrow format
- load_from_disk()
- DatasetDict save and load

**Streaming — Large Datasets**
- loading with streaming=True
- Iterating without downloading
- map(), filter(), shuffle() on streams
- take() and skip() on streams
- interleave_datasets() with streams
- Converting IterableDataset to Dataset

**Pushing to Hub**
- push_to_hub() — single Dataset
- push_to_hub() with specific split
- DatasetDict push_to_hub() — all splits at once
- Private vs public datasets
- Dataset cards and README metadata



## PHASE 1 — Core ML
> Status: NEXT



### 3. Transformers

**Pipeline API — Tasks**

Text tasks
- text-classification — sentiment, topic, intent
- token-classification — NER, POS tagging, chunking
- question-answering — extractive QA
- summarization — abstractive and extractive
- translation — between any language pair
- text-generation — open-ended generation
- text2text-generation — T5-style seq2seq
- fill-mask — masked language modeling
- zero-shot-classification — no training needed
- conversational — dialogue and chat

Vision tasks
- image-classification
- object-detection
- image-segmentation — semantic, panoptic, instance
- zero-shot-image-classification
- zero-shot-object-detection
- depth-estimation
- image-to-image
- image-to-text — captioning
- visual-question-answering

Audio tasks
- automatic-speech-recognition — Whisper and others
- audio-classification
- text-to-speech
- audio-to-audio

Multimodal tasks
- document-question-answering
- feature-extraction — get embeddings from any model

**AutoTokenizer**
- from_pretrained() — load any tokenizer
- Encoding single sentence
- Encoding sentence pairs
- Batch encoding
- Padding strategies — longest, max_length, do_not_pad
- Truncation strategies — longest_first, only_first, only_second
- return_tensors — pt, tf, np
- Special tokens — CLS, SEP, PAD, UNK, MASK, BOS, EOS
- offset_mapping — map tokens back to characters
- word_ids() — map tokens back to words
- Decoding — ids back to text
- Saving and loading custom tokenizers
- Adding new special tokens
- Fast vs slow tokenizers

**AutoModel — Task-Specific Classes**

Natural Language Understanding
- AutoModelForSequenceClassification — text classification, sentiment
- AutoModelForMultipleChoice — multiple choice QA
- AutoModelForTokenClassification — NER, POS
- AutoModelForQuestionAnswering — extractive span QA
- AutoModelForNextSentencePrediction — NSP

Natural Language Generation
- AutoModelForCausalLM — GPT-style decoder models
- AutoModelForMaskedLM — BERT-style masked prediction
- AutoModelForSeq2SeqLM — encoder-decoder, T5, BART
- AutoModelForSpeechSeq2Seq — Whisper-style ASR

Vision
- AutoModelForImageClassification
- AutoModelForObjectDetection
- AutoModelForSemanticSegmentation
- AutoModelForInstanceSegmentation
- AutoModelForDepthEstimation
- AutoModelForImageToImage

Vision + Language (Multimodal)
- AutoModelForVisualQuestionAnswering
- AutoModelForDocumentQuestionAnswering
- AutoModelForImageTextToText — LLaVA, InternVL, Qwen-VL style

Audio
- AutoModelForAudioClassification
- AutoModelForCTC — connectionist temporal classification ASR
- AutoModelForSpeechSeq2Seq — Whisper

Base and Utility
- AutoModel — raw hidden states, embeddings
- AutoModelForPreTraining — base pretraining objectives
- AutoFeatureExtractor — vision and audio preprocessing
- AutoImageProcessor — image-specific preprocessing
- AutoProcessor — combined tokenizer + feature extractor for multimodal

**Popular Model Architectures to Know**

Encoder-only (understanding tasks)
- BERT — base for most classification tasks
- RoBERTa — improved BERT
- DistilBERT — smaller, faster BERT
- ALBERT — parameter-efficient BERT
- DeBERTa — best for NLU benchmarks
- XLM-RoBERTa — multilingual

Decoder-only (generation tasks)
- GPT-2 — classic open generation
- LLaMA 2 / LLaMA 3 — Meta open LLMs
- Mistral — efficient 7B model
- Qwen2.5 — Alibaba open LLMs
- Phi-3 / Phi-4 — Microsoft small models
- Gemma 2 — Google open LLMs
- Falcon — TII open LLMs

Encoder-decoder (seq2seq tasks)
- T5 / Flan-T5 — text-to-text framework
- BART — denoising, summarization
- mT5 — multilingual T5
- Pegasus — specialized for summarization
- MarianMT — translation

Vision
- ViT — Vision Transformer base
- DeiT — data-efficient ViT
- Swin Transformer — hierarchical vision
- DINO / DINOv2 — self-supervised vision
- CLIP — vision-language alignment

Multimodal
- LLaVA — visual instruction tuning
- InstructBLIP
- Idefics
- Qwen-VL
- PaliGemma — Google vision-language

Audio
- Whisper — best ASR, all languages
- Wav2Vec 2.0 — speech representation
- HuBERT — self-supervised speech

**Trainer**
- TrainingArguments — all training hyperparameters
- Trainer — standard supervised training loop
- Seq2SeqTrainer — for seq2seq models with generate()
- TrainerCallback — hooks into training events
- EarlyStoppingCallback
- compute_metrics — custom evaluation function
- Data collators
  - DataCollatorWithPadding — for classification
  - DataCollatorForSeq2Seq — for summarization, translation
  - DataCollatorForLanguageModeling — for CLM and MLM
  - DataCollatorForTokenClassification — for NER

**Model Configuration and Loading**
- AutoConfig — load model architecture config
- from_pretrained() — loading weights from Hub or disk
- from_pretrained() with device_map="auto"
- from_pretrained() with torch_dtype for precision
- save_pretrained() — save to disk
- push_to_hub() — push model to Hub
- Model cards and README

**Generation — Text and Sequence**
- model.generate() — main generation method
- Greedy decoding
- Beam search — num_beams
- Sampling — do_sample=True
- Temperature — creativity control
- Top-k sampling
- Top-p nucleus sampling
- Repetition penalty
- Length penalty
- Constrained generation
- Streaming generation token by token
- GenerationConfig — reusable generation settings



### 4. Tokenizers

**Tokenization Algorithms**
- BPE — Byte-Pair Encoding, used by GPT models
- WordPiece — used by BERT
- SentencePiece — used by T5, LLaMA, Mistral
- Unigram — used by some multilingual models
- Character-level tokenization
- Byte-level BPE — used by GPT-2, RoBERTa

**Core Concepts**
- Vocabulary and vocab_size
- Subword tokenization — why words get split
- Special tokens and their roles
- Token IDs and input_ids
- attention_mask — what it means and why it matters
- token_type_ids — segment IDs for sentence pairs
- offset_mapping — token to character alignment
- word_ids() — token to word alignment
- Overflow tokens and stride for long documents

**Operations**
- Encoding single text
- Encoding text pairs
- Batch encoding
- Padding — to longest, to max_length
- Truncation — strategies for long text
- Decoding ids back to text
- skip_special_tokens in decoding
- Adding new tokens to vocabulary
- Training a tokenizer from scratch
- Saving and loading custom tokenizers

**Fast vs Slow Tokenizers**
- Rust-based fast tokenizers — speed difference
- When to use each
- Parallelism in fast tokenizers



### 5. Evaluate

**Text Metrics**
- accuracy — classification
- f1 — binary and multiclass, macro/micro/weighted
- precision and recall
- matthews_correlation — MCC
- exact_match — QA
- bleu — machine translation
- rouge — rouge1, rouge2, rougeL — summarization
- meteor — better translation metric
- ter — translation edit rate
- sacrebleu — standardized BLEU

**Semantic Metrics**
- bertscore — semantic similarity using BERT
- bleurt — learned evaluation metric

**Code and Structured**
- code_eval — for code generation tasks
- squad — exact match + F1 for QA

**Usage Patterns**
- evaluate.load() — load a single metric
- evaluate.combine() — multiple metrics at once
- metric.compute() — evaluate predictions
- metric.add_batch() — streaming evaluation
- compute_metrics function for Trainer integration
- evaluate.list_evaluation_modules() — discover metrics
- Creating custom metrics



## **PHASE 2 — Training and Fine-tuning**


### 6. Accelerate

**Core Concepts**
- Accelerator object — central controller
- accelerator.prepare() — wraps model, optimizer, dataloader
- accelerator.backward() — replaces loss.backward()
- Mixed precision — fp16, bf16
- Gradient accumulation
- Gradient clipping

**Training Setups**
- Single GPU training
- Multi-GPU training — DDP (Distributed Data Parallel)
- Multi-node training
- TPU training
- CPU-only training
- DeepSpeed integration — ZeRO stages 1, 2, 3
- FSDP — Fully Sharded Data Parallel

**State Management**
- accelerator.save_state() — full training checkpoint
- accelerator.load_state() — resume training
- accelerator.wait_for_everyone() — sync across processes
- accelerator.gather() — collect tensors from all GPUs
- accelerator.is_main_process — run code only on rank 0

**CLI**
- accelerate config — interactive setup wizard
- accelerate launch — run training script
- accelerate test — verify setup

**Utilities**
- find_executable_batch_size — auto batch size finder
- notebook_launcher — run distributed from notebook
- PartialState — lightweight distributed state



### 7. PEFT — Parameter Efficient Fine-Tuning

**LoRA — Low Rank Adaptation**
- Concept — train only rank decomposition matrices
- rank r — controls how many parameters are trained
- lora_alpha — scaling factor
- lora_dropout
- target_modules — which layers to apply LoRA to
- Choosing target_modules for different architectures
- LoraConfig
- get_peft_model()
- print_trainable_parameters()
- Saving LoRA adapters separately from base model
- Loading adapters with PeftModel.from_pretrained()
- merge_and_unload() — merge adapter into base model

**QLoRA — Quantized LoRA**
- 4-bit quantization with NF4 dtype
- Double quantization
- BitsAndBytesConfig integration
- prepare_model_for_kbit_training()
- Fine-tuning 7B models on single consumer GPU

**Other PEFT Methods**
- Prefix Tuning — prepend learnable tokens
- Prompt Tuning — learn soft prompt embeddings
- P-Tuning — learnable continuous prompts
- IA3 — rescaling activations
- AdaLoRA — adaptive rank allocation
- LoftQ — LoRA + quantization initialization

**Task Types in PEFT**
- TaskType.SEQ_CLS — sequence classification
- TaskType.TOKEN_CLS — token classification
- TaskType.CAUSAL_LM — language modeling
- TaskType.SEQ_2_SEQ_LM — seq2seq
- TaskType.QUESTION_ANS — question answering

**Multi-adapter Usage**
- Loading multiple adapters
- Switching between adapters
- Combining adapters — add, cat, svd, linear



### 8. TRL — Transformer Reinforcement Learning

**Supervised Fine-Tuning**
- SFTTrainer — easiest way to fine-tune any LLM
- SFTConfig — training arguments for SFT
- formatting_func — structure data as prompt + response
- Chat templates — apply_chat_template()
- packing — combine short examples for efficiency
- max_seq_length control
- Integration with PEFT and LoRA

**Preference Learning**
- DPOTrainer — Direct Preference Optimization
- DPOConfig — beta, loss type
- Dataset format — prompt, chosen, rejected columns
- IPO — Identity Preference Optimization
- KTO — Kahneman-Tversky Optimization (binary feedback)
- CPO — Contrastive Preference Optimization

**Reinforcement Learning**
- RewardTrainer — train a reward model from preferences
- RewardConfig
- PPOTrainer — Proximal Policy Optimization
- PPOConfig — KL penalty, clip range
- GRPOTrainer — Group Relative Policy Optimization
- GRPOConfig — used by DeepSeek-R1 style reasoning models

**Utilities**
- DataCollatorForCompletionOnlyLM — mask prompt, train only on response
- create_reference_model() — frozen copy for KL divergence
- get_kbit_device_map()



### 9. Safetensors

**Core Concepts**
- Why safetensors exists — no arbitrary code execution unlike pickle
- Format structure — header + raw tensor data
- Zero-copy memory mapping
- Framework support — PyTorch, TensorFlow, JAX, NumPy

**Operations**
- save_file() — save tensors to disk
- load_file() — load all tensors
- safe_open() — inspect and load selectively without full load
- Getting tensor names without loading weights
- Loading a single specific tensor
- Metadata in safetensors files

**Integration with Transformers**
- save_pretrained() automatically uses safetensors
- from_pretrained() loads safetensors by default
- Fallback to .bin when safetensors not available
- Converting .bin checkpoints to .safetensors



## **PHASE 3 — Specialized**
> Pick based on your goal



### 10. Sentence Transformers
> For: semantic search, RAG, embeddings, similarity, clustering

**Core Concepts**
- Sentence embeddings vs token embeddings
- Cosine similarity for semantic comparison
- Bi-encoder vs cross-encoder architecture
- Pooling strategies — mean, CLS, max

**Popular Models**
- all-MiniLM-L6-v2 — fast, small, general purpose
- all-mpnet-base-v2 — higher quality general
- multi-qa-MiniLM — optimized for QA search
- paraphrase-multilingual — 50+ languages
- BAAI/bge series — state of the art retrieval
- intfloat/e5 series — Microsoft embeddings

**Operations**
- encode() — single or batch encoding
- encode() with normalize_embeddings
- Semantic search with util.semantic_search()
- Cosine similarity with util.cos_sim()
- Dot score with util.dot_score()
- Paraphrase mining
- Clustering embeddings

**Cross Encoders**
- CrossEncoder for re-ranking
- Bi-encoder + cross-encoder pipeline for RAG
- Score normalization

**Fine-tuning**
- InputExample format
- Training losses
  - CosineSimilarityLoss — regression pairs
  - ContrastiveLoss — positive/negative pairs
  - TripletLoss — anchor, positive, negative
  - MultipleNegativesRankingLoss — best for retrieval
  - MatryoshkaLoss — flexible embedding sizes
- Evaluators during training
  - EmbeddingSimilarityEvaluator
  - InformationRetrievalEvaluator
  - RerankingEvaluator
- push_to_hub() for sharing models



### 11. Diffusers
> For: image generation, video generation, audio synthesis

**Pipeline Types**
- Text to Image — StableDiffusionPipeline, FluxPipeline
- Image to Image — style transfer
- Inpainting — fill masked regions
- Outpainting — extend image beyond borders
- Text to Video — CogVideoX, AnimateDiff
- Image to Video
- Text to Audio — AudioLDM2
- Unconditional generation

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
- Stable Diffusion 1.5 — base, widely supported
- Stable Diffusion XL (SDXL) — higher resolution
- Stable Diffusion 3 (SD3)
- FLUX.1 — black-forest-labs, best open model 2024-2025
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
- attention_slicing
- vae_slicing and vae_tiling
- enable_model_cpu_offload()
- xformers memory efficient attention
- torch.compile() for speed
- float16 vs bfloat16



### 12. timm — PyTorch Image Models
> For: image classification, feature extraction, transfer learning

**Model Discovery**
- timm.list_models() — all available
- Filtering by architecture pattern
- Filtering pretrained=True only
- timm.model_info() — details about a model

**Model Architectures**
- ResNet family — ResNet18, 34, 50, 101, 152
- EfficientNet family — B0 through B7
- Vision Transformer (ViT) family
- Swin Transformer — hierarchical patches
- ConvNeXt — modern CNN baseline
- DenseNet family
- MobileNet family — edge devices
- RegNet family
- MaxViT — hybrid CNN + attention
- DINO / DINOv2 — self-supervised features
- EfficientViT — efficient vision transformer

**Operations**
- timm.create_model() — create any model
- pretrained=True — load ImageNet weights
- num_classes — set output classes
- num_classes=0 — feature extractor mode, no head
- global_pool — how to pool spatial features
- Custom classifier head replacement
- Partial fine-tuning — freeze trunk, train head

**Data Preprocessing**
- resolve_model_data_config() — correct config for each model
- create_transform() — correct transforms for each model
- Correct mean and std normalization per model
- Input size requirements per architecture

**Feature Extraction**
- features_only=True — intermediate feature maps
- out_indices — which stages to extract
- Using timm as backbone for detection and segmentation



### 13. Gradio — Deep Dive
> For: building ML demos, internal tools, deployed apps

**Interface API — Simple**
- gr.Interface() — function in, function out
- Input and output component types
- examples — built-in example gallery
- title, description, article
- flagging — collect user feedback

**Blocks API — Full Control**
- gr.Blocks() context manager
- gr.Row() and gr.Column() — layout control
- gr.Tab() and gr.Tabs() — tabbed layouts
- gr.Accordion() — collapsible sections
- gr.Group() — visual grouping
- Component event listeners — .click(), .change(), .submit(), .upload()
- State management with gr.State()
- gr.update() — dynamic component updates

**Chat**
- gr.ChatInterface() — complete chatbot in one call
- gr.Chatbot() — raw chat display component
- Streaming responses token by token
- Multimodal chat with images
- Chat history management
- System prompt handling

**Components In Depth**
- gr.Textbox — text input and output
- gr.Image — image upload and display, types pil/numpy/filepath
- gr.Audio — record or upload, types numpy/filepath
- gr.Video — video upload and playback
- gr.File — any file type upload
- gr.DataFrame — interactive data table
- gr.Gallery — image grid display
- gr.Plot — matplotlib, plotly, bokeh charts
- gr.Label — classification results with confidence bars
- gr.HighlightedText — NER and span labeling display
- gr.AnnotatedImage — image with bounding boxes
- gr.Slider — numeric range control
- gr.Dropdown — single or multi-select menu
- gr.CheckboxGroup — multiple selection
- gr.Radio — single choice from options
- gr.ColorPicker
- gr.Model3D — 3D file viewer
- gr.JSON — formatted JSON display
- gr.HTML — raw HTML output
- gr.Code — syntax highlighted code display

**Advanced**
- Custom CSS and JS injection
- gr.themes — built-in and custom themes
- queue() — handle concurrent users
- concurrency_limit — parallel request control
- share=True — instant public URL tunnel
- auth — username and password protection
- Loading indicators and progress bars
- Cancellation of long running functions
- API mode — every Gradio app is also a REST API automatically
- Embedding in other webpages via iframe or web components



### 14. smolagents
> For: AI agents, tool-using models, multi-step reasoning

**Agent Types**
- CodeAgent — writes and executes Python code to use tools
- ToolCallingAgent — uses structured tool calls in JSON

**Model Backends**
- HfApiModel — any model on HuggingFace Hub
- LiteLLMModel — OpenAI, Anthropic, and 100+ providers
- TransformersModel — local model via transformers
- OpenAIServerModel — any OpenAI-compatible API endpoint

**Built-in Tools**
- DuckDuckGoSearchTool — web search
- WikipediaSearchTool — Wikipedia lookup
- VisitWebpageTool — fetch and read URLs
- PythonInterpreterTool — execute Python code
- FinalAnswerTool — return final result
- SpeechToTextTool — transcribe audio
- TextToImageTool — generate images

**Custom Tools**
- @tool decorator — wrap any Python function
- Tool class — full tool definition with schema
- Defining name, description, inputs, output_type
- Input type annotations — string, integer, image, audio

**Multi-agent**
- Manager agent + subagents
- Delegating tasks between agents
- Agent as a tool for another agent
- Parallel tool execution

**Memory and Context**
- ActionStep — record of agent actions
- agent.memory — full trace of reasoning
- Custom system prompts
- Few-shot examples in agent prompt



## **PHASE 4 — Low Level and Advanced**



### 15. Bitsandbytes
> Learn when you hit GPU memory limits during fine-tuning or inference

**Quantization Types**
- 8-bit quantization — load_in_8bit=True
- 4-bit quantization — load_in_4bit=True
- NF4 — NormalFloat4, best quality for 4-bit
- FP4 — Float4, alternative to NF4
- Double quantization — quantize the quantization constants
- Nested quantization for extra memory savings

**Compute dtypes**
- float16 for older GPUs
- bfloat16 for A100, H100, newer GPUs
- float32 for CPU

**Integration Points**
- BitsAndBytesConfig in transformers
- device_map="auto" with quantized models
- prepare_model_for_kbit_training() before applying PEFT
- QLoRA — quantization + LoRA combined

**8-bit Optimizers**
- bnb.optim.Adam8bit — uses 8-bit optimizer states
- bnb.optim.AdamW8bit
- Paged versions — paged_adam_8bit, paged_adamw_8bit
- Memory savings from quantized optimizer states



### 16. Kernels + CUDA + Triton
> Advanced — for maximum performance and understanding model internals

**HuggingFace Kernels Library**
- Concept — optimized compute kernels hosted on Hub
- get_kernel() — load a kernel from Hub
- Using community kernels in model forward passes
- Writing and publishing your own kernels
- Kernel versioning and compatibility

**Key GPU Concepts to Understand**
- What a GPU kernel is — parallel function across thousands of threads
- Thread, block, grid hierarchy in CUDA
- Memory types — global, shared, registers, constant
- Memory coalescing — why access patterns matter
- Occupancy — how many threads run simultaneously
- Warp — 32 threads that execute together
- Bank conflicts in shared memory
- Roofline model — memory bound vs compute bound analysis

**Notable Kernels and Why They Matter**
- Flash Attention — memory-efficient attention, used in all modern LLMs
- Flash Attention 2 — even faster, better parallelism
- Paged Attention — vLLM's key innovation for serving
- Fused LayerNorm — combine operations, fewer memory reads
- Rotary Embeddings (RoPE) — fused kernel for LLaMA, Mistral
- Fused cross-entropy — training stability
- Triton matmul — faster than cuBLAS in specific shapes

**Triton — Python-Based GPU Programming**
- Why Triton — write GPU kernels in Python, not C++
- tl.program_id() — which block am I running as
- tl.arange() — indices within a block
- tl.load() and tl.store() — memory access with masks
- BLOCK_SIZE as tl.constexpr — compile-time constant
- @triton.jit decorator
- triton.autotune() — automatic config search
- triton.testing.do_bench() — benchmark your kernel
- Fusion — combining multiple ops into one kernel
- What to build as practice
  - Vector addition kernel — hello world equivalent
  - Softmax kernel
  - Matrix multiplication kernel
  - Flash Attention from scratch

**Raw CUDA Path**
- CUDA C++ basics — __global__, __device__, __host__
- Launching kernels — grid and block dimensions
- cudaMalloc, cudaMemcpy, cudaFree
- Shared memory usage — __shared__
- Synchronization — __syncthreads()
- Atomic operations
- cuBLAS — highly optimized matrix ops
- cuDNN — deep learning primitives
- Writing PyTorch C++ extensions
- Binding CUDA to Python with pybind11

**Learning Resources for This Path**
- CUDA by Example — free, start here
- Triton tutorials on triton-lang.org
- Flash Attention paper — Dao et al 2022
- CUDA Programming Guide — NVIDIA official docs
- PyTorch custom ops tutorial



## Full Connection Map

```
FOUNDATION
  huggingface_hub ─────────────────────────────────────────────┐
  datasets ────────────────────────────────────────────────────┤
                                                               │
CORE ML                                                        │
  transformers                                                 │
    ├── pipeline API — all tasks                               │
    ├── AutoTokenizer                                          │
    ├── AutoModel + all task-specific variants                 │
    ├── Trainer + TrainingArguments                            │
    └── generate() + GenerationConfig                          │
                                                               ▼
  tokenizers ── how text becomes numbers                      Hub
                                                               │
  evaluate ──── measure model quality                          │
                                                               │
TRAINING                                                       │
  accelerate ──── multi-GPU, mixed precision, DeepSpeed        │
  peft ────────── LoRA, QLoRA, adapters                        │
  trl ─────────── SFT, DPO, GRPO, reward models               │
  safetensors ─── save and load weights safely                 │
                                                               │
SPECIALIZED                                                    │
  sentence-transformers ── embeddings, search, RAG             │
  diffusers ─────────────── image, video generation            │
  timm ──────────────────── vision models and features         │
  gradio ────────────────── demo UI and spaces ────────────────┤
  smolagents ────────────── agents with tools                  │
                                                               │
LOW LEVEL                                                      │
  bitsandbytes ── quantization for big models                  │
  kernels ──────── optimized GPU ops from Hub                  │
  triton ─────────── custom GPU kernels in Python              │
  CUDA ───────────── raw GPU programming in C++                │
```



## Week by Week Schedule

```
Weeks 01-02   huggingface_hub                     DONE
Weeks 03-04   datasets — all operations            DONE
Weeks 05-06   transformers — pipeline API
Weeks 07-08   transformers — AutoModel, Trainer
Week  09      tokenizers
Week  10      evaluate + first full fine-tune
Week  11      accelerate
Weeks 12-13   peft + LoRA + QLoRA
Week  14      trl — SFTTrainer
Week  15      trl — DPO and beyond
Week  16      safetensors
Week  17      pick one: sentence-transformers or diffusers or timm
Week  18      gradio deep dive
Week  19      smolagents
Week  20      bitsandbytes
Weeks 21+     kernels, triton, CUDA
```

## GitHub Repo Structure

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
│   ├── timm/
│   ├── gradio/
│   └── smolagents/
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


*Covers: huggingface_hub, datasets, transformers, tokenizers, evaluate,*
*accelerate, peft, trl, safetensors, sentence-transformers, diffusers,*
*timm, gradio, smolagents, bitsandbytes, kernels, triton, CUDA*