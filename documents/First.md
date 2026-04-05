# **HuggingFace Learning Roadmap ⚡**

### From Zero to Production — Python ML Engineer Path

## What We Are Building Toward

```
datasets → transformers → fine-tune → optimize → deploy → agents
```

Every library in this roadmap feeds the next one.
You are not learning tools in isolation — you are building one complete pipeline.

## **PHASE 0 — Platform Foundation**

### 1. HuggingFace Hub + huggingface_hub

**What it is:** The GitHub of ML. Central place for models, datasets, and Spaces.

**What we covered:**
- All 5 login methods: `login()`, `notebook_login()`, `interpreter_login()`, env var, CLI
- `hf auth login` CLI
- Downloading files: `hf_hub_download()`, `snapshot_download()`
- Uploading files: `upload_file()`, `upload_folder()`
- Creating and managing repos: `create_repo()`, `delete_repo()`
- Gated model access and `GatedRepoError` fix
- `CommitOperationAdd` for atomic multi-file commits
- Space management: `get_space_runtime()`, `pause_space()`, `restart_space()`

**Key install:**
```bash
pip install huggingface_hub
```

**Key imports:**
```python
from huggingface_hub import (
    login, logout, whoami,
    hf_hub_download, snapshot_download,
    upload_file, upload_folder,
    create_repo, delete_repo,
    HfApi, CommitOperationAdd
)
```



### 2. Datasets

**What it is:** The data layer. Load, process, and share datasets for any ML task.

**What we covered:**

**Loading — all sources:**
```python
load_dataset("imdb")                          # from Hub
load_dataset("csv", data_files="data.csv")    # local CSV
load_dataset("json", data_files="data.jsonl") # local JSONL
Dataset.from_dict({...})                      # from Python dict
Dataset.from_pandas(df)                       # from Pandas
Dataset.from_list([...])                      # from list of dicts
load_from_disk("./saved")                     # from disk
```

**Row operations:**
```python
ds.sort("label")
ds.shuffle(seed=42)
ds.select([0, 1, 2])
ds.filter(lambda x: x["label"] == 1)
ds.train_test_split(test_size=0.2)
ds.shard(num_shards=4, index=0)
```

**Column operations:**
```python
ds.rename_column("text", "review")
ds.remove_columns(["label"])
ds.add_column("length", values)
ds.cast_column("label", Value("float32"))
ds.flatten()
ds.select_columns(["text", "label"])
```

**Transformation:**
```python
ds.map(fn)                           # row by row
ds.map(fn, batched=True)             # batch processing
ds.map(fn, num_proc=4)               # multiprocessing
ds.map(fn, remove_columns=["text"])  # map and drop
```

**Combining:**
```python
concatenate_datasets([ds1, ds2])
interleave_datasets([ds1, ds2], probabilities=[0.7, 0.3])
```

**Export and format:**
```python
ds.to_pandas()
ds.to_dict()
ds.to_list()
ds.to_csv("out.csv")
ds.to_json("out.jsonl")
ds.to_parquet("out.parquet")
ds.set_format("torch", columns=["input_ids", "label"])
ds.set_format("numpy")
ds.reset_format()
```

**Save and push:**
```python
ds.save_to_disk("./my_dataset")
load_from_disk("./my_dataset")
ds.push_to_hub("username/my-dataset")
ds.push_to_hub("username/my-dataset", private=True)
```

**Streaming (large datasets):**
```python
ds = load_dataset("c4", "en", split="train", streaming=True)
for example in ds:
    print(example["text"])
list(ds.take(10))
```

**Python dict deep dive:**
```python
ds[0]               # first row as dict
ds[-1]              # last row
ds[:5]              # slice of rows
ds["text"]          # full column as list
ds.features.keys()  # column names
ds.features.items() # name + type pairs
ds.to_dict()        # full dict
ds.to_list()        # list of row dicts
"text" in ds.features  # check column exists
Counter(ds["label"])   # count values
[row for row in ds if row["label"] == 1]  # comprehension
```

**Key install:**
```bash
pip install datasets
```



## PHASE 1 — Core ML
> Status: NEXT

### 3. Transformers

**What it is:** The most important library in the ecosystem. Access to 1M+ pretrained models
for text, vision, audio, and multimodal tasks.

**Learning path inside Transformers:**

**Step A — Pipeline API (start here, 2 lines of code for anything):**
```python
from transformers import pipeline

# text
classifier  = pipeline("text-classification")
generator   = pipeline("text-generation", model="gpt2")
summarizer  = pipeline("summarization")
translator  = pipeline("translation_en_to_fr")
qa          = pipeline("question-answering")
ner         = pipeline("ner")

# vision
img_class   = pipeline("image-classification")
obj_detect  = pipeline("object-detection")

# audio
asr         = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

# run
result = classifier("HuggingFace is amazing!")
```

**Step B — AutoTokenizer + AutoModel (understand what pipeline does internally):**
```python
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

inputs  = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

**Step C — Task-specific model classes:**
```python
AutoModelForSequenceClassification   # text classification
AutoModelForTokenClassification      # NER, POS tagging
AutoModelForQuestionAnswering        # QA
AutoModelForSeq2SeqLM                # summarization, translation
AutoModelForCausalLM                 # text generation (GPT-style)
AutoModelForMaskedLM                 # fill-mask (BERT-style)
AutoModelForImageClassification      # image tasks
AutoModelForSpeechSeq2Seq            # ASR (Whisper)
```

**Step D — Trainer (fine-tuning with built-in training loop):**
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                   # mixed precision
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./my-model")
model.push_to_hub("username/my-model")
```

**What to practice:**
- Sentiment analysis on IMDB with BERT
- Text generation with GPT-2
- Named entity recognition
- Question answering with BERT
- Summarization with T5
- Speech recognition with Whisper
- Image classification with ViT

**Key install:**
```bash
pip install transformers torch
pip install transformers[torch]   # all torch deps
```



### 4. Tokenizers

**What it is:** The fast tokenization library that powers all AutoTokenizer classes.
Understanding this deeply makes you understand how models read text.

**What to learn:**
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# how tokens are created
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# encoding
enc = tokenizer("Hello world how are you?",
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt")

print(enc.input_ids)         # token IDs
print(enc.attention_mask)    # 1 for real tokens, 0 for padding
print(enc.token_type_ids)    # segment IDs for sentence pairs

# decoding
tokenizer.decode(enc.input_ids[0])

# special tokens
tokenizer.cls_token     # [CLS]
tokenizer.sep_token     # [SEP]
tokenizer.pad_token     # [PAD]
tokenizer.unk_token     # [UNK]
tokenizer.mask_token    # [MASK]

# batch encoding
tokenizer(["sentence one", "sentence two"], padding=True, truncation=True)

# train your own tokenizer from scratch
trainer = BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[CLS]", "[SEP]"])
```

**Key concepts to understand:**
- BPE (Byte-Pair Encoding) — used by GPT models
- WordPiece — used by BERT
- SentencePiece / Unigram — used by T5, LLaMA
- Why padding and truncation exist
- What attention_mask does
- Subword tokenization

**Key install:**
```bash
pip install tokenizers
# usually installed automatically with transformers
```



### 5. Evaluate

**What it is:** Standardized metrics for measuring model performance.

**What to learn:**
```python
import evaluate

# load metrics
accuracy  = evaluate.load("accuracy")
f1        = evaluate.load("f1")
rouge     = evaluate.load("rouge")
bleu      = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# compute
result = accuracy.compute(
    predictions=[0, 1, 1, 0],
    references= [0, 1, 0, 1]
)
print(result)  # {"accuracy": 0.75}

# use inside Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return accuracy.compute(predictions=predictions, references=labels)

# combine multiple metrics
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
clf_metrics.compute(predictions=[...], references=[...])

# list available metrics
evaluate.list_evaluation_modules()
```

**Key install:**
```bash
pip install evaluate
```



## PHASE 2 — Training & Fine-tuning
> Learn after Phase 1

### 6. Accelerate

**What it is:** Run the exact same PyTorch training code on single GPU, multi-GPU, TPU, or
CPU with zero code changes. Powers the Trainer inside Transformers.

**What to learn:**
```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="fp16",    # or "bf16" for newer GPUs
    gradient_accumulation_steps=4
)

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

# save properly across distributed setups
accelerator.save_state("./checkpoint")
```

**CLI tools:**
```bash
# configure your machine once
accelerate config

# launch distributed training
accelerate launch train.py

# launch with specific settings
accelerate launch --num_processes=4 --mixed_precision=fp16 train.py
```

**Key install:**
```bash
pip install accelerate
```



### 7. PEFT — Parameter Efficient Fine-Tuning

**What it is:** Fine-tune massive models (7B, 13B, 70B parameters) on a single GPU by
only training a tiny fraction of the weights. LoRA is the most used technique in 2025.

**What to learn:**

**LoRA — Low Rank Adaptation (most important):**
```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,                        # rank — lower = fewer params trained
    lora_alpha=32,               # scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # which layers to apply LoRA to
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 1,180,672 || all params: 124,441,346 || trainable%: 0.95%
```

**QLoRA — quantized LoRA (fine-tune 7B on 8GB GPU):**
```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

**Save and load LoRA adapters:**
```python
model.save_pretrained("./lora-adapter")
model.push_to_hub("username/my-lora-adapter")

# load adapter on top of base model later
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "username/my-lora-adapter")
merged = model.merge_and_unload()  # merge adapter into base model weights
```

**Other PEFT methods (know they exist):**
- Prefix Tuning
- Prompt Tuning
- IA3
- AdaLoRA

**Key install:**
```bash
pip install peft
```



### 8. TRL — Transformer Reinforcement Learning

**What it is:** Fine-tune LLMs with modern alignment techniques.
SFTTrainer is the simplest way to fine-tune any LLM on custom data.

**What to learn:**

**SFTTrainer — Supervised Fine-Tuning (most common):**
```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./sft-model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        max_seq_length=512,
    ),
    train_dataset=dataset,
    peft_config=lora_config,          # optional LoRA
    formatting_func=formatting_fn,    # format your data as prompt+response
)

trainer.train()
trainer.save_model("./my-sft-model")
```

**DPO — Direct Preference Optimization:**
```python
from trl import DPOTrainer, DPOConfig

# dataset must have: prompt, chosen, rejected columns
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(output_dir="./dpo-model", beta=0.1),
    train_dataset=dataset,
)
trainer.train()
```

**RewardTrainer:**
```python
from trl import RewardTrainer, RewardConfig
# train a reward model from human preference data
```

**GRPOTrainer — Group Relative Policy Optimization (latest, 2025):**
```python
from trl import GRPOTrainer
# used by DeepSeek-R1 style reasoning models
```

**Key install:**
```bash
pip install trl
```



### 9. Safetensors

**What it is:** The new standard for storing neural network weights. Safe, fast, no arbitrary
code execution risk (unlike pickle-based .bin files). All modern models use this.

**What to learn:**
```python
from safetensors.torch import save_file, load_file
import torch

# save model weights
tensors = {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}
save_file(tensors, "model.safetensors")

# load model weights
tensors = load_file("model.safetensors")

# with transformers — automatic, just use .safetensors extension
model.save_pretrained("./model")        # saves model.safetensors
model = AutoModel.from_pretrained("./model")  # loads it automatically

# inspect without loading (zero-copy, instant)
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    print(f.keys())                     # see all tensor names
    weight = f.get_tensor("weight")     # load specific tensor only
```

**Key install:**
```bash
pip install safetensors
```



## PHASE 3 — Specialized Libraries
> Pick based on your goal. Not all required.

### 10. Sentence Transformers
> If you are building: RAG, semantic search, embeddings, recommendation

**What it is:** The go-to library for text embeddings and semantic similarity.
Sentence-BERT and all its variants live here.

**What to learn:**
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")  # fast + good quality

# encode text to embeddings
sentences = ["This is a cat", "This is a dog", "Paris is beautiful"]
embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)

# semantic similarity
sim = util.cos_sim(embeddings[0], embeddings[1])
print(sim)  # 0.85 — similar

# semantic search (find most similar to a query)
query = model.encode("What is a pet?")
scores = util.semantic_search(query, embeddings, top_k=2)

# fine-tune on your own pairs
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

examples = [
    InputExample(texts=["anchor", "positive"], label=1.0),
    InputExample(texts=["anchor", "negative"], label=0.0),
]
loader    = DataLoader(examples, batch_size=16)
loss      = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(loader, loss)], epochs=3)
model.save("./my-embedding-model")
model.push_to_hub("username/my-embedding-model")
```

**Key install:**
```bash
pip install sentence-transformers
```



### 11. Diffusers
> If you are building: image generation, video generation, audio generation

**What it is:** The Transformers equivalent for diffusion models.
Stable Diffusion, SDXL, Flux, ControlNet all live here.

**What to learn:**
```python
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch

# text to image
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image = pipe("a photo of an astronaut riding a horse").images[0]
image.save("output.png")

# latest models
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

# image to image
from diffusers import StableDiffusionImg2ImgPipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(...)
output = pipe(prompt="...", image=init_image, strength=0.75).images[0]

# inpainting
from diffusers import StableDiffusionInpaintPipeline

# ControlNet (guided generation)
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
```

**Key install:**
```bash
pip install diffusers
pip install diffusers[torch]
```



### 12. timm — PyTorch Image Models
> If you are building: image classification, object detection, feature extraction

**What it is:** 1000+ state-of-the-art pretrained vision models. ResNet, ViT, EfficientNet,
Swin Transformer, ConvNeXt all in one place.

**What to learn:**
```python
import timm

# list available models
timm.list_models("resnet*")
timm.list_models("vit*", pretrained=True)

# load pretrained model
model = timm.create_model("resnet50", pretrained=True)
model = timm.create_model("vit_base_patch16_224", pretrained=True)

# custom classifier (transfer learning)
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=10        # your number of classes
)

# feature extraction (no classifier head)
model = timm.create_model("resnet50", pretrained=True, num_classes=0)
features = model(image_tensor)  # (batch, 2048) embeddings

# correct preprocessing config for each model
data_config = timm.data.resolve_model_data_config(model)
transforms  = timm.data.create_transform(**data_config, is_training=False)
```

**Key install:**
```bash
pip install timm
```



### 13. Gradio (deep dive)
> You already used this for Spaces. Now learn it properly.

**What to learn:**

**Blocks API — full layout control:**
```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# My App")

    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Input")
            btn = gr.Button("Run")
        with gr.Column():
            out = gr.Textbox(label="Output")

    btn.click(fn=my_function, inputs=inp, outputs=out)

demo.launch(share=True)
```

**ChatInterface — build a chatbot in 5 lines:**
```python
def chat(message, history):
    return f"You said: {message}"

gr.ChatInterface(fn=chat).launch()
```

**Components to master:**
```python
gr.Textbox()         # text input/output
gr.Image()           # image upload/display
gr.Audio()           # audio record/play
gr.Video()           # video upload
gr.File()            # any file upload
gr.DataFrame()       # tables
gr.Plot()            # matplotlib/plotly charts
gr.Label()           # classification output with confidence
gr.Slider()          # numeric slider
gr.Dropdown()        # select menu
gr.Checkbox()        # boolean
gr.Radio()           # single choice
gr.CheckboxGroup()   # multi choice
gr.Gallery()         # image gallery
gr.HighlightedText() # NER/token labeling output
gr.AnnotatedImage()  # image with bounding boxes
gr.Chatbot()         # chat history display
```

**Key install:**
```bash
pip install gradio
```



### 14. smolagents
> If you are building: AI agents, tool-using models, multi-step reasoning

**What it is:** Lightweight agent framework from HuggingFace.
Models that can use tools, browse the web, write and run code.

**What to learn:**
```python
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, tool

# define a custom tool
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: 25C, sunny"

# create agent with tools
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), get_weather],
    model=HfApiModel("Qwen/Qwen2.5-72B-Instruct"),
)

# run it
result = agent.run("What is the weather in Paris and what are the top news stories today?")
print(result)
```

**Key install:**
```bash
pip install smolagents
pip install smolagents[all]   # all tool integrations
```



## PHASE 4 — Low Level / Advanced
> Learn when you hit GPU memory/speed limitations

### 15. Bitsandbytes
> Used with PEFT for QLoRA. Learn when you need to load big models on small GPU.

**What to learn:**
```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization (most memory efficient)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # NormalFloat4 — best quality
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # nested quantization, saves more memory
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 8-bit quantization (faster inference, less memory)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)
```

**Key install:**
```bash
pip install bitsandbytes
```



### 16. Kernels + CUDA Programming
> Advanced. Learn when you need maximum speed or want to understand model internals.

**What it is:** HuggingFace Kernels lets you load and run optimized compute kernels
(written in CUDA/Triton) directly from the Hub. This is where raw GPU performance lives.

**The learning path for CUDA + Kernels:**

**Step A — Understand what a kernel is:**
```
A CUDA kernel is a function that runs in parallel across thousands of GPU threads.
When you call model(input), internally it runs dozens of these kernels:
  - matrix multiplication kernel (the most called)
  - softmax kernel
  - layer norm kernel
  - attention kernel (Flash Attention is a famous example)
```

**Step B — Triton (Python-based GPU programming, learn before raw CUDA):**
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

**Step C — HuggingFace Kernels library:**
```python
from kernels import get_kernel

# load an optimized kernel from HuggingFace Hub
# e.g. FlashAttention, fast RoPE, fused LayerNorm
kernel = get_kernel("kernels-community/flash-attention")

# use directly in your model forward pass
output = kernel.flash_attention_forward(q, k, v, ...)
```

**Step D — Raw CUDA with PyTorch (deepest level):**
```python
import torch

# write inline CUDA code
cuda_code = """
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""

# or use torch.cuda extensions
from torch.utils.cpp_extension import load_inline
add_ext = load_inline(
    name="add_ext",
    cuda_sources=[cuda_code],
    functions=["add_kernel"],
)
```

**Notable optimized kernels to know about:**
- **Flash Attention** — faster, memory-efficient attention (used in all modern LLMs)
- **Fused LayerNorm** — combines normalization ops into one kernel
- **Rotary Embeddings (RoPE)** — position encoding used by LLaMA, Mistral
- **Paged Attention** — vLLM's memory trick for serving
- **Triton matmul** — faster matrix multiply than cuBLAS for certain shapes

**What to study for CUDA:**
```
1. CUDA by Example (book, free online)
2. Triton tutorials — https://triton-lang.org/main/getting-started/tutorials
3. HuggingFace Kernels docs — https://huggingface.co/docs/kernels
4. PyTorch CUDA extension guide
5. Flash Attention paper (Dao et al, 2022) — most important paper to read
```

**Key install:**
```bash
pip install kernels
pip install triton         # for writing custom GPU kernels in Python
```



## The Full Pipeline — How Everything Connects

```
                    YOUR ML PIPELINE
                    ================

  DATA                  MODEL                 TRAINING
  -                  --                 --
  datasets         →    transformers     →    accelerate
  (load, process,       (pipeline,            (multi-GPU,
   map, filter,          AutoModel,            mixed precision)
   push_to_hub)          Tokenizer,
                         Trainer)         →   peft
                                              (LoRA, QLoRA,
  STORAGE                                      train 1% of params)
  -
  safetensors      →                      →   trl
  (save weights)                              (SFTTrainer, DPO,
                                               RLHF, GRPO)

  MEASURE               DEPLOY                SPECIALIZE
  -                               -
  evaluate         →    huggingface_hub  →    sentence-transformers
  (accuracy, F1,        (push model,           (embeddings, RAG)
   ROUGE, BLEU)          create space)
                                          →   diffusers
                    →   gradio                (image generation)
                        (demo UI,
                         ChatInterface)   →   timm
                                              (vision models)

                                          →   smolagents
                                              (AI agents)

                    LOW LEVEL
                    
                    kernels + triton + cuda
                    (maximum performance)
```



## Recommended Learning Order — Week by Week

```
Week 1-2   huggingface_hub + datasets         DONE
Week 3-4   transformers pipeline + AutoModel
Week 5     tokenizers deep dive
Week 6     evaluate + first full fine-tune with Trainer
Week 7     accelerate + multi-GPU training
Week 8     peft + LoRA fine-tuning
Week 9     trl + SFTTrainer on custom data
Week 10    safetensors + model saving/sharing
Week 11    pick one: sentence-transformers OR diffusers OR timm
Week 12    gradio deep dive + deploy your model as a Space
Week 13+   smolagents / bitsandbytes / kernels / triton
```



## GitHub Repo Structure for All Learnings

```
huggingface-learning/
├── 01_hub/
│   ├── login_methods.py
│   ├── download_upload.py
│   └── spaces_deploy.py
├── 02_datasets/
│   ├── loading_all_sources.py
│   ├── row_column_ops.py
│   ├── map_filter_transform.py
│   ├── streaming.py
│   └── dict_methods.py
├── 03_transformers/
│   ├── pipeline_basics.py
│   ├── auto_model_tokenizer.py
│   ├── trainer_finetune.py
│   └── task_specific_models.py
├── 04_tokenizers/
│   └── tokenization_deep_dive.py
├── 05_evaluate/
│   └── metrics.py
├── 06_accelerate/
│   └── distributed_training.py
├── 07_peft/
│   ├── lora_finetune.py
│   └── qlora_big_model.py
├── 08_trl/
│   ├── sft_trainer.py
│   └── dpo_trainer.py
├── 09_safetensors/
│   └── save_load_weights.py
├── 10_specialized/
│   ├── sentence_transformers_rag.py
│   ├── diffusers_image_gen.py
│   ├── timm_vision.py
│   └── smolagents_tools.py
├── 11_advanced/
│   ├── bitsandbytes_quant.py
│   ├── triton_kernel.py
│   └── hf_kernels.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```



## Key Rule

**Code goes to GitHub. Model weights go to HuggingFace Hub.**

Never push `.bin`, `.safetensors`, or `.pt` files to GitHub.
Push them with `model.push_to_hub("username/my-model")` instead.



*Last updated: April 2026*
*Covers: huggingface_hub, datasets, transformers, tokenizers, evaluate,*
*accelerate, peft, trl, safetensors, sentence-transformers, diffusers,*
*timm, gradio, smolagents, bitsandbytes, kernels, triton, CUDA*