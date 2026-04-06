# Phase 1 — Transformers
# Part 1: Pipeline API

---

## What is Pipeline

Pipeline is the highest-level API in the transformers library.
It wraps everything — model loading, tokenization, inference, post-processing —
into a single function call. You give it text (or image or audio), it gives you
a result. No manual tokenization, no logits, no decoding. Just input → output.

Under the hood every pipeline does:
```
input → tokenizer/processor → model → post-processing → readable output
```

---

## Installation

```bash
pip install transformers torch
pip install transformers[torch]      # installs torch automatically
pip install transformers[tf-cpu]     # for TensorFlow
pip install transformers[flax]       # for JAX/Flax
```

---

## Basic Usage Pattern

```python
from transformers import pipeline

# create a pipeline for a task
pipe = pipeline(task="text-classification")

# run inference
result = pipe("HuggingFace is amazing!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Specifying a model

```python
# without specifying model → uses a default model for that task
pipe = pipeline("text-classification")

# with specific model from Hub
pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# with local model path
pipe = pipeline("text-classification", model="./my-local-model")
```

### Device control

```python
# CPU (default)
pipe = pipeline("text-classification", device=-1)

# GPU 0
pipe = pipeline("text-classification", device=0)

# auto select best device
pipe = pipeline("text-classification", device_map="auto")

# specific dtype for memory saving
import torch
pipe = pipeline("text-generation", model="gpt2", torch_dtype=torch.float16)
```

### Batch processing

```python
# single input
result = pipe("one sentence")

# list of inputs → batch processing
results = pipe(["sentence one", "sentence two", "sentence three"])

# batch_size for larger datasets
results = pipe(["text1", "text2", "text3"], batch_size=8)
```

---

## TEXT TASKS

---

### text-classification

Classify text into categories. Used for sentiment analysis, topic classification,
intent detection, spam detection, toxicity detection.

```python
pipe = pipeline("text-classification")
pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# single
result = pipe("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# batch
results = pipe(["I love this!", "This is terrible."])
# [{'label': 'POSITIVE', 'score': 0.9998}, {'label': 'NEGATIVE', 'score': 0.9994}]

# return all scores not just top
result = pipe("I love this!", top_k=None)
# [{'label': 'POSITIVE', 'score': 0.9998}, {'label': 'NEGATIVE', 'score': 0.0002}]
```

**Common models:**
- distilbert-base-uncased-finetuned-sst-2-english — sentiment (positive/negative)
- cardiffnlp/twitter-roberta-base-sentiment — 3-class sentiment
- ProsusAI/finbert — financial sentiment
- martin-ha/toxic-comment-model — toxicity detection
- facebook/bart-large-mnli — via zero-shot, many categories

---

### token-classification

Label each token (word or subword) individually. Used for Named Entity Recognition,
Part-of-Speech tagging, chunking, and information extraction.

```python
pipe = pipeline("token-classification")
pipe = pipeline("token-classification", model="dslim/bert-base-NER")

result = pipe("My name is Sarah and I live in London.")
# [
#   {'entity': 'B-PER', 'score': 0.998, 'word': 'Sarah', 'start': 11, 'end': 16},
#   {'entity': 'B-LOC', 'score': 0.999, 'word': 'London', 'start': 32, 'end': 38}
# ]

# aggregate subword tokens into full words/entities
result = pipe("My name is Sarah.", aggregation_strategy="simple")
result = pipe("My name is Sarah.", aggregation_strategy="first")
result = pipe("My name is Sarah.", aggregation_strategy="average")
result = pipe("My name is Sarah.", aggregation_strategy="max")
# [{'entity_group': 'PER', 'score': 0.998, 'word': 'Sarah', 'start': 11, 'end': 16}]
```

**NER entity labels:**
- B-PER, I-PER — person names
- B-ORG, I-ORG — organizations
- B-LOC, I-LOC — locations
- B-MISC, I-MISC — miscellaneous
- B- prefix = beginning of entity, I- prefix = inside entity

**Common models:**
- dslim/bert-base-NER — English NER
- dbmdz/bert-large-cased-finetuned-conll03-english — CoNLL NER
- flair/ner-english-large — high accuracy NER
- Jean-Baptiste/roberta-large-ner-english — robust NER

---

### question-answering

Extract an answer span from a given context paragraph. The model reads the context
and finds the exact words that answer the question. Called extractive QA.

```python
pipe = pipeline("question-answering")
pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")

result = pipe(
    question="What is the capital of France?",
    context="France is a country in Western Europe. Its capital is Paris, "
            "which is also the largest city."
)
# {'score': 0.998, 'start': 54, 'end': 59, 'answer': 'Paris'}

# with more details
result = pipe(
    question="Who founded Apple?",
    context="Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
    top_k=3      # return top 3 possible answers
)

# handling no-answer cases
result = pipe(
    question="What color is the sky?",
    context="France is in Europe.",
    handle_impossible_answer=True   # returns empty string if no answer found
)
```

**What the output means:**
- score — confidence of the answer
- start / end — character positions in the context
- answer — extracted text span

**Common models:**
- deepset/roberta-base-squad2 — good general QA
- deepset/bert-large-uncased-whole-word-masking-squad2 — high accuracy
- deepset/minilm-uncased-squad2 — fast and small
- deepset/deberta-v3-base-squad2 — best accuracy

---

### summarization

Condense long text into a shorter version. Can be abstractive (generates new text)
or extractive (selects existing sentences).

```python
pipe = pipeline("summarization")
pipe = pipeline("summarization", model="facebook/bart-large-cnn")

long_text = """
The Amazon rainforest, often referred to as the 'lungs of the Earth', is the world's 
largest tropical rainforest, covering over 5.5 million square kilometres. It represents 
over half of the planet's remaining rainforests, and comprises the largest and most 
biodiverse tract of tropical rainforest in the world, with an estimated 390 billion 
individual trees divided into 16,000 species.
"""

result = pipe(long_text)
# [{'summary_text': 'The Amazon rainforest is the world largest tropical rainforest...'}]

# control length
result = pipe(
    long_text,
    max_length=100,    # max tokens in summary
    min_length=30,     # min tokens in summary
    do_sample=False    # deterministic output
)

# batch
results = pipe([text1, text2, text3], batch_size=4)
```

**Common models:**
- facebook/bart-large-cnn — best general summarization
- google/pegasus-xsum — extreme summarization, very short output
- sshleifer/distilbart-cnn-12-6 — faster, smaller BART
- philschmid/bart-large-cnn-samsum — dialogue summarization
- google/flan-t5-base — instruction-following summarization

---

### translation

Translate text between languages.

```python
# specific language pair pipelines
pipe = pipeline("translation_en_to_fr")   # English to French
pipe = pipeline("translation_en_to_de")   # English to German
pipe = pipeline("translation_en_to_ro")   # English to Romanian

result = pipe("Hello, how are you today?")
# [{'translation_text': 'Bonjour, comment allez-vous aujourd'hui?'}]

# using Helsinki-NLP MarianMT models (400+ language pairs)
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")   # English to Hindi
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")   # Hindi to English
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")   # English to Chinese

result = pipe("Machine learning is transforming the world.")

# using NLLB (No Language Left Behind) — 200 languages
pipe = pipeline("translation", model="facebook/nllb-200-distilled-600M")
result = pipe(
    "Hello world",
    src_lang="eng_Latn",
    tgt_lang="hin_Deva"   # Hindi
)

# length control
result = pipe("text", max_length=200)
```

**Language code formats:**
- Helsinki-NLP models use: en, fr, de, es, zh, hi, ar, ru, etc.
- NLLB uses: eng_Latn, fra_Latn, deu_Latn, hin_Deva, zho_Hans, ara_Arab

**Common models:**
- Helsinki-NLP/opus-mt-{src}-{tgt} — fast, specific pairs
- facebook/nllb-200-distilled-600M — 200 languages
- facebook/nllb-200-1.3B — higher quality 200 languages
- google/flan-t5-large — instruction-based translation

---

### text-generation

Generate text continuation from a given prompt. This is open-ended generation —
the model continues whatever text you give it.

```python
pipe = pipeline("text-generation")
pipe = pipeline("text-generation", model="gpt2")

result = pipe("Once upon a time in a land far away,")
# [{'generated_text': 'Once upon a time in a land far away, there lived a...'}]

# control generation
result = pipe(
    "The future of AI is",
    max_new_tokens=100,      # how many new tokens to generate
    num_return_sequences=3,  # generate 3 different completions
    do_sample=True,          # enable sampling (creative output)
    temperature=0.8,         # higher = more creative, lower = more focused
    top_k=50,                # sample from top 50 tokens only
    top_p=0.95,              # nucleus sampling
    repetition_penalty=1.2,  # penalize repeating words
)

# do not include prompt in output
result = pipe("Prompt text", return_full_text=False)

# batch
results = pipe(["Prompt 1", "Prompt 2"])
```

**Common models:**
- gpt2 — classic, small, good for learning
- gpt2-medium, gpt2-large, gpt2-xl — larger GPT-2 variants
- distilgpt2 — smaller, faster GPT-2
- microsoft/phi-2 — 2.7B, very good quality
- Qwen/Qwen2.5-1.5B — Alibaba, efficient small model
- mistralai/Mistral-7B-v0.1 — needs GPU for 7B
- meta-llama/Llama-3.2-1B — Meta small model

---

### text2text-generation

Input text, output text. Used for T5-style models that frame everything as
a text-to-text problem — summarization, translation, QA, grammar correction all
in one model.

```python
pipe = pipeline("text2text-generation", model="google/flan-t5-base")

# QA style
result = pipe("question: What is the capital of France? context: France is in Europe. Paris is its capital.")

# summarize style
result = pipe("summarize: The Amazon rainforest is the largest tropical forest...")

# translation style
result = pipe("translate English to French: Hello, how are you?")

# grammar correction
result = pipe("Fix grammar: She go to school everyday.")

# control output length
result = pipe("summarize: long text...", max_new_tokens=50, min_new_tokens=10)
```

**Common models:**
- google/flan-t5-small, base, large, xl, xxl — instruction-following T5
- t5-small, t5-base, t5-large — base T5 models
- facebook/bart-large — BART for text2text tasks

---

### fill-mask

Predict what word should fill a masked position in a sentence. Used to understand
what a model has learned about language, and for probing model knowledge.

```python
pipe = pipeline("fill-mask")
pipe = pipeline("fill-mask", model="bert-base-uncased")

# mask token is [MASK] for BERT-style models
result = pipe("The capital of France is [MASK].")
# [
#   {'token_str': 'paris', 'score': 0.98, 'sequence': 'The capital of France is paris.'},
#   {'token_str': 'lyon', 'score': 0.003, 'sequence': 'The capital of France is lyon.'},
# ]

# for RoBERTa-style models mask token is <mask>
pipe = pipeline("fill-mask", model="roberta-base")
result = pipe("The capital of France is <mask>.")

# return top N predictions
result = pipe("Paris is the [MASK] of France.", top_k=5)

# targets — restrict predictions to specific tokens only
result = pipe("I feel [MASK] today.", targets=["happy", "sad", "tired"])
```

**Mask tokens by model family:**
- BERT family → [MASK]
- RoBERTa family → <mask>
- ALBERT → [MASK]
- DeBERTa → [MASK]

**Common models:**
- bert-base-uncased — English, uncased
- bert-base-cased — English, case-sensitive
- roberta-base, roberta-large — stronger than BERT
- distilbert-base-uncased — smaller BERT
- xlm-roberta-base — multilingual

---

### zero-shot-classification

Classify text into categories you define at inference time — no training required.
The model uses natural language understanding to decide which label fits best.

```python
pipe = pipeline("zero-shot-classification")
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

result = pipe(
    "I want to book a flight to Paris.",
    candidate_labels=["travel", "cooking", "sports", "technology"]
)
# {
#   'sequence': 'I want to book a flight to Paris.',
#   'labels': ['travel', 'technology', 'sports', 'cooking'],
#   'scores': [0.97, 0.01, 0.01, 0.01]
# }

# multi-label — text can belong to multiple categories
result = pipe(
    "I love cooking Italian food and watching football.",
    candidate_labels=["food", "sports", "travel", "music"],
    multi_label=True
)

# hypothesis template — customize how labels are framed
result = pipe(
    "This movie was boring.",
    candidate_labels=["positive review", "negative review"],
    hypothesis_template="This is a {}."
)

# batch
results = pipe(["text1", "text2"], candidate_labels=["label1", "label2"])
```

**Common models:**
- facebook/bart-large-mnli — best general zero-shot
- cross-encoder/nli-deberta-v3-small — faster, still good
- typeform/distilbart-mnli-12-3 — even faster
- MoritzLaurer/deberta-v3-large-zeroshot-v2 — high accuracy

---

### conversational (deprecated in newer versions — use text-generation with chat template)

```python
# old way — still works but deprecated
from transformers import pipeline, Conversation

pipe = pipeline("conversational", model="microsoft/DialoGPT-medium")

conversation = Conversation("Hello, how are you?")
conversation = pipe(conversation)
print(conversation.generated_responses[-1])

# add more turns
conversation.add_user_input("What can you do?")
conversation = pipe(conversation)

# NEW way — use text-generation + chat template (2024+)
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
]

result = pipe(messages, max_new_tokens=100)
print(result[0]["generated_text"][-1]["content"])
```

---

## VISION TASKS

---

### image-classification

Classify what is in an image.

```python
pipe = pipeline("image-classification")
pipe = pipeline("image-classification", model="google/vit-base-patch16-224")

# from URL
result = pipe("https://example.com/cat.jpg")

# from local file path
result = pipe("./cat.jpg")

# from PIL Image
from PIL import Image
img = Image.open("cat.jpg")
result = pipe(img)

# [{'label': 'tabby cat', 'score': 0.92}, {'label': 'tiger cat', 'score': 0.05}]

# return top k results
result = pipe("cat.jpg", top_k=5)

# batch of images
results = pipe(["cat.jpg", "dog.jpg", "bird.jpg"])
```

**Common models:**
- google/vit-base-patch16-224 — Vision Transformer, 1000 ImageNet classes
- microsoft/resnet-50 — classic ResNet
- facebook/convnext-base-224 — modern CNN
- apple/mobilevit-small — mobile-friendly
- openai/clip-vit-base-patch32 — CLIP vision encoder

---

### object-detection

Find and locate objects in an image. Returns bounding boxes and labels.

```python
pipe = pipeline("object-detection")
pipe = pipeline("object-detection", model="facebook/detr-resnet-50")

result = pipe("street.jpg")
# [
#   {'label': 'car',    'score': 0.99, 'box': {'xmin': 10, 'ymin': 50, 'xmax': 200, 'ymax': 150}},
#   {'label': 'person', 'score': 0.98, 'box': {'xmin': 220, 'ymin': 30, 'xmax': 280, 'ymax': 200}},
# ]

# filter by threshold
result = pipe("street.jpg", threshold=0.9)

# batch
results = pipe(["image1.jpg", "image2.jpg"])
```

**Common models:**
- facebook/detr-resnet-50 — Detection Transformer
- facebook/detr-resnet-101 — larger DETR
- hustvl/yolos-tiny — YOLO with Transformers, very fast
- microsoft/table-transformer-detection — table detection in documents

---

### image-segmentation

Assign a label to every pixel in the image.

Three types of segmentation:
- Semantic — label every pixel by class (sky, road, person)
- Instance — detect separate instances of the same class
- Panoptic — combines semantic + instance

```python
pipe = pipeline("image-segmentation")
pipe = pipeline("image-segmentation", model="facebook/mask2former-swin-large-cityscapes-semantic")

result = pipe("street.jpg")
# [
#   {'label': 'road',   'score': 0.99, 'mask': <PIL.Image mask>},
#   {'label': 'sky',    'score': 0.97, 'mask': <PIL.Image mask>},
#   {'label': 'person', 'score': 0.95, 'mask': <PIL.Image mask>},
# ]

# subtask control
result = pipe("image.jpg", subtask="semantic")
result = pipe("image.jpg", subtask="instance")
result = pipe("image.jpg", subtask="panoptic")

# threshold
result = pipe("image.jpg", threshold=0.9, mask_threshold=0.5)
```

**Common models:**
- facebook/mask2former-swin-large-coco-panoptic
- facebook/mask2former-swin-large-cityscapes-semantic
- nvidia/segformer-b5-finetuned-ade-640-640
- mattmdjaga/segformer-b2-clothes — clothing segmentation

---

### zero-shot-image-classification

Classify images into categories defined at inference time.
Uses CLIP-style vision-language models.

```python
pipe = pipeline("zero-shot-image-classification")
pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

result = pipe(
    "cat.jpg",
    candidate_labels=["cat", "dog", "bird", "car"]
)
# [{'label': 'cat', 'score': 0.97}, {'label': 'dog', 'score': 0.02}]

# batch
results = pipe(["img1.jpg", "img2.jpg"], candidate_labels=["cat", "dog"])
```

**Common models:**
- openai/clip-vit-base-patch32
- openai/clip-vit-large-patch14
- laion/CLIP-ViT-H-14-laion2B-s32B-b79K — stronger CLIP

---

### zero-shot-object-detection

Detect objects by describing them in text — no predefined class list.

```python
pipe = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")

result = pipe(
    "office.jpg",
    candidate_labels=["laptop", "coffee cup", "keyboard", "phone"]
)
# [
#   {'label': 'laptop', 'score': 0.98, 'box': {'xmin':..., 'ymin':..., 'xmax':..., 'ymax':...}},
# ]
```

---

### depth-estimation

Estimate how far each pixel is from the camera. Produces a depth map.

```python
pipe = pipeline("depth-estimation")
pipe = pipeline("depth-estimation", model="Intel/dpt-large")

result = pipe("room.jpg")
# {'predicted_depth': tensor([[...values...]]), 'depth': <PIL.Image depth map>}

depth_image = result["depth"]   # PIL Image of the depth map
depth_tensor = result["predicted_depth"]  # raw depth values as tensor
```

**Common models:**
- Intel/dpt-large — DPT depth estimation
- Intel/dpt-hybrid-midas — hybrid approach
- LiheYoung/depth-anything-large-hf — best 2024 model

---

### image-to-text (captioning)

Generate a text description of an image.

```python
pipe = pipeline("image-to-text")
pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

result = pipe("cat.jpg")
# [{'generated_text': 'a cat sitting on a chair next to a window'}]

# control caption length
result = pipe("cat.jpg", max_new_tokens=50)

# multiple captions
result = pipe("cat.jpg", num_return_sequences=3, do_sample=True)

# batch
results = pipe(["img1.jpg", "img2.jpg"])
```

**Common models:**
- Salesforce/blip-image-captioning-base — BLIP captioning
- Salesforce/blip-image-captioning-large — larger BLIP
- Salesforce/blip2-opt-2.7b — BLIP-2 with OPT
- nlpconnect/vit-gpt2-image-captioning — ViT + GPT-2

---

### visual-question-answering

Answer a question about an image.

```python
pipe = pipeline("visual-question-answering")
pipe = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

result = pipe(
    image="cat.jpg",
    question="What color is the cat?"
)
# [{'answer': 'orange', 'score': 0.95}]

result = pipe(
    image="chart.jpg",
    question="What is the highest value in the bar chart?"
)

# top k answers
result = pipe(image="cat.jpg", question="What is this?", top_k=3)
```

**Common models:**
- Salesforce/blip-vqa-base — BLIP VQA
- Salesforce/blip-vqa-capfilt-large — larger BLIP
- dandelin/vilt-b32-finetuned-vqa — ViLT for VQA

---

## AUDIO TASKS

---

### automatic-speech-recognition

Transcribe spoken audio to text.

```python
pipe = pipeline("automatic-speech-recognition")
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

# from local audio file
result = pipe("audio.mp3")
result = pipe("audio.wav")
result = pipe("audio.flac")
# {'text': 'Hello, my name is John and I am a software engineer.'}

# from URL
result = pipe("https://example.com/speech.mp3")

# with timestamps — know when each word was spoken
result = pipe("audio.mp3", return_timestamps=True)
# {'text': '...', 'chunks': [{'text': 'Hello', 'timestamp': (0.0, 0.5)}, ...]}

# word-level timestamps
result = pipe("audio.mp3", return_timestamps="word")

# language specification (Whisper)
result = pipe("audio.mp3", generate_kwargs={"language": "french"})
result = pipe("audio.mp3", generate_kwargs={"language": "hindi"})

# task: transcribe vs translate to English
result = pipe("audio.mp3", generate_kwargs={"task": "transcribe"})  # keep original language
result = pipe("audio.mp3", generate_kwargs={"task": "translate"})   # translate to English

# long audio — chunk and process
result = pipe(
    "long_audio.mp3",
    chunk_length_s=30,      # process 30 seconds at a time
    stride_length_s=5,      # overlap between chunks
    batch_size=8            # process multiple chunks in parallel
)
```

**Whisper model sizes:**
- openai/whisper-tiny — fastest, lowest accuracy
- openai/whisper-base — good balance for simple audio
- openai/whisper-small — better accuracy
- openai/whisper-medium — recommended general use
- openai/whisper-large-v3 — best accuracy, needs more memory

**Other ASR models:**
- facebook/wav2vec2-base-960h — English Wav2Vec2
- facebook/wav2vec2-large-xlsr-53 — multilingual

---

### audio-classification

Classify audio clips into categories. Used for music genre, emotion in speech,
sound event detection, language identification.

```python
pipe = pipeline("audio-classification")
pipe = pipeline("audio-classification", model="superb/wav2vec2-base-superb-ks")

result = pipe("audio.wav")
# [{'label': 'yes', 'score': 0.98}, {'label': 'no', 'score': 0.01}]

# top k results
result = pipe("audio.wav", top_k=5)

# batch
results = pipe(["audio1.wav", "audio2.wav"])
```

**Common models:**
- superb/wav2vec2-base-superb-ks — keyword spotting
- facebook/wav2vec2-base-superb-sid — speaker identification
- ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition — emotion
- MIT/ast-finetuned-audioset-10-10-0.4593 — general audio events

---

### text-to-speech

Convert text to spoken audio.

```python
pipe = pipeline("text-to-speech")
pipe = pipeline("text-to-speech", model="microsoft/speecht5_tts")

result = pipe("Hello, this is a test of text to speech synthesis.")
# {'audio': array([...waveform...]), 'sampling_rate': 16000}

import soundfile as sf
sf.write("output.wav", result["audio"], result["sampling_rate"])

# with speaker embeddings (SpeechT5)
import torch
from datasets import load_dataset

# load speaker embeddings dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

result = pipe(
    "Hello world",
    forward_params={"speaker_embeddings": speaker_embedding}
)
```

**Common models:**
- microsoft/speecht5_tts — SpeechT5
- facebook/mms-tts-eng — English TTS
- suno/bark — expressive speech, music, sound effects
- hexgrad/Kokoro-82M — small, fast, high quality

---

## MULTIMODAL TASKS

---

### document-question-answering

Answer questions about documents — PDFs, scanned documents, forms, receipts.
Understands both text and layout.

```python
pipe = pipeline("document-question-answering")
pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

result = pipe(
    image="invoice.png",          # document image
    question="What is the total amount?"
)
# [{'score': 0.99, 'answer': '$1,234.56', 'start': 38, 'end': 39}]

result = pipe(
    image="form.png",
    question="What is the date of birth?"
)
```

**Common models:**
- impira/layoutlm-document-qa — general document QA
- microsoft/layoutlmv3-base — layout-aware understanding
- naver-clova-ix/donut-base-finetuned-docvqa — Donut model

---

### feature-extraction

Extract raw embeddings (vector representations) from any model.
Used for building search systems, clustering, similarity comparisons.

```python
pipe = pipeline("feature-extraction")
pipe = pipeline("feature-extraction", model="bert-base-uncased")

result = pipe("Hello world")
# list of shape [1, seq_len, hidden_size] — e.g., [1, 4, 768] for BERT

import numpy as np
embeddings = np.array(result)
print(embeddings.shape)   # (1, 4, 768)

# mean pooling to get sentence embedding
sentence_embedding = embeddings.mean(axis=1)  # (1, 768)

# batch
result = pipe(["sentence one", "sentence two"])

# return tensors directly
result = pipe("Hello", return_tensors=True)
```

---

## Pipeline Advanced Features

---

### Saving and Loading Pipelines

```python
# save the whole pipeline (model + tokenizer + config)
pipe.save_pretrained("./my-pipeline")

# load it back
from transformers import pipeline
pipe = pipeline("text-classification", model="./my-pipeline")
```

---

### Using Pipelines as Datasets Iterator

```python
from transformers import pipeline
from datasets import load_dataset

pipe = pipeline("text-classification", device=0)
dataset = load_dataset("imdb", split="test")

# efficient iteration over large datasets
for result in pipe(dataset["text"], batch_size=32):
    print(result)
```

---

### Pipeline with Custom Tokenizer and Model

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)
```

---

### Handling Long Texts

```python
# many pipelines truncate by default
# for text-classification, truncation is automatic

# for summarization — split manually if needed
def chunk_text(text, max_tokens=1024):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i+max_tokens]))
    return chunks

pipe = pipeline("summarization", model="facebook/bart-large-cnn")
chunks = chunk_text(very_long_text)
summaries = pipe(chunks)
final_summary = " ".join([s["summary_text"] for s in summaries])
```

---

## Quick Reference — All Tasks

| Task | Input | Output |
|---|---|---|
| text-classification | text | label + score |
| token-classification | text | label per token |
| question-answering | question + context | answer span |
| summarization | long text | short text |
| translation | text | translated text |
| text-generation | prompt | continued text |
| text2text-generation | instruction text | output text |
| fill-mask | text with [MASK] | predicted words |
| zero-shot-classification | text + labels | label + scores |
| conversational | conversation | response |
| image-classification | image | label + score |
| object-detection | image | boxes + labels |
| image-segmentation | image | masks + labels |
| zero-shot-image-classification | image + labels | label + score |
| zero-shot-object-detection | image + labels | boxes + labels |
| depth-estimation | image | depth map |
| image-to-text | image | caption |
| visual-question-answering | image + question | answer |
| automatic-speech-recognition | audio | text |
| audio-classification | audio | label + score |
| text-to-speech | text | audio waveform |
| document-question-answering | document image + question | answer |
| feature-extraction | text | embeddings |

---

## What is Next

Part 2 covers AutoTokenizer — understanding how text becomes numbers,
all encoding options, padding, truncation, special tokens, and decoding.