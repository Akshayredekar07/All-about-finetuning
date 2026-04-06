# Phase 1 — Transformers
# Part 3: AutoModel — All Task-Specific Variants

---

## What is AutoModel

AutoModel is the layer below pipeline. Pipeline hides everything.
AutoModel gives you direct access to the model's inputs and outputs.
You control the tokenization, the forward pass, and the post-processing yourself.

```
pipeline()    →   black box, just works, less control
AutoModel()   →   full control, raw logits, you do post-processing
```

When to use AutoModel instead of pipeline:
- You need the raw logits or hidden states
- You are training or fine-tuning
- You need custom inference logic
- You want to combine models
- Pipeline does not support your exact use case

---

## Installation

```bash
pip install transformers torch
```

---

## The Pattern — Always the Same

```python
from transformers import AutoTokenizer, AutoModelForXxx
import torch

# 1. load tokenizer
tokenizer = AutoTokenizer.from_pretrained("model-name")

# 2. load model
model = AutoModelForXxx.from_pretrained("model-name")

# 3. tokenize input
inputs = tokenizer("your text", return_tensors="pt")

# 4. run model
with torch.no_grad():      # no_grad for inference — saves memory
    outputs = model(**inputs)

# 5. process outputs
# outputs is a dataclass with .logits, .last_hidden_state, etc.
```

---

## AutoModel — Raw Embeddings

No task head. Returns raw hidden states from the transformer body.
Use when you want embeddings for downstream tasks.

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModel.from_pretrained("bert-base-uncased")

inputs  = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

# last_hidden_state — contextual embedding for every token
print(outputs.last_hidden_state.shape)
# torch.Size([1, seq_len, 768])   — (batch, tokens, hidden_size)

# pooler_output — embedding of [CLS] token, processed through a linear layer
print(outputs.pooler_output.shape)
# torch.Size([1, 768])

# mean pooling — one vector per sentence (most common for sentence embeddings)
attention_mask = inputs["attention_mask"]
token_embeddings = outputs.last_hidden_state
mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
sentence_embedding = torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
print(sentence_embedding.shape)
# torch.Size([1, 768])

# all hidden states from every layer
outputs = model(**inputs, output_hidden_states=True)
all_hidden = outputs.hidden_states   # tuple of (num_layers+1) tensors
# each tensor: (batch, seq_len, hidden_size)

# attention weights from every layer
outputs = model(**inputs, output_attentions=True)
all_attentions = outputs.attentions  # tuple of (num_layers) tensors
```

---

## TEXT — Natural Language Understanding

---

### AutoModelForSequenceClassification

Single label or multi-label text classification.
Use for: sentiment analysis, topic classification, intent detection, spam detection.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model     = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

inputs  = tokenizer("I love this product!", return_tensors="pt")
outputs = model(**inputs)

# logits — raw scores before softmax, shape: (batch, num_labels)
logits = outputs.logits
print(logits)           # tensor([[-3.9, 4.1]])

# convert to probabilities
probs = torch.softmax(logits, dim=-1)
print(probs)            # tensor([[0.0002, 0.9998]])

# get the predicted label
pred_id = logits.argmax(-1).item()
label   = model.config.id2label[pred_id]
print(label)            # "POSITIVE"

# all labels mapping
print(model.config.id2label)   # {0: 'NEGATIVE', 1: 'POSITIVE'}
print(model.config.label2id)   # {'NEGATIVE': 0, 'POSITIVE': 1}
print(model.config.num_labels) # 2

# multi-label classification
# logits shape: (batch, num_labels)
# use sigmoid instead of softmax for multi-label
probs = torch.sigmoid(logits)
```

---

### AutoModelForTokenClassification

Label each token individually. Use for NER, POS tagging, chunking.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model     = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

text   = "My name is Sarah and I live in London."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# logits — shape: (batch, seq_len, num_labels)
logits    = outputs.logits
print(logits.shape)   # torch.Size([1, 12, 9])   — 9 NER labels

# predicted label for each token
pred_ids = logits.argmax(-1)[0]   # first sample in batch
tokens   = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

for token, pred_id in zip(tokens, pred_ids):
    label = model.config.id2label[pred_id.item()]
    print(f"{token:15} → {label}")
# [CLS]           → O
# My              → O
# name            → O
# is              → O
# Sarah           → B-PER
# ...
# London          → B-LOC
# [SEP]           → O

# labels mapping
print(model.config.id2label)
# {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', ...}
```

---

### AutoModelForQuestionAnswering

Extract an answer span from a context. Returns start and end position logits.

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model     = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

question = "Where does Sarah live?"
context  = "My name is Sarah and I live in London, England."

# for QA, pass question and context as sentence pair
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# start_logits — score for each token being the answer start
# end_logits — score for each token being the answer end
start_logits = outputs.start_logits   # shape: (batch, seq_len)
end_logits   = outputs.end_logits     # shape: (batch, seq_len)

# get most likely start and end positions
start_pos = start_logits.argmax(-1).item()
end_pos   = end_logits.argmax(-1).item()

# decode the answer span
all_tokens = inputs["input_ids"][0]
answer_ids = all_tokens[start_pos: end_pos + 1]
answer     = tokenizer.decode(answer_ids, skip_special_tokens=True)
print(answer)  # "London, England"

# handle impossible answers (SQuAD 2.0 style)
# if start_pos == 0 (CLS token) → model says no answer exists
```

---

### AutoModelForMultipleChoice

Select the correct answer from multiple candidates.

```python
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch

tokenizer = AutoTokenizer.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")
model     = AutoModelForMultipleChoice.from_pretrained("LIAMF-USP/roberta-large-finetuned-race")

# input must be: for each choice, pair (context, choice)
context = "The sky is blue during the day."
choices = ["Because of the sun.", "Because of clouds.", "Because of rain.", "Because of stars."]

# encode each (context, choice) pair
encodings = tokenizer(
    [context] * len(choices),   # repeat context for each choice
    choices,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
)

# stack into (1, num_choices, seq_len) — batch of 1
input_ids      = encodings["input_ids"].unsqueeze(0)
attention_mask = encodings["attention_mask"].unsqueeze(0)
token_type_ids = encodings["token_type_ids"].unsqueeze(0)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids
)

# logits — shape: (batch, num_choices)
logits = outputs.logits
pred   = logits.argmax(-1).item()
print(f"Answer: {choices[pred]}")  # "Because of the sun."
```

---

## TEXT — Natural Language Generation

---

### AutoModelForCausalLM

Decoder-only autoregressive generation. GPT-style. Next-token prediction.
Use for: text generation, chatbots, code generation, completion.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model     = AutoModelForCausalLM.from_pretrained("gpt2")

# tokenize prompt
inputs = tokenizer("Once upon a time", return_tensors="pt")

# generate text
output_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8
)

# decode
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)

# forward pass — get logits for next token prediction
outputs = model(**inputs)
logits = outputs.logits   # shape: (batch, seq_len, vocab_size)

# loss — provide labels to get loss for training
labels = inputs["input_ids"].clone()
outputs = model(**inputs, labels=labels)
print(outputs.loss)    # cross-entropy loss
print(outputs.logits)  # same as without labels
```

---

### AutoModelForMaskedLM

Encoder-only masked language modeling. BERT-style.
Use for: fill-mask, pre-training, understanding tasks.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# put [MASK] where you want the model to predict
text   = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

# logits — shape: (batch, seq_len, vocab_size)
logits = outputs.logits

# find the [MASK] position
mask_token_id = tokenizer.mask_token_id
mask_pos      = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1]

# top 5 predictions at the masked position
mask_logits = logits[0, mask_pos, :]
top5_ids    = mask_logits.topk(5).indices[0]
top5_tokens = tokenizer.convert_ids_to_tokens(top5_ids)
print(top5_tokens)  # ['paris', 'lyon', 'nice', 'marseille', 'toulouse']
```

---

### AutoModelForSeq2SeqLM

Encoder-decoder models. Input sequence → output sequence.
Use for: summarization, translation, grammar correction, abstractive QA.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

text = "The Amazon rainforest is the world's largest tropical rainforest..."

# encode input
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

# generate summary
output_ids = model.generate(
    **inputs,
    max_new_tokens=150,
    min_length=50,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3
)

summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(summary)

# T5 — prefix the input with the task name
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

inputs    = tokenizer("summarize: The Amazon rainforest...", return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=100)
result    = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

---

## VISION

---

### AutoModelForImageClassification

Classify what is in an image. Returns logits over class labels.

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model     = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

image = Image.open("cat.jpg")

# processor handles resizing, normalization, converting to tensor
inputs  = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

logits  = outputs.logits   # shape: (1, 1000) — 1000 ImageNet classes
pred_id = logits.argmax(-1).item()
label   = model.config.id2label[pred_id]
print(label)   # "tabby, tabby cat"

# top 5
probs   = torch.softmax(logits[0], dim=0)
top5    = probs.topk(5)
for prob, idx in zip(top5.values, top5.indices):
    print(f"{model.config.id2label[idx.item()]}: {prob.item():.3f}")
```

---

### AutoModelForObjectDetection

Detect and localize multiple objects in an image.

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model     = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

image = Image.open("street.jpg")

inputs  = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# post-process bounding boxes
target_sizes = torch.tensor([image.size[::-1]])   # (height, width)
results = processor.post_process_object_detection(
    outputs,
    target_sizes=target_sizes,
    threshold=0.9
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box  = [round(i, 2) for i in box.tolist()]
    name = model.config.id2label[label.item()]
    print(f"{name}: {score:.3f} at {box}")
# car: 0.998 at [10.5, 50.2, 200.3, 150.8]
# person: 0.985 at [220.1, 30.0, 280.5, 200.2]
```

---

### AutoModelForSemanticSegmentation

Assign a class label to every pixel.

```python
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import numpy as np

processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
model     = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

image = Image.open("scene.jpg")

inputs  = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# logits — shape: (batch, num_classes, height, width)
logits = outputs.logits

# upsample to original image size and get predictions
upsampled = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],   # (height, width)
    mode="bilinear",
    align_corners=False
)

# class prediction per pixel
pred_seg = upsampled.argmax(dim=1)[0]   # shape: (height, width)

# colorize — create a colored segmentation map
palette  = np.array(model.config.id2label)
seg_img  = Image.fromarray(pred_seg.numpy().astype(np.uint8))
```

---

### AutoModelForDepthEstimation

Estimate depth from a single image.

```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import torch

processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
model     = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")

image = Image.open("room.jpg")

inputs  = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# predicted_depth — shape: (batch, height, width)
depth = outputs.predicted_depth

# scale to original image size
depth_upsampled = torch.nn.functional.interpolate(
    depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False
).squeeze()

depth_numpy = depth_upsampled.numpy()
# normalize for visualization
depth_normalized = (depth_numpy - depth_numpy.min()) / (depth_numpy.max() - depth_numpy.min())
depth_img = Image.fromarray((depth_normalized * 255).astype("uint8"))
```

---

## AUDIO

---

### AutoModelForAudioClassification

Classify audio clips.

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import librosa

extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
model     = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-ks")

# load audio — must be 16kHz sampling rate
audio, sr = librosa.load("audio.wav", sr=16000)

# process
inputs  = extractor(audio, sampling_rate=16000, return_tensors="pt")
outputs = model(**inputs)

logits  = outputs.logits
pred_id = logits.argmax(-1).item()
label   = model.config.id2label[pred_id]
print(label)  # "yes" or "no" or other keyword
```

---

### AutoModelForSpeechSeq2Seq (Whisper)

Transcribe audio to text using sequence-to-sequence models.

```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model     = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float16
)

# load audio
audio, sr = librosa.load("speech.mp3", sr=16000)

# process
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(torch.float16)

# generate transcription
output_ids = model.generate(
    input_features,
    language="english",      # force English output
    task="transcribe"        # transcribe (keep language) or translate (to English)
)

text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(text)
```

---

## MULTIMODAL

---

### AutoModelForVisualQuestionAnswering

Answer questions about images.

```python
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model     = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

image    = Image.open("kitchen.jpg")
question = "What appliances are on the counter?"

# processor handles both image and text
inputs  = processor(image, question, return_tensors="pt")
outputs = model(**inputs)

logits  = outputs.logits
pred_id = logits.argmax(-1).item()
answer  = model.config.id2label[pred_id]
print(answer)  # "microwave"
```

---

### AutoModelForImageTextToText — Vision Language Models (VLMs)

Modern multimodal models that take images + text and generate text responses.
LLaVA, InternVL, Qwen-VL, PaliGemma all use this class.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
model     = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16
)

image = Image.open("chart.jpg")

# multimodal message format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": "Describe what you see in this image."}
        ]
    }
]

# process
text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=256)
result     = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(result)
```

---

### AutoModelForDocumentQuestionAnswering

Answer questions about document images — invoices, forms, receipts, PDFs.

```python
from transformers import AutoProcessor, AutoModelForDocumentQuestionAnswering
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("impira/layoutlm-document-qa")
model     = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-document-qa")

image    = Image.open("invoice.png")
question = "What is the total amount due?"

inputs  = processor(image, question, return_tensors="pt")
outputs = model(**inputs)

# start and end logits like QA
start_logits = outputs.start_logits
end_logits   = outputs.end_logits

start_pos = start_logits.argmax(-1).item()
end_pos   = end_logits.argmax(-1).item()

answer = processor.tokenizer.decode(
    inputs["input_ids"][0][start_pos: end_pos + 1]
)
print(answer)  # "$1,234.56"
```

---

## UTILITY — Config, Processor, Feature Extractor

---

### AutoConfig

Load the model configuration without loading the weights.
Inspect model architecture, hyperparameters, labels.

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("bert-base-uncased")

print(config.hidden_size)          # 768
print(config.num_hidden_layers)    # 12
print(config.num_attention_heads)  # 12
print(config.max_position_embeddings)  # 512
print(config.vocab_size)           # 30522
print(config.id2label)             # label mappings
print(config.model_type)           # "bert"

# modify config and create a model from scratch (random weights)
from transformers import AutoModel
config.num_hidden_layers = 6       # smaller model
model = AutoModel.from_config(config)
```

---

### AutoProcessor

Combined processor for multimodal models that need both a tokenizer
and an image/audio processor.

```python
from transformers import AutoProcessor

# vision-language models
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

# processor wraps:
processor.tokenizer        # the text tokenizer
processor.image_processor  # the image preprocessor (for vision models)
processor.feature_extractor # the audio preprocessor (for audio models)
```

---

### AutoImageProcessor

Preprocess images for vision models.

```python
from transformers import AutoImageProcessor
from PIL import Image

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

image  = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")

# processor handles:
# - resizing to the model's expected size (224x224 for ViT)
# - normalizing pixel values with model-specific mean/std
# - converting to tensor

# batch of images
images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
inputs = processor(images=images, return_tensors="pt")
```

---

### AutoFeatureExtractor

Preprocess audio for audio models.

```python
from transformers import AutoFeatureExtractor
import librosa

extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v3")

audio, sr = librosa.load("speech.wav", sr=16000)

inputs = extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)
# returns: input_features for Whisper, input_values for Wav2Vec2
```

---

## Loading Modes — device_map, dtype, quantization

```python
from transformers import AutoModelForCausalLM
import torch

# CPU (default)
model = AutoModelForCausalLM.from_pretrained("gpt2")

# single GPU
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

# auto distribute across all available GPUs
model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")

# half precision — float16 (saves GPU memory, slightly less accurate)
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)

# bfloat16 — better for A100, H100 GPUs
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.bfloat16)

# 8-bit quantization (needs bitsandbytes)
model = AutoModelForCausalLM.from_pretrained("gpt2", load_in_8bit=True)

# 4-bit quantization (needs bitsandbytes)
model = AutoModelForCausalLM.from_pretrained("gpt2", load_in_4bit=True)

# offline — load from disk without internet
model = AutoModelForCausalLM.from_pretrained("./local-model")
```

---

## Saving and Pushing Models

```python
# save to disk
model.save_pretrained("./my-fine-tuned-model")
tokenizer.save_pretrained("./my-fine-tuned-model")

# push to Hub
model.push_to_hub("username/my-fine-tuned-model")
tokenizer.push_to_hub("username/my-fine-tuned-model")

# save and push with specific commit message
model.push_to_hub(
    "username/my-model",
    commit_message="fine-tuned on custom data"
)

# push with private repo
model.push_to_hub("username/my-model", private=True)
```

---

## All AutoModel Classes — Quick Reference

| Class | Use Case |
|---|---|
| AutoModel | Raw embeddings, hidden states |
| AutoModelForSequenceClassification | Text classification, sentiment |
| AutoModelForTokenClassification | NER, POS tagging |
| AutoModelForQuestionAnswering | Extractive QA |
| AutoModelForMultipleChoice | Multiple choice selection |
| AutoModelForNextSentencePrediction | NSP (BERT pre-training) |
| AutoModelForCausalLM | Text generation, GPT-style |
| AutoModelForMaskedLM | Fill-mask, BERT-style |
| AutoModelForSeq2SeqLM | Summarization, translation, T5 |
| AutoModelForSpeechSeq2Seq | ASR — Whisper |
| AutoModelForImageClassification | Image category prediction |
| AutoModelForObjectDetection | Bounding box detection |
| AutoModelForSemanticSegmentation | Per-pixel classification |
| AutoModelForInstanceSegmentation | Per-object masks |
| AutoModelForDepthEstimation | Depth map from image |
| AutoModelForImageToImage | Image transformation |
| AutoModelForVisualQuestionAnswering | Answer Q about image |
| AutoModelForImageTextToText | VLMs — LLaVA, Qwen-VL |
| AutoModelForDocumentQuestionAnswering | QA on document images |
| AutoModelForAudioClassification | Audio clip classification |
| AutoModelForCTC | CTC-based ASR |
| AutoModelForSpeechSeq2Seq | Whisper-style ASR |
| AutoModelForPreTraining | Raw pretraining objectives |
| AutoConfig | Model architecture only |
| AutoTokenizer | Text tokenization |
| AutoImageProcessor | Image preprocessing |
| AutoFeatureExtractor | Audio preprocessing |
| AutoProcessor | Multimodal preprocessing |

---

## What is Next

Part 4 covers the Trainer — training and fine-tuning models with TrainingArguments,
data collators, callbacks, and compute_metrics.