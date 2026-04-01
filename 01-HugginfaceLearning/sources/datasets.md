Here is the complete deep dive into every operation the `datasets` library supports.

---

## Installation

```bash
pip install datasets
pip install datasets[audio]       # for audio datasets
pip install datasets[vision]      # for image datasets
pip install datasets[s3]          # for loading from S3
```

---

## 1. LOADING — All Ways to Load a Dataset

```python
from datasets import load_dataset, Dataset, DatasetDict

# ── From HuggingFace Hub ──────────────────────────────────────────────
ds = load_dataset("imdb")                          # all splits
ds = load_dataset("imdb", split="train")           # single split
ds = load_dataset("nyu-mll/glue", "mrpc")          # with config
ds = load_dataset("imdb", split="train[:10%]")     # first 10%
ds = load_dataset("imdb", split="train[100:200]")  # row range
ds = load_dataset("imdb", split="train+test")      # merge splits

# ── From local files ──────────────────────────────────────────────────
ds = load_dataset("csv",  data_files="data.csv")
ds = load_dataset("json", data_files="data.jsonl")
ds = load_dataset("parquet", data_files="data.parquet")
ds = load_dataset("text", data_files="data.txt")

# Multiple files mapped to splits
ds = load_dataset("csv", data_files={
    "train": "train.csv",
    "test":  "test.csv"
})

# ── From in-memory structures ─────────────────────────────────────────
import pandas as pd

ds = Dataset.from_dict({
    "text":  ["hello world", "foo bar", "good morning"],
    "label": [0, 1, 0]
})

df = pd.DataFrame({"text": ["a", "b"], "label": [1, 0]})
ds = Dataset.from_pandas(df)

ds = Dataset.from_list([
    {"text": "hello", "label": 0},
    {"text": "world", "label": 1},
])

# ── From disk (previously saved) ──────────────────────────────────────
ds = load_dataset("path/to/dataset/folder")       # load_dataset style
ds = Dataset.load_from_disk("./my_saved_dataset") # exact save/load pair
```

---

## 2. INSPECTING — Know What's Inside

```python
ds = load_dataset("imdb", split="train")

print(ds)                    # overview: features, num_rows
print(ds.features)           # column names + types
print(ds.column_names)       # just column names as list
print(len(ds))               # number of rows
print(ds.shape)              # (rows, cols)
print(ds.num_rows)           
print(ds.num_columns)

# Access rows
print(ds[0])                 # first row as dict
print(ds[:5])                # first 5 rows as dict of lists
print(ds[10:20])             # row slice
print(ds["text"])            # entire column as list
print(ds[0]["text"])         # single value

# DatasetDict (multiple splits)
dsd = load_dataset("imdb")
print(dsd.keys())            # dict_keys(['train', 'test', 'unsupervised'])
print(dsd["train"])          # access a split
print(dsd["train"][0])       # first row of train split
```

---

## 3. ROW OPERATIONS

```python
from datasets import load_dataset

ds = load_dataset("imdb", split="train")

# ── sort ──────────────────────────────────────────────────────────────
sorted_ds = ds.sort("label")                        # ascending
sorted_ds = ds.sort("label", reverse=True)          # descending

# ── shuffle ───────────────────────────────────────────────────────────
shuffled = ds.shuffle(seed=42)

# ── select — pick specific rows by index ──────────────────────────────
small_ds = ds.select([0, 1, 2, 100, 200])           # exact indices
first100 = ds.select(range(100))                     # first 100 rows

# ── filter — keep rows matching a condition ───────────────────────────
positive = ds.filter(lambda x: x["label"] == 1)
long_texts = ds.filter(lambda x: len(x["text"]) > 500)

# filter with index
even_rows = ds.filter(
    lambda example, idx: idx % 2 == 0,
    with_indices=True
)

# ── skip and take ─────────────────────────────────────────────────────
ds_skip = ds.skip(100)     # skip first 100 rows (IterableDataset)
ds_take = ds.take(50)      # take first 50 rows  (IterableDataset)

# ── train / test split ────────────────────────────────────────────────
splits = ds.train_test_split(test_size=0.2, seed=42)
train_ds = splits["train"]
test_ds  = splits["test"]

# stratified split (keeps label proportions)
splits = ds.train_test_split(test_size=0.2, stratify_by_column="label")

# ── shard — divide into N equal chunks ───────────────────────────────
shard_0 = ds.shard(num_shards=4, index=0)   # chunk 0 of 4
shard_1 = ds.shard(num_shards=4, index=1)   # chunk 1 of 4
```

---

## 4. COLUMN OPERATIONS

```python
ds = load_dataset("imdb", split="train")

# ── rename ────────────────────────────────────────────────────────────
ds = ds.rename_column("text", "review")
ds = ds.rename_columns({"text": "review", "label": "sentiment"})

# ── remove ────────────────────────────────────────────────────────────
ds = ds.remove_columns("label")
ds = ds.remove_columns(["label", "text"])

# ── add a new column ──────────────────────────────────────────────────
lengths = [len(t) for t in ds["text"]]
ds = ds.add_column("text_length", lengths)

# ── cast column type ──────────────────────────────────────────────────
from datasets import Value
ds = ds.cast_column("label", Value("float32"))

# ── flatten nested columns ────────────────────────────────────────────
# e.g. answers.text, answers.start -> flat columns
flat_ds = ds.flatten()

# ── select specific columns only ──────────────────────────────────────
ds = ds.select_columns(["text", "label"])
```

---

## 5. MAP — Apply a Function to Every Row

`map()` is the most powerful operation — it applies a processing function to each example independently or in batches, and can even create new rows and columns.

```python
ds = load_dataset("imdb", split="train")

# ── basic map: add a new column ───────────────────────────────────────
def add_length(example):
    example["text_length"] = len(example["text"])
    return example

ds = ds.map(add_length)

# ── lambda map ────────────────────────────────────────────────────────
ds = ds.map(lambda x: {"upper_text": x["text"].upper()})

# ── batched map (much faster) ─────────────────────────────────────────
def batch_uppercase(batch):
    batch["upper"] = [t.upper() for t in batch["text"]]
    return batch

ds = ds.map(batch_uppercase, batched=True, batch_size=128)

# ── map with tokenizer (most common real use case) ────────────────────
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

tokenized_ds = ds.map(tokenize, batched=True)

# ── map with multiprocessing ──────────────────────────────────────────
ds = ds.map(add_length, num_proc=4)

# ── map and remove original column ───────────────────────────────────
ds = ds.map(tokenize, batched=True, remove_columns=["text"])

# ── map on all splits at once (DatasetDict) ───────────────────────────
dsd = load_dataset("imdb")
tokenized = dsd.map(tokenize, batched=True)  # applies to train+test
```

---

## 6. COMBINING DATASETS

```python
from datasets import concatenate_datasets, interleave_datasets

ds1 = load_dataset("imdb",    split="train")
ds2 = load_dataset("yelp_polarity", split="train")

# ── concatenate: stack rows vertically ───────────────────────────────
combined = concatenate_datasets([ds1, ds2])

# ── interleave: alternate rows from each dataset ─────────────────────
interleaved = interleave_datasets([ds1, ds2])

# weighted interleave (sample more from ds1)
interleaved = interleave_datasets(
    [ds1, ds2],
    probabilities=[0.7, 0.3],
    seed=42
)
```

---

## 7. FORMAT CONVERSION — Export to Other Frameworks

```python
ds = load_dataset("imdb", split="train")

# ── to Pandas DataFrame ───────────────────────────────────────────────
df = ds.to_pandas()

# ── to dict ───────────────────────────────────────────────────────────
d = ds.to_dict()

# ── to list of dicts ──────────────────────────────────────────────────
rows = ds.to_list()

# ── to CSV ────────────────────────────────────────────────────────────
ds.to_csv("output.csv")

# ── to JSON ───────────────────────────────────────────────────────────
ds.to_json("output.jsonl")

# ── to Parquet ────────────────────────────────────────────────────────
ds.to_parquet("output.parquet")

# ── set format for PyTorch training ───────────────────────────────────
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ── set format for TensorFlow ─────────────────────────────────────────
ds.set_format(type="tensorflow", columns=["input_ids", "label"])

# ── set format for NumPy ──────────────────────────────────────────────
ds.set_format(type="numpy", columns=["input_ids"])

# ── reset format back to default (Python dicts) ───────────────────────
ds.reset_format()

# ── with_format: returns a new dataset (non-destructive) ──────────────
torch_ds = ds.with_format("torch", columns=["input_ids", "label"])
```

---

## 8. SAVE & LOAD FROM DISK

```python
# ── save to disk ──────────────────────────────────────────────────────
ds.save_to_disk("./my_dataset")

# ── load back ────────────────────────────────────────────────────────
from datasets import load_from_disk
ds = load_from_disk("./my_dataset")

# ── save DatasetDict (all splits) ────────────────────────────────────
dsd = load_dataset("imdb")
dsd.save_to_disk("./imdb_all_splits")

dsd = DatasetDict.load_from_disk("./imdb_all_splits")
```

---

## 9. STREAMING — For Huge Datasets (No Download Needed)

If your dataset is bigger than your disk or you don't want to wait to download the data, you can use streaming — it processes data iteratively instead of loading everything into memory.

```python
# ── stream a huge dataset ─────────────────────────────────────────────
ds = load_dataset("c4", "en", split="train", streaming=True)

# iterate — never loads full dataset into RAM
for example in ds:
    print(example["text"][:100])
    break

# ── take first N examples ─────────────────────────────────────────────
first10 = list(ds.take(10))

# ── map on streamed dataset ───────────────────────────────────────────
streamed = ds.map(lambda x: {"length": len(x["text"])})

# ── filter on streamed dataset ────────────────────────────────────────
long_texts = ds.filter(lambda x: len(x["text"]) > 1000)

# ── shuffle buffer (approximate shuffle for streams) ──────────────────
shuffled = ds.shuffle(seed=42, buffer_size=10_000)
```

---

## 10. PUSH TO HUB — Upload Your Dataset

```python
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import os

login(token=os.getenv("HF_TOKEN"))

# ── push a single Dataset ─────────────────────────────────────────────
ds = Dataset.from_dict({
    "text":  ["review one", "review two", "review three"],
    "label": [1, 0, 1]
})

ds.push_to_hub("your-username/my-dataset")
ds.push_to_hub("your-username/my-dataset", private=True)

# ── push a specific split ─────────────────────────────────────────────
ds.push_to_hub("your-username/my-dataset", split="train")

# ── push DatasetDict (train + test together) ──────────────────────────
dsd = DatasetDict({
    "train": Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}),
    "test":  Dataset.from_dict({"text": ["c"],       "label": [1]}),
})
dsd.push_to_hub("your-username/my-dataset")
```

---

## 11. FEATURES — Define Schema Explicitly

```python
from datasets import Dataset, Features, Value, ClassLabel, Sequence

# ── define a typed schema ─────────────────────────────────────────────
features = Features({
    "text":  Value("string"),
    "label": ClassLabel(names=["negative", "positive"]),
    "score": Value("float32"),
    "tokens": Sequence(Value("string")),
})

ds = Dataset.from_dict(
    {"text": ["good", "bad"], "label": [1, 0], "score": [0.9, 0.1], "tokens": [["good"], ["bad"]]},
    features=features
)

print(ds.features)
# text: Value(dtype='string')
# label: ClassLabel(names=['negative', 'positive'])
```

---

## Quick Reference — Every Operation

| Category | Method | What it does |
|---|---|---|
| Load | `load_dataset()` | Load from Hub or local files |
| Load | `Dataset.from_dict()` | From Python dict |
| Load | `Dataset.from_pandas()` | From Pandas DataFrame |
| Load | `Dataset.from_list()` | From list of dicts |
| Load | `load_from_disk()` | From previously saved dataset |
| Row | `sort()` | Sort by column |
| Row | `shuffle()` | Randomly shuffle |
| Row | `select()` | Pick rows by index |
| Row | `filter()` | Keep rows matching condition |
| Row | `train_test_split()` | Split into train/test |
| Row | `shard()` | Split into N equal chunks |
| Column | `rename_column()` | Rename a column |
| Column | `remove_columns()` | Delete columns |
| Column | `add_column()` | Add a new column |
| Column | `cast_column()` | Change column type |
| Column | `flatten()` | Flatten nested fields |
| Transform | `map()` | Apply function to every row |
| Combine | `concatenate_datasets()` | Stack datasets vertically |
| Combine | `interleave_datasets()` | Alternate rows from datasets |
| Export | `to_pandas()` | Convert to DataFrame |
| Export | `to_csv()` | Save as CSV |
| Export | `to_json()` | Save as JSONL |
| Export | `to_parquet()` | Save as Parquet |
| Format | `set_format()` | Set output as torch/tf/numpy |
| Save | `save_to_disk()` | Save to local disk |
| Upload | `push_to_hub()` | Upload to HuggingFace Hub |
| Stream | `streaming=True` | Process without downloading |