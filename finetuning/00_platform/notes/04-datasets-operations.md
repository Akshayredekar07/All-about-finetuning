# **04 – Datasets Operations**

Every operation in the `datasets` library that Lab 0-5 sweeps. Written for **datasets 4.x**, where dataset loading scripts and `trust_remote_code` were removed.

---

## **Installation**

```bash
pip install -U datasets
pip install "datasets[audio]"       # audio datasets (4.x decodes audio via torchcodec; may need ffmpeg)
pip install "datasets[vision]"      # image datasets
pip install "datasets[s3]"          # loading from S3
pip install pandas polars           # for to_pandas() and to_polars()
```

---

## **Read This First: Scripts Are Gone**

In datasets 4.0 and later, `load_dataset()` refuses repos that rely on a Python loading script:

```
RuntimeError: Dataset scripts are no longer supported, but found conll2003.py
```

Fixes:

```python
# ── 1. use the namespaced, Parquet-based version of the dataset ───────
ds = load_dataset("lhoestq/conll2003")

# ── 2. or pin the old library: pip install "datasets<4.0" ─────────────
# ── 3. or convert a script dataset you own to Parquet ─────────────────
#    uv run --with "datasets==3.6.0" datasets-cli convert_to_parquet <user/name> --trust_remote_code
```

Use full `owner/name` ids throughout: `stanfordnlp/imdb`, `allenai/c4`, `fancyzhx/yelp_polarity`. Run each once to confirm it still loads. This is the "does it still load" item in the Stage 00 checklist.

---

## **1. Loading**

```python
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

# ── From the Hugging Face Hub ─────────────────────────────────────────
ds = load_dataset("stanfordnlp/imdb")                          # all splits -> DatasetDict
ds = load_dataset("stanfordnlp/imdb", split="train")           # single split
ds = load_dataset("nyu-mll/glue", "mrpc")                      # with config name
ds = load_dataset("stanfordnlp/imdb", split="train[:10%]")     # first 10%
ds = load_dataset("stanfordnlp/imdb", split="train[100:200]")  # row range
ds = load_dataset("stanfordnlp/imdb", split="train+test")      # merge splits

# ── From local files ──────────────────────────────────────────────────
ds = load_dataset("csv",     data_files="outputs/fixtures/data.csv")
ds = load_dataset("json",    data_files="outputs/fixtures/data.jsonl")
ds = load_dataset("parquet", data_files="outputs/fixtures/data.parquet")
ds = load_dataset("text",    data_files="outputs/fixtures/data.txt")

# ── Multiple files mapped to splits ───────────────────────────────────
ds = load_dataset("csv", data_files={
    "train": "outputs/fixtures/train.csv",
    "test":  "outputs/fixtures/test.csv",
})

# ── From in-memory structures ─────────────────────────────────────────
import pandas as pd

ds = Dataset.from_dict({
    "text":  ["hello world", "foo bar", "good morning"],
    "label": [0, 1, 0],
})

df = pd.DataFrame({"text": ["a", "b"], "label": [1, 0]})
ds = Dataset.from_pandas(df)

ds = Dataset.from_list([
    {"text": "hello", "label": 0},
    {"text": "world", "label": 1},
])

# ── From a Python generator (good for big or custom sources) ──────────
def gen():
    for i in range(1000):
        yield {"text": f"row {i}", "label": i % 2}

ds = Dataset.from_generator(gen)

# ── From disk (saved with save_to_disk) ───────────────────────────────
ds = load_from_disk("outputs/my_saved_dataset")
```

**Notes:**
- `load_dataset("path/to/folder")` reads a folder of data files (csv, parquet, ...). It does **not** read `save_to_disk` output. Use `load_from_disk` for that.
- Local files load as a single `train` split unless you pass the `data_files` dict.
- Hub downloads go to the HF cache, not your project folder.

---

## **2. Inspecting**

```python
ds = load_dataset("stanfordnlp/imdb", split="train")

print(ds)                    # overview: features, num_rows
print(ds.features)           # column names + types
print(ds.column_names)       # list of column names
print(len(ds))               # number of rows
print(ds.shape)              # (rows, cols)
print(ds.num_rows)
print(ds.num_columns)

# ── Access rows ───────────────────────────────────────────────────────
print(ds[0])                 # first row as dict
print(ds[:5])                # first 5 rows as dict of lists
print(ds[10:20])             # row slice
print(ds["text"][:3])        # a column (first 3 values)
print(ds[0]["text"])         # single value

# ── Label names ───────────────────────────────────────────────────────
print(ds.features["label"].names)          # ['neg', 'pos']
print(ds.features["label"].int2str(1))     # 'pos'

# ── Underlying Apache Arrow table ─────────────────────────────────────
print(ds.data.num_rows)
print(ds.data.schema)
print(ds.cache_files)                      # where the Arrow files live on disk

# ── DatasetDict (multiple splits) ─────────────────────────────────────
dsd = load_dataset("stanfordnlp/imdb")
print(dsd.keys())            # dict_keys(['train', 'test', 'unsupervised'])
print(dsd["train"])          # access a split
print(dsd["train"][0])       # first row of train
```

**Note:** `ds["text"]` builds a full Python list of the column. On a big dataset use `ds[:100]["text"]` or `ds.select(range(100))["text"]`.

---

## **3. Row Operations**

```python
ds = load_dataset("stanfordnlp/imdb", split="train")

# ── sort ──────────────────────────────────────────────────────────────
sorted_ds = ds.sort("label")                        # ascending
sorted_ds = ds.sort("label", reverse=True)          # descending

# ── shuffle ───────────────────────────────────────────────────────────
shuffled = ds.shuffle(seed=42)

# ── select: pick rows by index ────────────────────────────────────────
small_ds = ds.select([0, 1, 2, 100, 200])           # exact indices
first100 = ds.select(range(100))                    # first 100 rows

# ── filter: keep rows matching a condition ────────────────────────────
positive   = ds.filter(lambda x: x["label"] == 1)
long_texts = ds.filter(lambda x: len(x["text"]) > 500)

# filter with the row index
even_rows = ds.filter(lambda example, idx: idx % 2 == 0, with_indices=True)

# ── skip and take ─────────────────────────────────────────────────────
# these are for IterableDataset (streaming). On a normal Dataset use select().
stream  = load_dataset("stanfordnlp/imdb", split="train", streaming=True)
skipped = stream.skip(100)
taken   = stream.take(50)

# ── train / test split ────────────────────────────────────────────────
splits   = ds.train_test_split(test_size=0.2, seed=42)
train_ds = splits["train"]
test_ds  = splits["test"]

# stratified split (keeps label proportions; column must be a ClassLabel)
splits = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)

# ── shard: divide into N equal chunks ─────────────────────────────────
shard_0 = ds.shard(num_shards=4, index=0)
shard_1 = ds.shard(num_shards=4, index=1)
```

**Notes:**
- `shuffle`, `sort`, `select` and `filter` return a **new** dataset. They never change `ds` in place.
- `shuffle` creates an indices mapping, not new data. Reading a shuffled dataset in order is slower; `ds.flatten_indices()` rewrites it contiguously.
- Always pass `seed=` so the split is reproducible.

---

## **4. Column Operations**

```python
ds = load_dataset("stanfordnlp/imdb", split="train")

# ── rename ────────────────────────────────────────────────────────────
ds = ds.rename_column("text", "review")
ds = ds.rename_columns({"review": "text", "label": "sentiment"})
ds = ds.rename_column("sentiment", "label")

# ── remove ────────────────────────────────────────────────────────────
tmp = ds.remove_columns("label")
tmp = ds.remove_columns(["label", "text"])

# ── select specific columns only ──────────────────────────────────────
tmp = ds.select_columns(["text", "label"])

# ── add a new column ──────────────────────────────────────────────────
lengths = [len(t) for t in ds["text"]]
ds = ds.add_column("text_length", lengths)

# ── cast column type ──────────────────────────────────────────────────
from datasets import Value
ds = ds.cast_column("text_length", Value("int32"))

# ── flatten nested columns ────────────────────────────────────────────
# e.g. answers.text, answers.answer_start -> flat columns
# flat_ds = squad.flatten()

# ── remap label ids to your own mapping ───────────────────────────────
label2id = {"pos": 0, "neg": 1}
# ds = ds.align_labels_with_mapping(label2id, "label")
```

**Note:** a `ClassLabel` column can only be cast to another `ClassLabel` or compatible integer type. Test a cast on a small `select(range(10))` first.

---

## **5. Map: Apply a Function to Every Row**

`map()` is the most powerful operation. It processes examples one at a time or in batches, and can change, add or remove rows and columns.

```python
ds = load_dataset("stanfordnlp/imdb", split="train")

# ── basic map: add a new column ───────────────────────────────────────
def add_length(example):
    example["text_length"] = len(example["text"])
    return example

ds = ds.map(add_length)

# ── lambda map (return only the new keys) ─────────────────────────────
ds = ds.map(lambda x: {"upper_text": x["text"].upper()})

# ── batched map (much faster) ─────────────────────────────────────────
def batch_uppercase(batch):
    batch["upper"] = [t.upper() for t in batch["text"]]
    return batch

ds = ds.map(batch_uppercase, batched=True, batch_size=128)

# ── map with a tokenizer (most common real use) ───────────────────────
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

def tokenize(batch):
    # no padding here: pad per batch later with DataCollatorWithPadding
    return tokenizer(batch["text"], truncation=True, max_length=256)

tokenized_ds = ds.map(tokenize, batched=True)

# ── map with multiprocessing ──────────────────────────────────────────
ds = ds.map(add_length, num_proc=4)

# ── map and remove the original column ────────────────────────────────
tokenized_ds = ds.map(tokenize, batched=True, remove_columns=["text"])

# ── map on all splits at once (DatasetDict) ───────────────────────────
dsd = load_dataset("stanfordnlp/imdb")
tokenized = dsd.map(tokenize, batched=True)        # train + test + unsupervised

# ── lazy transform: runs on access, nothing stored ────────────────────
ds.set_transform(lambda batch: {"n_chars": [len(t) for t in batch["text"]]})
view = ds.with_transform(lambda batch: {"n_chars": [len(t) for t in batch["text"]]})
```

**Notes:**
- `map` caches its result on disk. Change the function and it reruns; reuse the same function and it returns instantly. Pass `load_from_cache_file=False` to force a rerun.
- With `batched=True` the function receives a dict of **lists**, not a single row.
- `padding="max_length"` (used in the old note) pads every row to the maximum and wastes memory. Pad per batch with `DataCollatorWithPadding` instead.
- `with_transform` and `set_transform` are lazy: good for augmentation and quick views, bad for expensive work you want to cache.

---

## **6. Combining Datasets**

```python
from datasets import concatenate_datasets, interleave_datasets

ds1 = load_dataset("stanfordnlp/imdb",         split="train")
ds2 = load_dataset("fancyzhx/yelp_polarity",   split="train")

# ── concatenate: stack rows vertically ────────────────────────────────
combined = concatenate_datasets([ds1, ds2])

# ── interleave: alternate rows from each dataset ──────────────────────
interleaved = interleave_datasets([ds1, ds2])

# ── weighted interleave: sample more from ds1 ─────────────────────────
interleaved = interleave_datasets(
    [ds1, ds2],
    probabilities=[0.7, 0.3],
    seed=42,
)

# ── stop rule: stop at the first exhausted / use all examples ─────────
interleaved = interleave_datasets([ds1, ds2], stopping_strategy="all_exhausted")

# ── stack columns side by side (same number of rows) ──────────────────
side_by_side = concatenate_datasets([ds1.select(range(100)), ds2.select(range(100))], axis=1)
```

**Note:** both datasets must have **identical features** (same column names and types). If they differ, `rename_column` and `cast_column` one of them first.

---

## **7. Format Conversion: Export to Other Frameworks**

```python
ds = load_dataset("stanfordnlp/imdb", split="train")

# ── to Pandas / Polars ────────────────────────────────────────────────
df  = ds.to_pandas()
pdf = ds.to_polars()

# ── to dict / list of dicts ───────────────────────────────────────────
d    = ds.to_dict()
rows = ds.to_list()

# ── to files ──────────────────────────────────────────────────────────
ds.to_csv("outputs/output.csv")
ds.to_json("outputs/output.jsonl")        # JSON Lines by default
ds.to_parquet("outputs/output.parquet")

# ── set format for PyTorch training ───────────────────────────────────
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ── set format for NumPy ──────────────────────────────────────────────
ds.set_format(type="numpy", columns=["input_ids"])

# ── reset format back to default (Python dicts) ───────────────────────
ds.reset_format()

# ── with_format: returns a NEW dataset (non-destructive) ──────────────
torch_ds = ds.with_format("torch", columns=["input_ids", "label"])
```

**Notes:**
- `set_format` changes the dataset **in place**. `with_format` returns a copy. Prefer `with_format` in notebooks so re-running a cell does not stack state.
- `to_pandas()` and `to_json()` load a lot into memory. On big datasets use `batched=True` / `batch_size`, or write Parquet.
- Formats with columns the tokenizer did not create (like `input_ids`) fail with a `KeyError`. Run `map(tokenize)` first.

---

## **8. Save and Load from Disk**

```python
# ── save to disk (Arrow format; goes in outputs/, gitignored) ─────────
ds.save_to_disk("outputs/my_dataset")

# ── load back ─────────────────────────────────────────────────────────
from datasets import load_from_disk
ds = load_from_disk("outputs/my_dataset")

# ── save a DatasetDict (all splits) ───────────────────────────────────
dsd = load_dataset("stanfordnlp/imdb")
dsd.save_to_disk("outputs/imdb_all_splits")

dsd = load_from_disk("outputs/imdb_all_splits")       # returns a DatasetDict
```

**Notes:**
- `save_to_disk` writes `.arrow` files, `state.json` and `dataset_info.json`. These are build artifacts. Do not commit them; regenerate from the notebook.
- `load_from_disk` works for both `Dataset` and `DatasetDict`, so you do not need `DatasetDict.load_from_disk`.
- Do not load a dataset from the same path you are saving to; `save_to_disk` refuses to overwrite its own source.

---

## **9. Streaming: Huge Datasets, No Full Download**

If the dataset is bigger than your disk, or you do not want to wait for a download, stream it. Data is fetched and processed as you iterate.

```python
# ── stream a huge dataset ─────────────────────────────────────────────
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

# iterate: never loads the full dataset
for example in ds:
    print(example["text"][:100])
    break

# ── take first N examples ─────────────────────────────────────────────
first10 = list(ds.take(10))

# ── map on a streamed dataset (lazy) ──────────────────────────────────
streamed = ds.map(lambda x: {"length": len(x["text"])})

# ── filter on a streamed dataset (lazy) ───────────────────────────────
long_texts = ds.filter(lambda x: len(x["text"]) > 1000)

# ── shuffle buffer (approximate shuffle for streams) ──────────────────
shuffled = ds.shuffle(seed=42, buffer_size=10_000)

# ── turn a normal Dataset into an IterableDataset ─────────────────────
local = load_dataset("stanfordnlp/imdb", split="train")
it = local.to_iterable_dataset(num_shards=4)
```

**Notes:**
- A streamed dataset has no `len()`, no `ds[0]` and no `.column_names` until you read a row.
- `shuffle` on a stream is only as random as `buffer_size`.
- Streaming can fail in notebooks with a `NotImplementedError` about the local file system (seen in Colab reports with C4 and similar). Update `datasets` and `huggingface_hub`, and restart the runtime.

---

## **10. Push to Hub: Upload Your Dataset**

```python
import os
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict

load_dotenv()
WRITE_TOKEN = os.environ["HF_WRITE_TOKEN"]      # pushing needs the WRITE token
REPO = "your-username/my-dataset"

# ── push a single Dataset ─────────────────────────────────────────────
ds = Dataset.from_dict({
    "text":  ["review one", "review two", "review three"],
    "label": [1, 0, 1],
})

ds.push_to_hub(REPO, token=WRITE_TOKEN)
ds.push_to_hub(REPO, private=True, token=WRITE_TOKEN)

# ── push a specific split ─────────────────────────────────────────────
ds.push_to_hub(REPO, split="train", token=WRITE_TOKEN)

# ── push a DatasetDict (train + test together) ────────────────────────
dsd = DatasetDict({
    "train": Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}),
    "test":  Dataset.from_dict({"text": ["c"],      "label": [1]}),
})
dsd.push_to_hub(REPO, token=WRITE_TOKEN)

# ── add a data card (README.md) ───────────────────────────────────────
from huggingface_hub import DatasetCard

card = DatasetCard("""---
license: mit
task_categories:
- text-classification
language:
- en
---
# My Dataset
Tiny sentiment fixture used in Stage 00 Lab 0-5.
""")
card.push_to_hub(REPO, token=WRITE_TOKEN)

# ── load it back to confirm ───────────────────────────────────────────
check = load_dataset(REPO, token=WRITE_TOKEN)
print(check)
```

**Notes:**
- `push_to_hub` writes Parquet files, so the Hub viewer works without a loading script.
- Pushing a split with `split="train"` replaces that split only, not the whole repo.
- A read token fails here with a 403.

---

## **11. Features: Define the Schema Explicitly**

```python
from datasets import Dataset, Features, Value, ClassLabel, Sequence

# ── define a typed schema ─────────────────────────────────────────────
features = Features({
    "text":   Value("string"),
    "label":  ClassLabel(names=["negative", "positive"]),
    "score":  Value("float32"),
    "tokens": Sequence(Value("string")),
})

ds = Dataset.from_dict(
    {
        "text":   ["good", "bad"],
        "label":  [1, 0],
        "score":  [0.9, 0.1],
        "tokens": [["good"], ["bad"]],
    },
    features=features,
)

print(ds.features)
# text:   Value(dtype='string')
# label:  ClassLabel(names=['negative', 'positive'])
# score:  Value(dtype='float32')
# tokens: List(Value(dtype='string'))      <- newer versions print List(...)

# ── turn a string column into a ClassLabel ────────────────────────────
# ds = ds.class_encode_column("label")
```

**Note:** `Sequence(...)` and `List(...)` are the same idea; recent versions print the second name. Do not compare feature objects by their printed text.

---

## **Quick Reference: Every Operation**

| Category | Method | What it does |
|---|---|---|
| Load | `load_dataset()` | Load from Hub or local files |
| Load | `Dataset.from_dict()` | From Python dict |
| Load | `Dataset.from_pandas()` | From Pandas DataFrame |
| Load | `Dataset.from_list()` | From list of dicts |
| Load | `Dataset.from_generator()` | From a Python generator |
| Load | `load_from_disk()` | From a `save_to_disk` folder |
| Inspect | `.features`, `.column_names`, `.shape` | Schema and size |
| Inspect | `.data` | Underlying Arrow table |
| Row | `sort()` | Sort by column |
| Row | `shuffle()` | Randomly shuffle |
| Row | `select()` | Pick rows by index |
| Row | `filter()` | Keep rows matching a condition |
| Row | `train_test_split()` | Split into train/test |
| Row | `shard()` | Split into N equal chunks |
| Row | `skip()` / `take()` | First N / after N (streaming) |
| Column | `rename_column()` | Rename a column |
| Column | `remove_columns()` | Delete columns |
| Column | `select_columns()` | Keep only some columns |
| Column | `add_column()` | Add a new column |
| Column | `cast_column()` | Change column type |
| Column | `flatten()` | Flatten nested fields |
| Column | `align_labels_with_mapping()` | Remap label ids |
| Transform | `map()` | Apply function to every row (cached) |
| Transform | `with_transform()` | Lazy function applied on access |
| Combine | `concatenate_datasets()` | Stack datasets vertically |
| Combine | `interleave_datasets()` | Alternate rows from datasets |
| Export | `to_pandas()` / `to_polars()` | Convert to DataFrame |
| Export | `to_csv()` / `to_json()` / `to_parquet()` | Save as file |
| Format | `set_format()` / `with_format()` | Output as torch / numpy |
| Save | `save_to_disk()` | Save to local disk (Arrow) |
| Upload | `push_to_hub()` | Upload to the Hub |
| Stream | `streaming=True` | Process without downloading |
| Stream | `to_iterable_dataset()` | Convert a Dataset to a stream |

**Verify by running:** `Dataset.to_polars`, `Dataset.take/skip` on map-style datasets, and `fancyzhx/yelp_polarity` against your installed `datasets` version. I checked the 4.0 script removal against HF forum and GitHub reports, but not every method signature.