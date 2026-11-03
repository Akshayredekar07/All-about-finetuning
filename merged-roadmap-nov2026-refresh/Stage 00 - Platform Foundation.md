# Stage 00 — Platform Foundation

**Weeks:** 1–4 (complete)
**Libraries:** `huggingface_hub` 2.4.0 · `datasets` 5.1.0
**Outcome:** authenticate, move artifacts in both directions, and own a dataset end to end without touching a GPU.

## HuggingFace Hub and `huggingface_hub` 2.4.0

The Hub is the GitHub of ML: models, datasets, Spaces, and Jobs. `huggingface_hub` is the Python and CLI client. Version 2 moved the HTTP stack to **httpx2/httpcore2**, requires **Python ≥3.10**, and renamed a batch of CLI subcommands and Python functions [3][24].

**Authentication**
- `login()`, `notebook_login()`, `interpreter_login()`
- Environment variable `HF_TOKEN`
- CLI: `hf auth login`, `hf auth whoami`
- Switching between multiple saved tokens
- `logout()` and `whoami()`
- Fine-grained tokens versus read and write tokens — give each project the least access it needs

**Downloading**
- `hf_hub_download()` for a single file
- `snapshot_download()` for an entire repo
- Specific revisions and branches
- `allow_patterns` and `ignore_patterns` filters
- Gated model access and `GatedRepoError` handling
- Offline mode with cached files (`HF_HUB_OFFLINE=1`)
- CLI: `hf download`
- Cache location controlled by `HF_HOME`
- Fast transfers use `hf_xet` automatically (the old `hf_transfer` is gone)

**Uploading**
- `upload_file()` for a single file
- `upload_folder()` for a directory
- `CommitOperationAdd` for atomic multi-file commits
- `create_commit()` for full commit control
- Ignoring files during upload
- CLI: `hf upload` — note the v2 rename from `hf upload-large-folder`

**Repository management**
- `create_repo()` for model, dataset, space
- `delete_repo()`
- `list_models()`, `list_datasets()`, `list_spaces()` → **renamed to `search_*` in hub v2**
- `model_info()`, `dataset_info()`
- `repo_exists()`

**Spaces management (API only)**
- `get_space_runtime()`
- `pause_space()`, `restart_space()`
- Space hardware management

**Error handling**
- Catch `HfHubHTTPError` and subclasses (`GatedRepoError`, `RepositoryNotFoundError`) — the library no longer uses `requests`, so `requests` exception handlers will never fire.

**v2 API rename table** [3][24]

| hub v1 | hub v2 | Note |
|---|---|---|
| `hf repo` | `hf repos` | pluralized |
| `hf repo-files delete` | `hf repos delete-files` | |
| `hf upload-large-folder` | `hf upload` | storage flag now `--volume` |
| `hf jobs ps` | `hf jobs --status` / `--label` | filters replace positional listing |
| `list_models()` | `search_models()` | |
| `upload_large_folder()` | `upload_folder()` | |
| `duplicate_space()` | `duplicate_repo()` | |
| `repo_type_and_id_from_hf_id()` | `parse_hf_uri()` | |
| `huggingface-cli` | `hf` | the old entry point is gone entirely |
| `from huggingface_hub.utils import httpx` | — | compatibility shim for the httpx2 move |

**Key install**
```bash
pip install -U huggingface_hub
```

**Key imports**
```python
from huggingface_hub import (
    login, logout, whoami,
    hf_hub_download, snapshot_download,
    upload_file, upload_folder,
    create_repo, delete_repo,
    HfApi, CommitOperationAdd
)
```

**CLI quick reference**
```bash
hf auth login
hf auth whoami
hf download Qwen/Qwen3-1.7B --local-dir ./qwen
hf upload username/my-model ./my-model
```

**Agent Skills for the CLI.** The Hub now ships a set of Agent Skills that teach a coding agent the `hf` surface: `hf-cli` (generated from your locally installed CLI, so it tracks your version), `huggingface-datasets`, `huggingface-llm-trainer` (TRL SFT/DPO/GRPO on HF Jobs), `huggingface-community-evals`, `huggingface-vision-trainer`, `huggingface-trackio`, `gradio`, and `transformers-js`. They install with `hf skills add <name>` (add `--global` for all projects) and are refreshed with `hf skills update -g`; `hf update` refreshes the globally installed skill. They follow the Agent Skills spec and work with Claude Code, Codex, Gemini CLI, and Cursor, with `AGENTS.md` as the fallback for clients that do not support skills [25][26][27].

```bash
hf skills add hf-cli --global
hf skills list
hf skills update -g
```

## Datasets 5.1.0

The data layer. Built on Apache Arrow, so large datasets are memory-mapped from disk instead of loaded into RAM. Version 5 is the current line; loading scripts and `trust_remote_code` remain removed, `ds["col"]` still returns a lazy `Column`, `Sequence` remains replaced by `List`, and audio/video decoding goes through torchcodec [4][5].

**Loading — all sources**
```python
from datasets import load_dataset, Dataset, load_from_disk

load_dataset("stanfordnlp/imdb")                  # from Hub
load_dataset("stanfordnlp/imdb", split="train")   # a specific split
load_dataset("stanfordnlp/imdb", split="train[:10%]")  # split slice
load_dataset("csv", data_files="data.csv")        # local CSV
load_dataset("json", data_files="data.jsonl")     # local JSONL
load_dataset("parquet", data_files="data.parquet")  # local Parquet
load_dataset("text", data_files="notes.txt")      # plain text, one row per line
load_dataset("csv", data_files={"train": "tr.csv", "test": "te.csv"})  # files mapped to splits
Dataset.from_dict({...})                          # from Python dict
Dataset.from_pandas(df)                           # from Pandas
Dataset.from_list([...])                          # from list of dicts
Dataset.from_generator(gen_fn)                    # from a generator
load_from_disk("./saved")                         # from disk (save_to_disk output)
```

Loading with **configs and subsets** is the same call with a second argument. If you see "Dataset scripts are no longer supported", look for a Parquet version of the dataset on the Hub or convert it yourself — do not pass `trust_remote_code`.

**Inspecting**
```python
ds.features         # schema
ds.column_names
ds.shape, ds.num_rows, ds.num_columns
ds[0]               # first row as dict
ds[-1]              # last row
ds[:5]              # slice of rows (dict of lists)
ds["text"]          # lazy Column object, not a list
list(ds["text"])    # materialize the full column as a list
ds[0]["text"]       # one value
```

**Row operations**
```python
ds.sort("label")
ds.sort("label", reverse=True)
ds.shuffle(seed=42)
ds.select([0, 1, 2])
ds.filter(lambda x: x["label"] == 1)
ds.filter(lambda x, idx: idx % 2 == 0, with_indices=True)
ds.skip(1000).take(10)
ds.train_test_split(test_size=0.2, seed=42)
ds.train_test_split(test_size=0.2, stratify_by_column="label")
ds.shard(num_shards=4, index=0)
```

**Column operations**
```python
from datasets import Value, ClassLabel, List

ds.rename_column("text", "review")
ds.rename_columns({"text": "review", "label": "target"})
ds.remove_columns(["label"])
ds.add_column("length", values)
ds.cast_column("label", Value("float32"))
ds.flatten()
ds.select_columns(["text", "label"])
ds.align_labels_with_mapping(label2id, "label")
```

Schema types to know: `Value`, `ClassLabel`, `List` (replaces the old `Sequence`), `Image`, `Audio`.

**Transformation**
```python
ds.map(fn)                                  # row by row
ds.map(fn, batched=True)                    # batch processing (much faster)
ds.map(fn, batched=True, batch_size=1000)
ds.map(fn, num_proc=4)                      # multiprocessing
ds.map(fn, remove_columns=["text"])         # map and drop columns
ds.map(fn, with_indices=True)
ds.with_transform(fn)                       # on-the-fly, applied when rows are accessed
ds.with_format("torch")                     # non-destructive format change
```

A batched `map` can return more rows than it received (generate several rows from one), as long as every column grows consistently.

**Combining**
```python
from datasets import concatenate_datasets, interleave_datasets

concatenate_datasets([ds1, ds2])
interleave_datasets([ds1, ds2], probabilities=[0.7, 0.3], seed=42)
```

**Export and format**
```python
ds.to_pandas()
ds.to_dict()
ds.to_list()
ds.to_csv("out.csv")
ds.to_json("out.jsonl")
ds.to_parquet("out.parquet")
ds.to_polars()          # Arrow -> Polars without a pandas round-trip
ds.to_iterable_dataset()  # material dataset -> stream
ds.data                 # the underlying Arrow table
ds.set_format("torch", columns=["input_ids", "label"])
ds.set_format("numpy")
ds.set_format("pandas")
ds.reset_format()
```

**Save and push**
```python
ds.save_to_disk("./my_dataset")
load_from_disk("./my_dataset")
ds.push_to_hub("username/my-dataset")
ds.push_to_hub("username/my-dataset", private=True)
ds.push_to_hub("username/my-dataset", num_proc=8)   # faster upload
dataset_dict.push_to_hub("username/my-dataset")     # DatasetDict: all splits at once
```

Also learn: dataset cards and README metadata for your pushed datasets, pushing a specific split, and private versus public datasets.

**Streaming (large datasets)**
```python
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
for example in ds:
    print(example["text"])
list(ds.take(10))
ds.skip(1000).take(10)
ds.map(fn)            # lazy, runs as you iterate
ds.filter(fn)
ds.shuffle(seed=42, buffer_size=10_000)
```

Streaming returns an `IterableDataset` supporting `map`, `filter`, `shuffle`, `take`, `skip`, and `interleave_datasets`. To keep a subset, build a regular `Dataset` from it with `Dataset.from_generator`.

> [!warning] datasets 5.x shuffle default changed
> Streaming `shuffle()` now defaults to `buffer_size=1000` and `max_buffer_input_shards=10`. A training run that "shuffles differently" after the upgrade is usually this, not a seeding bug. Pass explicit `buffer_size` if reproducibility matters.

**Agent traces.** datasets 5.x adds first-class loading of agent traces (tool-call conversations harvested from coding agents) into a normalized `messages` schema, built on the `teich` package pinned at 0.1.5 in the 5.1.0 metadata. This is the data substrate for agent-behavior SFT and is the natural bridge between Stage 3 and Stage 5 [5].

**Python dict deep dive**
```python
from collections import Counter

ds[0]                   # first row as dict
ds[:5]                  # dict of lists
ds.features.keys()      # column names
ds.features.items()     # name + type pairs
ds.to_dict()            # full dict
ds.to_list()            # list of row dicts
"text" in ds.features   # check column exists
Counter(ds["label"])    # count values
[row for row in ds if row["label"] == 1]   # comprehension
```

**Key install**
```bash
pip install -U datasets
```

---

---

## Worksheet additions

## Scope
Stage 0 is the floor everything else stands on. It covers the two libraries that define how models and data are named, versioned, moved, and found. Nothing in Stages 1–8 makes sense if the Hub client is wrong, because every model reference in every later stage is a Hub path.

**Exit criterion:** you can create a dataset repo, publish a versioned release, restore it on a clean machine, and explain every token in `hf auth whoami` output.

## Environment
```bash
python -m venv .venv && source .venv/bin/activate   # Python >= 3.11 for the full stack
pip install -U huggingface_hub datasets accelerate
pip list | grep -i -E "huggingface|datasets|accelerate"
```

`huggingface_hub` 2.4.0 requires Python ≥3.10 and pulls `huggingface_hub<3.0,>=1.31.0` into `transformers` and `datasets`, so pin the hub explicitly if you hit an httpx2 stack trace in an old environment.

---

### Authentication
- [ ] `login()` — programmatic
- [ ] `notebook_login()` — Jupyter widget
- [ ] `interpreter_login()` — force a terminal prompt (use this in CI or containers)
- [ ] Environment variable `HF_TOKEN` — the only method that works in non-interactive jobs
- [ ] CLI: `hf auth login`, `hf auth whoami`
- [ ] Switching between multiple saved tokens (`login(token=...)` with different profiles)
- [ ] `logout()` and `whoami()`
- [ ] Fine-grained tokens versus read/write tokens — scope each project to the least access it needs
- [ ] Reading the token type list programmatically (`HfApi().whoami()`)

### Downloading
- [ ] `hf_hub_download()` for a single file
- [ ] `snapshot_download()` for an entire repo
- [ ] Specific revisions and branches (`revision="main"`, a commit SHA, a tag)
- [ ] `allow_patterns` and `ignore_patterns` filters
- [ ] Gated model access and `GatedRepoError` handling
- [ ] Offline mode with cached files (`HF_HUB_OFFLINE=1`)
- [ ] CLI: `hf download`
- [ ] Cache location controlled by `HF_HOME`
- [ ] Fast transfers through `hf_xet` automatically (the old `hf_transfer` is gone)

### Uploading
- [ ] `upload_file()` for a single file
- [ ] `upload_folder()` for a directory
- [ ] `CommitOperationAdd` for atomic multi-file commits
- [ ] `create_commit()` for full commit control
- [ ] Ignoring files during upload (`ignore_patterns`)
- [ ] CLI: `hf upload` (v2 renamed `hf upload-large-folder`; the storage flag is now `--volume`)

### Repository management
- [ ] `create_repo()` for model, dataset, and space types
- [ ] `delete_repo()`
- [ ] Listing/searching: `list_models()` → `search_models()`, `list_datasets()`, `list_spaces()`
- [ ] `model_info()`, `dataset_info()`
- [ ] `repo_exists()`
- [ ] `duplicate_space()` → `duplicate_repo()`

### Spaces management (API only)
- [ ] `get_space_runtime()`
- [ ] `pause_space()`, `restart_space()`
- [ ] Space hardware management through `huggingface_hub`

### Error handling
- [ ] Catch `HfHubHTTPError` and subclasses: `GatedRepoError`, `RepositoryNotFoundError`, `RevisionNotFoundError`
- [ ] Never catch `requests` exceptions — the library moved to httpx and your handler will never fire

### CLI v2 surface
- [ ] `hf auth login` / `hf auth whoami`
- [ ] `hf download <repo> --revision <sha> --include "*.safetensors"`
- [ ] `hf upload <repo> <path> --exclude "*.pyc"`
- [ ] `hf repos create` (v2 pluralized `hf repo`)
- [ ] `hf repos delete-files` (v2 renamed from `hf repo-files delete`)
- [ ] `hf jobs` listing with `--status` / `--label` filters (v2 replaced `hf jobs ps`)
- [ ] `hf skills list` / `hf skills add <name>` / `hf skills update -g`

---

### Atomic multi-file release
A release is one commit containing several files. If you upload file-by-file, a reader can observe a half-written repo.

```python
from huggingface_hub import HfApi, CommitOperationAdd

api = HfApi()
api.create_repo(repo_id="username/my-model", repo_type="model", private=False)

ops = [
    CommitOperationAdd(path_in_repo="model.safetensors", path_or_fileobj="./model/model.safetensors"),
    CommitOperationAdd(path_in_repo="config.json",        path_or_fileobj="./model/config.json"),
    CommitOperationAdd(path_in_repo="tokenizer.json",    path_or_fileobj="./model/tokenizer.json"),
    CommitOperationAdd(path_in_repo="README.md",         path_or_fileobj="./model/README.md"),
]

commit = api.create_commit(
    repo_id="username/my-model",
    repo_type="model",
    operations=ops,
    commit_message="Release v1.0.0: base weights + tokenizer + card",
)
print(commit.oid)
```

### Gated repos, handled properly
```python
from huggingface_hub import hf_hub_download, GatedRepoError, HfHubHTTPError

try:
    path = hf_hub_download(repo_id="Qwen/Qwen3-1.7B",
                           filename="config.json")
except GatedRepoError:
    print("Accept the licence on the Hub page, then use a token with access.")
except HfHubHTTPError as e:
    print(f"Hub error {e.response.status_code}: {e.response.text[:200]}")
```

### Snapshot with filters and an explicit revision
```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    revision="main",                       # or a commit SHA for reproducibility
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
    ignore_patterns=["*consolidated*", "original/*"],
    local_dir_use_symlinks=False,
)
```

### Space lifecycle
```python
from huggingface_hub import HfApi

api = HfApi()
print(api.get_space_runtime("username/my-space"))
api.pause_space("username/my-space")
api.restart_space("username/my-space")
```

### Token hygiene
```python
import os
from huggingface_hub import login, whoami, logout

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
print(whoami())
logout()
```

**Never** commit `HF_TOKEN`. Keep it in `.env`, list `.env` in `.gitignore`, and rotate anything that ever touched a commit.

---

### Loading — all sources
- [ ] `load_dataset()` from the Hub
- [ ] Loading with configs and subsets (`load_dataset("repo", "config_name")`)
- [ ] Loading specific splits
- [ ] Loading split slices — first 10%, row ranges, percentages
- [ ] Loading from local CSV
- [ ] Loading from local JSON and JSONL
- [ ] Loading from Parquet
- [ ] Loading from text files
- [ ] Loading from multiple files mapped to splits
- [ ] `Dataset.from_dict()`, `from_pandas()`, `from_list()`, `from_generator()`
- [ ] `load_from_disk()`
- [ ] Why dataset scripts no longer load, and how to find or make a Parquet version
- [ ] **Agent traces**: loading tool-call traces from datasets 5.x into a normalized `messages` schema

### Inspecting
- [ ] `features`, `column_names`, `shape`, `num_rows`, `num_columns`
- [ ] Accessing rows by index, by slice, and by column name
- [ ] Materializing a lazy `Column` with `list()`
- [ ] Iterating over rows

### Row operations
- [ ] `sort()` ascending and descending
- [ ] `shuffle()` with a seed for reproducibility
- [ ] `select()` by index list
- [ ] `filter()`, including `with_indices=True`
- [ ] `skip()` and `take()`
- [ ] `train_test_split()` by proportion or absolute count, and stratified by column
- [ ] `shard()` into N chunks

### Column operations
- [ ] `rename_column()`, `rename_columns()`
- [ ] `remove_columns()`, `add_column()`
- [ ] `cast_column()`
- [ ] `flatten()`
- [ ] `select_columns()`
- [ ] Feature schema with `Value`, `ClassLabel`, `List`, `Image`, `Audio`
- [ ] `align_labels_with_mapping()`

### Transformation
- [ ] `map()` row by row, `batched=True`, `num_proc`, `remove_columns`, adding columns, generating multiple rows from one
- [ ] `with_transform()` and `with_format()`

### Combining, exporting, formatting
- [ ] `concatenate_datasets()` — stack vertically
- [ ] `interleave_datasets()` with probabilities and seed
- [ ] `to_pandas()`, `to_dict()`, `to_list()`, `to_csv()`, `to_json()`, `to_parquet()`
- [ ] `to_polars()` — Arrow to Polars without a pandas round-trip
- [ ] `to_iterable_dataset()` — turn a materialized dataset into a stream
- [ ] Reading the underlying Arrow table through `ds.data`
- [ ] `set_format("torch" | "numpy" | "pandas")` and `reset_format()`

### Saving and pushing
- [ ] `save_to_disk()` / `load_from_disk()`, including `DatasetDict`
- [ ] `push_to_hub()` for a dataset and for a specific split
- [ ] `DatasetDict.push_to_hub()` for all splits at once
- [ ] `push_to_hub(num_proc=...)` for faster uploads
- [ ] Private versus public datasets
- [ ] Dataset cards and README metadata

### Streaming
- [ ] `streaming=True` without downloading the whole dataset
- [ ] `map()`, `filter()`, `shuffle(buffer_size=...)` on streams
- [ ] `take()` and `skip()` on streams
- [ ] `interleave_datasets()` with streams
- [ ] Building a regular `Dataset` from a stream with `from_generator()`

---

## Code — dataset operations
```python
from datasets import load_dataset, Dataset, concatenate_datasets, interleave_datasets

# Every source in one place
hub   = load_dataset("stanfordnlp/imdb")
csv   = load_dataset("csv", data_files="data.csv")
jsonl = load_dataset("json", data_files="data.jsonl")
pq    = load_dataset("parquet", data_files="data.parquet")
txt   = load_dataset("text", data_files="notes.txt")
split_map = load_dataset("csv", data_files={"train": "tr.csv", "test": "te.csv"})

# Splitting and slicing
train = hub["train"].train_test_split(test_size=0.2, seed=42)
tiny  = hub["train"].select(range(1000))
strat = hub["train"].train_test_split(test_size=0.2, stratify_by_column="label")

# Row and column surgery
clean = hub["train"].filter(lambda x: len(x["text"]) > 50).shuffle(seed=42)
clean = clean.rename_column("text", "body").remove_columns(["label"])

# Batched transform that adds a column
def add_len(batch):
    batch["length"] = [len(t) for t in batch["text"]]
    return batch

clean = clean.map(add_len, batched=True, batch_size=1000)

# Combining
both  = concatenate_datasets([hub["train"], hub["test"]])
mixed = interleave_datasets([hub["train"], hub["test"]], probabilities=[0.7, 0.3], seed=42)

# Exports
mixed.to_parquet("out.parquet")
mixed.to_polars()                 # Arrow -> Polars
stream = mixed.to_iterable_dataset()
print(stream.data.num_rows if hasattr(stream, "data") else "stream")
```

### Streaming with explicit buffers
```python
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

for ex in ds.skip(1000).take(10):
    print(ex["text"][:80])

# datasets 5.x changed the streaming shuffle defaults:
# buffer_size=1000, max_buffer_input_shards=10. Set them explicitly when
# reproducibility matters, because two runs with different buffer sizes
# produce different orderings and different loss curves.
ds = ds.shuffle(seed=42, buffer_size=10_000)
```

### Agent traces
```python
# datasets 5.x normalizes agent tool-call traces into a `messages` schema.
# The loader is built on the `teich` package (0.1.5 in the 5.1.0 metadata).
from datasets import load_dataset

traces = load_dataset("trace-commons/agent-traces", split="train")
print(traces.features)
# messages: [{'role': ..., 'content': ...}, ...] plus tool-call metadata
```

Audit the harness terms before training on donated traces: the data may be anonymized and public-source, but the collection terms still govern use.

---

### Lab 0-1 — Authentication matrix (0.5 day)
- Log in five different ways and record what each produces.
- Create a read-only token and a read-write token; prove the read-only one fails on `upload_file`.
- Implement `interpreter_login()` fallback in a container without a browser.
- Verify `HF_TOKEN` works with no interactive input.

### Lab 0-2 — Gated and private repos (0.5 day)
- Attempt to download a gated model without access; catch `GatedRepoError`.
- Accept the licence, then download with a token that has the grant.
- Create a private repo, upload, and verify it is invisible to an unauthenticated client.

### Lab 0-3 — Atomic release (1 day)
- Build a small model directory (config, weights, tokenizer, README).
- Push it with a single `create_commit` containing four `CommitOperationAdd` operations.
- Clone it back with `snapshot_download` and diff the tree.
- Tag a release and pin `revision=` to the tag.

### Lab 0-4 — Space lifecycle (0.5 day)
- Deploy a Gradio Space, then pause, restart, and read `get_space_runtime()`.
- Change hardware programmatically through `huggingface_hub`.
- Record what the runtime object reports about state and hardware.

### Lab 0-5 — Datasets operation sweep (1 day)
- Load the same content from CSV, JSONL, Parquet, and a text file; compare `features`.
- Run every row operation and every column operation from the checklist.
- Convert to Polars and back; confirm `ds.data` gives you the Arrow table.
- Stream a multi-shard dataset, shuffle with an explicit buffer, and materialize a subset.
- Load an agent-trace dataset and inspect the `messages` schema.

### Lab 0-6 — Hub Skills (0.5 day)
- `hf skills add hf-cli --global`, then `hf skills list`.
- Have a coding agent perform a Hub task using the skill, and check it used current CLI syntax rather than `huggingface-cli`.
- Run `hf skills update -g` and confirm the version stamp moves.

---

## Pitfalls
- **`huggingface-cli` is gone.** Any tutorial that starts with it is pre-2025. Use `hf`.
- **`hf repo` is now `hf repos`**, and `hf repo-files delete` is `hf repos delete-files`. Silent "unknown command" errors are renames, not a broken install.
- **`trust_remote_code` is a red flag on any repo you did not write.** It executes code from the remote repository.
- **Streaming shuffle defaults changed in datasets 5.x.** Reproducibility issues after the upgrade are usually the buffer size.
- **`ds["col"]` is a lazy `Column`.** Code that expects a list breaks silently; wrap with `list()`.
- **The `datasets` upper bound on the hub is `<3.0`.** Do not let pip install a hub 3 prerelease under your feet.
- **A gated repo error is not a 404.** Handle `GatedRepoError` separately, because the remedy (accepting the licence) is completely different from a typo'd repo id.
- **Tokens leak through notebooks.** `whoami()` output, shell history, and `.ipynb_checkpoints` are all common leaks.

---

## Acceptance gate
- [ ] Every authentication method from the checklist works and is documented.
- [ ] A multi-file release lands in one commit and restores byte-identically on a clean machine.
- [ ] A gated repo is downloaded successfully after the licence is accepted.
- [ ] Every row, column, transformation, combining, export, and streaming operation in the checklist has been executed at least once.
- [ ] A dataset with a data card is public on the Hub, versioned, and re-loadable by `load_dataset` from a fresh environment.
- [ ] `hf skills list` shows the CLI skill and it is current.

---

## Resources
- Hub CLI guide — https://huggingface.co/docs/hub/en/guides/cli
- Hub Agent Skills — https://huggingface.co/docs/hub/en/agents-skills
- CLI for AI Agents — https://huggingface.co/docs/hub/agents-cli.md
- Skills repository — https://github.com/huggingface/skills
- `huggingface_hub` on PyPI — https://pypi.org/project/huggingface_hub/
- `datasets` on PyPI — https://pypi.org/project/datasets/

**Next:** Stage 02 — Core ML Stack.
