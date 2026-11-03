## **Stage 00: Platform Foundation**

**Outcome:** authenticate, move artifacts in both directions, and own a dataset end to end without touching a GPU.

**Scope:** Stage 00 is the floor everything else stands on. It covers the two libraries that define how models and data are named, versioned, moved, and found: `huggingface_hub` and `datasets`. Nothing in the later stages makes sense if the Hub client is wrong, because every model reference in every later stage is a Hub path.

**Exit criterion:** you can create a dataset repo, publish a versioned release, restore it on a clean machine, and explain every token in the `hf auth whoami` output.

### **0.1 Environment**

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11 or newer for the full stack
pip install -U huggingface_hub datasets accelerate
pip list | grep -i -E "huggingface|datasets|accelerate"
```

`huggingface_hub` requires Python 3.10 or newer. If you hit an `httpx` stack trace in an old environment, pin the hub package explicitly.

### **0.2 HuggingFace Hub and `huggingface_hub`**

The Hub is the GitHub of ML: models, datasets, Spaces, and Jobs. `huggingface_hub` is the Python and CLI client. The current major release moved the HTTP stack to `httpx`, and renamed a batch of CLI subcommands and Python functions.

**Authentication**

- `login()`, `notebook_login()` (Jupyter widget), `interpreter_login()` (forces a terminal prompt, use it in CI or containers)
- Environment variable `HF_TOKEN`, the only method that works in non-interactive jobs
- CLI: `hf auth login`, `hf auth whoami`
- Switching between multiple saved tokens (`login(token=...)` with different profiles)
- `logout()` and `whoami()`
- Fine-grained tokens versus read and write tokens: give each project the least access it needs
- Reading the token type list programmatically with `HfApi().whoami()`

**Downloading**

- `hf_hub_download()` for a single file
- `snapshot_download()` for an entire repo
- Specific revisions and branches (`revision="main"`, a commit SHA, or a tag)
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
- Ignoring files during upload (`ignore_patterns`)
- CLI: `hf upload`

**Repository management**

- `create_repo()` for model, dataset, and space types
- `delete_repo()`
- `search_models()`, `search_datasets()`, `search_spaces()` (renamed from `list_*`)
- `model_info()`, `dataset_info()`
- `repo_exists()`
- `duplicate_repo()` (renamed from `duplicate_space()`)

**Spaces management (API only)**

- `get_space_runtime()`
- `pause_space()`, `restart_space()`
- Space hardware management through `huggingface_hub`

**Error handling**

- Catch `HfHubHTTPError` and its subclasses: `GatedRepoError`, `RepositoryNotFoundError`, `RevisionNotFoundError`
- Never catch `requests` exceptions. The library no longer uses `requests`, so those handlers will never fire.

**CLI rename table**

| Old name | New name | Note |
|---|---|---|
| `hf repo` | `hf repos` | pluralized |
| `hf repo-files delete` | `hf repos delete-files` | |
| `hf upload-large-folder` | `hf upload` | storage flag is now `--volume` |
| `hf jobs ps` | `hf jobs --status` / `--label` | filters replace positional listing |
| `list_models()` | `search_models()` | |
| `upload_large_folder()` | `upload_folder()` | |
| `duplicate_space()` | `duplicate_repo()` | |
| `repo_type_and_id_from_hf_id()` | `parse_hf_uri()` | |
| `huggingface-cli` | `hf` | the old entry point is gone entirely |
| `from huggingface_hub.utils import httpx` | no replacement | compatibility shim for the `httpx` move |

**CLI v2 surface**

- `hf auth login` and `hf auth whoami`
- `hf download <repo> --revision <sha> --include "*.safetensors"`
- `hf upload <repo> <path> --exclude "*.pyc"`
- `hf repos create`
- `hf repos delete-files`
- `hf jobs` listing with `--status` and `--label` filters
- `hf skills list`, `hf skills add <name>`, `hf skills update -g`

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
hf upload tanvi/my-model ./my-model
```

**Agent Skills for the CLI**

The Hub ships a set of Agent Skills that teach a coding agent the `hf` command surface: `hf-cli` (generated from your locally installed CLI, so it tracks your version), `huggingface-datasets`, `huggingface-llm-trainer` (TRL SFT, DPO, GRPO on HF Jobs), `huggingface-community-evals`, `huggingface-vision-trainer`, `huggingface-trackio`, `gradio`, and `transformers-js`.

They install with `hf skills add <name>` (add `--global` for all projects) and are refreshed with `hf skills update -g`. `hf update` refreshes the globally installed skill. They follow the Agent Skills spec and work with Claude Code, Codex, Gemini CLI, and Cursor, with `AGENTS.md` as the fallback for clients that do not support skills.

```bash
hf skills add hf-cli --global
hf skills list
hf skills update -g
```

### **0.3 Hub Code Patterns**

**Atomic multi-file release**

A release is one commit containing several files. If you upload file by file, a reader can observe a half-written repo.

```python
from huggingface_hub import HfApi, CommitOperationAdd

api = HfApi()
api.create_repo(repo_id="rohit/my-model", repo_type="model", private=False)

ops = [
    CommitOperationAdd(path_in_repo="model.safetensors", path_or_fileobj="./model/model.safetensors"),
    CommitOperationAdd(path_in_repo="config.json",       path_or_fileobj="./model/config.json"),
    CommitOperationAdd(path_in_repo="tokenizer.json",    path_or_fileobj="./model/tokenizer.json"),
    CommitOperationAdd(path_in_repo="README.md",         path_or_fileobj="./model/README.md"),
]

commit = api.create_commit(
    repo_id="rohit/my-model",
    repo_type="model",
    operations=ops,
    commit_message="Release v1.0.0: base weights, tokenizer, and card",
)
print(commit.oid)
```

**Gated repos, handled properly**

```python
from huggingface_hub import hf_hub_download, GatedRepoError, HfHubHTTPError

try:
    path = hf_hub_download(repo_id="Qwen/Qwen3-1.7B", filename="config.json")
except GatedRepoError:
    print("Accept the licence on the Hub page, then use a token with access.")
except HfHubHTTPError as e:
    print(f"Hub error {e.response.status_code}: {e.response.text[:200]}")
```

**Snapshot with filters and an explicit revision**

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

**Space lifecycle**

```python
from huggingface_hub import HfApi

api = HfApi()
print(api.get_space_runtime("meera/my-space"))
api.pause_space("meera/my-space")
api.restart_space("meera/my-space")
```

**Token hygiene**

```python
import os
from huggingface_hub import login, whoami, logout

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
print(whoami())
logout()
```

Never commit `HF_TOKEN`. Keep it in `.env`, list `.env` in `.gitignore`, and rotate anything that ever touched a commit.

### **0.4 Datasets**

The data layer. It is built on Apache Arrow, so large datasets are memory-mapped from disk instead of loaded into RAM. Loading scripts and `trust_remote_code` are removed, `ds["col"]` returns a lazy `Column`, `Sequence` is replaced by `List`, and audio and video decoding goes through `torchcodec`.

**Loading from all sources**

```python
from datasets import load_dataset, Dataset, load_from_disk

load_dataset("stanfordnlp/imdb")                          # from Hub
load_dataset("stanfordnlp/imdb", split="train")           # a specific split
load_dataset("stanfordnlp/imdb", split="train[:10%]")     # split slice
load_dataset("csv", data_files="data.csv")                # local CSV
load_dataset("json", data_files="data.jsonl")             # local JSONL
load_dataset("parquet", data_files="data.parquet")        # local Parquet
load_dataset("text", data_files="notes.txt")              # plain text, one row per line
load_dataset("csv", data_files={"train": "tr.csv", "test": "te.csv"})  # files mapped to splits
Dataset.from_dict({...})                                  # from Python dict
Dataset.from_pandas(df)                                   # from Pandas
Dataset.from_list([...])                                  # from list of dicts
Dataset.from_generator(gen_fn)                            # from a generator
load_from_disk("./saved")                                 # from disk (save_to_disk output)
```

Loading with configs and subsets uses the same call with a second argument. If you see "Dataset scripts are no longer supported", look for a Parquet version of the dataset on the Hub or convert it yourself. Do not pass `trust_remote_code`.

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
ds.with_transform(fn)                       # on the fly, applied when rows are accessed
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
ds.to_polars()            # Arrow to Polars without a pandas round trip
ds.to_iterable_dataset()  # materialized dataset to stream
ds.data                   # the underlying Arrow table
ds.set_format("torch", columns=["input_ids", "label"])
ds.set_format("numpy")
ds.set_format("pandas")
ds.reset_format()
```

**Save and push**

```python
ds.save_to_disk("./my_dataset")
load_from_disk("./my_dataset")
ds.push_to_hub("karan/my-dataset")
ds.push_to_hub("karan/my-dataset", private=True)
ds.push_to_hub("karan/my-dataset", num_proc=8)   # faster upload
dataset_dict.push_to_hub("karan/my-dataset")     # DatasetDict: all splits at once
```

Also learn dataset cards and README metadata for your pushed datasets, pushing a specific split, and private versus public datasets.

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

Streaming shuffle defaults changed in recent releases: `shuffle()` now defaults to `buffer_size=1000` and `max_buffer_input_shards=10`. A training run that shuffles differently after an upgrade is usually this, not a seeding bug. Pass an explicit `buffer_size` if reproducibility matters.

**Agent traces**

Recent `datasets` releases add first-class loading of agent traces (tool-call conversations harvested from coding agents) into a normalized `messages` schema. This is the data substrate for agent-behavior SFT and is the natural bridge between Stage 3 and Stage 5.

```python
from datasets import load_dataset

traces = load_dataset("trace-commons/agent-traces", split="train")
print(traces.features)
# messages: [{'role': ..., 'content': ...}, ...] plus tool-call metadata
```

Audit the harness terms before training on donated traces: the data may be anonymized and public-source, but the collection terms still govern use.

**Python dict deep dive**

```python
from collections import Counter

ds[0]                   # first row as dict
ds[:5]                  # dict of lists
ds.features.keys()      # column names
ds.features.items()     # name and type pairs
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

### **0.5 Combined Dataset Example**

```python
from datasets import load_dataset, Dataset, concatenate_datasets, interleave_datasets

# Every source in one place
hub       = load_dataset("stanfordnlp/imdb")
csv       = load_dataset("csv", data_files="data.csv")
jsonl     = load_dataset("json", data_files="data.jsonl")
pq        = load_dataset("parquet", data_files="data.parquet")
txt       = load_dataset("text", data_files="notes.txt")
split_map = load_dataset("csv", data_files={"train": "tr.csv", "test": "te.csv"})

# Splitting and slicing
train = hub["train"].train_test_split(test_size=0.2, seed=42)
tiny  = hub["train"].select(range(1000))
strat = hub["train"].train_test_split(test_size=0.2, stratify_by_column="label")

# Row and column changes
clean = hub["train"].filter(lambda x: len(x["text"]) > 50).shuffle(seed=42)
clean = clean.rename_column("text", "body").remove_columns(["label"])

# Batched transform that adds a column
def add_len(batch):
    batch["length"] = [len(t) for t in batch["body"]]
    return batch

clean = clean.map(add_len, batched=True, batch_size=1000)

# Combining
both  = concatenate_datasets([hub["train"], hub["test"]])
mixed = interleave_datasets([hub["train"], hub["test"]], probabilities=[0.7, 0.3], seed=42)

# Exports
mixed.to_parquet("out.parquet")
mixed.to_polars()
stream = mixed.to_iterable_dataset()
```

### **0.6 Checklist**

**Authentication**

- `login()` programmatic
- `notebook_login()` Jupyter widget
- `interpreter_login()` terminal prompt for CI or containers
- `HF_TOKEN` environment variable
- `hf auth login` and `hf auth whoami`
- Switching between multiple saved tokens
- `logout()` and `whoami()`
- Fine-grained tokens versus read and write tokens
- Reading the token type list with `HfApi().whoami()`

**Downloading**

- `hf_hub_download()` and `snapshot_download()`
- Revisions: branch, commit SHA, tag
- `allow_patterns` and `ignore_patterns`
- Gated model access and `GatedRepoError`
- Offline mode with `HF_HUB_OFFLINE=1`
- `hf download`
- Cache location with `HF_HOME`
- Fast transfers through `hf_xet`

**Uploading**

- `upload_file()` and `upload_folder()`
- `CommitOperationAdd` and `create_commit()`
- `ignore_patterns` during upload
- `hf upload`

**Repository management**

- `create_repo()` for model, dataset, and space types
- `delete_repo()`
- `search_models()`, `search_datasets()`, `search_spaces()`
- `model_info()`, `dataset_info()`, `repo_exists()`
- `duplicate_repo()`

**Spaces management**

- `get_space_runtime()`
- `pause_space()` and `restart_space()`
- Space hardware management

**Error handling**

- `HfHubHTTPError`, `GatedRepoError`, `RepositoryNotFoundError`, `RevisionNotFoundError`
- No `requests` exception handlers

**Datasets: loading**

- `load_dataset()` from the Hub, with configs and subsets
- Specific splits and split slices (first 10%, row ranges, percentages)
- Local CSV, JSON and JSONL, Parquet, and text files
- Multiple files mapped to splits
- `Dataset.from_dict()`, `from_pandas()`, `from_list()`, `from_generator()`
- `load_from_disk()`
- Why dataset scripts no longer load, and how to find or make a Parquet version
- Agent traces in the normalized `messages` schema

**Datasets: inspecting**

- `features`, `column_names`, `shape`, `num_rows`, `num_columns`
- Access by index, by slice, and by column name
- Materializing a lazy `Column` with `list()`
- Iterating over rows

**Datasets: row operations**

- `sort()` ascending and descending
- `shuffle()` with a seed
- `select()` by index list
- `filter()`, including `with_indices=True`
- `skip()` and `take()`
- `train_test_split()` by proportion, by absolute count, and stratified by column
- `shard()` into N chunks

**Datasets: column operations**

- `rename_column()` and `rename_columns()`
- `remove_columns()` and `add_column()`
- `cast_column()`
- `flatten()`
- `select_columns()`
- Feature schema with `Value`, `ClassLabel`, `List`, `Image`, `Audio`
- `align_labels_with_mapping()`

**Datasets: transformation**

- `map()` row by row, `batched=True`, `num_proc`, `remove_columns`, adding columns, generating multiple rows from one
- `with_transform()` and `with_format()`

**Datasets: combining, exporting, formatting**

- `concatenate_datasets()` to stack vertically
- `interleave_datasets()` with probabilities and seed
- `to_pandas()`, `to_dict()`, `to_list()`, `to_csv()`, `to_json()`, `to_parquet()`
- `to_polars()`
- `to_iterable_dataset()`
- Reading the Arrow table through `ds.data`
- `set_format("torch" | "numpy" | "pandas")` and `reset_format()`

**Datasets: saving and pushing**

- `save_to_disk()` and `load_from_disk()`, including `DatasetDict`
- `push_to_hub()` for a dataset and for a specific split
- `DatasetDict.push_to_hub()` for all splits at once
- `push_to_hub(num_proc=...)` for faster uploads
- Private versus public datasets
- Dataset cards and README metadata

**Datasets: streaming**

- `streaming=True` without downloading the whole dataset
- `map()`, `filter()`, `shuffle(buffer_size=...)` on streams
- `take()` and `skip()` on streams
- `interleave_datasets()` with streams
- Building a regular `Dataset` from a stream with `from_generator()`

### **0.7 Labs**

**Lab 0-1: Authentication matrix**

- Log in five different ways and record what each produces.
- Create a read-only token and a read-write token, and prove the read-only one fails on `upload_file`.
- Implement an `interpreter_login()` fallback in a container without a browser.
- Verify `HF_TOKEN` works with no interactive input.

**Lab 0-2: Gated and private repos**

- Attempt to download a gated model without access and catch `GatedRepoError`.
- Accept the licence, then download with a token that has the grant.
- Create a private repo, upload, and verify it is invisible to an unauthenticated client.

**Lab 0-3: Atomic release**

- Build a small model directory (config, weights, tokenizer, README).
- Push it with a single `create_commit` containing four `CommitOperationAdd` operations.
- Clone it back with `snapshot_download` and diff the tree.
- Tag a release and pin `revision=` to the tag.

**Lab 0-4: Space lifecycle**

- Deploy a Gradio Space, then pause it, restart it, and read `get_space_runtime()`.
- Change hardware programmatically through `huggingface_hub`.
- Record what the runtime object reports about state and hardware.

**Lab 0-5: Datasets operation sweep**

- Load the same content from CSV, JSONL, Parquet, and a text file, and compare `features`.
- Run every row operation and every column operation from the checklist.
- Convert to Polars and back, and confirm `ds.data` gives you the Arrow table.
- Stream a multi-shard dataset, shuffle with an explicit buffer, and materialize a subset.
- Load an agent-trace dataset and inspect the `messages` schema.

**Lab 0-6: Hub Skills**

- Run `hf skills add hf-cli --global`, then `hf skills list`.
- Have a coding agent perform a Hub task using the skill, and check it used current CLI syntax rather than `huggingface-cli`.
- Run `hf skills update -g` and confirm the version stamp moves.

### **0.8 Pitfalls**

- **`huggingface-cli` is gone.** Any tutorial that starts with it is outdated. Use `hf`.
- **`hf repo` is now `hf repos`**, and `hf repo-files delete` is now `hf repos delete-files`. An "unknown command" error is a rename, not a broken install.
- **`trust_remote_code` is a red flag on any repo you did not write.** It executes code from the remote repository.
- **Streaming shuffle defaults changed.** Reproducibility issues after an upgrade are usually the buffer size.
- **`ds["col"]` is a lazy `Column`.** Code that expects a list breaks silently, so wrap it with `list()`.
- **Keep the hub package below the next major release.** Do not let pip install a prerelease of a new major version under your feet.
- **A gated repo error is not a 404.** Handle `GatedRepoError` separately, because the remedy (accepting the licence) is completely different from fixing a mistyped repo id.
- **Tokens leak through notebooks.** `whoami()` output, shell history, and `.ipynb_checkpoints` are all common leaks.

### **0.9 Acceptance Gate**

- Every authentication method from the checklist works and is documented.
- A multi-file release lands in one commit and restores byte-identically on a clean machine.
- A gated repo is downloaded successfully after the licence is accepted.
- Every row, column, transformation, combining, export, and streaming operation in the checklist has been executed at least once.
- A dataset with a data card is public on the Hub, versioned, and reloadable by `load_dataset` from a fresh environment.
- `hf skills list` shows the CLI skill and it is current.

### **0.10 Resources**

- Hub CLI guide: https://huggingface.co/docs/hub/en/guides/cli
- Hub Agent Skills: https://huggingface.co/docs/hub/en/agents-skills
- CLI for AI Agents: https://huggingface.co/docs/hub/agents-cli.md
- Skills repository: https://github.com/huggingface/skills
- `huggingface_hub` on PyPI: https://pypi.org/project/huggingface_hub/
- `datasets` on PyPI: https://pypi.org/project/datasets/

**Next:** Stage 02, Core ML Stack.