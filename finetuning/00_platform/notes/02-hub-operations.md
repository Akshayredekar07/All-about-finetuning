# **02 – Hub Operations: Download, Cache, Upload**

This notebook covers downloading models and datasets, managing the local Hugging Face cache, and uploading or updating files on the Hub.

Use `HF_TOKEN` for downloads and `HF_WRITE_TOKEN` for any action that creates, changes, or deletes something on the Hub.

Topics: `hf_hub_download()`, `snapshot_download()`, cache management, `upload_file()`, `upload_folder()`, and file deletion.

---

## **Setup Cell (top of every notebook)**

```python
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

READ_TOKEN  = os.environ["HF_TOKEN"]
WRITE_TOKEN = os.environ["HF_WRITE_TOKEN"]
USER        = HfApi(token=READ_TOKEN).whoami()["name"]

MODEL_REPO   = f"{USER}/my-cool-model"
DATASET_REPO = f"{USER}/my-dataset"
```

Repo ids live here, not hard-coded in later cells.

---

## **Download (READ)**

| Need | Function |
|---|---|
| One file | `hf_hub_download()` |
| Whole repo | `snapshot_download()` |
| From the terminal | `hf download` |

### **Download a single file**

```python
from huggingface_hub import hf_hub_download

# ── from a model repo ─────────────────────────────────────────────────
path = hf_hub_download(
    repo_id="google-bert/bert-base-uncased",
    filename="config.json",
)
print(f"File saved to: {path}")

# ── from a dataset repo (repo_type is required) ───────────────────────
path = hf_hub_download(
    repo_id="stanfordnlp/imdb",
    filename="README.md",
    repo_type="dataset",
)

# ── pin a branch, tag or commit hash ──────────────────────────────────
path = hf_hub_download(
    repo_id="openai-community/gpt2",
    filename="config.json",
    revision="main",          # or "v1", or a full commit hash
)
```

### **Download an entire repo (snapshot)**

```python
from huggingface_hub import snapshot_download

# ── everything, into a folder you choose ──────────────────────────────
local_dir = snapshot_download(
    repo_id="openai-community/gpt2",
    local_dir="outputs/gpt2_model",
)
print(f"Repo downloaded to: {local_dir}")

# ── skip formats you do not need ──────────────────────────────────────
snapshot_download(
    repo_id="openai-community/gpt2",
    local_dir="outputs/gpt2_slim",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)

# ── only the formats you do need ──────────────────────────────────────
snapshot_download(
    repo_id="openai-community/gpt2",
    local_dir="outputs/gpt2_json",
    allow_patterns=["*.json", "*.txt"],
)

# ── private or gated repo: READ token is enough ───────────────────────
snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir="outputs/llama2",
    token=READ_TOKEN,         # you must have accepted the terms on the model page
)
```

### **CLI download**

```bash
# ── single file ───────────────────────────────────────────────────────
hf download openai-community/gpt2 config.json

# ── several files ─────────────────────────────────────────────────────
hf download openai-community/gpt2 config.json model.safetensors

# ── whole repo into a folder ──────────────────────────────────────────
hf download openai-community/gpt2 --local-dir outputs/gpt2_model

# ── a dataset ─────────────────────────────────────────────────────────
hf download stanfordnlp/imdb --repo-type dataset --local-dir outputs/imdb_raw
```

**Notes:**
- Without `local_dir`, files go to the shared cache and the returned path is a symlink into it. With `local_dir`, you get a real copy in your folder.
- Use the full `owner/name` id. Short names like `gpt2` are legacy aliases that redirect.
- `hf download` prints the local path, so you can capture it in a shell variable.

---

## **Cache**

```bash
# ── default location ──────────────────────────────────────────────────
~/.cache/huggingface/hub

# ── move it (put in .env or your shell profile) ───────────────────────
export HF_HOME=D:/hf_cache

# ── use only what is already cached, never touch the network ──────────
export HF_HUB_OFFLINE=1

# ── inspect and clean the cache ───────────────────────────────────────
hf cache --help
```

```python
# ── same offline behaviour for a single call ──────────────────────────
path = hf_hub_download(
    "google-bert/bert-base-uncased", "config.json",
    local_files_only=True,    # raises an error if it is not cached
)
```

**Where things go:**

| What | Where | Committed? |
|---|---|---|
| Hub downloads and datasets cache | `~/.cache/huggingface/` or `HF_HOME` | No (outside repo) |
| `save_to_disk` output, big downloads, temp model dirs | `outputs/` at repo root | No (gitignored) |
| Space source code | `00_platform/hub/spaces/...` | Yes |
| Lab results | `00_platform/results/*.md` | Yes |

---

## **Create a Repo**

```python
from huggingface_hub import create_repo

# ── model repo ────────────────────────────────────────────────────────
create_repo(
    repo_id=MODEL_REPO,
    repo_type="model",        # "model", "dataset" or "space"
    private=True,             # False for public
    exist_ok=True,            # makes re-runs safe
    token=WRITE_TOKEN,
)

# ── dataset repo ──────────────────────────────────────────────────────
create_repo(DATASET_REPO, repo_type="dataset", private=True, exist_ok=True, token=WRITE_TOKEN)
```

```bash
hf repos --help               # create, delete, move, branch, tag (see the CLI reference)
```

---

## **Upload (WRITE)**

Uploading needs the **write** token and an existing repo.

### **Upload a single file**

```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="outputs/release/README.md",   # local path
    path_in_repo="README.md",                       # path inside the repo
    repo_id=MODEL_REPO,
    repo_type="model",
    commit_message="Add model card",
    token=WRITE_TOKEN,
)
```

### **Upload an entire folder**

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="outputs/release",
    repo_id=MODEL_REPO,
    repo_type="model",
    ignore_patterns=["*.pyc", "__pycache__/", ".ipynb_checkpoints/"],
    token=WRITE_TOKEN,
)
```

### **Upload to a dataset repo**

```python
upload_file(
    path_or_fileobj="outputs/fixtures/train.csv",
    path_in_repo="data/train.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",      # ← important!
    token=WRITE_TOKEN,
)
```

### **CLI upload**

```bash
# ── single file ───────────────────────────────────────────────────────
hf upload your-username/my-cool-model ./README.md README.md

# ── whole folder ──────────────────────────────────────────────────────
hf upload your-username/my-cool-model ./outputs/release . --repo-type model

# ── very large folders: resumable ─────────────────────────────────────
hf upload-large-folder --help
```

**Notes:**
- Forgetting `repo_type="dataset"` sends the file to a model repo with the same name, or fails.
- Every `upload_file` or `upload_folder` call is **one commit**. Three calls make three commits, and someone downloading in between sees a half-updated repo.
- `upload-large-folder` is marked deprecated in recent CLI docs, so check `hf upload-large-folder --help` for the current replacement before relying on it.

---

## **Atomic Commit (Lab 0-3)**

Use `create_commit` when several files must change together.

```python
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

api = HfApi(token=WRITE_TOKEN)

# ── one commit: add two files, delete one ─────────────────────────────
api.create_commit(
    repo_id=MODEL_REPO,
    repo_type="model",
    operations=[
        CommitOperationAdd(path_in_repo="README.md",   path_or_fileobj="outputs/release/README.md"),
        CommitOperationAdd(path_in_repo="config.json", path_or_fileobj="outputs/release/config.json"),
        CommitOperationDelete(path_in_repo="old_file.txt"),
    ],
    commit_message="Release v1",
)

# ── tag it so downloads can pin it ────────────────────────────────────
api.create_tag(MODEL_REPO, tag="v1", repo_type="model")

# ── later: pin the tag ────────────────────────────────────────────────
hf_hub_download(MODEL_REPO, "config.json", revision="v1", token=READ_TOKEN)
```

**Note:** all operations land together or none do. If one path is wrong, the whole commit is rejected and the repo is unchanged.

---

## **Quick Reference**

| Operation | Function |
|---|---|
| Download 1 file | `hf_hub_download()` |
| Download whole repo | `snapshot_download()` |
| Create repo | `create_repo()` |
| Upload 1 file | `upload_file()` |
| Upload folder | `upload_folder()` |
| Multi-file atomic change | `HfApi.create_commit()` |
| Tag a release | `HfApi.create_tag()` |
| Pin a version | `revision="tag-or-hash"` |
| Work offline | `HF_HUB_OFFLINE=1` or `local_files_only=True` |

**Verify by running:** `hf cache --help`, `hf repos --help` and `hf upload-large-folder --help` on your installed version. These subcommands changed between 0.x and 1.x.