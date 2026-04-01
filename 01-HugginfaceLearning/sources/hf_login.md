Here's a complete guide to all Hugging Face Python SDK login methods, plus download and upload examples — all from the official docs.

---

## 📦 Installation

```bash
# Basic install
pip install huggingface_hub

# With faster upload/download support (recommended)
pip install huggingface_hub[hf_xet]

# Full install (all optional deps)
pip install huggingface_hub[all]

# Via conda
conda install -c conda-forge huggingface_hub
```

---

## 🔑 Getting Your Token

Go to 👉 **https://huggingface.co/settings/tokens** and create a token.
- **Read token** → for downloading public/private models
- **Write token** → for uploading, creating repos

---

## 🔐 Login Methods (All 5 Ways)

### Method 1 — CLI Login (Interactive, Recommended for Local Dev)

You can log in using the CLI directly from your terminal:

```bash
# Interactive prompt — asks for your token
hf auth login

# Or pass token directly via env variable
hf auth login --token $HUGGINGFACE_TOKEN
```

---

### Method 2 — `login()` with Token (Programmatic — Best for Scripts)

In Python code, import `login` and pass your token directly:

```python
from huggingface_hub import login

# Pass token directly (stores it on disk for future use)
login(token="hf_YOUR_TOKEN_HERE")

# Separate read/write tokens
access_token_read  = "hf_READ_TOKEN"
access_token_write = "hf_WRITE_TOKEN"

login(token=access_token_read)   # read-only
login(token=access_token_write)  # for uploads
```

---

### Method 3 — Environment Variable (Best for CI/CD & Production)

```python
import os
from huggingface_hub import login

# Set env variable before running
# export HF_TOKEN=hf_YOUR_TOKEN_HERE

login(token=os.environ["HF_TOKEN"])

# OR — huggingface_hub auto-reads HF_TOKEN env var without explicit login!
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
```

You can also put it in a `.env` file:
```bash
# .env
HF_TOKEN=hf_YOUR_TOKEN_HERE
```
```python
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
import os
login(token=os.getenv("HF_TOKEN"))
```

---

### Method 4 — `notebook_login()` (For Jupyter Notebooks)

`login()` auto-detects if you're in a notebook, but you can force the widget UI explicitly with `notebook_login()`:

```python
from huggingface_hub import notebook_login

# Shows an interactive widget inside Jupyter/Colab
notebook_login()
```

---

### Method 5 — `interpreter_login()` (Force Terminal Prompt)

`interpreter_login()` is useful if you want to force the terminal prompt instead of a notebook widget:

```python
from huggingface_hub import interpreter_login

# Forces a terminal/CLI-style prompt even inside a notebook
interpreter_login()
```

---

### Bonus — Google Colab (Using Secrets)

In Google Colab you can safely store tokens using Colab Secrets and retrieve them like this:

```python
from huggingface_hub import login
from google.colab import userdata

HF_TOKEN = userdata.get("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)
    print("Logged in!")
else:
    print("Token not set in Colab Secrets.")
```

---

### Check Who You Are

```python
from huggingface_hub import whoami
print(whoami())  # Returns your username, email, etc.
```

or via CLI:
```bash
hf auth whoami
```

---

## 📥 Download Examples (READ)

### Download a Single File

`hf_hub_download()` is the main function for downloading files — it caches on disk and returns the local path:

```python
from huggingface_hub import hf_hub_download

# Download config.json from a model repo
path = hf_hub_download(
    repo_id="google-bert/bert-base-uncased",
    filename="config.json"
)
print(f"File saved to: {path}")

# Download from a dataset repo
path = hf_hub_download(
    repo_id="stanfordnlp/imdb",
    filename="README.md",
    repo_type="dataset"
)

# Download a specific revision/branch
path = hf_hub_download(
    repo_id="gpt2",
    filename="model.safetensors",
    revision="main"   # or a commit hash
)
```

---

### Download an Entire Repo (Snapshot)

```python
from huggingface_hub import snapshot_download

# Download all files from a model
local_dir = snapshot_download(
    repo_id="openai-community/gpt2",
    local_dir="./gpt2_model"       # save to specific folder
)
print(f"Repo downloaded to: {local_dir}")

# Download only specific file types (ignore others)
snapshot_download(
    repo_id="openai-community/gpt2",
    local_dir="./gpt2_slim",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"]
)

# For a private/gated model (needs write token)
snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir="./llama2",
    token="hf_YOUR_WRITE_TOKEN"
)
```

---

### CLI Download

```bash
# Download single file
hf download gpt2 config.json

# Download multiple files
hf download gpt2 config.json model.safetensors

# Download whole repo
hf download gpt2 --local-dir ./gpt2_model
```

---

## 📤 Upload Examples (WRITE)

> ⚠️ Uploading requires a **write token** and that the repo exists.

### Step 0 — Create a Repo (if it doesn't exist)

```python
from huggingface_hub import create_repo

create_repo(
    repo_id="your-username/my-cool-model",
    repo_type="model",   # "model", "dataset", or "space"
    private=True         # set False for public
)
```

---

### Upload a Single File

Use `upload_file()` to push a single file to your repo:

```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="./my_model/README.md",   # local path
    path_in_repo="README.md",                  # path in the repo
    repo_id="your-username/my-cool-model",
    repo_type="model",
    token="hf_YOUR_WRITE_TOKEN"
)
```

---

### Upload an Entire Folder

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="./my_model_folder",          # local directory
    repo_id="your-username/my-cool-model",
    repo_type="model",
    ignore_patterns=["*.pyc", "__pycache__/"], # skip unwanted files
    token="hf_YOUR_WRITE_TOKEN"
)
```

---

### Upload a Dataset

```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="./data/train.csv",
    path_in_repo="data/train.csv",
    repo_id="your-username/my-dataset",
    repo_type="dataset",    # ← important!
    token="hf_YOUR_WRITE_TOKEN"
)
```

---

### CLI Upload

```bash
# Upload a single file
hf upload your-username/my-model ./README.md README.md

# Upload a whole folder
hf upload your-username/my-model ./my_folder/ . --repo-type model
```

---

## 🧹 Logout

```python
from huggingface_hub import logout
logout()  # clears saved token from disk
```

```bash
hf auth logout
```

---

## ⚡ Quick Reference Summary

| Method | Best For |
|---|---|
| `hf auth login` (CLI) | Local machine one-time setup |
| `login(token=...)` | Scripts, automation |
| `HF_TOKEN` env var | CI/CD, Docker, production |
| `notebook_login()` | Jupyter / Colab notebook UI |
| `interpreter_login()` | Force terminal prompt |
| Colab `userdata.get()` | Google Colab securely |

| Operation | Function |
|---|---|
| Download 1 file | `hf_hub_download()` |
| Download whole repo | `snapshot_download()` |
| Upload 1 file | `upload_file()` |
| Upload folder | `upload_folder()` |
| Create repo | `create_repo()` |