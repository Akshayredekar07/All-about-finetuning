# **03 – Repositories, Spaces and Errors**

This notebook covers everything you can do **after a repository already exists**:

- Inspect a repository: check that it exists, view metadata, files, commits, branches, and tags
- Work with branches and tags
- Change repository settings
- Run a Space
- Handle common `huggingface_hub` errors

## **Token rule**

| Action | Token |
|---|---|
| Read a repository: check existence, inspect files, metadata, commits, branches | `HF_TOKEN` |
| Change a repository or Space: upload, delete, update settings, run a Space | `HF_WRITE_TOKEN` |

We create two separate `HfApi` clients so that read operations use the read token and write operations use the write token.

```python
read_api  = HfApi(token=READ_TOKEN)
write_api = HfApi(token=WRITE_TOKEN)
```

## **Repositories used**

```python
MODEL_REPO   = f"{USER}/my-cool-model"
DATASET_REPO = f"{USER}/my-dataset"
SPACE_REPO   = f"{USER}/sentiment-analysis"
```

**Note:** The method names and signatures below were checked against `huggingface_hub` 2.5.0. The error examples in Section 5 are based on documentation and class definitions; run the cells yourself to confirm the behaviour in your environment.

Two `HfApi` objects, one per token, so you never depend on which token `login()` saved last (see note 01).

---

## **1. Inspect a Repo (READ)**

| Need | Function |
|---|---|
| Does it exist | `repo_exists()` |
| Does a file exist | `file_exists()` |
| Metadata | `model_info()`, `dataset_info()`, `space_info()` |
| File list | `list_repo_files()` |
| History | `list_repo_commits()` |
| Branches and tags | `list_repo_refs()` |

```python
# ── exists ────────────────────────────────────────────────────────────
read_api.repo_exists(MODEL_REPO)
read_api.repo_exists(DATASET_REPO, repo_type="dataset")
read_api.repo_exists(SPACE_REPO, repo_type="space")
read_api.file_exists(MODEL_REPO, "config.json")

# ── metadata ──────────────────────────────────────────────────────────
info = read_api.model_info(MODEL_REPO)
print(info.id, info.sha)
print(info.private, info.gated)
print(info.tags)
print([s.rfilename for s in info.siblings])      # files in the repo

ds_info    = read_api.dataset_info(DATASET_REPO)
space_info = read_api.space_info(SPACE_REPO)
print(space_info.sdk)

# ── files at a given revision ─────────────────────────────────────────
read_api.list_repo_files(MODEL_REPO)
read_api.list_repo_files(MODEL_REPO, revision="v1")

# ── commit history ────────────────────────────────────────────────────
for c in read_api.list_repo_commits(MODEL_REPO)[:5]:
    print(c.commit_id[:8], c.created_at, c.title)

# ── branches and tags ─────────────────────────────────────────────────
refs = read_api.list_repo_refs(MODEL_REPO)
print([b.name for b in refs.branches])
print([t.name for t in refs.tags])
```

**Notes:**
- `repo_exists` returns `True` for a gated repo you cannot open yet, and `False` for a private repo your token cannot see. The Hub answers 404 for private repos you have no access to, so "does not exist" and "not yours" look the same.
- `file_exists` behaves differently on gated repos: it raises `GatedRepoError` instead of returning `False`.
- Leaving out `repo_type` makes every call look in the model namespace. A dataset or Space id then comes back as "not found".

---

## **2. Branches, Tags and Pull Requests (WRITE)**

```python
# ── branch from main ──────────────────────────────────────────────────
write_api.create_branch(MODEL_REPO, branch="dev", exist_ok=True)

# ── upload onto that branch instead of main ───────────────────────────
write_api.upload_file(
    path_or_fileobj="outputs/release/README.md",
    path_in_repo="README.md",
    repo_id=MODEL_REPO,
    revision="dev",
    commit_message="Draft model card",
)

# ── open a pull request instead of committing directly ────────────────
commit = write_api.upload_file(
    path_or_fileobj="outputs/release/README.md",
    path_in_repo="README.md",
    repo_id=MODEL_REPO,
    create_pr=True,
)
print(commit.pr_url)

# ── tag a release (defaults to the head of main) ──────────────────────
write_api.create_tag(MODEL_REPO, tag="v1", tag_message="First release")

# ── pin it later ──────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download
hf_hub_download(MODEL_REPO, "config.json", revision="v1", token=READ_TOKEN)

# ── remove things ─────────────────────────────────────────────────────
write_api.delete_file("old_file.txt", MODEL_REPO)
write_api.delete_folder("old_dir", MODEL_REPO)
write_api.delete_tag(MODEL_REPO, tag="v1")
write_api.delete_branch(MODEL_REPO, branch="dev")
```

**Notes:**
- `delete_file` and `delete_folder` are one commit each. To remove files together with other changes in a single commit, use `CommitOperationDelete` inside `create_commit` (note 02).
- A tag points at one commit forever. Download with `revision="v1"` to get exactly the files you released, even after `main` moves on.
- Pass `repo_type="dataset"` or `"space"` to every call above when the repo is not a model.

---

## **3. Repo Settings, Rename and Delete (WRITE)**

```python
# ── visibility ────────────────────────────────────────────────────────
write_api.update_repo_settings(MODEL_REPO, private=False)

# ── gated access: owner side ──────────────────────────────────────────
write_api.update_repo_settings(MODEL_REPO, gated="auto")      # auto-approve requests
write_api.update_repo_settings(MODEL_REPO, gated="manual")    # you approve each request
write_api.update_repo_settings(MODEL_REPO, gated=False)       # remove gating

# ── rename or move to another namespace ───────────────────────────────
write_api.move_repo(from_id=MODEL_REPO, to_id=f"{USER}/my-cool-model-v2")

# ── delete (irreversible) ─────────────────────────────────────────────
write_api.delete_repo(DATASET_REPO, repo_type="dataset", missing_ok=True)
```

**Notes:**
- `delete_repo` cannot be undone. `missing_ok=True` only skips the error when the repo is already gone.
- `gated` here is the owner side. A user downloading your gated repo still needs a read token and must accept the terms on the repo page, as in note 02.
- After `move_repo`, update `MODEL_REPO` in your setup cell. The old id may redirect for a while, but do not rely on it.

---

## **4. Spaces**

A Space is a git repo with an app and a README whose header tells the Hub how to run it.

### **Folder layout (committed in this repo)**

```
00_platform/hub/spaces/sentiment-analysis/
    README.md            # header below plus a short description
    app.py               # the Gradio app
    requirements.txt     # dependencies the build installs
```

```yaml
---
title: Sentiment Analysis
sdk: gradio                   # gradio, streamlit, docker or static
# sdk_version: copy the value from the README the Hub generates when you create a Space in the browser
app_file: app.py
pinned: false
---
```

### **Create and deploy**

```python
# ── create the Space ──────────────────────────────────────────────────
write_api.create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    space_sdk="gradio",
    private=True,
    exist_ok=True,
)

# ── push the folder (every push triggers a rebuild) ───────────────────
write_api.upload_folder(
    folder_path="hub/spaces/sentiment-analysis",
    repo_id=SPACE_REPO,
    repo_type="space",
    ignore_patterns=["*.pyc", "__pycache__/", ".ipynb_checkpoints/"],
)

print(f"https://huggingface.co/spaces/{SPACE_REPO}")
```

### **Secrets and variables**

```python
# ── secret: hidden, injected as an environment variable ───────────────
write_api.add_space_secret(SPACE_REPO, key="HF_TOKEN", value=READ_TOKEN)

# ── variable: visible config, also an environment variable ────────────
write_api.add_space_variable(SPACE_REPO, key="MODEL_REPO_ID", value=MODEL_REPO)

# ── remove ────────────────────────────────────────────────────────────
write_api.delete_space_secret(SPACE_REPO, key="HF_TOKEN")
```

In `app.py`, read both with `os.environ`. Never put a token in `app.py` or the README, because the Space repo is git history that others can clone if the Space is public.

### **Runtime control**

```python
from huggingface_hub import SpaceHardware

# ── what is it doing right now ────────────────────────────────────────
rt = write_api.get_space_runtime(SPACE_REPO)
print(rt.stage)                # for example RUNNING
print(rt.hardware)             # for example cpu-basic
print(rt.requested_hardware)   # differs from hardware right after a change
print(rt.sleep_time)           # seconds, or None

# ── pause: stops the app, not billed while paused ─────────────────────
write_api.pause_space(SPACE_REPO)

# ── restart: the only programmatic way to resume a paused Space ───────
write_api.restart_space(SPACE_REPO)

# ── rebuild from scratch, ignoring cached requirements ────────────────
write_api.restart_space(SPACE_REPO, factory_reboot=True)

# ── go back to the free hardware ──────────────────────────────────────
write_api.request_space_hardware(SPACE_REPO, SpaceHardware.CPU_BASIC)

# ── wait until it is up or has failed ─────────────────────────────────
import time

for _ in range(30):
    rt = write_api.get_space_runtime(SPACE_REPO)
    print(rt.stage)
    if rt.stage in ("RUNNING", "BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR"):
        break
    time.sleep(10)
```

### **Stages**

| Stage | Meaning | Usual cause |
|---|---|---|
| `BUILDING` | installing dependencies | normal after every push |
| `APP_STARTING` | build done, app booting | normal |
| `RUNNING` | serving requests | |
| `RUNNING_BUILDING`, `RUNNING_APP_STARTING` | old version still serving while the new one builds | normal after a push |
| `BUILD_ERROR` | dependency install failed | bad or conflicting `requirements.txt` |
| `CONFIG_ERROR` | README header rejected | typo in `sdk`, `app_file` or YAML |
| `RUNTIME_ERROR` | app crashed after start | exception in `app.py`; read the Space logs |
| `NO_APP_FILE` | `app_file` not found | wrong filename or folder not uploaded |
| `PAUSED` | stopped by `pause_space` | you paused it |
| `STOPPED`, `DELETING` | not running, or being removed | |

**Notes:**
- A free `cpu-basic` Space goes to sleep after 48 hours without visitors. That is different from paused: a visitor wakes a sleeping Space, but only `restart_space` resumes a paused one.
- `set_space_sleep_time` works only on upgraded hardware. The sleep time on free hardware is fixed.
- Only the owner can pause or restart a Space (403 otherwise). To work with someone else's Space, duplicate it into your account first. Check `HfApi.duplicate_repo` on your version, because older releases named this `duplicate_space`.
- Static Spaces cannot be paused. `pause_space` raises `BadRequestError`.
- `rt.stage` is a `SpaceStage` that compares equal to its plain string. It can be a bare `str` if the server returns a stage the enum does not know yet.
- Upgraded hardware bills as soon as the Space restarts. Keep demos on `cpu-basic`.

---

## **5. Errors**

The client no longer uses `requests`. Catch `HfHubHTTPError` and its subclasses, not `requests.exceptions.HTTPError`.

```python
from huggingface_hub.errors import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    GatedRepoError,
    RevisionNotFoundError,
    RemoteEntryNotFoundError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    BadRequestError,
)
```

### **The classes**

| Exception | Raised when | Also inherits from |
|---|---|---|
| `HfHubHTTPError` | any failed Hub call; base of most below | `OSError` |
| `RepositoryNotFoundError` | wrong id, wrong `repo_type`, or a private repo you cannot see | `HfHubHTTPError` |
| `GatedRepoError` | gated repo and you are not on the approved list | `RepositoryNotFoundError` |
| `RevisionNotFoundError` | branch, tag or commit does not exist | `HfHubHTTPError` |
| `RemoteEntryNotFoundError` | file is missing at that revision | `HfHubHTTPError`, `EntryNotFoundError` |
| `LocalEntryNotFoundError` | file not in the cache and the network is off | `FileNotFoundError`, `EntryNotFoundError` |
| `OfflineModeIsEnabled` | a network call while `HF_HUB_OFFLINE=1` | `ConnectionError` |
| `BadRequestError` | invalid request, for example pausing a static Space | `HfHubHTTPError`, `ValueError` |
| `HFValidationError` | malformed repo id, such as `"a/b/c"` | `ValueError` |

Every `HfHubHTTPError` carries `e.response` (use `e.response.status_code`), `e.server_message` and `e.request_id`. Quote the request id when you report a problem to Hugging Face.

### **Catch order matters**

`GatedRepoError` is a subclass of `RepositoryNotFoundError`. If you list the parent first, the gated case is swallowed by it. Write subclasses before parents, and `HfHubHTTPError` last.

```python
from huggingface_hub import hf_hub_download

# ── trigger each failure on purpose ───────────────────────────────────
cases = [
    ("missing repo",     dict(repo_id=f"{USER}/does-not-exist-123", filename="config.json")),
    ("missing file",     dict(repo_id="openai-community/gpt2", filename="nope.txt")),
    ("missing revision", dict(repo_id="openai-community/gpt2", filename="config.json", revision="no-such-tag")),
    ("gated, no access", dict(repo_id="meta-llama/Llama-2-7b-hf", filename="config.json")),
]

for label, kwargs in cases:
    try:
        hf_hub_download(token=READ_TOKEN, **kwargs)
    except GatedRepoError as e:
        print(label, "->", type(e).__name__, e.response.status_code)
    except RepositoryNotFoundError as e:
        print(label, "->", type(e).__name__, e.response.status_code)
    except RevisionNotFoundError as e:
        print(label, "->", type(e).__name__, e.response.status_code)
    except RemoteEntryNotFoundError as e:
        print(label, "->", type(e).__name__, e.response.status_code)
    except HfHubHTTPError as e:
        print(label, "->", type(e).__name__, e.response.status_code, e.server_message)
```

For the gated case, use any gated repo whose terms you have not accepted. If you already accepted Llama 2's terms, this one will simply download.

### **Permission failure (Lab 0-1)**

```python
# ── a read token trying to write ──────────────────────────────────────
try:
    read_api.upload_file(
        path_or_fileobj=b"probe",
        path_in_repo="probe.txt",
        repo_id=MODEL_REPO,
    )
except HfHubHTTPError as e:
    print(e.response.status_code)     # expect 403
    print(e.server_message)

# ── the same call with the write token ────────────────────────────────
write_api.upload_file(
    path_or_fileobj=b"probe",
    path_in_repo="probe.txt",
    repo_id=MODEL_REPO,
)
write_api.delete_file("probe.txt", MODEL_REPO)      # clean up
```

### **Offline failure**

```python
# ── file not cached, network disabled for this one call ───────────────
try:
    hf_hub_download("openai-community/gpt2", "vocab.json", local_files_only=True)
except LocalEntryNotFoundError as e:
    print(type(e).__name__)
```

`HF_HUB_OFFLINE` is read when `huggingface_hub` is imported. Setting it with `os.environ` after the import has no effect, so export it before Python starts. `local_files_only=True` works per call and needs no restart.

### **Diagnosis table**

| Symptom | Likely cause | Check |
|---|---|---|
| `RepositoryNotFoundError` on a repo you know exists | typo, missing `repo_type`, or a private repo with the wrong or no token | `read_api.whoami()`, then `repo_exists(..., repo_type=...)` |
| `GatedRepoError` | terms not accepted, or a manual request still pending | accept on the model page with the same account as the token |
| 403 on upload or `create_repo` | read token, or a fine-grained token without write on that repo | `echo $HF_TOKEN`; check the token's permissions on the settings page |
| `RemoteEntryNotFoundError` | wrong filename, or the file is not at that revision | `list_repo_files(repo, revision=...)` |
| `RevisionNotFoundError` | tag never created, or a typo in the branch | `list_repo_refs(repo)` |
| `LocalEntryNotFoundError` | file never downloaded and the network is off | run once online, or drop `local_files_only` |
| `BadRequestError` on `pause_space` | static Space | check `space_info.sdk` |
| 403 on `pause_space` or `restart_space` | you do not own the Space | duplicate it |
| Space stuck in `BUILD_ERROR` | `requirements.txt` | read the build log on the Space page |
| Space in `CONFIG_ERROR` | README header | compare it with the header in section 4 |

---

## **Quick Reference**

| Task | Function |
|---|---|
| Repo exists | `repo_exists()` |
| File exists | `file_exists()` |
| Repo metadata | `model_info()`, `dataset_info()`, `space_info()` |
| List files | `list_repo_files()` |
| Commit history | `list_repo_commits()` |
| Branches and tags | `list_repo_refs()` |
| Create branch or tag | `create_branch()`, `create_tag()` |
| Delete branch or tag | `delete_branch()`, `delete_tag()` |
| Delete file or folder | `delete_file()`, `delete_folder()` |
| Open a pull request | `create_pr=True` on upload calls |
| Gating and visibility | `update_repo_settings()` |
| Rename or move | `move_repo()` |
| Delete a repo | `delete_repo()` |
| Create a Space | `create_repo(repo_type="space", space_sdk=...)` |
| Space secret or variable | `add_space_secret()`, `add_space_variable()` |
| Space state | `get_space_runtime()` |
| Pause, resume | `pause_space()`, `restart_space()` |
| Change hardware | `request_space_hardware()` |
| Catch Hub errors | `HfHubHTTPError` and its subclasses |

**Verify by running:** the 403 on a read-token upload and the exact status codes in the diagnosis table, since I could not call the live Hub. Also confirm `HfApi.duplicate_repo` and `request_space_hardware` on your installed version with `help(HfApi.<name>)`. Method names and signatures here match `huggingface_hub` 2.5.0.

**Sources:** `huggingface_hub` source and docstrings, the Stage 00 section of `roadmap-nov2026/complete-readmap.md`.