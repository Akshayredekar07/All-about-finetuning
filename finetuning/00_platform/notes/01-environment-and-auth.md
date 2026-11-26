# **01 – Environment and Auth**

Everything you need before touching the Hub: install, tokens, every login method, and how to check who you are.
Verified against the current `hf` CLI docs (huggingface_hub 1.x). The old `huggingface-cli` still works but prints a deprecation warning, so use `hf`.

---

## **Installation**

```bash
# ── basic ─────────────────────────────────────────────────────────────
pip install -U huggingface_hub

# ── faster transfers (Xet backend) ────────────────────────────────────
pip install -U "huggingface_hub[hf_xet]"    # recent versions may already pull this in

# ── .env support for notebooks ────────────────────────────────────────
pip install python-dotenv

# ── check the CLI works ───────────────────────────────────────────────
hf version
hf --help

# ── alternatives ──────────────────────────────────────────────────────
uvx hf auth login                            # run without installing
brew install hf                              # macOS / Linux via Homebrew
```

**Note:** if you get `command not found: hf`, open a new terminal so your PATH reloads.

---

## **Tokens**

Create them at https://huggingface.co/settings/tokens.

| Token | Env var | Can do |
|---|---|---|
| **Read** | `HF_TOKEN` | Download public, private and gated repos you have access to |
| **Write** | `HF_WRITE_TOKEN` | Everything above, plus create repos, upload, change Space settings |
| **Fine-grained** | your choice | Only the repos and actions you tick; best for CI and shared machines |

```bash
# ── .env (never commit this; commit only .env.example) ────────────────
HF_TOKEN=
HF_WRITE_TOKEN=
HF_HOME=              # optional: move the cache, e.g. D:/hf_cache
WANDB_API_KEY=        # needed from Stage 03 onward
```

```bash
# ── .gitignore ────────────────────────────────────────────────────────
.env
outputs/
*.arrow
*.safetensors
*.bin
```

**Notes:**
- A **read** token used for an upload fails with a 403. This is the Lab 0-1 test case.
- A **gated** model needs only a read token, but you must first accept its terms on the model page. The old note said write token; that was wrong.
- Never paste `hf_...` strings into a notebook cell. They end up in git history and in saved outputs.

---

## **Login Methods**

| Method | Best for |
|---|---|
| `hf auth login` | One-time setup on your own machine |
| `login(token=...)` | Scripts and automation |
| `HF_TOKEN` env var | CI/CD, Docker, production |
| `notebook_login()` | Jupyter or Colab widget |
| `interpreter_login()` | Force a terminal prompt |
| Colab `userdata.get()` | Google Colab secrets |

### **Method 1 – CLI login**

```bash
# ── interactive: asks for a token ─────────────────────────────────────
hf auth login

# ── non-interactive ───────────────────────────────────────────────────
hf auth login --token $HF_TOKEN

# ── also save to git credential helper (only if you run raw git push) ─
hf auth login --token $HF_TOKEN --add-to-git-credential

# ── already logged in? force a new login (e.g. to switch tokens) ──────
hf auth login --force

# ── give the token a name so you can switch back later ────────────────
hf auth login --token $HF_WRITE_TOKEN --token-name write
```

### **Method 2 – `login()` in Python**

```python
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# ── read token ────────────────────────────────────────────────────────
login(token=os.environ["HF_TOKEN"])

# ── write token (replaces the read token as the active one) ───────────
login(token=os.environ["HF_WRITE_TOKEN"])
```

**Note:** the second `login()` replaces the first. To use both in one script, skip `login()` and pass `token=` to each call instead.

### **Method 3 – environment variable**

```python
import os
from dotenv import load_dotenv

# ── option A: load from .env ──────────────────────────────────────────
load_dotenv()                      # HF_TOKEN is now in os.environ

# ── option B: export in the shell before starting Python ──────────────
# export HF_TOKEN=hf_...

# ── no login() call needed: huggingface_hub reads HF_TOKEN itself ─────
from huggingface_hub import whoami
print(whoami()["name"])
```

**Note:** an env var beats a token saved by `hf auth login`. If `whoami` shows the wrong account, check `echo $HF_TOKEN`.

### **Method 4 – `notebook_login()`**

```python
from huggingface_hub import notebook_login

# ── widget inside Jupyter / Colab ─────────────────────────────────────
notebook_login()
```

### **Method 5 – `interpreter_login()`**

```python
from huggingface_hub import interpreter_login

# ── terminal-style prompt, even inside a notebook ─────────────────────
interpreter_login()
```

Use this when the widget does not render (some VS Code and remote setups).

### **Bonus – Google Colab secrets**

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

## **Which Token Wins?**

When more than one is present, `huggingface_hub` picks in this order:

1. `token=` passed to the function call
2. `HF_TOKEN` environment variable
3. The token saved on disk by `hf auth login` or `login()`

This order is what makes the Lab 0-1 matrix possible: the same upload can succeed or fail depending on which of the three you supply.

---

## **Check Who You Are**

```python
from huggingface_hub import whoami

# ── username, orgs, token role ────────────────────────────────────────
info = whoami()
print(info["name"])
print(info.get("auth", {}).get("accessToken", {}).get("role"))
```

```bash
hf auth whoami
hf auth whoami --format json
hf auth token              # print the active token (careful: it is a secret)
```

---

## **Multiple Saved Tokens**

```bash
# ── list stored tokens ────────────────────────────────────────────────
hf auth list               # alias: hf auth ls

# ── switch the active token ───────────────────────────────────────────
hf auth switch
hf auth switch --token-name write
```

Use this in Lab 0-1: switch between the read and write token and watch the same upload fail, then succeed.

---

## **Logout**

```python
from huggingface_hub import logout

# ── clears the saved token from disk ──────────────────────────────────
logout()
```

```bash
hf auth logout
hf auth logout --token-name write      # remove only one saved token
```

**Note:** `logout()` does not unset `HF_TOKEN`. If the env var is set, you are still authenticated.

---

## **Quick Reference**

| Task | Command or function |
|---|---|
| Log in (CLI) | `hf auth login` |
| Log in (code) | `login(token=...)` |
| Who am I | `hf auth whoami` / `whoami()` |
| List saved tokens | `hf auth list` |
| Switch token | `hf auth switch --token-name NAME` |
| Print active token | `hf auth token` |
| Log out | `hf auth logout` / `logout()` |
| Move the cache | set `HF_HOME` |
| Offline mode | set `HF_HUB_OFFLINE=1` |

**Sources:** Hub CLI reference and the "Say hello to `hf`" post on huggingface.co.