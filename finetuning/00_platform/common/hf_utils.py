"""Shared helpers for the stage 00 labs.

Rule: call setup() BEFORE importing huggingface_hub / datasets / transformers,
because the cache folders are read when those libraries are imported.
huggingface_hub is imported lazily inside functions for the same reason.
"""
import os
from pathlib import Path

import yaml

CONFIG_NAME = Path("configs") / "hub.yaml"


def find_root(start=None):
    """Walk upward from start (default: current folder) to the folder holding configs/hub.yaml."""
    here = Path(start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        if (folder / CONFIG_NAME).exists():
            return folder
    raise FileNotFoundError(f"Could not find {CONFIG_NAME} above {here}")


def load_config(root=None):
    """Read configs/hub.yaml. The folder it was found in is stored as cfg['_root']."""
    root = Path(root) if root else find_root()
    with open(root / CONFIG_NAME, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = root
    return cfg


def setup(cfg=None):
    """Create folders, move the Hub caches into datasets/downloads, load .env.

    Returns a dict of Paths. Run this in Cell 1, before other imports.
    """
    cfg = cfg or load_config()
    root = cfg["_root"]
    paths = {name: root / rel for name, rel in cfg["paths"].items()}
    for key in ("downloads", "upload", "results"):
        paths[key].mkdir(parents=True, exist_ok=True)

    # ── caches ──
    os.environ["HF_HUB_CACHE"] = str(paths["downloads"] / "hub_cache")
    os.environ["HF_DATASETS_CACHE"] = str(paths["downloads"] / "datasets_cache")

    # ── tokens: .env found by searching upward ──
    from dotenv import load_dotenv
    load_dotenv()
    os.environ.pop("HF_TOKEN", None)          # our explicit tokens must win
    paths["root"] = root
    return paths


def get_token(cfg, which="write", required=True):
    """which = read | write | finegrained. Returns the token (or None if not required)."""
    name = cfg["env"][f"{which}_token"]
    token = os.getenv(name)
    if required and not token:
        raise KeyError(f"{name} is missing from .env")
    return token


def role_of(token):
    """Role of a token as the Hub sees it. token=None uses env, then the saved token."""
    from huggingface_hub import HfApi
    info = HfApi(token=token).whoami()
    return info.get("auth", {}).get("accessToken", {}).get("role")


def repo_id(cfg, user, key):
    """repo_id(cfg, 'akshay', 'model') -> 'akshay/release-lab-model'."""
    return f"{user}/{cfg['repos'][key]}"


def folder_size_mb(folder):
    """Total size of a folder in MB (0 if it does not exist)."""
    folder = Path(folder)
    if not folder.exists():
        return 0.0
    total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    return round(total / 1_000_000, 2)


def save_result(cfg, filename, text):
    """Write a markdown result into the results folder from hub.yaml."""
    out = cfg["_root"] / cfg["paths"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    (out / filename).write_text(text, encoding="utf-8")
    return out / filename


def delete_repo_quietly(api, repo, repo_type="model"):
    """Delete a repo. Returns False (instead of an error) if it is already gone."""
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        api.delete_repo(repo, repo_type=repo_type)
        return True
    except RepositoryNotFoundError:
        return False