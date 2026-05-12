"""
NDP config — manages ~/.ndp/config.json.

First run auto-generates a stable user_id UUID.
Service role key must be pasted in once; after that it's silent.
"""

import json
import uuid
from pathlib import Path

NDP_DIR = Path.home() / ".ndp"
CONFIG_FILE = NDP_DIR / "config.json"

SUPABASE_URL = "https://izackuempnhgoojtbgvi.supabase.co"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return _init_config()
    cfg = json.loads(CONFIG_FILE.read_text())
    if not cfg.get("user_id"):
        cfg["user_id"] = str(uuid.uuid4())
        _write(cfg)
    return cfg


def _init_config() -> dict:
    NDP_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "supabase_url": SUPABASE_URL,
        "supabase_key": "",
        "user_id": str(uuid.uuid4()),
        # Privacy: default OFF. NDPA stores file paths + behavioral signal only.
        # Set to true to upload file CONTENT (e.g. for hot-tier context staging).
        # File contents only stored when this is explicitly enabled.
        "upload_file_contents": False,
    }
    _write(cfg)
    return cfg


def get_setting(key: str, default=None):
    """Read a single setting from config. Used for feature flags."""
    try:
        cfg = load_config()
        return cfg.get(key, default)
    except Exception:
        return default


def save_key(key: str):
    cfg = load_config()
    cfg["supabase_key"] = key.strip()
    _write(cfg)


def _write(cfg: dict):
    NDP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    CONFIG_FILE.chmod(0o600)


def is_configured() -> bool:
    if not CONFIG_FILE.exists():
        return False
    try:
        cfg = json.loads(CONFIG_FILE.read_text())
        return bool(cfg.get("supabase_key"))
    except Exception:
        return False


if __name__ == "__main__":
    cfg = load_config()
    if not cfg.get("supabase_key"):
        print("NDP Supabase setup")
        print(f"Project: {SUPABASE_URL}")
        print("Get your service role key from: Supabase dashboard → Settings → API → service_role")
        key = input("Paste service role key: ").strip()
        save_key(key)
        print(f"\nDone. Your user_id: {cfg['user_id']}")
        print(f"Config saved to: {CONFIG_FILE}")
    else:
        print(f"Already configured. user_id: {cfg['user_id']}")
