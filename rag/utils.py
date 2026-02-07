from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return safe or "paper"


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
