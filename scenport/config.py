"""Configuration loading with dotted-key overrides."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports ``cfg["a.b.c"]`` lookups."""

    def __getitem__(self, key: str) -> Any:
        if "." not in key:
            return super().__getitem__(key)
        node: Any = self
        for part in key.split("."):
            node = node[part]
        return node

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def resolve(self, key: str, root: Path) -> Path:
        """Return a path entry resolved against the repository root."""
        return (root / str(self[key])).resolve()


def coerce(text: str) -> Any:
    """Turn a command-line string into the obvious Python value."""
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Read a YAML config and apply ``key=value`` overrides in order."""
    with open(path, encoding="utf-8") as fh:
        cfg = Config(yaml.safe_load(fh))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, _, value = item.partition("=")
        cfg.set(key.strip(), coerce(value.strip()))
    return cfg


def stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed from arbitrary parts.

    ``hash()`` is salted per interpreter, so anything derived from it changes
    between runs and between array tasks. This does not.
    """
    import hashlib

    payload = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") >> 1


def repo_root() -> Path:
    """Repository root, i.e. the parent of the ``scenport`` package directory."""
    return Path(__file__).resolve().parent.parent


def fingerprint(cfg: Config, keys: list[str]) -> str:
    """Short deterministic tag built from selected config values.

    Used to name run directories so that two different settings never collide
    and re-running the same settings overwrites in place.
    """
    import hashlib

    payload = "|".join(f"{k}={cfg.get(k)!r}" for k in sorted(keys))
    return hashlib.sha1(payload.encode()).hexdigest()[:10]
