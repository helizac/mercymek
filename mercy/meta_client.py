"""One OpenAI-compatible client for every Meta model (base_url = api.meta.ai), with retry/backoff."""
from __future__ import annotations

import base64
import json
import os
import pathlib
from functools import lru_cache

from openai import OpenAI

from mercy import config


@lru_cache(maxsize=4)
def meta_client(timeout: float = 120.0) -> OpenAI:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set (see .env.example)")
    return OpenAI(base_url=config.META_BASE_URL, api_key=key,
                  max_retries=3,  # SDK exponential backoff for 429 and 5xx
                  timeout=timeout)


def has_key() -> bool:
    return bool(os.environ.get("MODEL_API_KEY"))


def upload_file(path: pathlib.Path) -> str:
    """Files API, purpose=user_data. Synchronous: a 200 means the file is ready."""
    with open(path, "rb") as f:
        return meta_client().files.create(purpose="user_data", file=f).id


def data_url(path: pathlib.Path, mime: str | None = None) -> str:
    mime = mime or {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".webp": "image/webp", ".mp4": "video/mp4"}[path.suffix.lower()]
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def structured(model: str, input, schema: dict, name: str, effort: str | None = None,
               max_output_tokens: int | None = None, timeout: float | None = None, **kw) -> dict:
    """Responses API with text.format json_schema; returns the parsed object."""
    args = dict(model=model, input=input,
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}, **kw)
    if effort:
        args["reasoning"] = {"effort": effort}
    if max_output_tokens:
        args["max_output_tokens"] = max_output_tokens
    client = meta_client() if timeout is None else meta_client().with_options(timeout=timeout)
    r = client.responses.create(**args)
    return json.loads(r.output_text)


def usage_line(r) -> str:
    u = getattr(r, "usage", None)
    return f"in={getattr(u, 'input_tokens', '?')} out={getattr(u, 'output_tokens', '?')}" if u else ""
