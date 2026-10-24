#!/usr/bin/env python3
"""
langfuse_callback.py — opt-in Langfuse tracing for MLX-QUANT benchmarks.

Off by default: nothing is sent unless LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are all set and non-empty in the environment. Standard
library only (urllib.request) — no new dependencies. Uses the same env-var
family and /api/public/trace ingestion endpoint as the fleet's LiteLLM
langfuse callback (crabcc/install/ollama-stack/litellm.config.yaml).
"""

import base64
import json
import os
import threading
import time
import urllib.request
from functools import wraps

_ENV_VARS = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def langfuse_enabled():
    """True only when every LANGFUSE_* env var is set and non-empty."""
    return all(os.environ.get(k) for k in _ENV_VARS)


def langfuse_trace(
    name,
    model=None,
    input_preview=None,
    output_preview=None,
    duration_ms=None,
    metadata=None,
):
    """POST a trace to Langfuse. Fire-and-forget on a daemon thread — never
    blocks the caller; all errors are swallowed. No-op when disabled."""
    if not langfuse_enabled():
        return
    payload = {
        "name": name,
        "timestamp": _iso8601(time.time()),
        "metadata": metadata or {},
    }
    if input_preview is not None:
        payload["input"] = input_preview
    if output_preview is not None:
        payload["output"] = output_preview
    if duration_ms is not None:
        end = time.time()
        start = end - duration_ms / 1000.0
        observation = {
            "name": name,
            "type": "GENERATION",
            "model": model,
            "startTime": _iso8601(start),
            "endTime": _iso8601(end),
        }
        if input_preview is not None:
            observation["input"] = input_preview
        if output_preview is not None:
            observation["output"] = output_preview
        payload["observations"] = [observation]

    threading.Thread(
        target=_post_trace,
        args=(
            os.environ["LANGFUSE_HOST"],
            os.environ["LANGFUSE_PUBLIC_KEY"],
            os.environ["LANGFUSE_SECRET_KEY"],
            payload,
        ),
        daemon=True,
    ).start()


def langfuse_trace_benchmark(fn):
    """Decorator: record a benchmark's name + wall time as a Langfuse trace.
    No-op (and no network) unless the LANGFUSE_* env vars are set."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Opt-in gating: run the benchmark untouched when tracing is off.
        if not langfuse_enabled():
            return fn(*args, **kwargs)
        start = time.time()
        result = fn(*args, **kwargs)
        langfuse_trace(
            name=fn.__name__,
            duration_ms=(time.time() - start) * 1000.0,
            metadata={"args": [str(a) for a in args]},
            output_preview=str(result)[:200] if result is not None else None,
        )
        return result

    return wrapper


def _post_trace(host, public_key, secret_key, payload):
    body = json.dumps(payload).encode("utf-8")
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode(
        "ascii"
    )
    req = urllib.request.Request(
        host.rstrip("/") + "/api/public/trace",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Basic " + credentials,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:
        pass


def _iso8601(epoch):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch)) + "Z"
