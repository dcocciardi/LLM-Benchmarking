# inference.py
"""
Low-level interface to llama.cpp (llama-cli).

This module is responsible for:
- launching inference
- capturing raw stdout/stderr
- returning logs for parsing
"""

import subprocess
from pathlib import Path

from config import LLAMA_CLI


def run_llama(
    model_path: Path,
    prompt: str,
    *,
    max_tokens: int = 128,
    threads: int = -1,
    ngl_layers: int = 0,
) -> str:
    """
    Run llama-cli and return raw output.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cmd = [
        str(LLAMA_CLI),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(max_tokens),
        "-t", str(threads),
    ]

    if ngl_layers > 0:
        cmd.extend(["-ngl", str(ngl_layers)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    return result.stdout + result.stderr
