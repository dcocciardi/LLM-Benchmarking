# ppl.py
"""
Perplexity (PPL) evaluation utilities using llama-perplexity.

Uses a standard language modelling corpus (WikiText-2 raw).
If the corpus is not found locally, it is downloaded automatically.
"""

from pathlib import Path
import subprocess
import re
import urllib.request


WIKITEXT2_URL = (
    "https://huggingface.co/datasets/wikitext/resolve/main/"
    "wikitext-2-raw-v1/wiki.test.raw"
)


# ---------------------------
# Corpus utilities
# ---------------------------

def ensure_wikitext2_corpus(corpus_path: Path) -> None:
    """
    Ensure that the WikiText-2 raw test corpus exists locally.
    If not, download it from Hugging Face.
    """

    if corpus_path.exists():
        return

    print("[INFO] WikiText-2 corpus not found. Downloading...")

    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(WIKITEXT2_URL, corpus_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download WikiText-2 corpus: {e}"
        )

    print("[INFO] WikiText-2 corpus downloaded successfully.")


# ---------------------------
# Perplexity computation
# ---------------------------

def compute_ppl(
    model_path: Path,
    corpus_path: Path,
    llama_perplexity_bin: Path,
    *,
    context_size: int = 2048,
    batch_size: int = 256,
    ngl_layers: int = 0,
) -> float:
    """
    Compute perplexity using llama-perplexity.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not llama_perplexity_bin.exists():
        raise FileNotFoundError(
            f"llama-perplexity binary not found: {llama_perplexity_bin}"
        )

    # Ensure corpus is available
    ensure_wikitext2_corpus(corpus_path)

    cmd = [
        str(llama_perplexity_bin),
        "-m", str(model_path),
        "-f", str(corpus_path),
        "-c", str(context_size),
        "-b", str(batch_size),
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

    output = result.stdout + result.stderr

    # Typical output:
    # "perplexity = 12.3456"
    match = re.search(r"perplexity\s*=\s*([\d\.]+)", output)
    if not match:
        raise RuntimeError(
            "Unable to parse perplexity from llama-perplexity output."
        )

    return float(match.group(1))
