# ppl.py
"""
Perplexity (PPL) evaluation utilities using llama-perplexity.

- Uses WikiText-2 raw test set (standard LM benchmark)
- Automatically downloads the corpus if not present
- Relies on centralised paths defined in config.py
"""

from pathlib import Path
import subprocess
import re
import urllib.request

from config import CORPORA_DIR


# ---------------------------
# WikiText-2 configuration
# ---------------------------

WIKITEXT2_URL = (
    "https://huggingface.co/datasets/wikitext/resolve/main/"
    "wikitext-2-raw-v1/wiki.test.raw"
)

WIKITEXT2_PATH = CORPORA_DIR / "wikitext2" / "wiki.test.raw"


# ---------------------------
# Corpus utilities
# ---------------------------

def ensure_wikitext2_corpus() -> Path:
    """
    Ensure that the WikiText-2 raw test corpus exists locally.
    If not, download it from Hugging Face.

    Returns:
        Path to the corpus file.
    """

    if WIKITEXT2_PATH.exists():
        return WIKITEXT2_PATH

    print("[INFO] WikiText-2 corpus not found. Downloading...")

    WIKITEXT2_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(WIKITEXT2_URL, WIKITEXT2_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download WikiText-2 corpus: {e}"
        )

    print("[INFO] WikiText-2 corpus downloaded successfully.")
    return WIKITEXT2_PATH


# ---------------------------
# Perplexity computation
# ---------------------------

def compute_ppl(
    model_path: Path,
    llama_perplexity_bin: Path,
    *,
    context_size: int = 2048,
    batch_size: int = 256,
    ngl_layers: int = 0,
) -> float:
    """
    Compute perplexity for a GGUF model using llama-perplexity.

    Args:
        model_path: Path to the GGUF model.
        llama_perplexity_bin: Path to llama-perplexity binary.
        context_size: Context window size.
        batch_size: Batch size.
        ngl_layers: Number of GPU layers.

    Returns:
        Perplexity value (float).
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not llama_perplexity_bin.exists():
        raise FileNotFoundError(
            f"llama-perplexity binary not found: {llama_perplexity_bin}"
        )

    corpus_path = ensure_wikitext2_corpus()

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

    # Expected output line:
    # "perplexity = 12.3456"
    match = re.search(r"perplexity\s*=\s*([\d\.]+)", output)
    if not match:
        raise RuntimeError(
            "Unable to parse perplexity from llama-perplexity output."
        )

    return float(match.group(1))
