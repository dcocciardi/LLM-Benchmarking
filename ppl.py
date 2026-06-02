"""
Perplexity (PPL) evaluation utilities using llama-perplexity.

- Uses WikiText-2 raw test set
- Automatically downloads the corpus if not present
- Relies on centralised paths defined in config.py
"""

from pathlib import Path
import subprocess
import re

from datasets import load_dataset

from config import CORPORA_DIR, LLAMA_PPL


# ---------------------------
# WikiText-2 configuration
# ---------------------------

WIKITEXT2_PATH = CORPORA_DIR / "wikitext2" / "wiki.test.raw"


# ---------------------------
# Corpus utilities
# ---------------------------

def ensure_wikitext2_corpus() -> Path:
    """
    Download WikiText-2 raw test split from Hugging Face datasets
    and save it locally if not already present.
    """

    if WIKITEXT2_PATH.exists():
        return WIKITEXT2_PATH

    print("[INFO] WikiText-2 corpus not found. Downloading...")

    WIKITEXT2_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )

        text = "\n".join(dataset["text"])

        with open(WIKITEXT2_PATH, "w", encoding="utf-8") as f:
            f.write(text)

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
    *,
    context_size: int = 2048,
    batch_size: int = 256,
    ngl_layers: int = 0,
) -> float:
    """
    Compute perplexity for a GGUF model using llama-perplexity.
    """

    if not LLAMA_PPL.exists():
        raise RuntimeError(f"llama-perplexity not found at {LLAMA_PPL}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    corpus_path = ensure_wikitext2_corpus()

    cmd = [
        str(LLAMA_PPL),
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
        check=True,
    )

    output = result.stdout + result.stderr

    patterns = [
        r"perplexity\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"ppl\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return float(match.group(1))

    print("\n[DEBUG] llama-perplexity output:")
    print(output[-2000:])

    raise RuntimeError(
        "Unable to parse perplexity from llama-perplexity output."
    )