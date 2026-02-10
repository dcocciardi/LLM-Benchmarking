"""
Utilities for downloading and preparing LLMs from Hugging Face.

This module handles:
- model download from Hugging Face
- conversion to GGUF format
- optional quantisation

Authentication is NOT handled here.
If a model is gated, the user must login beforehand using:
    huggingface-cli login
"""

from pathlib import Path
import subprocess
import sys

from huggingface_hub import snapshot_download, HfHubHTTPError

from config import (
    MODELS_DIR,
    LLAMA_CPP_ROOT,
    LLAMA_QUANTIZE,
    CONVERT_SCRIPT,
)


# ---------------------------
# Hugging Face download
# ---------------------------

def download_model_from_hf(
    model_id: str,
    *,
    revision: str | None = None,
) -> Path:
    """
    Download a Hugging Face model repository locally.

    Returns the local path to the downloaded model.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=MODELS_DIR / model_id.replace("/", "__"),
            local_dir_use_symlinks=False,
        )
        return Path(local_path)

    except HfHubHTTPError as e:
        print("\n[Hugging Face Hub Error]")
        print("Unable to download the requested model repository.\n")
        print("Possible reasons:")
        print(" - The repository is gated or private")
        print(" - Authentication is required")
        print(" - The repository or revision does not exist\n")
        print("If the repository is gated, please authenticate first using:")
        print("  huggingface-cli login\n")
        raise e


# ---------------------------
# GGUF conversion
# ---------------------------

def convert_to_gguf(
    model_dir: Path,
    output_path: Path,
    *,
    outtype: str = "f16",
) -> Path:
    """
    Convert a Hugging Face model to GGUF using llama.cpp utilities.
    """

    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}"
        )

    cmd = [
        sys.executable,
        str(CONVERT_SCRIPT),
        str(model_dir),
        "--outfile",
        str(output_path),
        "--outtype",
        outtype,
    ]

    subprocess.run(cmd, check=True)
    return output_path


# ---------------------------
# Quantisation
# ---------------------------

def quantise_gguf(
    gguf_path: Path,
    quant_type: str,
    output_path: Path,
) -> Path:
    """
    Quantise a GGUF model using llama.cpp.
    """

    if not LLAMA_QUANTIZE.exists():
        raise FileNotFoundError(
            f"llama-quantize not found at {LLAMA_QUANTIZE}"
        )

    cmd = [
        str(LLAMA_QUANTIZE),
        str(gguf_path),
        str(output_path),
        quant_type,
    ]

    subprocess.run(cmd, check=True)
    return output_path