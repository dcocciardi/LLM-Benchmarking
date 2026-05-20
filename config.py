from pathlib import Path

# =========================
# PROJECT ROOT
# =========================

# Root del progetto (cartella dove si trova questo file)
PROJECT_ROOT = Path(__file__).resolve().parent


# =========================
# DIRECTORY STRUCTURE
# =========================

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
CORPORA_DIR = DATA_DIR / "corpora"
PLOTS_DIR = DATA_DIR / "plots"

# I modelli stanno a livello root (NON sotto data/)
MODELS_DIR = PROJECT_ROOT / "models"


# =========================
# FILE PATHS
# =========================

RESULTS_CSV = RESULTS_DIR / "results.csv"
PPL_CSV = RESULTS_DIR / "perplexity.csv"

PROMPT_FILE = PROJECT_ROOT / "prompt.txt"


# =========================
# LLAMA.CPP CONFIGURATION
# =========================

import os


def find_llama_cpp_root():
    """
    Automatically locate llama.cpp installation.
    Priority:
    1. LLAMA_CPP_ROOT env variable
    2. Common local paths
    3. Recursive search under home
    """

    env_path = os.environ.get("LLAMA_CPP_ROOT")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path

    candidates = [
        Path.home() / "llama.cpp",
        Path.home() / "llama.cpp2" / "llama.cpp",
        Path.home() / "llm-tirocinio" / "llama.cpp",
        Path.home() / "tools" / "llama.cpp",
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    for path in Path.home().rglob("llama.cpp"):
        if path.is_dir():
            return path.resolve()

    raise RuntimeError(
        "Could not locate llama.cpp automatically.\n"
        "Set it manually with:\n"
        "export LLAMA_CPP_ROOT=/path/to/llama.cpp"
    )


LLAMA_CPP_ROOT = find_llama_cpp_root()

LLAMA_CLI = LLAMA_CPP_ROOT / "build" / "bin" / "llama-cli"
LLAMA_PPL = LLAMA_CPP_ROOT / "build" / "bin" / "llama-perplexity"
LLAMA_QUANTIZE = LLAMA_CPP_ROOT / "build" / "bin" / "llama-quantize"
CONVERT_SCRIPT = LLAMA_CPP_ROOT / "convert_hf_to_gguf.py"

for p in [LLAMA_CLI, LLAMA_PPL, LLAMA_QUANTIZE, CONVERT_SCRIPT]:
    if not p.exists():
        raise RuntimeError(
            f"Missing llama.cpp component: {p}\n"
            f"Detected root: {LLAMA_CPP_ROOT}"
        )
# =========================
# QUANTISATION OPTIONS
# =========================

# Nota:
# - F16 viene gestito in fase di conversione HF -> GGUF
# - le altre passano da llama-quantize
SUPPORTED_QUANTS = ["F16", "Q8_0", "Q4_K_M", "Q2_K"]