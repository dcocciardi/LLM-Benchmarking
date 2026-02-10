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

# Directory root di llama.cpp (buildata manualmente)
LLAMA_CPP_ROOT = Path.home() / "llama.cpp"

if not LLAMA_CPP_ROOT.exists():
    raise RuntimeError(
        f"llama.cpp not found at {LLAMA_CPP_ROOT}. "
        "Please update LLAMA_CPP_ROOT in config.py"
    )

LLAMA_CLI = LLAMA_CPP_ROOT / "build" / "bin" / "llama-cli"
LLAMA_PPL = LLAMA_CPP_ROOT / "build" / "bin" / "llama-perplexity"
LLAMA_QUANTIZE = LLAMA_CPP_ROOT / "build" / "bin" / "llama-quantize"
CONVERT_SCRIPT = LLAMA_CPP_ROOT / "convert_hf_to_gguf.py"

for p in [LLAMA_CLI, LLAMA_PPL, LLAMA_QUANTIZE, CONVERT_SCRIPT]:
    if not p.exists():
        raise RuntimeError(f"Missing llama.cpp component: {p}")


# =========================
# QUANTISATION OPTIONS
# =========================

# Nota:
# - F16 viene gestito in fase di conversione HF -> GGUF
# - le altre passano da llama-quantize
SUPPORTED_QUANTS = ["F16", "Q8_0", "Q4_K_M", "Q2_K"]