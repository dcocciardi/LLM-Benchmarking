from pathlib import Path
import os

# --- PATH BASE ---
# Ottiene la cartella dove si trova questo file (la root del progetto)
PROJECT_ROOT = Path(__file__).parent.resolve()

# --- CARTELLE DATI ---
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
CORPORA_DIR = DATA_DIR / "corpora"
MODELS_DIR = DATA_DIR / "models"
PLOTS_DIR = DATA_DIR / "plots"

# Assicuriamoci che esistano
for d in [DATA_DIR, RESULTS_DIR, CORPORA_DIR, MODELS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- FILE SPECIFICI ---
RESULTS_CSV = RESULTS_DIR / "results.csv"
PROMPT_FILE = PROJECT_ROOT / "prompt.txt"  # Assicurati che prompt.txt sia nella cartella principale

# --- CONFIGURAZIONE LLAMA.CPP ---
# Definiamo dove sono i binari compilati.
# Se non li trovi, modifica questo percorso con quello assoluto:
# es: Path("/home/dcocciardi/llama.cpp")
LLAMA_CPP_ROOT = Path.home() / "llama.cpp" 

LLAMA_CLI = LLAMA_CPP_ROOT / "build" / "bin" / "llama-cli"
LLAMA_PPL = LLAMA_CPP_ROOT / "build" / "bin" / "llama-perplexity"
LLAMA_QUANTIZE = LLAMA_CPP_ROOT / "build" / "bin" / "llama-quantize"
CONVERT_SCRIPT = LLAMA_CPP_ROOT / "convert_hf_to_gguf.py"

# --- OPZIONI ---
SUPPORTED_QUANTS = ["F16", "Q8_0", "Q4_K_M", "Q2_K"]