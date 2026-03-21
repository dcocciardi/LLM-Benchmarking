"""
Main wrapper for the LLM Edge Benchmark pipeline.

Provides an interactive CLI menu to:
- prepare models (download, convert, quantise) [WIP]
- run benchmarks via llama-cli
- compute perplexity (PPL)
- generate comparison plots
"""

import sys
import csv
from pathlib import Path
from typing import List

from config import MODELS_DIR
from hf_utils import download_model_from_hf

from config import (
    SUPPORTED_QUANTS,
    RESULTS_CSV,
    PPL_CSV,
    PROMPT_FILE,
    DATA_DIR,
    LLAMA_CLI,
)

from benchmark_cli import run_llama_benchmark
from plots import generate_basic_plots
from ppl import compute_ppl


HF_MODELS = {
    "1": ("Phi-3 Mini 3.8B", "microsoft/Phi-3-mini-4k-instruct"),
    "2": ("Gemma 3 4B", "google/gemma-3-4b-it"),
    "3": ("Gemma 2 2B", "google/gemma-2-2b-it"),
    "4": ("Mistral 7B", "mistralai/Mistral-7B-v0.1"),
    "5": ("Llama 3.1 8B", "meta-llama/Meta-Llama-3.1-8B"),
    "6": ("ShearedLLaMA 2.7B", "princeton-nlp/Sheared-LLaMA-2.7B"),
    "7": ("Phi-2 2.7B", "microsoft/phi-2"),
    "8": ("Qwen 2 1.5B", "Qwen/Qwen2-1.5B"),
}

# ---------------------------
# Menu utilities
# ---------------------------

def print_header():
    print("\n" + "=" * 50)
    print(" LLM EDGE BENCHMARK PIPELINE ")
    print("=" * 50 + "\n")


def print_menu():
    print("1) Prepare model (download, convert, quantise)")
    print("2) Run benchmark (llama-cli)")
    print("3) Compute perplexity (PPL)")
    print("4) Generate plots")
    print("5) Run full pipeline (1 → 4)")
    print("0) Exit\n")


def ask_choice() -> int:
    try:
        return int(input("Select an option: ").strip())
    except ValueError:
        return -1


def ask_list(prompt: str) -> List[str]:
    raw = input(prompt).strip()
    return [x.strip() for x in raw.split(",") if x.strip()]


def list_local_models():

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = list(MODELS_DIR.glob("*"))

    if models:

        print("\nAlready downloaded models:\n")

        for m in models:
            print(f" - {m.name}")

        print()

    else:

        print("\nNo model was found in the models directory.")
        print("Please download at least one model to start the benchmark.\n")

# ---------------------------
# Menu actions
# ---------------------------

def prepare_model_menu():

    print("\n--- Download model from HuggingFace ---\n")

    for k, (name, _) in HF_MODELS.items():
        print(f"{k}) {name}")

    print("9) Enter HuggingFace repository manually\n")

    choice = input("Select model to download: ").strip()

    if choice in HF_MODELS:

        model_name, repo = HF_MODELS[choice]

        print(f"\n[INFO] Downloading {model_name} from HuggingFace...\n")

        try:

            local_path = download_model_from_hf(repo)

            print("\nDownload completed.")
            print(f"Model saved to: {local_path}\n")

        except Exception as e:

            print("\n[ERROR] Download failed:")
            print(e)

    elif choice == "9":

        repo = input("Enter HuggingFace repository (e.g. org/model): ").strip()

        try:

            local_path = download_model_from_hf(repo)

            print("\nDownload completed.")
            print(f"Model saved to: {local_path}\n")

        except Exception as e:

            print("\n[ERROR] Download failed:")
            print(e)

    else:

        print("\nInvalid selection.\n")


def run_benchmark_menu():
    print("\n--- Run benchmark ---")

    model_name = input("Model name (label for results): ").strip()
    model_path = Path(input("Path to GGUF model: ").strip())

    try:
        ngl_layers = int(
            input("Number of GPU layers (-ngl, default 0): ").strip() or 0
        )
    except ValueError:
        print("Invalid number for GPU layers.")
        return

    print("\n[INFO] Running benchmark...\n")

    try:
        results = run_llama_benchmark(
            model_name=model_name,
            model_path=str(model_path),
            prompt_file=str(PROMPT_FILE),
            llama_cli_path=str(LLAMA_CLI),
            ngl_layers=ngl_layers,
        )
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        return

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    file_exists = RESULTS_CSV.exists()

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Model",
                "Quant",
                "PromptID",
                "PromptText",
                "Load_s",
                "Eval_s",
                "TPS",
                "ModelRAM_MB",
                "KVCache_MB",
                "RuntimeRAM_MB",
                "NumParams_B",
            ]
        )

        if not file_exists:
            writer.writeheader()

        for row in results:
            writer.writerow(row)

    avg_tps = sum(r["TPS"] for r in results) / len(results)
    avg_load = sum(r["Load_s"] for r in results) / len(results)

    print("\n--- Benchmark completed ---")
    print(f"Model          : {model_name}")
    print(f"Prompts tested : {len(results)}")
    print(f"Avg load time  : {avg_load:.2f} s")
    print(f"Avg TPS        : {avg_tps:.2f} tok/s")
    print(f"Results saved  : {RESULTS_CSV}\n")


def compute_ppl_menu():
    print("\n--- Compute perplexity (PPL) ---")

    model_path = Path(input("Path to GGUF model: ").strip())

    try:
        ngl_layers = int(
            input("Number of GPU layers (-ngl, default 0): ").strip() or 0
        )
    except ValueError:
        print("Invalid number for GPU layers.")
        return

    print("\n[INFO] Computing perplexity...\n")

    try:
        ppl_value = compute_ppl(
            model_path=model_path,
            ngl_layers=ngl_layers,
        )
    except Exception as e:
        print(f"[ERROR] PPL computation failed: {e}")
        return

    file_exists = PPL_CSV.exists()

    with open(PPL_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["Model", "PPL"]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "Model": model_path.name,
            "PPL": ppl_value,
        })

    print("\n--- Perplexity result ---")
    print(f"Model : {model_path.name}")
    print(f"PPL   : {ppl_value:.4f}")
    print(f"Saved : {PPL_CSV}\n")


def generate_plots_menu():
    print("\n--- Generate plots ---")

    if not RESULTS_CSV.exists():
        print(f"[ERROR] Results file not found: {RESULTS_CSV}")
        return

    try:
        generate_basic_plots(
            results_csv=RESULTS_CSV,
            ppl_csv=PPL_CSV,
            output_dir=DATA_DIR / "plots",
        )
    except Exception as e:
        print(f"[ERROR] Plot generation failed: {e}")
        return

    print("[INFO] Plot generation completed.\n")


def full_pipeline_menu():
    print("\n--- Full pipeline ---")
    print("Not implemented yet.\n")


# ---------------------------
# Main loop
# ---------------------------

def main():

    list_local_models()

    while True:
        print_header()
        list_local_models()
        print_menu()
        choice = ask_choice()

        if choice == 1:
            prepare_model_menu()
        elif choice == 2:
            run_benchmark_menu()
        elif choice == 3:
            compute_ppl_menu()
        elif choice == 4:
            generate_plots_menu()
        elif choice == 5:
            full_pipeline_menu()
        elif choice == 0:
            print("\nExiting.")
            sys.exit(0)
        else:
            print("\nInvalid option. Please try again.\n")


if __name__ == "__main__":
    main()
