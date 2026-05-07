"""
Main wrapper for the LLM Edge Benchmark pipeline.

Provides an interactive CLI menu to:
- download models from Hugging Face
- convert models to GGUF
- quantise models
- run llama.cpp benchmarks
- compute perplexity (PPL)
- generate benchmark plots
- run the full automated pipeline
"""

import sys
import csv
from pathlib import Path
from typing import List

import shutil

from config import (
    MODELS_DIR,
    RESULTS_CSV,
    PPL_CSV,
    PROMPT_FILE,
    DATA_DIR,
    LLAMA_CLI,
    LLAMA_PPL,
    SUPPORTED_QUANTS,
)

from model_registry import MODEL_REGISTRY

from hf_utils import (
    download_model_from_hf,
    convert_to_gguf,
)

from benchmark_cli import run_llama_benchmark
from ppl import compute_ppl
from plots import generate_basic_plots


# =========================================================
# Menu utilities
# =========================================================

def print_header():

    print("\n" + "=" * 60)
    print("          LLM EDGE BENCHMARK PIPELINE")
    print("=" * 60 + "\n")


def print_menu():

    print("1) Download model from Hugging Face")
    print("2) Run benchmark")
    print("3) Compute perplexity (PPL)")
    print("4) Generate plots")
    print("5) Run full automated pipeline")
    print("0) Exit\n")


def ask_choice() -> int:

    try:
        return int(input("Select an option: ").strip())

    except ValueError:
        return -1


def ask_list(prompt: str) -> List[str]:

    raw = input(prompt).strip()

    return [
        x.strip()
        for x in raw.split(",")
        if x.strip()
    ]


# =========================================================
# Model selection
# =========================================================

def choose_model_menu() -> dict:
    """
    Display the available benchmark models.

    The user can:
    - choose a predefined model
    - manually enter a custom Hugging Face repository
    """

    print("\n--- Select model ---\n")

    model_items = list(MODEL_REGISTRY.items())

    for i, (model_key, model_info) in enumerate(model_items, start=1):

        print(f"{i}) {model_info['display_name']}")
        print(f"   HF repo : {model_info['hf_repo']}")

        if model_info["gated"]:
            print("   Access  : gated (HF login required)")
        else:
            print("   Access  : public")

        print()

    custom_option = len(model_items) + 1

    print(f"{custom_option}) Enter custom Hugging Face repository")
    print("0) Cancel\n")

    try:
        choice = int(input("Select model: ").strip())

    except ValueError:
        raise ValueError("Invalid selection.")

    if choice == 0:
        raise KeyboardInterrupt("Operation cancelled.")

    # -----------------------------------------------------
    # Predefined model
    # -----------------------------------------------------

    if 1 <= choice <= len(model_items):

        model_key, model_info = model_items[choice - 1]

        return {
            "model_key": model_key,
            "display_name": model_info["display_name"],
            "hf_repo": model_info["hf_repo"],
            "gated": model_info["gated"],
            "is_custom": False,
        }

    # -----------------------------------------------------
    # Custom model
    # -----------------------------------------------------

    if choice == custom_option:

        hf_repo = input(
            "Enter Hugging Face repository "
            "(e.g. microsoft/phi-2): "
        ).strip()

        if not hf_repo or "/" not in hf_repo:
            raise ValueError("Invalid Hugging Face repository.")

        model_key = hf_repo.replace("/", "__")

        return {
            "model_key": model_key,
            "display_name": hf_repo,
            "hf_repo": hf_repo,
            "gated": False,
            "is_custom": True,
        }

    raise ValueError("Selection out of range.")


# =========================================================
# Local model listing
# =========================================================

def list_local_models():

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = list(MODELS_DIR.glob("*"))

    if models:

        print("\nDownloaded models:\n")

        for model in models:
            print(f" - {model.name}")

        print()

    else:

        print("\nNo local models found.\n")


# =========================================================
# Download menu
# =========================================================

def prepare_model_menu():

    print("\n--- Download and convert model from Hugging Face ---")

    try:
        selected_model = choose_model_menu()

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        return

    print("\n[INFO] Selected model")
    print(f"Name    : {selected_model['display_name']}")
    print(f"HF repo : {selected_model['hf_repo']}")

    if selected_model["gated"]:
        print("\n[WARNING]")
        print("This model may require Hugging Face authentication.")
        print("If you do not have access yet, run:")
        print("  huggingface-cli login\n")

    confirm = input("Continue download and F32 GGUF conversion? [y/N]: ").strip().lower()

    if confirm != "y":
        print("\nOperation cancelled.\n")
        return

    try:
        print("\n[INFO] Downloading model from Hugging Face...\n")

        local_path = download_model_from_hf(
            model_id=selected_model["hf_repo"],
        )

        print("\n[INFO] Download completed.")
        print(f"HF model path: {local_path}")

        gguf_output_path = (
            MODELS_DIR
            / selected_model["model_key"]
            / f"{selected_model['model_key']}-F32.gguf"
        )

        print("\n[INFO] Converting model to GGUF F32...")
        print(f"Output path: {gguf_output_path}\n")

        gguf_path = convert_to_gguf(
            model_dir=local_path,
            output_path=gguf_output_path,
            outtype="f32",
        )
        
        print("\n[INFO] Removing original Hugging Face model files...")
        shutil.rmtree(local_path)
        print("[INFO] Hugging Face files removed successfully.")

    except Exception as e:
        print("\n[ERROR] Model preparation failed.")
        print(e)
        return

    print("\nModel preparation completed successfully.")
    print(f"F32 GGUF saved to: {gguf_path}\n")


# =========================================================
# Benchmark menu
# =========================================================

def run_benchmark_menu():

    print("\n--- Run benchmark ---")

    model_name = input(
        "Model label: "
    ).strip()

    quant = input(
        "Quantisation label "
        "(e.g. F16, Q8_0, Q4_K_M): "
    ).strip()

    model_path = Path(
        input("Path to GGUF model: ").strip()
    )

    try:

        ngl_layers = int(
            input(
                "Number of GPU layers (-ngl, default 0): "
            ).strip() or 0
        )

    except ValueError:

        print("Invalid GPU layer count.")
        return

    print("\n[INFO] Running benchmark...\n")

    try:

        result = run_llama_benchmark(
            model_name=model_name,
            quant=quant,
            model_path=str(model_path),
            prompt_file=str(PROMPT_FILE),
            llama_cli_path=str(LLAMA_CLI),
            ngl_layers=ngl_layers,
        )

    except Exception as e:

        print(f"[ERROR] Benchmark failed: {e}")
        return

    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = RESULTS_CSV.exists()

    with open(
        RESULTS_CSV,
        "a",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Model",
                "Quant",
                "NumPrompts",
                "Avg_Load_s",
                "Avg_Eval_s",
                "Avg_TPS",
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    print("\n--- Benchmark completed ---")
    print(f"Model          : {model_name}")
    print(f"Quantisation   : {quant}")
    print(f"Prompts tested : {result['NumPrompts']}")
    print(f"Avg load time  : {result['Avg_Load_s']:.2f} s")
    print(f"Avg eval time  : {result['Avg_Eval_s']:.2f} s")
    print(f"Avg TPS        : {result['Avg_TPS']:.2f} tok/s")
    print(f"Results saved  : {RESULTS_CSV}\n")


# =========================================================
# PPL menu
# =========================================================

def compute_ppl_menu():

    print("\n--- Compute perplexity (PPL) ---")

    model_path = Path(
        input("Path to GGUF model: ").strip()
    )

    try:

        ngl_layers = int(
            input(
                "Number of GPU layers (-ngl, default 0): "
            ).strip() or 0
        )

    except ValueError:

        print("Invalid GPU layer count.")
        return

    print("\n[INFO] Computing perplexity...\n")

    try:

        ppl_value = compute_ppl(
            model_path=model_path,
            llama_perplexity_bin=LLAMA_PPL,
            ngl_layers=ngl_layers,
        )

    except Exception as e:

        print(f"[ERROR] PPL computation failed: {e}")
        return

    file_exists = PPL_CSV.exists()

    with open(
        PPL_CSV,
        "a",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Model",
                "PPL",
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "Model": model_path.name,
            "PPL": ppl_value,
        })

    print("\n--- Perplexity completed ---")
    print(f"Model : {model_path.name}")
    print(f"PPL   : {ppl_value:.4f}")
    print(f"Saved : {PPL_CSV}\n")


# =========================================================
# Plot menu
# =========================================================

def generate_plots_menu():

    print("\n--- Generate plots ---")

    if not RESULTS_CSV.exists():

        print(f"[ERROR] Results CSV not found: {RESULTS_CSV}")
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

    print("\n[INFO] Plot generation completed.\n")


# =========================================================
# Full pipeline
# =========================================================

def full_pipeline_menu():

    print("\n--- Full automated pipeline ---")
    print("Not implemented yet.\n")


# =========================================================
# Main loop
# =========================================================

def main():

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

            print("\nInvalid option.\n")


if __name__ == "__main__":
    main()