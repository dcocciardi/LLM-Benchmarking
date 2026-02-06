# main.py
"""
Main wrapper for the LLM Edge Benchmark pipeline.

This script provides an interactive CLI menu to:
- prepare models (download, convert, quantise)
- run benchmarks via llama-cli
- compute perplexity (PPL)
- generate comparison plots
- run the full pipeline end-to-end

Actual implementations are delegated to dedicated modules.
"""

import sys
from typing import List

from config import SUPPORTED_QUANTS

import csv
from benchmark_cli import run_llama_benchmark
from config import RESULTS_CSV, PROMPT_FILE, LLAMA_CLI

from plots import generate_basic_plots
from config import RESULTS_CSV, DATA_DIR



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


# ---------------------------
# Menu actions (placeholders)
# ---------------------------

def prepare_model_menu():
    print("\n--- Prepare model ---")
    model_id = input(
        "Hugging Face model ID "
        "(e.g. meta-llama/Llama-3.1-8B-Instruct): "
    ).strip()

    print("\nAvailable quantisations:")
    for i, q in enumerate(SUPPORTED_QUANTS, start=1):
        print(f"{i}) {q}")

    selected = ask_list(
        "Select quantisations (comma separated, e.g. 1,3): "
    )

    try:
        quants = [SUPPORTED_QUANTS[int(i) - 1] for i in selected]
    except (IndexError, ValueError):
        print("Invalid quantisation selection.")
        return

    print("\n[INFO]")
    print(f"Model ID       : {model_id}")
    print(f"Quantisations  : {quants}")
    print("Action         : prepare_model() [NOT IMPLEMENTED YET]\n")

    # TODO: call hf_utils.prepare_model()


def run_benchmark_menu():
    print("\n--- Run benchmark ---")

    model_name = input("Model name (label for results): ").strip()
    model_path = input("Path to GGUF model: ").strip()

    try:
        ngl_layers = int(input("Number of GPU layers (-ngl, default 0): ").strip() or 0)
    except ValueError:
        print("Invalid number for GPU layers.")
        return

    print("\n[INFO] Running benchmark...\n")

    try:
        results = run_llama_benchmark(
            model_name=model_name,
            model_path=model_path,
            prompt_file=str(PROMPT_FILE),
            llama_cli_path=str(LLAMA_CLI),
            ngl_layers=ngl_layers,
        )
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        return

    # ---- Save results to CSV ----
    file_exists = RESULTS_CSV.exists()

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["Model", "PromptID", "Load_s", "Eval_s", "TPS"]
        )

        if not file_exists:
            writer.writeheader()

        for row in results:
            writer.writerow(row)

    # ---- Print summary ----
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
    model_path = input("Path to GGUF model: ").strip()

    print("\n[INFO]")
    print(f"Model path     : {model_path}")
    print("Action         : compute_ppl() [NOT IMPLEMENTED YET]\n")

    # TODO: call ppl.compute_ppl()


def generate_plots_menu():
    print("\n--- Generate plots ---")

    if not RESULTS_CSV.exists():
        print(f"[ERROR] Results file not found: {RESULTS_CSV}")
        return

    output_dir = DATA_DIR / "plots"

    try:
        generate_basic_plots(
            csv_path=RESULTS_CSV,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"[ERROR] Plot generation failed: {e}")
        return

    print("[INFO] Plot generation completed.\n")


def full_pipeline_menu():
    print("\n--- Full pipeline ---")
    print(
        "This will execute:\n"
        "  1) Model preparation\n"
        "  2) Benchmark\n"
        "  3) Perplexity computation\n"
        "  4) Plot generation\n"
    )

    confirm = input("Proceed? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.\n")
        return

    print("\n[INFO]")
    print("Action         : full_pipeline() [NOT IMPLEMENTED YET]\n")

    # TODO: orchestrate full pipeline


# ---------------------------
# Main loop
# ---------------------------

def main():
    while True:
        print_header()
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
