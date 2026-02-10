"""
Plotting utilities for LLM benchmark analysis.

Generates comparison plots across models:
1) Perplexity vs Tokens/sec
2) 1 / Perplexity vs Runtime RAM
3) Runtime RAM vs Number of Parameters
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def generate_basic_plots(
    results_csv: Path,
    ppl_csv: Path,
    output_dir: Path,
):
    """
    Generate comparison plots from benchmark and perplexity results.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Load data
    # ---------------------------

    df_results = pd.read_csv(results_csv)
    df_ppl = pd.read_csv(ppl_csv)

    if df_results.empty or df_ppl.empty:
        raise ValueError("Input CSV files are empty.")

    # ---------------------------
    # Aggregate benchmark results
    # ---------------------------

    bench = (
        df_results
        .groupby("Model", as_index=False)
        .agg({
            "TPS": "mean",
            "RuntimeRAM_MB": "mean",
            "NumParams_B": "first",
        })
    )

    data = bench.merge(df_ppl, on="Model", how="inner")

    # ---------------------------
    # Plot 1: PPL vs TPS
    # ---------------------------

    plt.figure(figsize=(8, 6))
    plt.scatter(data["TPS"], data["PPL"])

    for _, row in data.iterrows():
        plt.annotate(
            row["Model"],
            (row["TPS"], row["PPL"]),
            fontsize=9,
            alpha=0.8,
        )

    plt.xlabel("Tokens / second")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs Throughput")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "ppl_vs_tps.png", dpi=300)
    plt.close()

    # ---------------------------
    # Plot 2: 1/PPL vs Runtime RAM
    # ---------------------------

    plt.figure(figsize=(8, 6))
    plt.scatter(data["RuntimeRAM_MB"], 1.0 / data["PPL"])

    for _, row in data.iterrows():
        plt.annotate(
            row["Model"],
            (row["RuntimeRAM_MB"], 1.0 / row["PPL"]),
            fontsize=9,
            alpha=0.8,
        )

    plt.xlabel("Runtime RAM (MB)")
    plt.ylabel("1 / Perplexity")
    plt.title("Accuracy vs Memory Footprint")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "inv_ppl_vs_ram.png", dpi=300)
    plt.close()

    # ---------------------------
    # Plot 3: Runtime RAM vs #Parameters
    # ---------------------------

    plt.figure(figsize=(8, 6))
    plt.scatter(data["NumParams_B"], data["RuntimeRAM_MB"])

    for _, row in data.iterrows():
        plt.annotate(
            row["Model"],
            (row["NumParams_B"], row["RuntimeRAM_MB"]),
            fontsize=9,
            alpha=0.8,
        )

    plt.xlabel("Number of Parameters (B)")
    plt.ylabel("Runtime RAM (MB)")
    plt.title("Model Size vs Memory Footprint")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "ram_vs_params.png", dpi=300)
    plt.close()

    print(f"[INFO] Plots saved to: {output_dir}")
