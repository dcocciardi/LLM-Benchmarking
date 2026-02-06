# plots.py
"""
Plotting utilities for LLM benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def generate_basic_plots(
    csv_path: Path,
    output_dir: Path,
):
    """
    Generate basic benchmark plots:
    - Load time per prompt
    - TPS per prompt
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Results CSV is empty. Nothing to plot.")

    models = df["Model"].unique()

    # -------- Load Time Plot --------
    plt.figure(figsize=(12, 6))
    for model in models:
        model_data = df[df["Model"] == model]
        plt.plot(
            model_data["PromptID"],
            model_data["Load_s"],
            marker="o",
            label=model
        )

    plt.xlabel("Prompt ID")
    plt.ylabel("Load Time (s)")
    plt.title("Load Time per Prompt by Model")
    plt.grid(True)
    plt.xticks(df["PromptID"].unique())
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "load_time.png", dpi=300)
    plt.close()

    # -------- TPS Plot --------
    plt.figure(figsize=(12, 6))
    for model in models:
        model_data = df[df["Model"] == model]
        plt.plot(
            model_data["PromptID"],
            model_data["TPS"],
            marker="o",
            label=model
        )

    plt.xlabel("Prompt ID")
    plt.ylabel("Tokens / second (TPS)")
    plt.title("TPS per Prompt by Model")
    plt.grid(True)
    plt.xticks(df["PromptID"].unique())
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tps.png", dpi=300)
    plt.close()

    print(f"[INFO] Plots saved to: {output_dir}")
