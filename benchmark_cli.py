# benchmark_cli.py
"""
Benchmark utilities for llama-cli.

This module provides a reusable function to benchmark a GGUF model
using llama-cli and extract runtime metrics such as:
- load time
- eval time
- tokens/sec (TPS)

Derived and refactored from test_cli.py.
"""

import subprocess
import re
from typing import List, Dict


# ---------------------------
# Parsing utilities
# ---------------------------

def parse_llama_output(stderr_output: str) -> Dict[str, float]:
    """
    Parse llama-cli stderr output and extract benchmark metrics.
    Returns values in seconds and tokens/sec.
    """

    metrics = {
        "load_s": 0.0,
        "eval_s": 0.0,
        "tokens": 0,
        "tps": 0.0,
    }

    # Load time
    load_match = re.search(r"load time\s*=\s*([\d\.]+)\s*ms", stderr_output)
    if load_match:
        metrics["load_s"] = float(load_match.group(1)) / 1000.0

    # Eval time (generation only, excluding prompt eval)
    eval_line = None
    for line in stderr_output.splitlines():
        if "eval time" in line and "prompt eval" not in line:
            eval_line = line
            break

    if eval_line:
        eval_match = re.search(r"eval time\s*=\s*([\d\.]+)\s*ms", eval_line)
        if eval_match:
            metrics["eval_s"] = float(eval_match.group(1)) / 1000.0

        tps_match = re.search(
            r"\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*(?:tok/s|tokens per second)\)",
            eval_line
        )
        if tps_match:
            metrics["tps"] = float(tps_match.group(1))

        tokens_match = re.search(
            r"eval time\s*=\s*[\d\.]+\s*ms\s*/\s*(\d+)\s*runs",
            eval_line
        )
        if tokens_match:
            metrics["tokens"] = int(tokens_match.group(1))

    return metrics


# ---------------------------
# Benchmark runner
# ---------------------------

def run_llama_benchmark(
    model_name: str,
    model_path: str,
    prompt_file: str,
    llama_cli_path: str,
    *,
    context_size: int = 2048,
    max_tokens: int = 100,
    ngl_layers: int = 0,
    temperature: float = 0.7,
) -> List[Dict]:
    """
    Run llama-cli benchmark on all prompts in a file.

    Returns a list of dictionaries, one per prompt.
    """

    results = []

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    if not prompts:
        raise ValueError("No prompts found in prompt file.")

    for prompt_id, prompt_text in enumerate(prompts, start=1):
        command = [
            llama_cli_path,
            "-m", model_path,
            "-p", prompt_text,
            "-n", str(max_tokens),
            "-c", str(context_size),
            "--temp", str(temperature),
            "--ignore-eos",
            "--no-warmup",
            "-no-cnv",
        ]

        if ngl_layers > 0:
            command.extend(["-ngl", str(ngl_layers)])

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        stderr = process.stderr
        metrics = parse_llama_output(stderr)

        # Debug safeguard
        if metrics["tps"] == 0.0 and metrics["tokens"] == 0:
            print(f"[WARN] llama-cli produced zero metrics for prompt {prompt_id}")
            print(stderr)

        results.append({
            "Model": model_name,
            "PromptID": prompt_id,
            "Load_s": metrics["load_s"],
            "Eval_s": metrics["eval_s"],
            "TPS": metrics["tps"],
        })

    return results
