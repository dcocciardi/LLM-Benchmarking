"""
Benchmark utilities for llama-cli.

Runs inference benchmarks and extracts runtime metrics
from llama.cpp logs.
"""

import subprocess
import re
from typing import Dict, List


# ---------------------------
# Parsing utilities
# ---------------------------

def parse_llama_output(output: str) -> Dict[str, float]:
    """
    Parse llama-cli output and extract runtime metrics.

    Returns:
        Dictionary with timing and memory estimates.
    """

    metrics = {
        "Load_s": 0.0,
        "Eval_s": 0.0,
        "TPS": 0.0,
        "ModelRAM_MB": 0.0,
        "KVCache_MB": 0.0,
        "RuntimeRAM_MB": 0.0,
    }

    # Load time
    m = re.search(r"load time\s*=\s*([\d\.]+)\s*ms", output)
    if m:
        metrics["Load_s"] = float(m.group(1)) / 1000.0

    # Eval time + TPS
    for line in output.splitlines():
        if "eval time" in line and "prompt eval" not in line:
            m_eval = re.search(r"eval time\s*=\s*([\d\.]+)\s*ms", line)
            if m_eval:
                metrics["Eval_s"] = float(m_eval.group(1)) / 1000.0

            m_tps = re.search(r"([\d\.]+)\s*(?:tok/s|tokens per second)", line)
            if m_tps:
                metrics["TPS"] = float(m_tps.group(1))
            break

    # Model size
    m_model = re.search(r"model size\s*=\s*([\d\.]+)\s*MB", output)
    if m_model:
        metrics["ModelRAM_MB"] = float(m_model.group(1))

    # KV cache
    m_kv = re.search(r"KV cache\s*=\s*([\d\.]+)\s*MB", output)
    if m_kv:
        metrics["KVCache_MB"] = float(m_kv.group(1))

    metrics["RuntimeRAM_MB"] = (
        metrics["ModelRAM_MB"] + metrics["KVCache_MB"]
    )

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
    max_tokens: int = 128,
    ngl_layers: int = 0,
    temperature: float = 0.7,
) -> List[Dict]:
    """
    Run benchmark on all prompts and return aggregated metrics.
    """

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [p.strip() for p in f if p.strip()]

    if not prompts:
        raise ValueError("Prompt file is empty.")

    all_metrics = []

    for prompt in prompts:
        cmd = [
            llama_cli_path,
            "-m", model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-c", str(context_size),
            "--temp", str(temperature),
            "--ignore-eos",
            "--no-warmup",
        ]

        if ngl_layers > 0:
            cmd.extend(["-ngl", str(ngl_layers)])

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True,
        )

        output = proc.stdout + proc.stderr
        metrics = parse_llama_output(output)
        all_metrics.append(metrics)

    # Aggregate across prompts
    avg = lambda k: sum(m[k] for m in all_metrics) / len(all_metrics)

    return [{
        "Model": model_name,
        "Load_s": avg("Load_s"),
        "Eval_s": avg("Eval_s"),
        "TPS": avg("TPS"),
        "RuntimeRAM_MB": avg("RuntimeRAM_MB"),
        "NumParams_B": None,  # optional, can be filled later
    }]
