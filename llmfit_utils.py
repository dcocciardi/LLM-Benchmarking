"""
LLMFit integration utilities.

This module:
- checks whether llmfit is installed
- installs llmfit if missing
- generates hardware-aware recommendations
- caches recommendations as JSON and TXT
- checks whether a selected model appears in the recommendation list
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass

from config import DATA_DIR


LOCAL_BIN = os.path.expanduser("~/.local/bin")
LLMFIT_INSTALL_CMD = (
    "curl -fsSL https://llmfit.axjns.dev/install.sh | sh -s -- --local"
)

SYSTEM_DIR = DATA_DIR / "system"
LLMFIT_JSON = SYSTEM_DIR / "llmfit_recommendations.json"
LLMFIT_TXT = SYSTEM_DIR / "llmfit_recommendations.txt"


@dataclass
class LLMFitResult:
    available: bool
    can_run: bool | None
    raw_output: str
    message: str


def refresh_local_bin_in_path() -> None:
    """Make ~/.local/bin visible to the current Python process."""
    current_path = os.environ.get("PATH", "")

    if LOCAL_BIN not in current_path.split(":"):
        os.environ["PATH"] = f"{LOCAL_BIN}:{current_path}"


def is_llmfit_available() -> bool:
    """Check whether llmfit is available."""
    refresh_local_bin_in_path()
    return shutil.which("llmfit") is not None


def install_llmfit() -> bool:
    """Install llmfit locally using the official installer."""
    print("\n[INFO] Installing llmfit...")

    try:
        subprocess.run(
            LLMFIT_INSTALL_CMD,
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] llmfit installation failed: {e}")
        return False

    refresh_local_bin_in_path()

    if is_llmfit_available():
        print("[OK] llmfit installed successfully.")
        return True

    print("[WARNING] llmfit installation completed, but llmfit was not found.")
    return False


def ensure_llmfit_available(auto_install: bool = True) -> bool:
    """Ensure that llmfit is available."""
    if is_llmfit_available():
        return True

    print("\n[INFO] llmfit not found.")

    if not auto_install:
        return False

    choice = input("Install llmfit automatically? [Y/n]: ").strip().lower()

    if choice == "n":
        return False

    return install_llmfit()


def generate_llmfit_recommendations() -> list:
    """
    Run llmfit and save hardware recommendations to JSON and TXT files.
    """

    if not ensure_llmfit_available(auto_install=True):
        print("[WARNING] llmfit unavailable. Recommendations will not be generated.")
        return []

    SYSTEM_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Generating LLMFit hardware recommendations...")

    result = subprocess.run(
        [
            "llmfit",
            "recommend",
            "--use-case", "general",
            "--runtime", "llamacpp",
            "--min-fit", "marginal",
            "--limit", "50",
            "--json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )

    output = result.stdout.strip()

    if result.returncode != 0 or not output:
        print("[WARNING] llmfit recommendation generation failed.")
        print(result.stderr)
        return []

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        print("[WARNING] Could not parse llmfit JSON output.")
        print(output[:1000])
        return []

    with open(LLMFIT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    write_recommendations_txt(data)

    print(f"[OK] LLMFit JSON saved to: {LLMFIT_JSON}")
    print(f"[OK] LLMFit TXT saved to : {LLMFIT_TXT}")

    return normalise_recommendations(data)


def write_recommendations_txt(data) -> None:
    """Write a human-readable TXT file with recommended models."""

    recommendations = normalise_recommendations(data)

    with open(LLMFIT_TXT, "w", encoding="utf-8") as f:
        f.write("LLMFit recommended models for this architecture\n")
        f.write("=" * 60 + "\n\n")

        if not recommendations:
            f.write("No recommendations found.\n")
            return

        for i, item in enumerate(recommendations, start=1):
            f.write(f"{i}) {item.get('name', 'unknown')}\n")

            for key, value in item.items():
                if key != "name":
                    f.write(f"   {key}: {value}\n")

            f.write("\n")


def load_llmfit_recommendations() -> list:
    """
    Load cached LLMFit recommendations.
    If missing, generate them.
    """

    if not LLMFIT_JSON.exists():
        return generate_llmfit_recommendations()

    try:
        with open(LLMFIT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        return normalise_recommendations(data)

    except Exception as e:
        print(f"[WARNING] Could not load LLMFit cache: {e}")
        return []


def normalise_recommendations(data) -> list[dict]:
    """
    Convert possible llmfit JSON structures into a list of dictionaries.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["recommendations", "models", "results", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    return []


def extract_model_strings(item: dict) -> list[str]:
    """
    Extract possible model identifiers from a recommendation item.
    """

    keys = [
        "name",
        "model",
        "model_name",
        "id",
        "repo",
        "repository",
        "hf_repo",
    ]

    values = []

    for key in keys:
        value = item.get(key)
        if isinstance(value, str):
            values.append(value.lower())

    return values


def is_model_recommended(selected_model: dict) -> bool:
    """
    Check whether the selected model appears in the cached LLMFit recommendations.
    """

    recommendations = load_llmfit_recommendations()

    if not recommendations:
        return False

    hf_repo = selected_model["hf_repo"].lower()
    display_name = selected_model["display_name"].lower()
    model_key = selected_model["model_key"].lower()

    search_terms = {
        hf_repo,
        display_name,
        model_key,
        hf_repo.split("/")[-1],
    }

    for item in recommendations:
        candidate_strings = extract_model_strings(item)

        for candidate in candidate_strings:
            for term in search_terms:
                if term and (term in candidate or candidate in term):
                    return True

    return False


def print_llmfit_cache_summary(limit: int = 20) -> None:
    """Print a short summary of cached recommendations."""

    recommendations = load_llmfit_recommendations()

    if not recommendations:
        print("\n[llmfit] No cached recommendations available.\n")
        return

    print("\n[llmfit] Recommended models for this architecture:\n")

    for i, item in enumerate(recommendations[:limit], start=1):
        name = (
            item.get("name")
            or item.get("model")
            or item.get("id")
            or item.get("repo")
            or "unknown"
        )

        print(f"{i}) {name}")

    print()