# model_registry.py
"""
Registry dei modelli benchmarkati nel progetto.

Ogni modello contiene:
- display_name : nome leggibile
- hf_repo      : repository Hugging Face
- gated        : True se richiede autenticazione HF
"""

MODEL_REGISTRY = {
    "phi-3-mini-3.8b": {
        "display_name": "Phi-3 Mini 3.8B",
        "hf_repo": "microsoft/Phi-3-mini-128k-instruct",
        "gated": False,
    },

    "gemma-3-4b": {
        "display_name": "Gemma 3 4B",
        "hf_repo": "google/gemma-3-4b-it",
        "gated": True,
    },

    "gemma-2-2b": {
        "display_name": "Gemma 2 2B",
        "hf_repo": "google/gemma-2-2b",
        "gated": True,
    },

    "mistral-7b-v0.3": {
        "display_name": "Mistral 7B v0.3",
        "hf_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "gated": False,
    },

    "llama-3.1-8b": {
        "display_name": "Llama 3.1 8B",
        "hf_repo": "meta-llama/Llama-3.1-8B",
        "gated": True,
    },

    "sheared-llama-2.7b": {
        "display_name": "ShearedLLaMA 2.7B",
        "hf_repo": "princeton-nlp/Sheared-LLaMA-2.7B",
        "gated": False,
    },

    "phi-2-2.7b": {
        "display_name": "Phi-2 2.7B",
        "hf_repo": "microsoft/phi-2",
        "gated": False,
    },

    "qwen-2-1.5b": {
        "display_name": "Qwen 2 1.5B",
        "hf_repo": "Qwen/Qwen2-1.5B-Instruct",
        "gated": False,
    },
}