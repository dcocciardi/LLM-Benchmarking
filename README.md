# LLM Edge Benchmarking Pipeline

A reproducible benchmarking framework for evaluating quantised Large Language Models (LLMs) on resource-constrained hardware using **llama.cpp**.

This project was developed to support systematic experimentation on embedded and heterogeneous systems, enabling controlled benchmarking of inference performance, memory consumption, perplexity, and hardware compatibility across multiple model architectures and quantisation levels.

---

## Features

- Download models directly from Hugging Face
- Automatic conversion from Hugging Face checkpoints to GGUF format
- Local quantisation using llama.cpp
- Controlled inference benchmarking with standardised prompts
- Perplexity (PPL) evaluation
- CSV-based result aggregation
- Automatic plot generation
- Hardware-aware model recommendations using **LLMFit**
- Interactive CLI workflow
- Cross-platform compatibility (x86_64 / ARM64)

---

## Supported functionality

### Model acquisition
The pipeline can download models directly from Hugging Face repositories.

Supported workflows:

- predefined model registry
- manual Hugging Face repository input
- gated model warning support

Downloaded checkpoints are automatically converted into GGUF format.

---

### Quantisation
The framework supports multiple quantisation formats using llama.cpp:

- F16
- Q8_0
- Q4_K_M
- Q2_K

All quantised variants are generated locally from the same baseline GGUF model to ensure methodological consistency.

---

### Inference benchmarking
Inference is executed through **llama-cli** under controlled conditions.

Collected metrics include:

- model loading time
- prompt evaluation time
- generation throughput (tokens/sec)
- total evaluation time
- runtime memory usage
- peak memory usage

Benchmarks are executed over a predefined prompt set (`prompt.txt`) to minimise variance caused by prompt-specific behaviour.

---

### Perplexity evaluation
Perplexity computation is supported through **llama-perplexity**.

Typical usage:

- WikiText-2
- custom corpora

Results are automatically exported to CSV.

---

### Plot generation
The pipeline can automatically generate visualisations from collected benchmark data.

Example plots:

- perplexity vs throughput
- memory usage vs parameter count
- throughput vs quantisation level
- memory vs performance tradeoff

---

### Hardware-aware recommendations
The project integrates **LLMFit** for architecture-aware model recommendation.

At startup:

- host hardware is automatically analysed
- recommended models are generated
- cache files are stored locally
- future launches reuse cached recommendations

Users can manually refresh recommendations from the interactive menu.

These recommendations are advisory only and do not block execution.

---

## Project architecture

```text
LLM-Benchmarking/
├── main.py                  # interactive CLI entry point
├── config.py               # configuration paths and tool discovery
├── hf_utils.py             # Hugging Face download + GGUF conversion
├── inference.py            # inference benchmarking logic
├── ppl.py                  # perplexity evaluation
├── plots.py                # graph generation
├── llmfit_utils.py         # hardware recommendation integration
├── prompt.txt              # benchmark prompts
├── requirements.txt
│
├── models/                 # downloaded / converted models
│
├── data/
│   ├── corpora/            # benchmark corpora
│   ├── plots/              # generated figures
│   ├── results/
│   │   ├── results.csv
│   │   └── perplexity.csv
│   └── system/
│       ├── llmfit_recommendations.json
│       └── llmfit_recommendations.txt
│
└── README.md
```

---

## Requirements

### Software

Required:

- Python 3.10+
- Git
- llama.cpp
- Hugging Face account (optional for gated models)

Optional:

- LLMFit
- CUDA toolkit (for GPU acceleration)

---

### Hardware

Tested environments include:

- NVIDIA Jetson Orin Nano (ARM64 embedded)
- NVIDIA RTX 4090 workstation

Other systems may work depending on available RAM / VRAM.

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/dcocciardi/LLM-Benchmarking.git
cd LLM-Benchmarking
```

---

### 2. Create virtual environment

```bash
python3 -m venv llm-venv
source llm-venv/bin/activate
```

---

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Build llama.cpp

Clone llama.cpp:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Build with CUDA support:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

Then export:

```bash
export LLAMA_CPP_ROOT=/path/to/llama.cpp
```

The pipeline can also attempt automatic discovery of llama.cpp installations.

---

### 5. Optional: install LLMFit

If not installed, the pipeline will ask to install it automatically.

Manual installation:

```bash
curl -fsSL https://llmfit.com/install.sh | sh
```

Ensure it is in PATH:

```bash
export PATH=$HOME/.local/bin:$PATH
```

---

## Running the pipeline

Launch:

```bash
python main.py
```

Menu:

```text
1) Download model from Hugging Face
2) Run benchmark
3) Compute perplexity (PPL)
4) Generate plots
5) Run full automated pipeline
6) Show LLMFit recommendations
7) Refresh LLMFit recommendations
0) Exit
```

---

## Example workflow

Typical workflow:

### Download a model

```text
Option 1
→ select Qwen 2 1.5B
→ convert to GGUF
→ generate quantised variants
```

---

### Run benchmark

```text
Option 2
→ select quantisation
→ run prompt benchmark
→ collect CSV metrics
```

---

### Compute perplexity

```text
Option 3
→ choose dataset
→ run llama-perplexity
→ store results
```

---

### Generate plots

```text
Option 4
→ visualise tradeoffs
```

---

## Metrics collected

### Performance

- loading time
- prompt eval time
- generation speed
- total runtime
- memory usage
- peak memory usage

---

### Quality

- perplexity (PPL)

---

## Reproducibility

The framework was designed to maximise reproducibility:

- fixed llama.cpp toolchain
- deterministic local quantisation
- standardised prompts
- automated CSV persistence
- unified benchmarking workflow
- architecture-aware compatibility checks

No external pre-quantised checkpoints are required.

---

## Limitations

- Perplexity does not directly measure generative usefulness
- Benchmark performance depends on prompt characteristics
- LLMFit recommendations are heuristic, not guarantees
- Some Hugging Face models require authentication
- Extremely large models may exceed local hardware limits

---

## Future work

Potential extensions:

- HumanEval integration
- MMLU support
- HLE benchmark support
- accuracy benchmarking
- Ollama backend support
- automated multi-device comparison
- structured experiment export
- dashboard visualisation
