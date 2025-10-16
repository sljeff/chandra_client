# Chandra

Chandra is a highly accurate OCR model that converts images and PDFs into structured HTML/Markdown/JSON while preserving layout information.

## Features

- Convert documents to markdown, html, or json with detailed layout information
- Math equation support (LaTeX)
- Reconstructs forms, including checkboxes
- Precise table reconstruction
- Support for 40+ languages
- Two inference modes: local (HuggingFace) and remote (vLLM server)

## Installation

```bash
uv sync
source .venv/bin/activate
```

## Usage

### Streamlit Web App

Launch the interactive demo:

```bash
streamlit run chandra_app.py --server.fileWatcherType none --server.headless true
```

**Inference Modes:**
- **hf**: Loads model locally using HuggingFace Transformers (requires GPU)
- **vllm**: Connects to a running vLLM server for optimized batch inference

### vLLM Server (Optional)

For production deployments or batch processing, use the vLLM server:

```bash
python scripts/start_vllm.py
```

This launches a Docker container with optimized inference settings. Configure via environment variables:

- `VLLM_API_BASE`: Server URL (default: `http://localhost:8000/v1`)
- `VLLM_MODEL_NAME`: Model name for the server (default: `chandra`)
- `VLLM_GPUS`: GPU device IDs (default: `0`)
- `HF_TOKEN`: HuggingFace token for model access

### Configuration

Settings can be configured via environment variables or a `local.env` file:

```bash
# Model settings
MODEL_CHECKPOINT=datalab-to/chandra-0.2.8
MAX_OUTPUT_TOKENS=8192

# vLLM settings
VLLM_API_BASE=http://localhost:8000/v1
VLLM_MODEL_NAME=chandra
VLLM_GPUS=0
```

## Output Formats

Chandra provides three output formats:

1. **HTML**: Structured HTML with layout blocks and bounding boxes
2. **Markdown**: Clean, readable Markdown conversion
3. **Layout Image**: Visual representation of detected layout blocks