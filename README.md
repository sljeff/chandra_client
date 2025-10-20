# Chandra

Chandra is a highly accurate OCR model that converts images and PDFs into structured HTML/Markdown/JSON while preserving layout information.

## Features

- Convert documents to markdown, html, or json with detailed layout information
- Good handwriting support
- Reconstructs forms accurately, including checkboxes
- Math equation support (LaTeX)
- Precise table reconstruction
- Support for 40+ languages
- Two inference modes: local (HuggingFace) and remote (vLLM server)


## Benchmarks

| **Model** |  ArXiv   | Old Scans Math |  Tables  | Old Scans | Headers and Footers | Multi column | Long tiny text |   Base   |    Overall     |
|:----------|:--------:|:--------------:|:--------:|:---------:|:-------------------:|:------------:|:--------------:|:--------:|:--------------:|
| Datalab Chandra v0.1.0 |   81.4   |    **80.3**    | **89.4** | **50.0**  |        88.3         |   **81.0**   |    **91.6**    | **99.9** | **82.7 ± 0.9** |
| Datalab Marker v1.10.0 | **83.8** |      69.7      |   74.8   |   32.3    |        86.6         |     79.4     |      85.7      |   99.6   |   76.5 ± 1.0   |
| Mistral OCR API |   77.2   |      67.5      |   60.6   |   29.3    |        93.6         |     71.3     |      77.1      |   99.4   |   72.0 ± 1.1   |
| Deepseek OCR |   75.2   |      67.9      |   79.1   |   32.9    |        96.1         |     66.3     |      78.5      |   97.7   |   74.2 ± 1.0   |
| Nanonets OCR |   67.0   |      68.6      |   77.7   |   39.5    |        40.7         |     69.9     |      53.4      |   99.3   |   64.5 ± 1.1   |
| GPT-4o (Anchored) |   53.5   |      74.5      |   70.0   |   40.7    |        93.8         |     69.3     |      60.6      |   96.8   |   69.9 ± 1.1   |
| Gemini Flash 2 (Anchored) |   54.5   |      56.1      |   72.1   |   34.2    |        64.7         |     61.5     |      71.5      |   95.6   |   63.8 ± 1.2   |
| Qwen 2.5 VL (No Anchor) |   63.1   |      65.7      |   67.3   |   38.6    |        73.6         |     68.3     |      49.1      |   98.3   |   65.5 ± 1.2   |
| Qwen 3 VL |   70.2   |      75.1      |   45.6   |   37.5    |        89.1         |     62.1     |      43.0      |   94.3   |   64.6 ± 1.1   |
| olmOCR v0.3.0 |   78.6   |      79.9      |   72.9   |   43.9    |      **95.1**       |     77.3     |      81.2      |   98.9   |   78.5 ± 1.1   |


## Installation

### From PyPI (Recommended)

```bash
pip install chandra-ocr
```

### From Source

```bash
git clone https://github.com/yourusername/chandra.git
cd chandra
uv sync
source .venv/bin/activate
```

## Usage

### CLI

Process single files or entire directories:

```bash
# Single file, with vllm server (see below for how to launch)
chandra input.pdf ./output --method vllm

# Process all files in a directory with local model
chandra ./documents ./output --method hf

# Process specific pages with custom settings
chandra document.pdf ./output --page-range "1-10,15,20-25" --max-workers 8
```

**CLI Options:**
- `--method [hf|vllm]`: Inference method (default: vllm)
- `--page-range TEXT`: Page range for PDFs (e.g., "1-5,7,9-12")
- `--max-output-tokens INTEGER`: Max tokens per page
- `--max-workers INTEGER`: Parallel workers for vLLM
- `--include-images/--no-images`: Extract and save images (default: include)
- `--include-headers-footers/--no-headers-footers`: Include page headers/footers (default: exclude)
- `--batch-size INTEGER`: Pages per batch (default: 1)

**Output Structure:**

Each processed file creates a subdirectory with:
- `<filename>.md` - Markdown output
- `<filename>.html` - HTML output
- `<filename>_metadata.json` - Metadata (page info, token count, etc.)
- `images/` - Extracted images from the document

### Streamlit Web App

Launch the interactive demo for single-page processing:

```bash
chandra_app
```

The web interface allows you to:
- Upload PDFs or images
- Select specific pages from PDFs
- View OCR results with layout visualization
- Download markdown output
- See extracted images embedded in the output

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