# Chandra Client

Lightweight Python client for Chandra OCR with vLLM support.

Fork of [datalab-to/chandra](https://github.com/datalab-to/chandra) - optimized for client-only usage.

**Key improvements:**
- Minimal dependencies (~50MB vs ~4GB) - no torch/transformers
- Simple high-level API
- All config via constructor - no environment variables
- Direct file processing with `parse_file(path)`

## Installation

```bash
pip install git+https://github.com/sljeff/chandra_client.git
```

## Prerequisites

You must have a running Chandra vLLM server. See [original repo](https://github.com/datalab-to/chandra) for deployment instructions.

## Usage

### Basic Usage

```python
from chandra import ChandraOCRClient

client = ChandraOCRClient(base_url="http://localhost:8000")
pages = client.parse_file("document.pdf")

for page in pages:
    print(page["md_content"])
```

### Advanced Usage

```python
from chandra import ChandraOCRClient
from PIL import Image

# Custom configuration
client = ChandraOCRClient(
    base_url="http://localhost:8000",
    temperature=0.3,
    max_tokens=16384,
    num_threads=32,
)

# Parse PDF with page range
pages = client.parse_file("document.pdf", page_range="0-9")

# Parse single image
image = Image.open("page.png")
result = client.parse_image(image)

# Custom prompt
result = client.parse_file("doc.pdf", prompt="Extract all text as JSON")
```

## API Reference

### ChandraOCRClient

```python
ChandraOCRClient(
    base_url: str = "http://localhost:8000",
    api_key: str = "",
    model_name: str = "chandra",
    temperature: float = 0.0,
    top_p: float = 0.1,
    max_tokens: int = 8192,
    num_threads: int = 8,
    max_retries: int = 6,
    include_headers_footers: bool = False,
    image_dpi: int = 200,
    min_image_dim: int = 1024,
)
```

### Methods

- `parse_file(path, prompt_mode="layout", prompt=None, page_range=None)` → List[dict]
- `parse_images(images, prompt_mode="layout", prompt=None)` → List[dict]
- `parse_image(image, prompt_mode="layout", prompt=None)` → dict

**Parameters:**
- `path` - File path (PDF or image)
- `images` - List of PIL Image objects
- `image` - Single PIL Image object
- `prompt_mode` - `"layout"` (default) or `"plain"`, or custom prompt type
- `prompt` - Custom prompt string (overrides prompt_mode)
- `page_range` - Page range for PDFs (e.g., `"0-9"` or `"1,3,5-10"`)

## Return Structure

Each page returns a dictionary:

```python
{
    "page_no": 0,
    "input_width": 1632,
    "input_height": 2112,
    "cells": [
        {
            "bbox": [286, 202, 1332, 612],
            "category": "Figure",
            "text": ""
        },
        {
            "bbox": [286, 624, 1333, 686],
            "category": "Caption",
            "text": "Figure 4: SEED calculation process..."
        },
        {
            "bbox": [286, 748, 1333, 924],
            "category": "Text",
            "text": "LaTeX preprocessing: we strip wrappers..."
        },
        {
            "bbox": [100, 850, 900, 1200],
            "category": "Table",
            "text": "<table>...</table>"
        }
    ],
    "md_content": "# Title\n\nContent..."
}
```

**Common categories:** `Text`, `Caption`, `Footnote`, `Equation-Block`, `List-Group`, `Page-Header`, `Page-Footer`, `Image`, `Section-Header`, `Table`, `Complex-Block`, `Code-Block`, `Form`, `Table-Of-Contents`, `Figure`

## Why This Fork

Original chandra-ocr includes local inference with heavy dependencies (torch, transformers, etc.). This fork:
- Removes local inference support
- Removes CLI tools and apps
- Only supports vLLM backend
- Reduces dependencies from ~4GB to ~50MB
- Provides simpler, high-level API

Perfect for production deployments where you have a dedicated vLLM server.

## Original Project

For the full-featured version with local inference:
- Repository: https://github.com/datalab-to/chandra
- Hosted API: https://www.datalab.to/
- Playground: https://www.datalab.to/playground

## License

Apache 2.0 (same as original)

Model weights: See [MODEL_LICENSE](MODEL_LICENSE)

## Credits

Fork of [datalab-to/chandra](https://github.com/datalab-to/chandra) by Datalab.
