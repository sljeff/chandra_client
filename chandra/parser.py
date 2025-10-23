from typing import List, TypedDict
from PIL import Image
from openai import OpenAI

from chandra.model.schema import BatchInputItem
from chandra.model.vllm import generate_vllm
from chandra.output import parse_markdown, parse_chunks, extract_text_from_cell
from chandra.input import load_file


class CellDict(TypedDict):
    bbox: List[int]
    category: str
    text: str


class PageResultDict(TypedDict):
    page_no: int
    input_width: int
    input_height: int
    cells: List[CellDict]
    md_content: str


PROMPT_MODE_MAP = {
    "layout": "ocr_layout",
    "plain": "ocr",
}


class ChandraOCRClient:
    """High-level client for Chandra OCR vLLM API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "EMPTY",
        model_name: str = "chandra",
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_tokens: int = 8192,
        num_threads: int = 8,
        max_retries: int = 6,
        include_headers_footers: bool = False,
        image_dpi: int = 200,
        min_image_dim: int = 1024,
    ):
        # Ensure base_url ends with /v1
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"
        
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_threads = num_threads
        self.max_retries = max_retries
        self.include_headers_footers = include_headers_footers
        self.image_dpi = image_dpi
        self.min_image_dim = min_image_dim

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def parse_images(
        self,
        images: List[Image.Image],
        prompt_mode: str = "layout",
        prompt: str | None = None,
    ) -> List[PageResultDict]:
        prompt_type = PROMPT_MODE_MAP.get(prompt_mode, prompt_mode)

        batch = [
            BatchInputItem(
                image=img,
                prompt=prompt,
                prompt_type=prompt_type if not prompt else None,
            )
            for img in images
        ]

        results = generate_vllm(
            batch,
            client=self.client,
            model_name=self.model_name,
            max_output_tokens=self.max_tokens,
            max_retries=self.max_retries,
            max_workers=self.num_threads,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        pages = []
        for idx, (result, input_item) in enumerate(zip(results, batch)):
            chunks = parse_chunks(result.raw, input_item.image)
            raw_html = result.raw

            page_result: PageResultDict = {
                "page_no": idx,
                "input_width": input_item.image.width,
                "input_height": input_item.image.height,
                "cells": [
                    {
                        "bbox": c["bbox"],
                        "category": c["label"],
                        "text": extract_text_from_cell(c["label"], c["content"]),
                    }
                    for c in chunks
                ],
                "md_content": parse_markdown(
                    raw_html,
                    include_headers_footers=self.include_headers_footers
                ),
            }
            pages.append(page_result)

        return pages

    def parse_image(
        self,
        image: Image.Image,
        prompt_mode: str = "layout",
        prompt: str | None = None,
    ) -> PageResultDict:
        return self.parse_images([image], prompt_mode, prompt)[0]

    def parse_file(
        self,
        path: str,
        prompt_mode: str = "layout",
        prompt: str | None = None,
        page_range: str | None = None,
    ) -> List[PageResultDict]:
        config = {
            "page_range": page_range,
            "image_dpi": self.image_dpi,
            "min_image_dim": self.min_image_dim,
        }
        images = load_file(path, config)
        return self.parse_images(images, prompt_mode, prompt)
