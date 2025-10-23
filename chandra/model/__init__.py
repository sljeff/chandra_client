from typing import List

from chandra.model.schema import BatchInputItem, BatchOutputItem
from chandra.model.vllm import generate_vllm
from chandra.output import parse_markdown, parse_html, parse_chunks, extract_images


class InferenceManager:
    """Lightweight client for Chandra vLLM server."""
    
    def __init__(self, method: str = "vllm"):
        if method != "vllm":
            raise ValueError(
                "This client only supports 'vllm' method. "
                "For local inference, use the original chandra-ocr package."
            )
        self.method = method
        self.model = None

    def generate(
        self, batch: List[BatchInputItem], max_output_tokens=None, **kwargs
    ) -> List[BatchOutputItem]:
        output_kwargs = {}
        if "include_headers_footers" in kwargs:
            output_kwargs["include_headers_footers"] = kwargs.pop(
                "include_headers_footers"
            )

        results = generate_vllm(batch, max_output_tokens=max_output_tokens, **kwargs)

        output = []
        for result, input_item in zip(results, batch):
            chunks = parse_chunks(result.raw, input_item.image)
            output.append(
                BatchOutputItem(
                    markdown=parse_markdown(result.raw, **output_kwargs),
                    html=parse_html(result.raw, **output_kwargs),
                    chunks=chunks,
                    raw=result.raw,
                    page_box=[0, 0, input_item.image.width, input_item.image.height],
                    token_count=result.token_count,
                    images=extract_images(result.raw, chunks, input_item.image),
                )
            )
        return output
