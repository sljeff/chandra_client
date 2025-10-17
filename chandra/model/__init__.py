from typing import List

from chandra.model.hf import load_model, generate_hf
from chandra.model.schema import BatchInputItem, BatchOutputItem
from chandra.model.vllm import generate_vllm
from chandra.output import parse_markdown, parse_html, parse_chunks


class InferenceManager:
    def __init__(self, method: str = "vllm"):
        assert method in ("vllm", "hf"), "method must be 'vllm' or 'hf'"
        self.method = method

        if method == "hf":
            self.model = load_model()
        else:
            self.model = None

    def generate(self, batch: List[BatchInputItem], **kwargs) -> List[BatchOutputItem]:
        if self.method == "vllm":
            results = generate_vllm(batch, **kwargs)
        else:
            results = generate_hf(batch, self.model, **kwargs)

        output = []
        for result, input_item in zip(results, batch):
            output.append(
                BatchOutputItem(
                    markdown=parse_markdown(result.raw),
                    html=parse_html(result.raw),
                    chunks=parse_chunks(result.raw, input_item.image),
                    raw=result.raw,
                    page_box=[0, 0, input_item.image.width, input_item.image.height],
                    token_count=result.token_count
                )
            )
        return output
