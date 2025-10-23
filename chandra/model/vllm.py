import base64
import io
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from typing import List

from PIL import Image
from openai import OpenAI

from chandra.model.schema import BatchInputItem, GenerationResult
from chandra.model.util import scale_to_fit, detect_repeat_token
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate_vllm(
    batch: List[BatchInputItem],
    client: OpenAI | None = None,
    model_name: str | None = None,
    max_output_tokens: int | None = None,
    max_retries: int | None = None,
    max_workers: int | None = None,
    temperature: float = 0.0,
    top_p: float = 0.1,
    retry_temperature: float = 0.3,
    retry_top_p: float = 0.95,
) -> List[GenerationResult]:
    if client is None:
        client = OpenAI(
            api_key=settings.VLLM_API_KEY,
            base_url=settings.VLLM_API_BASE,
        )
    
    if model_name is None:
        model_name = settings.VLLM_MODEL_NAME

    if max_retries is None:
        max_retries = settings.MAX_VLLM_RETRIES

    if max_workers is None:
        max_workers = min(64, len(batch))

    if max_output_tokens is None:
        max_output_tokens = settings.MAX_OUTPUT_TOKENS

    if model_name is None:
        models = client.models.list()
        model_name = models.data[0].id

    def _generate(
        item: BatchInputItem, temp: float = 0, top_p_val: float = 0.1
    ) -> GenerationResult:
        prompt = item.prompt
        if not prompt:
            prompt = PROMPT_MAPPING[item.prompt_type]

        content = []
        image = scale_to_fit(item.image)
        image_b64 = image_to_base64(image)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )

        content.append({"type": "text", "text": prompt})

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_output_tokens,
                temperature=temp,
                top_p=top_p_val,
            )
        except Exception as e:
            print(f"Error during VLLM generation: {e}")
            return GenerationResult(raw="", token_count=0, error=True)

        return GenerationResult(
            raw=completion.choices[0].message.content,
            token_count=completion.usage.completion_tokens,
            error=False,
        )

    def process_item(item, max_retries_val):
        result = _generate(item, temp=temperature, top_p_val=top_p)
        retries = 0

        while retries < max_retries_val and (
            detect_repeat_token(result.raw)
            or (
                len(result.raw) > 50
                and detect_repeat_token(result.raw, cut_from_end=50)
            )
            or result.error
        ):
            print(
                f"Detected repeat token or error, retrying generation (attempt {retries + 1})..."
            )
            result = _generate(item, temp=retry_temperature, top_p_val=retry_top_p)
            retries += 1

        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_item, batch, repeat(max_retries)))

    return results
