from dataclasses import dataclass
from typing import List

from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from chandra.image import scale_to_fit
from chandra.prompts import PROMPT_MAPPING
from chandra.settings import settings

from qwen_vl_utils import process_vision_info

@dataclass
class BatchItem:
    images: List[Image.Image]
    prompt: str | None = None
    prompt_type: str | None = None


def load():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        settings.MODEL_CHECKPOINT,
        dtype=settings.TORCH_DTYPE,
        device_map="auto",
        attn_implementation=settings.TORCH_ATTN_IMPLEMENTATION,
    ).to(settings.TORCH_DEVICE_MODEL)
    model = model.eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
    model.processor = processor
    return model

def process_batch_element(item: BatchItem, processor):
    prompt = item.prompt
    prompt_type = item.prompt_type
    images = item.images

    if not prompt:
        prompt = PROMPT_MAPPING[prompt_type]

    content = []
    for image in images:
        image = scale_to_fit(image)  # Guarantee max size
        content.append({"type": "image", "image": image})

    content.append({"type": "text", "text": prompt})
    message = {
        "role": "user",
        "content": content
    }
    return message


def generate(batch: List[BatchItem], model):
    messages = [process_batch_element(item, model.processor) for item in batch]
    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)
    inputs = model.processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        padding_side="left"
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=settings.MAX_OUTPUT_TOKENS)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = model.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

