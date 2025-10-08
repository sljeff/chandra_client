import json
from dataclasses import dataclass

from PIL import Image
from PIL.ImageDraw import ImageDraw
from bs4 import BeautifulSoup

@dataclass
class LayoutBlock:
    bbox: list[int]
    label: str
    content: str

def parse_layout(html: str, image: Image.Image):
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    width, height = image.size
    width_scaler = width / 1024
    height_scaler = height / 1024
    layout_blocks = []
    for div in top_level_divs:
        bbox = div.get("data-bbox")
        bbox = json.loads(bbox)
        bbox = list(map(int, bbox))
        # Normalize bbox
        bbox = [
            max(0, int(bbox[0] * width_scaler)),
            max(0, int(bbox[1] * height_scaler)),
            min(int(bbox[2] * width_scaler), width),
            min(int(bbox[3] * height_scaler), height),
        ]
        label = div.get("data-label", "block")
        content = str(div.decode_contents())
        layout_blocks.append(LayoutBlock(bbox=bbox, label=label, content=content))
    return layout_blocks

def draw_layout(image: Image.Image, layout_blocks: list[LayoutBlock]):
    draw_image = image.copy()
    draw = ImageDraw(draw_image)
    for block in layout_blocks:
        draw.rectangle(block.bbox, outline="red", width=2)
        draw.text((block.bbox[0], block.bbox[1]), block.label, fill="blue")

    return draw_image
