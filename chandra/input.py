from typing import List
import filetype
from PIL import Image
import pypdfium2 as pdfium

from chandra.settings import settings


def load_pdf_images(filepath: str, page_range: List[int]):
    doc = pdfium.PdfDocument(filepath)
    images = [doc[i].render(scale=settings.IMAGE_DPI / 72).to_pil().convert("RGB") for i in range(len(doc)) if not page_range or i in page_range]
    doc.close()
    return images


def parse_range_str(range_str: str) -> List[int]:
    range_lst = range_str.split(",")
    page_lst = []
    for i in range_lst:
        if "-" in i:
            start, end = i.split("-")
            page_lst += list(range(int(start), int(end) + 1))
        else:
            page_lst.append(int(i))
    page_lst = sorted(list(set(page_lst)))  # Deduplicate page numbers and sort in order
    return page_lst


def load_file(filepath: str, config: dict):
    page_range = config.get("page_range")
    if page_range:
        page_range = parse_range_str(page_range)

    input_type = filetype.guess(filepath)
    if input_type and input_type.extension == "pdf":
        images = load_pdf_images(filepath, page_range)
    else:
        images = [Image.open(filepath).convert("RGB")]
    return images