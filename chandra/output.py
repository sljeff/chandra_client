import hashlib
import json
import re
from dataclasses import dataclass, asdict
from functools import lru_cache

import six
from PIL import Image
from bs4 import BeautifulSoup, NavigableString
from markdownify import MarkdownConverter, re_whitespace


@lru_cache
def _hash_html(html: str):
    return hashlib.md5(html.encode("utf-8")).hexdigest()


def get_image_name(html: str, div_idx: int):
    html_hash = _hash_html(html)
    return f"{html_hash}_{div_idx}_img.webp"


def extract_images(html: str, chunks: dict, image: Image.Image):
    images = {}
    div_idx = 0
    for idx, chunk in enumerate(chunks):
        div_idx += 1
        if chunk["label"] in ["Image", "Figure"]:
            img = chunk["content"].find("img")
            if not img:
                continue
            bbox = chunk["bbox"]
            block_image = image.crop(bbox)
            img_name = get_image_name(html, div_idx)
            images[img_name] = block_image
    return images


def parse_html(
    html: str, include_headers_footers: bool = False
):
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    out_html = ""
    image_idx = 0
    div_idx = 0
    for div in top_level_divs:
        div_idx += 1
        label = div.get("data-label")

        # Skip headers and footers if not included
        if label and not include_headers_footers:
            if label in ["Page-Header", "Page-Footer"]:
                continue

        if label in ["Image", "Figure"]:
            # Replace image content with text placeholder
            div.clear()
            placeholder = f"[{label}]"
            div.append(placeholder)
        content = str(div.decode_contents())
        out_html += content
    return out_html


def escape_dollars(text):
    return text.replace("$", r"\$")


def get_formatted_table_text(element):
    text = []
    for content in element.contents:
        if content is None:
            continue

        if isinstance(content, NavigableString):
            stripped = content.strip()
            if stripped:
                text.append(escape_dollars(stripped))
        elif content.name == "br":
            text.append("<br>")
        elif content.name == "math":
            text.append("$" + content.text + "$")
        else:
            content_str = escape_dollars(str(content))
            text.append(content_str)

    full_text = ""
    for i, t in enumerate(text):
        if t == "<br>":
            full_text += t
        elif i > 0 and text[i - 1] != "<br>":
            full_text += " " + t
        else:
            full_text += t
    return full_text


class Markdownify(MarkdownConverter):
    def __init__(
        self,
        inline_math_delimiters,
        block_math_delimiters,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inline_math_delimiters = inline_math_delimiters
        self.block_math_delimiters = block_math_delimiters

    def convert_math(self, el, text, parent_tags):
        block = el.has_attr("display") and el["display"] == "block"
        if block:
            return (
                "\n"
                + self.block_math_delimiters[0]
                + text.strip()
                + self.block_math_delimiters[1]
                + "\n"
            )
        else:
            return (
                " "
                + self.inline_math_delimiters[0]
                + text.strip()
                + self.inline_math_delimiters[1]
                + " "
            )

    def convert_table(self, el, text, parent_tags):
        return "\n\n" + str(el) + "\n\n"

    def convert_a(self, el, text, parent_tags):
        text = self.escape(text)
        # Escape brackets and parentheses in text
        text = re.sub(r"([\[\]()])", r"\\\1", text)
        return super().convert_a(el, text, parent_tags)

    def escape(self, text, parent_tags=None):
        text = super().escape(text, parent_tags)
        if self.options["escape_dollars"]:
            text = text.replace("$", r"\$")
        return text

    def process_text(self, el, parent_tags=None):
        text = six.text_type(el) or ""

        # normalize whitespace if we're not inside a preformatted element
        if not el.find_parent("pre"):
            text = re_whitespace.sub(" ", text)

        # escape special characters if we're not inside a preformatted or code element
        if not el.find_parent(["pre", "code", "kbd", "samp", "math"]):
            text = self.escape(text)

        # remove trailing whitespaces if any of the following condition is true:
        # - current text node is the last node in li
        # - current text node is followed by an embedded list
        if el.parent.name == "li" and (
            not el.next_sibling or el.next_sibling.name in ["ul", "ol"]
        ):
            text = text.rstrip()

        return text


def parse_markdown(
    html: str, include_headers_footers: bool = False
):
    html = parse_html(html, include_headers_footers)

    md_cls = Markdownify(
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=True,
        escape_asterisks=True,
        escape_dollars=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
        inline_math_delimiters=("$", "$"),
        block_math_delimiters=("$$", "$$"),
    )
    try:
        markdown = md_cls.convert(html)
    except Exception as e:
        print(f"Error converting HTML to Markdown: {e}")
        markdown = ""
    return markdown.strip()


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
        try:
            bbox = json.loads(bbox)
        except Exception:
            bbox = [0, 0, 1, 1]  # Fallback to a default bbox if parsing fails

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


def parse_chunks(html: str, image: Image.Image):
    layout = parse_layout(html, image)
    chunks = [asdict(block) for block in layout]
    return chunks


# ========== dots-ocr-client compatibility functions ==========


def html_to_plain_text(html: str) -> str:
    """
    Extract plain text from HTML content.
    Preserves line breaks and extracts LaTeX from math tags.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Convert br tags to newlines
    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Replace math tags with their text content (LaTeX)
    for math_tag in soup.find_all("math"):
        latex = math_tag.get_text().strip()
        math_tag.replace_with(latex)

    # Get text and normalize whitespace
    text = soup.get_text(separator=" ", strip=True)
    text = re_whitespace.sub(" ", text)
    # Restore intentional line breaks
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def html_to_table_html(html: str) -> str:
    """
    Extract table HTML from content.
    Returns the raw HTML table(s), or plain text if no table found.
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return html_to_plain_text(html)
    return "\n\n".join(str(table) for table in tables)


def html_to_latex(html: str) -> str:
    """
    Extract LaTeX content from math tags.
    Falls back to plain text if no math tags found.
    """
    soup = BeautifulSoup(html, "html.parser")
    maths = soup.find_all("math")
    if not maths:
        return html_to_plain_text(html)

    parts = []
    for math_tag in maths:
        latex = math_tag.get_text().strip()
        if latex:
            parts.append(latex)
    return "\n".join(parts)


def extract_text_from_cell(label: str, html: str) -> str:
    """
    Extract appropriate text representation based on cell label.

    - Text/Title/Section-header: plain text
    - Table: HTML table
    - Formula: LaTeX
    - Image/Figure/Picture: empty string
    """
    if label == "Table":
        return html_to_table_html(html)
    elif label in {"Formula", "Equation", "Math"}:
        return html_to_latex(html)
    elif label in {"Image", "Figure", "Picture"}:
        return ""
    else:
        # Default: extract plain text
        return html_to_plain_text(html)
