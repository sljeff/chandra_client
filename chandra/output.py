import json
import re
from dataclasses import dataclass, asdict

import six
from PIL import Image
from bs4 import BeautifulSoup, NavigableString
from markdownify import MarkdownConverter, re_whitespace


def parse_html(html: str, include_headers_footers: bool = False):
    soup = BeautifulSoup(html, "html.parser")
    top_level_divs = soup.find_all("div", recursive=False)
    out_html = ""
    for div in top_level_divs:
        label = div.get("data-label")

        # Skip headers and footers if not included
        if label and not include_headers_footers:
            if label in ["Page-Header", "Page-Footer"]:
                continue

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


def parse_markdown(html: str, include_headers_footers: bool = False):
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
    markdown = md_cls.convert(html)
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
        except Exception as e:
            bbox = [0, 0, 1, 1] # Fallback to a default bbox if parsing fails

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

