import pypdfium2 as pdfium
import streamlit as st
from PIL import Image

from chandra.layout import parse_layout, draw_layout
from chandra.load import load_pdf_images
from chandra.model import load, BatchItem, generate


@st.cache_resource()
def load_model():
    return load()

@st.cache_data()
def get_page_image(pdf_file, page_num):
    return load_pdf_images(pdf_file, [page_num])[0]

@st.cache_data()
def page_counter(pdf_file):
    doc = pdfium.PdfDocument(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len

# Function for OCR
def ocr_layout(
    img: Image.Image,
) -> (Image.Image, str):
    batch = BatchItem(
        images=[img],
        prompt_type="ocr_layout",
    )
    html = generate([batch], model=model)[0]
    print(f"Generated HTML: {html[:500]}...")
    layout = parse_layout(html, img)
    layout_image = draw_layout(img, layout)
    return html, layout_image

def ocr(
    img: Image.Image,
) -> str:
    batch = BatchItem(
        images=[img],
        prompt_type="ocr"
    )
    return generate([batch], model=model)[0]

st.set_page_config(layout="wide")
col1, col2 = st.columns([0.5, 0.5])

model = load_model()

st.markdown("""
# Chandra OCR Demo

This app will let you try chandra, a multilingual OCR toolkit.
""")

in_file = st.sidebar.file_uploader(
    "PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
)

if in_file is None:
    st.stop()

filetype = in_file.type
page_count = None
if "pdf" in filetype:
    page_count = page_counter(in_file)
    page_number = st.sidebar.number_input(
        f"Page number out of {page_count}:", min_value=0, value=0, max_value=page_count
    )

    pil_image = get_page_image(in_file, page_number)
else:
    pil_image = Image.open(in_file).convert("RGB")
    page_number = None

run_ocr = st.sidebar.button("Run OCR")
prompt_type = st.sidebar.selectbox(
    "Prompt type",
    ["ocr_layout", "ocr"],
    index=0,
    help="Select the prompt type for OCR.",
)

if pil_image is None:
    st.stop()

if run_ocr:
    if prompt_type == "ocr_layout":
        pred, layout_image = ocr_layout(
            pil_image,
        )
    else:
        pred = ocr(
            pil_image,
        )
        layout_image = None

    with col1:
        html_tab, text_tab, layout_tab = st.tabs(["HTML", "HTML as text", "Layout Image"])
        with html_tab:
            st.markdown(pred, unsafe_allow_html=True)
        with text_tab:
            st.text(pred)

        if layout_image:
            with layout_tab:
                st.image(layout_image, caption="Detected Layout", use_container_width=True)

with col2:
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)
