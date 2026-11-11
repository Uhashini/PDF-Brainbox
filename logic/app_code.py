from mistralai import Mistral
from PyPDF2 import PdfReader
from pptx import Presentation
from pptx.util import Inches

api_key = "Y70bo7Bnkil7MgiZ3VdOdWwH3edP9UK4"
embed_model = "mistral-embed"
chat_model = "mistral-small-latest"
client = Mistral(api_key=api_key)

def get_text_embedding(txt):
    response = client.embeddings.create(model=embed_model, inputs=[txt])
    return response.data[0].embedding


# Chat function with mistral
def mistral_chat(user_message, is_json=False):
    messages = [{"role": "user", "content": user_message}]
    if is_json:
        chat_response = client.chat.complete(
            model=chat_model,
            messages=messages,
        )
    else:
        chat_response = client.chat.complete(
            model=chat_model,
            messages=messages
        )
    return chat_response.choices[0].message.content

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    if reader.pages:
        return reader.pages[0].extract_text()
    return ""

def generate_pptx_from_text(text):
    prs = Presentation()
    slide_layout = prs.slide_layouts[1]  # Title and content layout

    if isinstance(text, list):
        slides = text
    else:
        slides = text.strip().split("\n\n") if text and text.strip() else []

    for slide in slides:
        if isinstance(slide, str):
            lines = slide.strip().split("\n")
        else:
            lines = [str(slide)]
        if not lines:
            continue
        title = lines[0]
        content = lines[1:]

        slide_obj = prs.slides.add_slide(slide_layout)
        slide_obj.shapes.title.text = title[:40]
        body = slide_obj.placeholders[1]
        body.text = "\n".join(content) if content else title

    pptx_path = "presentation.pptx"
    prs.save(pptx_path)
    return prs if isinstance(text, list) else pptx_path
