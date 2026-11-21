import io
import re
from fastapi import UploadFile
from pypdf import PdfReader
from docx import Document

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


async def extract_text_from_upload(upload_file: UploadFile) -> str:
    filename = (upload_file.filename or "").lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise ValueError(
            f"Định dạng file '{upload_file.filename}' không được hỗ trợ. Chỉ chấp nhận .pdf, .docx, .txt."
        )

    content = await upload_file.read()
    if not content:
        raise ValueError(f"File '{upload_file.filename}' bị rỗng.")

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        raw_text = "\n".join(pages_text)
    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        raw_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        raw_text = content.decode("utf-8", errors="ignore")

    # Chuẩn hóa khoảng trắng ngang nhưng giữ nguyên ngắt đoạn và dấu câu cho Stylometry
    cleaned_text = re.sub(r"[ \t]+", " ", raw_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
    return cleaned_text