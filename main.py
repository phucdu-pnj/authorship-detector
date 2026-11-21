from typing import Optional
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from nlp_engine import AuthorshipAnalyzer
from file_parser import extract_text_from_upload

app = FastAPI(title="Hệ thống Phát hiện Bài báo Sai Tác giả")

# Gắn thư mục static để phục vụ file CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

analyzer = AuthorshipAnalyzer()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "ref_text": "",
        "target_text": "",
        "ref_filename": None,
        "target_filename": None
    })


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    ref_text: str = Form(""),
    target_text: str = Form(""),
    ref_file: Optional[UploadFile] = File(None),
    target_file: Optional[UploadFile] = File(None)
):
    error = None
    result = None
    ref_filename = None
    target_filename = None

    try:
        if ref_file and ref_file.filename:
            ref_text = await extract_text_from_upload(ref_file)
            ref_filename = ref_file.filename

        if target_file and target_file.filename:
            target_text = await extract_text_from_upload(target_file)
            target_filename = target_file.filename

        if len(ref_text.split()) < 30 or len(target_text.split()) < 30:
            error = (
                "Mỗi bài báo cần tối thiểu 30 từ để trích xuất dấu vân tay ngôn ngữ "
                "(lưu ý: file PDF dạng ảnh scan chưa OCR sẽ không trích xuất được chữ)."
            )
        else:
            result = analyzer.compare_articles(ref_text, target_text)

    except ValueError as ve:
        error = str(ve)
    except Exception as e:
        error = f"Lỗi khi xử lý tài liệu: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "error": error,
        "ref_text": ref_text,
        "target_text": target_text,
        "ref_filename": ref_filename,
        "target_filename": target_filename
    })