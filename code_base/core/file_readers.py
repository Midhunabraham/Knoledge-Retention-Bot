import io
import os
import pandas as pd
from typing import List, Tuple
from pypdf import PdfReader
from core.config import MAX_CHARS_PER_CHUNK

# Office Loaders
import docx
from pptx import Presentation

# Image / OCR Loaders
try:
    from PIL import Image
    import pytesseract
    # Set this path if Tesseract is not in your PATH env variable
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """Splits text into chunks respecting newlines."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], []
    size = 0
    for p in paras:
        if size + len(p) > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [p], len(p)
        else:
            buf.append(p)
            size += len(p)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

# ---------------------------------------------------------
# Individual File Readers
# ---------------------------------------------------------

def read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)

def read_docx(file: io.BytesIO) -> str:
    """Reads Word (.docx) files."""
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pptx(file: io.BytesIO) -> str:
    """Reads PowerPoint (.pptx) files."""
    prs = Presentation(file)
    text_runs = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text_runs.append(shape.text)
    return "\n".join(text_runs)

def read_excel(file: io.BytesIO) -> str:
    """Reads Excel (.xlsx) files."""
    try:
        # Reads all sheets
        dfs = pd.read_excel(file, sheet_name=None)
        text_content = []
        for sheet_name, df in dfs.items():
            text_content.append(f"--- Sheet: {sheet_name} ---")
            text_content.append(df.to_string(index=False))
        return "\n\n".join(text_content)
    except Exception as e:
        return f"Error reading Excel: {str(e)}"

def read_image(file: io.BytesIO) -> str:
    """Extracts text from images using OCR (requires Tesseract)."""
    if not OCR_AVAILABLE:
        return "[Error: PIL or Pytesseract not installed]"
    
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text if text.strip() else "[Image contained no readable text]"
    except Exception as e:
        # Often happens if Tesseract binary is not found in PATH
        return f"[Error: Tesseract OCR not found or configured. Details: {str(e)}]"

def read_code_or_text(file: io.BytesIO) -> str:
    """Reads .txt, .md, .py, .c, etc."""
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return "[Binary or unreadable file content]"

# ---------------------------------------------------------
# Main Router
# ---------------------------------------------------------

def extract_text_from_upload(upload) -> Tuple[str, str]:
    name = upload.name
    # Create a fresh stream for the reader
    data = upload.read()
    file_stream = io.BytesIO(data)
    
    ext = os.path.splitext(name.lower())[1]
    
    # 1. Documents
    if ext == ".pdf":
        return read_pdf(file_stream), name
    elif ext == ".docx":
        return read_docx(file_stream), name
    elif ext == ".pptx":
        return read_pptx(file_stream), name
    elif ext in [".xlsx", ".xls"]:
        return read_excel(file_stream), name
        
    # 2. Images (OCR)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        return read_image(file_stream), name
        
    # 3. Code & Text (.c, .py, .md, .txt, .json, .log)
    elif ext in [".c", ".h", ".py", ".md", ".txt", ".json", ".log", ".js", ".html", ".css"]:
        return read_code_or_text(file_stream), name
        
    else:
        # Try generic text fallback
        return read_code_or_text(file_stream), name