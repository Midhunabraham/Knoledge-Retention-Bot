import io
import os
import pandas as pd
import traceback
from typing import List, Tuple
from pypdf import PdfReader
from core.config import MAX_CHARS_PER_CHUNK
from openpyxl import load_workbook

# --- New Imports for EML ---
import email
from email import policy
from email.parser import BytesParser

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
    """
    Ultra-Robust Excel Reader (Fixed for UnboundLocalError).
    """
    # Initialize error variables to avoid "UnboundLocalError"
    err_pandas = "Skipped or Unknown"
    err_openpyxl = "Skipped or Unknown"
    err_html = "Skipped or Unknown"
    err_csv = "Skipped or Unknown"

    # Create a copy of the stream content so we can reuse it for multiple attempts
    file_bytes = file.read()
    
    # --- Attempt 1: Standard Pandas ---
    try:
        # print("Attempt 1: Pandas read_excel...")
        dfs = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        return _dfs_to_string(dfs)
    except Exception as e:
        err_pandas = str(e)

    # --- Attempt 2: OpenPyXL Read-Only (Fixes Bad Styles) ---
    try:
        if load_workbook:
            # print("Attempt 2: OpenPyXL read_only...")
            wb = load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
            return _wb_to_string(wb)
    except Exception as e:
        err_openpyxl = str(e)

    # --- Attempt 3: "Fake Excel" (HTML Table) ---
    # Many websites export HTML tables but name them .xlsx
    try:
        # print("Attempt 3: Treating as HTML Table...")
        # read_html returns a list of DataFrames
        dfs_list = pd.read_html(io.BytesIO(file_bytes))
        dfs_dict = {f"Table_{i}": df for i, df in enumerate(dfs_list)}
        return _dfs_to_string(dfs_dict)
    except Exception as e:
        err_html = str(e)

    # --- Attempt 4: "Fake Excel" (Text/CSV) ---
    try:
        # print("Attempt 4: Treating as CSV/Text...")
        df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python')
        return _dfs_to_string({"CSV_Content": df})
    except Exception as e:
        err_csv = str(e)

    # --- Final Failure Report ---
    return (
        f"### EXCEL PROCESSING FAILED ###\n"
        f"The file format is likely corrupted or mismatched (e.g., HTML named as XLSX).\n\n"
        f"Debug Log:\n"
        f"1. Pandas: {err_pandas}\n"
        f"2. OpenPyXL: {err_openpyxl}\n"
        f"3. HTML Fallback: {err_html}\n"
        f"4. CSV Fallback: {err_csv}"
    )

# ---------------------------------------------------------
# Helper Functions (Paste these below read_excel)
# ---------------------------------------------------------

def _dfs_to_string(dfs_dict):
    """Helper to convert a dict of DataFrames to string."""
    text_content = []
    for sheet_name, df in dfs_dict.items():
        text_content.append(f"--- Sheet/Table: {sheet_name} ---")
        # clean NaN and convert to string
        text_content.append(df.fillna("").to_string(index=False))
    return "\n\n".join(text_content)

def _wb_to_string(wb):
    """Helper to convert OpenPyXL workbook to string."""
    fallback_text = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        fallback_text.append(f"--- Sheet: {sheet_name} ---")
        rows = []
        for row in ws.iter_rows(values_only=True):
            row_data = [str(cell) for cell in row if cell is not None]
            if row_data:
                rows.append(" | ".join(row_data))
        fallback_text.append("\n".join(rows))
    return "\n\n".join(fallback_text)
    
def read_eml(file: io.BytesIO) -> str:
    """Reads Email (.eml) files."""
    try:
        # Parse the email bytes
        msg = BytesParser(policy=policy.default).parse(file)
        
        parts = []
        
        # 1. Extract Headers
        for header in ['Subject', 'From', 'To', 'Date']:
            val = msg.get(header)
            if val:
                parts.append(f"{header}: {val}")
        
        parts.append("-" * 20) # Separator
        
        # 2. Extract Body
        # We prefer plain text, but will take HTML if that's all there is
        body = msg.get_body(preferencelist=('plain', 'html'))
        if body:
            try:
                content = body.get_content()
                parts.append(content)
            except Exception:
                parts.append("[Error decoding email body]")
        else:
            parts.append("[No text body found]")
            
        return "\n".join(parts)
        
    except Exception as e:
        return f"Error reading EML file: {str(e)}"

def read_image(file: io.BytesIO) -> str:
    """Extracts text from images using OCR (requires Tesseract)."""
    if not OCR_AVAILABLE:
        return "[Error: PIL or Pytesseract not installed]"
    
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        return text if text.strip() else "[Image contained no readable text]"
    except Exception as e:
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
    elif ext == ".eml":
        return read_eml(file_stream), name 
        
    # 2. Images (OCR)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        return read_image(file_stream), name
        
    # 3. Code & Text
    elif ext in [".c", ".h", ".py", ".md", ".txt", ".json", ".log", ".js", ".html", ".css"]:
        return read_code_or_text(file_stream), name
        
    else:
        # Try generic text fallback
        return read_code_or_text(file_stream), name