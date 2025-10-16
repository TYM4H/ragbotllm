import pdfplumber
from pypdf import PdfReader

def extract_text_simple(pdf_path: str) -> str:
    text_parts = []

    #Основной способ через pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        print(f"шибка pdfplumber: {e}")

    #fallback через PyPDF
    if not text_parts:
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        except Exception as e:
            print(f"Ошибка PyPDF: {e}")

    full_text = "\n".join(text_parts)
    print(f"dfplumber извлёк {len(full_text)} символов текста из {pdf_path}")
    return full_text
