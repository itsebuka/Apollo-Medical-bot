from pypdf import PdfReader
from pathlib import Path

pdf_path = Path(r"c:\Users\Ebuka Eleogu\Ebuka-s-adtc-2026\backend\data\knowledge\NGA_Nigeria_Nigeria-Standard-treatment-guidelines-2nd-Edition_2016.pdf")
try:
    reader = PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    text_found = False
    for i in range(min(5, len(reader.pages))):
        text = reader.pages[i].extract_text()
        print(f"Page {i+1} text length: {len(text) if text else 0}")
        if text and len(text.strip()) > 0:
            text_found = True
            print(f"Sample text from page {i+1}: {text[:100]!r}")
            break
            
    if not text_found:
        print("No text could be extracted from the first 5 pages. This is likely a scanned document (images) or has no text layer.")
except Exception as e:
    print(f"Error reading PDF: {e}")
