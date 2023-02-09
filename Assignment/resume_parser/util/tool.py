from PyPDF2 import PdfReader

def read_pdf (file) :
    pdf = PdfReader(file)
    pdf_text = ""
    for i in range(len(pdf.pages)) :
        page = pdf.pages[i]
        pdf_text += page.extract_text()
    return pdf_text