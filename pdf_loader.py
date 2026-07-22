import pymupdf

class PDFLoader:

    def load(self, file_path):
        doc = pymupdf.open(file_path)
        text = self._extract_text(doc)
        doc.close()
        return text

    def _extract_text(self, doc):
        text = ""
        for page in doc:
            text += page.get_text() + "\n\n"
        return text