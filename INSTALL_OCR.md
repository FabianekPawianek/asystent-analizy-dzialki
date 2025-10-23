# Instalacja OCR dla analizy zeskanowanych PDF

## Problem
Niektóre pliki PDF MPZP są zeskanowanymi dokumentami bez warstwy tekstowej. Aby je przetworzyć, potrzebujemy OCR (Optical Character Recognition).

## Rozwiązanie
Aplikacja automatycznie wykrywa PDFy bez tekstu i używa Tesseract OCR jako fallback.

## Instalacja

### 1. Zainstaluj biblioteki Python
```bash
pip install pytesseract Pillow
```

### 2. Zainstaluj Tesseract OCR

#### **Windows:**
1. Pobierz instalator: https://github.com/UB-Mannheim/tesseract/wiki
2. Uruchom instalator (zalecane: `tesseract-ocr-w64-setup-5.3.3.20231005.exe`)
3. Podczas instalacji zaznacz "Additional language data" → wybierz **Polish (pol)**
4. Domyślna ścieżka instalacji: `C:\Program Files\Tesseract-OCR\`

5. **WAŻNE:** Dodaj Tesseract do PATH lub ustaw w kodzie:
```python
# Jeśli potrzebne, dodaj na początku app.py:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### **Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-pol
```

#### **macOS:**
```bash
brew install tesseract
brew install tesseract-lang  # dla języka polskiego
```

### 3. Sprawdź instalację
```bash
tesseract --version
```

Powinieneś zobaczyć:
```
tesseract 5.x.x
...
```

## Jak działa

1. **Najpierw**: aplikacja próbuje wyodrębnić tekst standardowo z PDF
2. **Jeśli PDF jest pusty** (< 100 znaków): automatycznie:
   - Konwertuje każdą stronę PDF na obraz (300 DPI)
   - Używa Tesseract OCR z językiem polskim
   - Wyświetla postęp dla każdej strony

## Testowanie lokalnie

Możesz przetestować OCR na lokalnym pliku:

```python
import fitz
import pytesseract
from PIL import Image
import io

pdf_path = "665_S.M.8052.MC.pdf"

with fitz.open(pdf_path) as doc:
    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        text = pytesseract.image_to_string(img, lang='pol')
        print(f"Strona {page_num}: {len(text)} znaków")
        print(text[:200])  # Pierwsze 200 znaków
```

## Uwagi

- **Jakość OCR** zależy od jakości skanu (zalecane minimum 300 DPI)
- **Czas przetwarzania**: OCR jest wolniejszy niż ekstrakcja tekstu (ok. 2-5 sek na stronę)
- **Dokładność**: Tesseract dla języka polskiego ma ~95% dokładność przy dobrej jakości skanów
- **Język**: Używamy `lang='pol'` dla polskich znaków (ą, ć, ę, ł, ń, ó, ś, ź, ż)

## Troubleshooting

### "TesseractNotFoundError"
- Windows: Upewnij się że ścieżka do `tesseract.exe` jest w PATH lub ustaw ją w kodzie
- Linux/Mac: Sprawdź czy Tesseract jest zainstalowany: `which tesseract`

### "Failed loading language 'pol'"
- Pobierz dane językowe dla polskiego:
  - Windows: ponowna instalacja z zaznaczonym Polish
  - Linux: `sudo apt install tesseract-ocr-pol`
  - Mac: `brew install tesseract-lang`

### Słaba jakość rozpoznawania
- Zwiększ DPI: zmień `300/72` na `600/72` w linii 524 app.py (wolniejsze, ale dokładniejsze)
- Użyj preprocessing obrazu przed OCR (konwersja do grayscale, threshold, denoise)
