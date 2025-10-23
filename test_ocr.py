#!/usr/bin/env python3
"""
Test skrypt OCR - sprawdza czy Tesseract działa i czy może odczytać tekst z PDF
"""
import sys

print("=" * 70)
print("TEST INSTALACJI OCR")
print("=" * 70)

# Test 1: Sprawdź import bibliotek
print("\n1. Sprawdzam importy bibliotek...")
try:
    import fitz
    print("   ✓ PyMuPDF (fitz) zainstalowany")
except ImportError as e:
    print(f"   ✗ PyMuPDF brak: {e}")
    sys.exit(1)

try:
    import pytesseract
    print("   ✓ pytesseract zainstalowany")
except ImportError as e:
    print(f"   ✗ pytesseract brak: {e}")
    print("   → Zainstaluj: pip install pytesseract")
    sys.exit(1)

try:
    from PIL import Image
    print("   ✓ Pillow zainstalowany")
except ImportError as e:
    print(f"   ✗ Pillow brak: {e}")
    print("   → Zainstaluj: pip install Pillow")
    sys.exit(1)

# Test 2: Sprawdź Tesseract OCR
print("\n2. Sprawdzam Tesseract OCR...")
try:
    version = pytesseract.get_tesseract_version()
    print(f"   ✓ Tesseract wersja: {version}")
except Exception as e:
    print(f"   ✗ Tesseract nie znaleziony: {e}")
    print("   → Instrukcje instalacji: Zobacz INSTALL_OCR.md")
    sys.exit(1)

# Test 3: Sprawdź język polski
print("\n3. Sprawdzam dostępność języka polskiego...")
try:
    langs = pytesseract.get_languages()
    if 'pol' in langs:
        print(f"   ✓ Język polski dostępny")
    else:
        print(f"   ✗ Język polski niedostępny")
        print(f"   → Dostępne języki: {', '.join(langs)}")
        print("   → Zainstaluj język polski (zobacz INSTALL_OCR.md)")
except Exception as e:
    print(f"   ⚠ Nie można sprawdzić języków: {e}")

# Test 4: Test OCR na lokalnym PDF
print("\n4. Testuję OCR na pliku 665_S.M.8052.MC.pdf...")
pdf_path = "665_S.M.8052.MC.pdf"

try:
    import io

    with fitz.open(pdf_path) as doc:
        print(f"   ✓ PDF otwarty ({len(doc)} stron)")

        # Test pierwszej strony
        page = doc[0]

        # Sprawdź czy jest warstwa tekstowa
        text_layer = page.get_text()
        print(f"   → Warstwa tekstowa: {len(text_layer)} znaków")

        if len(text_layer.strip()) < 50:
            print(f"   → PDF wymaga OCR")

            # Renderuj stronę jako obraz
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            print(f"   → Obraz wygenerowany: {img.size[0]}x{img.size[1]} px")

            # OCR
            print("   → Uruchamiam OCR...")
            text_ocr = pytesseract.image_to_string(img, lang='pol')

            print(f"   ✓ OCR zakończony: {len(text_ocr)} znaków")

            if text_ocr.strip():
                print("\n   Pierwsze 300 znaków:")
                print("   " + "-" * 66)
                preview = text_ocr[:300].replace('\n', '\n   ')
                print(f"   {preview}")
                print("   " + "-" * 66)
                print("\n   ✓ SUCCESS: OCR działa poprawnie!")
            else:
                print("   ✗ OCR nie wykrył tekstu")
        else:
            print(f"   ✓ PDF ma warstwę tekstową, OCR nie jest potrzebny")
            print(f"\n   Pierwsze 200 znaków:")
            print("   " + "-" * 66)
            preview = text_layer[:200].replace('\n', '\n   ')
            print(f"   {preview}")

except FileNotFoundError:
    print(f"   ✗ Plik {pdf_path} nie znaleziony")
    print("   → Upewnij się że uruchamiasz skrypt w katalogu z plikiem PDF")
except Exception as e:
    print(f"   ✗ Błąd: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST ZAKOŃCZONY")
print("=" * 70)
