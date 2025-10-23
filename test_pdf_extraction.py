#!/usr/bin/env python3
"""
Test skrypt do sprawdzania czy PyMuPDF poprawnie wyodrębnia tekst z PDF
"""
import fitz

# Ścieżka do testowego PDF
pdf_path = "665_S.M.8052.MC.pdf"

print(f"Testowanie wyodrębniania tekstu z: {pdf_path}")
print("=" * 60)

try:
    with fitz.open(pdf_path) as doc:
        print(f"✅ PDF otwarty pomyślnie")
        print(f"📄 Liczba stron: {len(doc)}")
        print("=" * 60)

        total_text = ""
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            total_text += page_text
            print(f"\n--- Strona {page_num} ---")
            print(f"Długość tekstu: {len(page_text)} znaków")
            if len(page_text) > 0:
                # Wyświetl pierwsze 200 znaków
                preview = page_text[:200].replace('\n', ' ').strip()
                print(f"Podgląd: {preview}...")
            else:
                print("⚠️ Brak tekstu na tej stronie!")

        print("\n" + "=" * 60)
        print(f"✅ PODSUMOWANIE:")
        print(f"   Całkowita długość tekstu: {len(total_text)} znaków")
        print(f"   Całkowita długość (bez białych znaków): {len(total_text.strip())} znaków")

        if len(total_text.strip()) > 0:
            print(f"\n✅ SUCCESS: Tekst został poprawnie wyodrębniony!")
            print(f"\nPierwsze 500 znaków:")
            print("-" * 60)
            print(total_text[:500])
        else:
            print(f"\n❌ BŁĄD: PDF nie zawiera tekstu lub jest to zeskanowany obraz!")

except Exception as e:
    print(f"❌ BŁĄD: {e}")
    import traceback
    traceback.print_exc()
