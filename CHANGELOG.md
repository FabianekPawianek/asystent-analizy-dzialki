# Changelog

Wszystkie istotne zmiany w projekcie będą dokumentowane w tym pliku.

Format oparty na [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
projekt używa [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Planowane (v0.3.0)
- Wsparcie dla wielokrotnych działek
- Export analizy do PDF
- REST API
- Cache dla dokumentów MPZP

---

## [0.2.0] - 2025-01-23

### 🌟 Dodane
- **OCR dla zeskanowanych PDF**
  - Automatyczna detekcja PDF bez warstwy tekstowej
  - Integracja z Tesseract OCR + język polski
  - Fallback: standardowa ekstrakcja → OCR
  - Feedback w czasie rzeczywistym (postęp OCR)

- **Analiza MPZP z AI (Gemini 2.5 Pro)**
  - Autonomiczny agent nawigujący po geoportalu
  - Automatyczna ekstrakcja linków do dokumentów
  - Analiza ustaleń ogólnych i szczegółowych
  - Ekstrakcja kluczowych informacji (przeznaczenie, wysokość, wskaźniki)

- **Dokumentacja**
  - `README.md` - kompleksowa dokumentacja projektu
  - `DEPLOYMENT.md` - instrukcje deployment na Streamlit Cloud
  - `INSTALL_OCR.md` - konfiguracja Tesseract OCR
  - `CHANGELOG.md` - historia zmian

- **Testy**
  - `test_ocr.py` - weryfikacja instalacji OCR
  - `test_pdf_extraction.py` - test ekstrakcji tekstu z PDF

- **Deployment**
  - `packages.txt` - pakiety systemowe dla Streamlit Cloud
  - Auto-konfiguracja Tesseract dla Windows
  - Wsparcie dla Linux/macOS

### 🔧 Zmienione
- Refaktoryzacja `analyze_documents_with_ai()`
  - Lepsze zarządzanie błędami
  - Fallback OCR dla PDF bez tekstu
  - Usunięto tłumienie błędów (`except: continue`)

- Zwiększono fragment tekstu dla analizy AI (3000 → 5000 znaków)
- Uproszczono ekstrakcję tekstu z PDF
- Usunięto niepotrzebne debugowanie w produkcji

### 🐛 Naprawione
- Problem z pustymi dokumentami MPZP (OCR jako fallback)
- Cache Streamlit blokujący nowe wyniki (usunięto `@st.cache_data`)
- Błędna metoda `get_text("blocks")` (zwracała tuple zamiast string)

### 📦 Zależności
- Dodano: `pytesseract`, `Pillow`
- Systemowe: `tesseract-ocr`, `tesseract-ocr-pol`

### ⚠️ Breaking Changes
- **Wymaga Tesseract OCR** dla pełnej funkcjonalności
  - Lokalnie: instalacja manualna
  - Streamlit Cloud: automatycznie z `packages.txt`
- **Wymaga Google Cloud credentials** dla Gemini AI

### 📊 Wydajność
- Ekstrakcja tekstu standardowa: < 1s
- OCR (3 strony): ~6-15s
- Analiza AI (Gemini): ~5-10s

---

## [0.1.0] - 2024-12-XX

### 🌟 Dodane (Pierwsza wersja)
- **Identyfikacja działki**
  - Wyszukiwanie po adresie (Nominatim)
  - Interaktywna mapa (Folium)
  - Warstwa satelitarna (Esri)
  - Warstwa działek ewidencyjnych (ULDK/GUGIK)
  - Automatyczne pobieranie współrzędnych działki

- **Wizualizacja 3D**
  - Model 3D otoczenia (PyDeck)
  - Budynki z OpenStreetMap
  - Szacowanie wysokości budynków
  - Motywy mapy (jasny/ciemny)
  - Sterowanie kamerą (obrót, zoom, przesuwanie)

- **Analiza nasłonecznienia**
  - Symulacja ray-tracing (Trimesh)
  - Uwzględnianie cieni od budynków
  - Mapa cieplna z godzinami słońca
  - Diagram ścieżki słońca (analemma)
  - Konfigurowalne parametry (data, zakres godzin)
  - Zaznaczenie pozycji słońca na diagramie

- **UI/UX**
  - Nowoczesny design z gradientem
  - Smooth scroll między sekcjami
  - Animacje fade-in
  - Responsywny layout
  - Ekran powitalny

- **Integracje**
  - ULDK/GUGIK - dane działek
  - Nominatim - geokodowanie
  - OpenStreetMap - budynki
  - PVLib - pozycja słońca
  - Google Vertex AI - backend

### 📦 Zależności
- Core: `streamlit`, `pandas`, `numpy`
- Geo: `folium`, `streamlit-folium`, `osmnx`, `shapely`, `pyproj`, `geopandas`
- 3D: `pydeck`, `trimesh`, `open3d`
- PDF: `PyMuPDF`
- Solar: `pvlib`
- AI: `langchain-google-vertexai`, `google-cloud-aiplatform`, `vertexai`
- Scraping: `selenium`
- Viz: `matplotlib`

### 🎨 Style
- Gradient tło (niebieski → zielony)
- Zaokrąglone karty
- Cienie i blur effects
- Zielona paleta kolorów (#28a745)
- Custom scrollbar (ukryty)

---

## Legenda typów zmian

- **🌟 Dodane** - nowe funkcje
- **🔧 Zmienione** - zmiany w istniejących funkcjach
- **🗑️ Usunięte** - usunięte funkcje
- **🐛 Naprawione** - poprawki błędów
- **🔒 Bezpieczeństwo** - poprawki bezpieczeństwa
- **📦 Zależności** - aktualizacje pakietów
- **⚠️ Breaking Changes** - zmiany łamiące kompatybilność
- **📚 Dokumentacja** - zmiany w dokumentacji
- **🎨 Style** - zmiany wizualne
- **⚡ Wydajność** - optymalizacje

---

## Jak dodawać wpisy

### Format
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Dodane
- Krótki opis nowej funkcji
  - Szczegół 1
  - Szczegół 2

### Zmienione
- Co się zmieniło i dlaczego
```

### Zasady
1. Najnowsze zmiany na górze
2. Grupuj zmiany według typu
3. Użyj jasnego, zwięzłego języka
4. Linkuj do issues/PRs jeśli istnieją
5. Zaznacz breaking changes

---

## Links

- [Repository](https://github.com/TWOJ_USERNAME/Asystent-Analizy-Dzialki)
- [Releases](https://github.com/TWOJ_USERNAME/Asystent-Analizy-Dzialki/releases)
- [Issues](https://github.com/TWOJ_USERNAME/Asystent-Analizy-Dzialki/issues)

[Unreleased]: https://github.com/TWOJ_USERNAME/Asystent-Analizy-Dzialki/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/TWOJ_USERNAME/Asystent-Analizy-Dzialki/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/TWOJ_USERNAME/Asystent-Analizy-Dzialki/releases/tag/v0.1.0
