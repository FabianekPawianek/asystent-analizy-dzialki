# Asystent Analizy Działki - Szczecin

**Wersja Beta 0.2**

Inteligentna aplikacja do kompleksowej analizy działek ewidencyjnych w Szczecinie, wykorzystująca AI (Google Gemini 2.5 Pro) do automatycznej analizy dokumentów planistycznych MPZP.

## Funkcjonalności

### 1. **Identyfikacja działki**
- Wyszukiwanie działki po adresie
- Interaktywna mapa z warstwami satelitarnymi
- Automatyczna identyfikacja granic działki (ULDK/GUGIK)
- Wyświetlanie numeru ewidencyjnego

### 2. **Wizualizacja 3D otoczenia**
- Generowanie modelu 3D zabudowy w promieniu 300m
- Dane budynków z OpenStreetMap
- Interaktywna wizualizacja (PyDeck)
- Wybór motywu (jasny/ciemny)

### 3. **Analiza nasłonecznienia**
- Symulacja ray-tracing z uwzględnieniem cieni budynków
- Mapa cieplna z liczbą godzin słońca dla każdego punktu działki
- Diagram ścieżki słońca (przesilenia, równonoce)
- Wizualizacja analemmy
- Konfigurowalne parametry (zakres dat, przedział godzinowy)

### 4. **Analiza MPZP z AI** 
- **Autonomiczny agent** nawigujący po geoportalu Szczecina
- **Automatyczna ekstrakcja** dokumentów MPZP
- **OCR dla zeskanowanych PDF** (Tesseract + język polski)
- **Analiza AI** (Gemini 2.5 Pro):
  - Cel i charakter planu
  - Oznaczenie terenu
  - Przeznaczenie terenu
  - Wysokość zabudowy
  - Wskaźniki zabudowy
  - Geometria dachu

## Instalacja lokalna

### Wymagania systemowe
- Python 3.9+
- Chrome/Chromium (dla Selenium)
- Tesseract OCR (dla analizy zeskanowanych PDF)

### Krok 1: Sklonuj repozytorium
```bash
git clone <repository-url>
cd "CAD - AI/github/0.2"
```

### Krok 2: Zainstaluj zależności Python
```bash
pip install -r requirements.txt
```

### Krok 3: Zainstaluj Tesseract OCR

#### Windows:
1. Pobierz: https://github.com/UB-Mannheim/tesseract/wiki
2. Uruchom instalator
3. Zaznacz **"Additional language data" → Polish (pol)**

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-pol
```

#### macOS:
```bash
brew install tesseract tesseract-lang
```

**Sprawdź instalację:**
```bash
tesseract --version
```

### Krok 4: Skonfiguruj Google Cloud

1. Utwórz projekt w Google Cloud Console
2. Włącz API:
   - Vertex AI API
   - Cloud AI Platform API
3. Utwórz Service Account i pobierz credentials JSON
4. Ustaw zmienną środowiskową:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

Lub w `.streamlit/secrets.toml`:
```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
# ... reszta pól
```

### Krok 5: Uruchom aplikację
```bash
streamlit run app.py
```

Aplikacja będzie dostępna pod: `http://localhost:8501`

## Testowanie

Sprawdź czy OCR działa poprawnie:
```bash
tesseract --version
```

Powinno wyświetlić wersję Tesseract (5.x.x) i dostępne języki (pol).

## Deployment na Streamlit Cloud

### Wymagane pliki
- `requirements.txt` - biblioteki Python
- `packages.txt` - pakiety systemowe (Tesseract OCR)
- `.streamlit/config.toml` - konfiguracja Streamlit

### Konfiguracja deployment

1. Push repozytorium na GitHub
2. Przejdź do https://share.streamlit.io/
3. Utwórz nową aplikację wskazując na `app.py`
4. W Advanced Settings → Secrets dodaj Google Cloud credentials:
   ```toml
   [gcp_service_account]
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "..."
   private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
   client_email = "..."
   # ... pozostałe pola z credentials.json
   ```
5. Deploy

Streamlit Cloud automatycznie zainstaluje Tesseract OCR z `packages.txt`.

### Limity Free Tier
- 1 GB RAM (może być niewystarczające dla dużych PDF)
- 1 CPU core
- Timeout: około 10 minut

W razie problemów z wydajnością rozważ upgrade do Teams lub alternatywne platformy (Google Cloud Run, Heroku).

## Technologie

### Backend:
- **Streamlit** - framework aplikacji
- **PyMuPDF (fitz)** - przetwarzanie PDF
- **Tesseract OCR** - OCR dla zeskanowanych PDF
- **Selenium** - automatyzacja przeglądarki (web scraping)

### Geospatial:
- **Folium** - interaktywne mapy 2D
- **PyDeck** - wizualizacje 3D
- **OSMnx** - dane OpenStreetMap
- **Shapely** - operacje geometryczne
- **PyProj** - transformacje układów współrzędnych

### 3D & Analiza:
- **Trimesh** - ray-tracing i operacje 3D
- **Open3D** - zaawansowane przetwarzanie 3D
- **PVLib** - obliczenia pozycji słońca
- **NumPy** - obliczenia numeryczne

### AI & NLP:
- **Google Vertex AI** - Gemini 2.5 Pro
- **LangChain** - orkiestracja LLM
- **pytesseract** - OCR

## Wydajność

| Operacja | Czas |
|----------|------|
| Identyfikacja działki | < 2s |
| Model 3D otoczenia | 3-5s |
| Analiza nasłonecznienia (1 dzień) | 10-30s |
| Agent MPZP (nawigacja) | 20-40s |
| OCR (3 strony PDF) | 6-15s |
| Analiza AI (Gemini) | 5-10s |

## Bezpieczeństwo

- **NIE commituj** credentials do GitHub
- Używaj `.streamlit/secrets.toml` lub zmiennych środowiskowych
- Dodaj `credentials.json` do `.gitignore`
- Używaj Service Account (nie user credentials)

## Licencja

Copyright © 2025 Fabian Korycki

## Współpraca

Zgłaszanie błędów i propozycje funkcji: [Issues](../../issues)

## Kontakt

Autor: Fabian Korycki
E-mail: fabiankoryckiarchitecture@gmail.com

Powered by **Google Gemini AI**

---

**Szczecin • Wersja Beta 0.2**
