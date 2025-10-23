# Deployment na Streamlit Cloud

## Przygotowanie repozytorium GitHub

### 1. Sprawdź strukturę projektu
Upewnij się, że masz następujące pliki:

```
github/0.2/
├── app.py                  # Główna aplikacja
├── requirements.txt        # Biblioteki Python
├── packages.txt           # Pakiety systemowe (Tesseract OCR)
├── .streamlit/
│   └── config.toml        # Konfiguracja Streamlit
├── .gitignore
└── README.md
```

### 2. Sprawdź `.gitignore`
Upewnij się, że **NIE** ignorujesz:
- `requirements.txt` ✓
- `packages.txt` ✓
- `.streamlit/config.toml` ✓

Powinieneś ignorować:
```
cache/
*.pyc
__pycache__/
.env
*.pdf  # (opcjonalnie - testowe PDF)
```

### 3. Commit i push na GitHub

```bash
cd "/mnt/x/Users/nkmsa/Desktop/Studia/architektura/ROK5/CAD - AI/github/0.2"

# Dodaj wszystkie pliki
git add .

# Commit
git commit -m "Add OCR support for scanned PDF documents

- Added Tesseract OCR fallback for PDFs without text layer
- Created packages.txt for Streamlit Cloud deployment
- Updated requirements.txt with pytesseract and Pillow"

# Push
git push origin main  # lub 'master' w zależności od nazwy brancha
```

## Deployment na Streamlit Cloud

### Krok 1: Przejdź do Streamlit Cloud
1. Otwórz: https://share.streamlit.io/
2. Zaloguj się przez GitHub

### Krok 2: Utwórz nową aplikację
1. Kliknij **"New app"**
2. Wybierz **repozytorium GitHub**
3. Wybierz **branch** (np. `main`)
4. Ustaw **Main file path**: `app.py`
5. (Opcjonalnie) Ustaw **Custom subdomain**

### Krok 3: Advanced Settings
**WAŻNE:** Musisz skonfigurować zmienne środowiskowe dla Google Cloud:

1. Kliknij **"Advanced settings"**
2. W sekcji **"Secrets"** dodaj:

```toml
# Google Cloud credentials
GOOGLE_APPLICATION_CREDENTIALS = "path/to/credentials.json"

# Lub zawartość JSON bezpośrednio:
[gcp_service_account]
type = "service_account"
project_id = "utility-league-474606-r8"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

### Krok 4: Deploy
1. Kliknij **"Deploy!"**
2. Streamlit Cloud automatycznie:
   - Zainstaluje pakiety z `requirements.txt`
   - Zainstaluje pakiety systemowe z `packages.txt` (Tesseract)
   - Uruchomi aplikację

### Krok 5: Weryfikacja OCR
Po deploymencie, sprawdź logi czy Tesseract został zainstalowany:
- W aplikacji spróbuj analizy MPZP z zeskanowanym PDF
- Powinieneś zobaczyć komunikat: "Dokument nie zawiera warstwy tekstowej. Używam OCR..."

## Pliki wymagane dla Streamlit Cloud

### `requirements.txt` (biblioteki Python)
```txt
streamlit
streamlit-folium
pandas==2.1.4
numpy
requests
PyMuPDF
folium
osmnx
pydeck
trimesh
shapely
pyproj
selenium
langchain-google-vertexai
google-cloud-aiplatform
vertexai
open3d
matplotlib
geopandas
mapbox_earcut
pvlib
rtree
pytesseract    # ← OCR wrapper
Pillow         # ← Image processing
```

### `packages.txt` (pakiety systemowe)
```txt
tesseract-ocr       # ← Tesseract OCR engine
tesseract-ocr-pol   # ← Polski pakiet językowy
```

## Troubleshooting

### Problem: "TesseractNotFoundError"
**Przyczyna:** `packages.txt` nie został załadowany lub ma błędną nazwę

**Rozwiązanie:**
1. Sprawdź czy `packages.txt` jest w **głównym katalogu** (obok `app.py`)
2. Sprawdź czy nazwa pliku to dokładnie `packages.txt` (nie `package.txt`)
3. Zrestartuj deployment w Streamlit Cloud

### Problem: "Failed loading language 'pol'"
**Przyczyna:** Brak polskiego pakietu językowego

**Rozwiązanie:**
Dodaj do `packages.txt`:
```txt
tesseract-ocr-pol
```

### Problem: Google Cloud Authentication Error
**Przyczyna:** Brak credentials lub źle skonfigurowane secrets

**Rozwiązanie:**
1. W Streamlit Cloud → **App settings** → **Secrets**
2. Dodaj credentials Google Cloud (zobacz Krok 3 powyżej)
3. W kodzie dodaj obsługę secrets:

```python
import streamlit as st
import json
import os

# Załaduj credentials z Streamlit Secrets
if 'gcp_service_account' in st.secrets:
    credentials_dict = dict(st.secrets['gcp_service_account'])

    # Zapisz do pliku tymczasowego
    credentials_path = '/tmp/gcp_credentials.json'
    with open(credentials_path, 'w') as f:
        json.dump(credentials_dict, f)

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
```

### Problem: ChromeDriver not found (Selenium)
**Przyczyna:** Brak ChromeDriver w kontenerze

**Rozwiązanie:**
Dodaj do `packages.txt`:
```txt
chromium-browser
chromium-chromedriver
```

I zaktualizuj kod Selenium:
```python
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.binary_location = '/usr/bin/chromium-browser'

driver = webdriver.Chrome(options=chrome_options)
```

## Limity Streamlit Cloud

### Free Tier:
- **1 GB RAM** (może być mało dla OCR dużych PDF)
- **1 CPU core**
- **1 GB storage**
- **Czas wykonania:** ~10 min timeout

### Jeśli OCR jest za wolny:
1. Zmniejsz DPI: `300/72` → `200/72` (linia 543)
2. Ogranicz strony: OCR tylko pierwszych 5 stron
3. Rozważ upgrade do **Teams** ($0.29/h)

## Alternatywne opcje deployment

Jeśli Streamlit Cloud nie wystarcza:

### 1. **Google Cloud Run** (Recommended)
- Więcej zasobów (RAM, CPU)
- Płatność za użycie
- Pełna kontrola nad kontenerem Docker

### 2. **Heroku**
- Darmowy tier
- Wymaga `Aptfile` zamiast `packages.txt`

### 3. **AWS EC2 / Azure VM**
- Pełna kontrola
- Wyższe koszty
- Wymaga większej konfiguracji

## Monitorowanie

Po deploymencie monitoruj:
1. **Logi** - czy OCR działa poprawnie
2. **Czas odpowiedzi** - czy nie przekracza limitów
3. **Użycie pamięci** - czy nie ma OOM errors
4. **Koszty API** - Google Cloud (Gemini, Vertex AI)

## Bezpieczeństwo

### NIE commituj do GitHub:
- ❌ Google Cloud credentials (`credentials.json`)
- ❌ API keys
- ❌ Secrets

### Używaj Streamlit Secrets:
- ✅ Store credentials w zakładce Secrets
- ✅ Dodaj `credentials.json` do `.gitignore`
- ✅ Dokumentuj wymagane secrets w README

## Gotowe do publikacji!

Checklist przed deploymentem:
- [ ] `requirements.txt` - zaktualizowany
- [ ] `packages.txt` - utworzony z Tesseract
- [ ] `.gitignore` - credentials ukryte
- [ ] Secrets skonfigurowane w Streamlit Cloud
- [ ] README.md - instrukcje dla użytkowników
- [ ] Kod przetestowany lokalnie
- [ ] Git commit & push

Powodzenia! 🚀
