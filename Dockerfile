FROM python:3.11-slim

# Ustaw zmienne środowiskowe
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Zainstaluj zależności systemowe
RUN apt-get update && apt-get install -y \
    # Tesseract OCR + polski
    tesseract-ocr \
    tesseract-ocr-pol \
    # Chrome dla Selenium
    wget \
    gnupg \
    unzip \
    # Biblioteki dla spatial/geo
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Zainstaluj Chrome (używamy chromium zamiast google-chrome-stable dla lepszej kompatybilności)
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Ustaw zmienne środowiskowe dla Chrome
ENV CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Ustaw katalog roboczy
WORKDIR /app

# Skopiuj requirements i zainstaluj zależności Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj resztę aplikacji
COPY . .

# Port dla Streamlit
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Uruchom Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
