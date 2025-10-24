---
title: Asystent Analizy Działki - Szczecin
emoji: 🏗️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: gpl-3.0
---

# Asystent Analizy Działki - Szczecin

**Wersja Beta 0.2**

Inteligentna aplikacja do kompleksowej analizy działek ewidencyjnych w Szczecinie, wykorzystująca AI (Google Gemini 2.5 Pro) do automatycznej analizy dokumentów planistycznych MPZP.

## Funkcjonalności

### 1. Identyfikacja działki
- Wyszukiwanie działki po adresie
- Interaktywna mapa z warstwami satelitarnymi
- Automatyczna identyfikacja granic działki (ULDK/GUGIK)

### 2. Wizualizacja 3D otoczenia
- Generowanie modelu 3D zabudowy w promieniu 300m
- Dane budynków z OpenStreetMap
- Interaktywna wizualizacja (PyDeck)

### 3. Analiza nasłonecznienia
- Symulacja ray-tracing z uwzględnieniem cieni budynków
- Mapa cieplna z liczbą godzin słońca
- Diagram ścieżki słońca (przesilenia, równonoce)

### 4. Analiza MPZP z AI
- Autonomiczny agent nawigujący po geoportalu Szczecina
- OCR dla zeskanowanych PDF (Tesseract + język polski)
- Analiza AI (Gemini 2.5 Pro)

## Wymagania

Aplikacja wymaga Google Cloud credentials dla Vertex AI (Gemini).

Skonfiguruj secrets w ustawieniach Space:
1. Przejdź do Settings → Repository secrets
2. Dodaj secrets zgodnie z dokumentacją

## Więcej informacji

- [GitHub Repository](https://github.com/FabianekPawianek/asystent-analizy-dzialki)
- [Changelog](CHANGELOG.md)

## Autor

Fabian Korycki | fabiankoryckiarchitecture@gmail.com

Powered by Google Gemini AI
