# Asystent Analizy Działki: Manifest dla Architektury Skoncentrowanej na Człowieku

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://fabianekpawianek-asystent-analizy-dzialki.hf.space)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-0.2-green.svg)](https://github.com/FabianekPawianek/asystent-analizy-dzialki/releases/tag/v0.2.0)

**Wersja Beta 0.2 (Szczecin)**

To narzędzie jest praktycznym zastosowaniem filozofii **Parametrycznego Humanizmu**. To protest przeciwko architekturze zysku i manifest na rzecz jakości życia, tworzonej dla ludzi, a nie dla arkuszy kalkulacyjnych.

Aplikacja dostarcza obiektywnych, mierzalnych danych, które stanowią fundament świadomego projektowania i niepodważalny argument w rozmowach z inwestorami.

**Dowiedz się więcej o filozofii projektu w pliku [VISION.md](VISION.md)**

## Live Demo

**[Otwórz aplikację →](https://fabianekpawianek-asystent-analizy-dzialki.hf.space)**

Hostowane na Hugging Face Spaces (16 GB RAM, darmowe).

## Kluczowe Możliwości i ich Zastosowanie

### 1. Identyfikacja działki
- **Funkcja:** Wyszukiwanie działki po adresie, interaktywna mapa z warstwami satelitarnymi, automatyczna identyfikacja granic (ULDK/GUGIK).
- **Zastosowanie:** Błyskawiczna i precyzyjna weryfikacja lokalizacji i granic prawnych działki, stanowiąca punkt wyjścia dla każdej dalszej analizy.

### 2. Wizualizacja 3D otoczenia
- **Funkcja:** Generowanie modelu 3D zabudowy w promieniu 300m na podstawie danych OpenStreetMap.
- **Zastosowanie:** Głębokie zrozumienie kontekstu urbanistycznego. Analiza relacji przestrzennych, skali otoczenia i potencjalnego wpływu nowej zabudowy na istniejącą tkankę.

### 3. Analiza nasłonecznienia
- **Funkcja:** Symulacja ray-tracing z uwzględnieniem cieni rzucanych przez otaczające budynki. Mapa cieplna z liczbą godzin słońca i diagram ścieżki słońca.
- **Zastosowanie:** Obiektywna kwantyfikacja dostępu do światła naturalnego. Narzędzie do optymalizacji formy budynku w celu maksymalizacji komfortu użytkowników i dostarczenia twardych danych w procesie projektowym.

### 4. Analiza MPZP z AI
- **Funkcja:** Autonomiczny agent AI (Gemini 2.5 Pro) nawigujący po geoportalu, wykorzystujący OCR (Tesseract) do analizy zeskanowanych PDF.
- **Zastosowanie:** Automatyzacja i przyspieszenie żmudnego procesu analizy dokumentów planistycznych. Ekstrakcja kluczowych wskaźników i uwarunkowań w ułamku czasu wymaganego przy analizie manualnej.

## Wymagania

Aplikacja wymaga Google Cloud credentials dla Vertex AI (Gemini).

Skonfiguruj secrets w ustawieniach Space:
1. Przejdź do Settings → Repository secrets
2. Dodaj secrets zgodnie z dokumentacją

## Więcej informacji

- [GitHub Repository](https://github.com/FabianekPawianek/asystent-analizy-dzialki)
- [Changelog](CHANGELOG.md)
- [Wizja Projektu](VISION.md)

## Autor

Fabian Korycki | fabiankoryckiarchitecture@gmail.com

Powered by Google Gemini AI