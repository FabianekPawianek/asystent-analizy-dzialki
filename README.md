# Asystent Analizy Działki: Manifest dla Architektury Skoncentrowanej na Człowieku

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://fabianekpawianek-asystent-analizy-dzialki.hf.space)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-0.2.2-green.svg)](https://github.com/FabianekPawianek/asystent-analizy-dzialki/releases/tag/v0.2.2)

**Wersja Beta 0.2.2 (Szczecin/Polska)**

To narzędzie jest praktycznym zastosowaniem filozofii **Parametrycznego Humanizmu**. To protest przeciwko architekturze zysku i manifest na rzecz jakości życia, tworzonej dla ludzi, a nie dla arkuszy kalkulacyjnych.

Aplikacja dostarcza obiektywnych, mierzalnych danych, które stanowią fundament świadomego projektowania i niepodważalny argument w rozmowach z inwestorami.

**Dowiedz się więcej o filozofii projektu w pliku [VISION.md](VISION.md)**

## Live Demo

**[Otwórz aplikację](https://fabianekpawianek-asystent-analizy-dzialki.hf.space)**

Hostowane na Hugging Face Spaces.

## Kluczowe Możliwości i ich Zastosowanie

### 1. Identyfikacja działki
- **Funkcja:** Wyszukiwanie działki po adresie, interaktywna mapa z warstwami satelitarnymi, automatyczna identyfikacja granic (ULDK/GUGIK), możliwość wyboru i analizy wielu działek jednocześnie.
- **Zastosowanie:** Błyskawiczna i precyzyjna weryfikacja lokalizacji i granic prawnych działki. Możliwość analizy wielu działek naraz dla kompleksowej oceny obszaru, z automatycznym centrowaniem mapy na wybranej działce.

### 2. Zaawansowana Analiza Nasłonecznienia LiDAR (NOWOŚĆ)
- **Funkcja:** Symulacja ray-tracing oparta na Numerycznym Modelu Pokrycia Terenu (DSM) z Geoportalu.
- **Unikalność:** W przeciwieństwie do standardowych narzędzi opartych o proste bryły, system ten **uwzględnia realną zieleń (drzewa), ukształtowanie terenu oraz istniejącą zabudowę**.
- **Technologia:** Wykorzystuje dwa modele terenu (DSM i DTM) do "oczyszczenia" badanej działki pod nową inwestycję, zachowując jednocześnie cienie rzucane przez otoczenie. Precyzja dostępna dla każdego zakątka Polski.

### 3. Wizualizacja 3D otoczenia
- **Funkcja:** Generowanie modelu 3D zabudowy w promieniu 300m na podstawie danych OpenStreetMap.
- **Zastosowanie:** Głębokie zrozumienie kontekstu urbanistycznego. Analiza relacji przestrzennych, skali otoczenia i potencjalnego wpływu nowej zabudowy na istniejącą tkankę. Teraz umożliwia jednoczesne wizualizację wielu działek.

### 4. Analiza MPZP z AI
- **Funkcja:** Autonomiczny agent AI (Gemini 2.5 Pro) nawigujący po geoportalu, wykorzystujący OCR (Tesseract) do analizy zeskanowanych PDF, z bezpiecznym wyodrębnianiem ID projektu z poświadczeń.
- **Zastosowanie:** Automatyzacja i przyspieszenie żmudnego procesu analizy dokumentów planistycznych. Ekstrakcja kluczowych wskaźników i uwarunkowań w ułamku czasu wymaganego przy analizie manualnej, z lepszym bezpieczeństwem danych.

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

Powered by Google Gemini AI & Geoportal.gov.pl