# Changelog

Wszystkie istotne zmiany w projekcie będą dokumentowane w tym pliku.

Format oparty na [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
projekt używa [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planowane (v0.3.0) - Focus: Szczecin
Projekt w wersji 0.3 będzie kontynuował skupienie na mieście Szczecin, budując bazę analiz potrzebną do rzetelnych wniosków AI.

#### Analiza stanu istniejącego
- Automatyczne pobieranie zdjęć satelitarnych z Google Maps
- Integracja z Geoportal.gov.pl dla danych topograficznych
- Analiza wizualna stanu działki i otoczenia
- Identyfikacja istniejącej zabudowy i infrastruktury
- Ocena zagospodarowania terenu

#### Analiza historyczna
- Scraping i analiza historycznych zdjęć satelitarnych
- Przetwarzanie skanów historycznych map
- Analiza książek telefonicznych i dokumentów archiwalnych
- Rekonstrukcja historii zabudowy działki
- Timeline zmian zagospodarowania terenu

#### Przejście na CityGML 3D
- Przejście z danych OpenStreetMap na CityGML 3D dla bardziej szczegółowej analizy budynków
- Ulepszenie dokładności modeli 3D poprzez użycie danych.CityGML
- Zwiększenie szczegółowości analizy cieni i nasłonecznienia
- Scalanie danych.CityGML z danymi MPZP
- Możliwość usunięcia zabudowy na analizowanych działkach w przypadku nieistotnych budynków mogących zaburzać analizę nasłonecznienia

#### Planowane na późniejsze wersje
- Generowanie propozycji zabudowy (forma + podział funkcjonalny)
- Wariantowanie rozwiązań projektowych
- Rozszerzenie poza Szczecin (wymaga refaktoryzacji modułu MPZP)

## [0.2.1] - 2025-11-07

### Dodane
- **Możliwość wyboru wielu działek**: Użytkownik może teraz zaznaczać i analizować wiele działek jednocześnie
- **Automatyczne centrowanie mapy**: Mapa automatycznie centruje się na wybranej działce po jej kliknięciu
- **Ulepszona wizualizacja 3D**: Wszystkie zaznaczone działki są teraz wyświetlane z zielonymi obramówkami (RGB #28a745) w widoku 3D
- **Połączona analiza nasłonecznienia**: Analiza nasłonecznienia działa teraz na połączonym obszarze wszystkich zaznaczonych działek
- **Poprawiony model analemy słońca**: Zastosowanie bardziej zaawansowanego modelu astronomicznego do obliczeń ścieżek słońca
- **Skalowanie tytułu**: Responsywne skalowanie tytułu "Asystent Analizy Działki" z gradientem niebiesko-zielonym

### Zmienione
- **Stylizacja widoku 3D**: Granice działek teraz wyświetlają się jako zielone obramowania o grubości 1px dopasowane do kolorystyki aplikacji
- **Logika analizy nasłonecznienia**: Zastosowanie unii wybranych działek zamiast analizy pojedynczej działki
- **Interakcja z mapą**: Kliknięcie działki centruje mapę na tej konkretnej działce
- **Obsługa poświadczeń GCP**: Ulepszono bezpieczeństwo przez ekstrakcję ID projektu z poświadczeń zamiast twardego kodowania
- **Czyszczenie kodu**: Usunięto niepotrzebne pliki testowe OCR i inne nieużywane pliki

### Naprawione
- **Błędy analizy nasłonecznienia**: Naprawiono problemy z symulacją solarną przy wyborze wielu działek
- **Analiza MPZP**: Poprawnie ekstrahuje ID projektu z poświadczeń dla lepszego bezpieczeństwa
- **Transformacja współrzędnych**: Dodano brakującą funkcję `transform_single_coord` dla poprawnej konwersji CRS
- **Wybór motywu**: Usunięto niepotrzebne przyciski wyboru motywu ("Jasny"/"Ciemny")

### Usunięte
- **Przyciski wyboru motywu**: Uproszczono UI przez usunięcie opcji wyboru motywu mapy
- **Pliki testowe**: Usunięto testy OCR i inne niepotrzebne pliki deweloperskie

## [0.2.0] - 2025-10-23

### Dodane

#### Wizualizacja 3D otoczenia
- Generowanie modelu 3D otaczającej zabudowy na podstawie danych OpenStreetMap
- Funkcja `generate_3d_context_view()` z wykorzystaniem PyDeck
- Szacowanie wysokości budynków na podstawie parametrów OSM
- Wybór motywu mapy (jasny/ciemny)
- Interaktywne sterowanie kamerą (obrót, przesuwanie, zoom)

#### Analiza nasłonecznienia
- Symulacja ray-tracing z wykorzystaniem Trimesh
- Funkcja `run_solar_simulation()` obliczająca średnie dzienne nasłonecznienie
- Mapa cieplna przedstawiająca liczbę godzin słońca dla każdego punktu działki
- Uwzględnianie cieni rzucanych przez otaczającą zabudowę
- Konfigurowalne parametry analizy (zakres dat, przedział godzinowy)
- Diagram ścieżki słońca dla kluczowych dni roku (przesilenia, równonoce)
- Wizualizacja analemmy dla każdej godziny
- Oznaczenia azymutów i kierunków kardynalnych
- Zaznaczenie pozycji słońca w analizowanym okresie
- Funkcje pomocnicze:
  - `create_trimesh_scene()` - budowa geometrii 3D zabudowy
  - `create_analysis_grid()` - generowanie siatki punktów pomiarowych
  - `generate_sun_path_data()` - obliczanie ścieżki słońca
  - `generate_complete_sun_path_diagram()` - pełny diagram roczny
  - `value_to_rgb()` - mapowanie wartości na kolory
  - `create_discrete_legend_html()` - legenda mapy cieplnej

#### OCR dla zeskanowanych dokumentów PDF
- Automatyczna detekcja PDF bez warstwy tekstowej
- Integracja z Tesseract OCR z obsługą języka polskiego
- Fallback: standardowa ekstrakcja tekstu → OCR
- Feedback w czasie rzeczywistym (postęp OCR strona po stronie)
- Auto-konfiguracja ścieżki Tesseract dla Windows (linie 34-48)

#### Interfejs użytkownika
- Ekran powitalny z przyciskiem "Rozpocznij"
- System wyboru typu analizy (Nasłonecznienie / MPZP)
- Rozbudowane style CSS z gradientami i animacjami
- Smooth scroll między sekcjami
- Responsywny layout
- Ukryty scrollbar z zachowaniem funkcjonalności
- Karty analiz z efektami hover
- Niestandardowe style dla elementów formularzy

### Zmienione

#### Refaktoryzacja analizy MPZP
- Przepisano `analyze_documents_with_ai()` z lepszym zarządzaniem błędami
- Dodano fallback OCR dla dokumentów bez warstwy tekstowej
- Zwiększono fragment tekstu dla analizy AI (do 5000 znaków)
- Usunięto `@st.cache_data` z funkcji analizy (problemy z cache)
- Poprawiono obsługę błędów (zastąpiono `except: continue` szczegółową obsługą)

#### Wizualizacja
- Zwiększono wysokość mapy (500 → 700px dla wyszukiwania, 550px dla potwierdzenia)
- Zmieniono kolor obrysu działki na #28a745 (zielony)
- Dodano wypełnienie dla poligonu działki (fill_opacity=0.3)

#### Struktura kodu
- Dodano zarządzanie stanem sesji dla wyboru analizy
- Refaktoryzacja układu strony (conditional rendering)
- Usunięto emoji z komunikatów systemowych

### Naprawione
- Problem z pustymi dokumentami MPZP (OCR jako rozwiązanie)
- Błędna metoda ekstrakcji tekstu z PDF
- Cache Streamlit blokujący aktualizacje wyników analizy

### Zależności

#### Dodane biblioteki Python
- `osmnx` - pobieranie danych budynków z OpenStreetMap
- `pydeck` - wizualizacja 3D
- `pvlib` - obliczenia pozycji słońca
- `trimesh` - geometria 3D i ray-tracing
- `open3d` - operacje na chmurach punktów
- `pytesseract` - interfejs do Tesseract OCR
- `Pillow` - przetwarzanie obrazów dla OCR
- `numpy` - rozszerzone użycie w obliczeniach numerycznych
- `pandas` - zakresy dat i szeregi czasowe

#### Pakiety systemowe
- `tesseract-ocr` - silnik OCR
- `tesseract-ocr-pol` - dane językowe dla języka polskiego

### Breaking Changes
- Wymaga instalacji Tesseract OCR dla pełnej funkcjonalności
  - Lokalnie: instalacja manualna z konfiguracją ścieżki
  - Streamlit Cloud: automatyczna instalacja przez `packages.txt`
- Wymaga Google Cloud credentials dla Gemini AI (bez zmian od v0.1)
- Znacznie zwiększone wymagania pamięci RAM (symulacje 3D)

### Wydajność
- Ekstrakcja tekstu standardowa: < 1s
- OCR (dokument 3-stronicowy): 6-15s
- Analiza AI (Gemini 2.5 Pro): 5-10s
- Generowanie modelu 3D otoczenia: 2-5s
- Symulacja nasłonecznienia (1 dzień, 14 godzin): 15-30s
- Symulacja nasłonecznienia (zakres wielodniowy): proporcjonalnie dłużej

---

## [0.1.0] - 2025-10-10

### Dodane

#### Identyfikacja działki
- Wyszukiwanie po adresie z wykorzystaniem Nominatim API
- Konwersja adresu na współrzędne geograficzne
- Pobieranie danych ewidencyjnych z ULDK/GUGIK
- Funkcja `get_parcel_by_id()` - pobranie geometrii działki po ID
- Funkcja `get_parcel_from_coords()` - identyfikacja działki po współrzędnych
- Transformacja współrzędnych między EPSG:2180 a EPSG:4326

#### Wizualizacja
- Interaktywna mapa z wykorzystaniem Folium
- Warstwa satelitarna (Esri World Imagery)
- Warstwa działek ewidencyjnych (WMS z GUGIK)
- Wizualizacja poligonu wybranej działki
- Mapa potwierdzenia z zaznaczoną działką

#### Analiza MPZP z AI
- Agent AI Nawigator oparty na Selenium
- Autonomiczna nawigacja po geoportalu Szczecina
- Funkcja `run_ai_agent_flow()` - główny workflow agenta
- Funkcja `extract_links_by_clicking()` - ekstrakcja linków do dokumentów
- Funkcja `analyze_documents_with_ai()` - analiza treści PDF z Gemini AI
- Ekstrakcja tekstu z PDF (PyMuPDF - tylko warstwy tekstowe)
- Analiza ustaleń ogólnych (cel planu)
- Analiza ustaleń szczegółowych (przeznaczenie, wysokość, wskaźniki, dach)
- Obsługa trzech stanów MPZP (uchwalony, wszczęty, brak)

#### Interfejs użytkownika
- Podstawowy layout Streamlit
- Formularz wyszukiwania adresu
- Workflow 2-krokowy (wybór działki → analiza)
- Wyświetlanie wyników analizy w expanderach
- System powiadomień (success, error, info, warning)

### Zależności

#### Biblioteki Python
- `streamlit` - framework aplikacji webowej
- `folium`, `streamlit-folium` - mapy interaktywne
- `pyproj` - transformacje współrzędnych
- `shapely` - operacje geometryczne
- `requests` - zapytania HTTP
- `PyMuPDF` (fitz) - ekstrakcja tekstu z PDF
- `selenium` - automatyzacja przeglądarki
- `vertexai` - Google Vertex AI SDK
- `langchain-google-vertexai` - integracja LangChain z Gemini

#### Usługi zewnętrzne
- ULDK/GUGIK - dane ewidencyjne działek
- Nominatim - geokodowanie adresów
- Google Vertex AI - Gemini 2.5 Pro (analiza dokumentów)
- Geoportal Szczecin - dane MPZP

### Wymagania
- Google Cloud credentials dla Gemini AI
- ChromeDriver dla Selenium
- Połączenie internetowe (API calls)

---

## Legenda typów zmian

- **Dodane** - nowe funkcje
- **Zmienione** - zmiany w istniejących funkcjach
- **Usunięte** - usunięte funkcje
- **Naprawione** - poprawki błędów
- **Bezpieczeństwo** - poprawki bezpieczeństwa
- **Zależności** - aktualizacje pakietów
- **Breaking Changes** - zmiany łamiące kompatybilność
- **Wydajność** - optymalizacje

---

## Links

- [Repository](https://github.com/FabianekPawianek/Asystent-Analizy-Dzialki)
- [Releases](https://github.com/FabianekPawianek/Asystent-Analizy-Dzialki/releases)
- [Issues](https://github.com/FabianekPawianek/Asystent-Analizy-Dzialki/issues)

[Unreleased]: https://github.com/FabianekPawianek/Asystent-Analizy-Dzialki/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/FabianekPawianek/Asystent-Analizy-Dzialki/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/FabianekPawianek/Asystent-Analizy-Dzialki/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/FabianekPawianek/Asystent-Analizy-Dzialki/releases/tag/v0.1.0