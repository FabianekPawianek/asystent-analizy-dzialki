import requests
import json
import xml.etree.ElementTree as ET
import io
import os
import config
from google import genai
from google.genai import types

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, box, shape
    from shapely import wkt
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

POG_WFS_URL = "https://mapy.geoportal.gov.pl/wss/ext/PlanyOgolneGmin"

client = None

def init_ai(api_key=None):
    global client
    if not api_key:
        try:
            import streamlit as st
            api_key = config.get_google_api_key(st.secrets)
        except Exception:
            api_key = config.get_google_api_key()
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise Exception(f"Nie udało się zainicjalizować Google AI w pog_agent: {e}")

def _extract_bbox_and_poly(parcel_gdf):

    coords = None
    geom = None
    minx, miny, maxx, maxy = None, None, None, None

    if isinstance(parcel_gdf, dict):
        if "Współrzędne EPSG:2180" in parcel_gdf:
            coords = parcel_gdf["Współrzędne EPSG:2180"]
        elif "geometry" in parcel_gdf:
            geom = parcel_gdf["geometry"]
    elif isinstance(parcel_gdf, list):
        coords = parcel_gdf
    elif isinstance(parcel_gdf, str):
        if HAS_GEOPANDAS:
            try:
                geom = wkt.loads(parcel_gdf)
            except Exception:
                pass
    elif HAS_GEOPANDAS and isinstance(parcel_gdf, gpd.GeoDataFrame):
        if not parcel_gdf.empty:
            bounds = parcel_gdf.total_bounds # (minx, miny, maxx, maxy)
            return bounds[0], bounds[1], bounds[2], bounds[3], parcel_gdf.geometry.iloc[0]

    if coords:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        if HAS_GEOPANDAS and len(coords) >= 3:
            geom = Polygon(coords)

    if geom is not None and (minx is None or miny is None):
        bounds = geom.bounds
        minx, miny, maxx, maxy = bounds[0], bounds[1], bounds[2], bounds[3]

    if minx is None:
        raise ValueError("Nie można wyznaczyć geometrii ani BBOX z podanych danych działki.")

    if minx == maxx:
        minx -= 1.0
        maxx += 1.0
    if miny == maxy:
        miny -= 1.0
        maxy += 1.0

    return minx, miny, maxx, maxy, geom


def fetch_pog_data_for_parcel(parcel_gdf):

    minx, miny, maxx, maxy, geom = _extract_bbox_and_poly(parcel_gdf)
    bbox_str = f"{minx},{miny},{maxx},{maxy}"
    bbox_crs_str = f"{minx},{miny},{maxx},{maxy},EPSG:2180"

    pog_data = {
        "bbox": [minx, miny, maxx, maxy],
        "strefa_symbol": None,
        "strefa_nazwa": None,
        "max_wysokosc_m": None,
        "min_biologicznie_czynna_pct": None,
        "max_intensywnosc_zabudowy": None,
        "max_powierzchnia_zabudowy_pct": None,
        "obszar_uzupelnienia_zabudowy_ouz": None,
        "obszar_zabudowy_srodmiejskiej_ozs": None,
        "akt_planowania_nazwa": None,
        "akt_planowania_uchwala": None,
        "gmina": None,
        "raw_attributes": {}
    }

    headers = {'User-Agent': 'AsystentAnalizyDzialki/2.2'}

    wfs_params_options = [
        {
            'SERVICE': 'WFS',
            'VERSION': '1.1.0',
            'REQUEST': 'GetFeature',
            'TYPENAME': 'pog:StrefaPlanistyczna',
            'BBOX': bbox_crs_str,
            'SRSNAME': 'EPSG:2180'
        },
        {
            'SERVICE': 'WFS',
            'VERSION': '1.1.0',
            'REQUEST': 'GetFeature',
            'TYPENAME': 'strefaPlanistyczna',
            'BBOX': bbox_crs_str,
            'SRSNAME': 'EPSG:2180'
        },
        {
            'SERVICE': 'WFS',
            'VERSION': '2.0.0',
            'REQUEST': 'GetFeature',
            'TYPENAMES': 'pog:StrefaPlanistyczna',
            'BBOX': bbox_crs_str,
            'SRSNAME': 'EPSG:2180'
        }
    ]

    response_text = None
    for params in wfs_params_options:
        try:
            resp = requests.get(POG_WFS_URL, params=params, headers=headers, timeout=12)
            if resp.status_code == 200 and ("Feature" in resp.text or "strefa" in resp.text.lower() or "msGMLOutput" in resp.text):
                response_text = resp.text
                break
        except Exception:
            continue

    if not response_text:
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        wms_params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetFeatureInfo',
            'LAYERS': 'strefaPlanistyczna,obszarUzupelnieniaZabudowy,obszarZabSrodmiejskiej,aktPlanowaniaprzestrzennego',
            'QUERY_LAYERS': 'strefaPlanistyczna,obszarUzupelnieniaZabudowy,obszarZabSrodmiejskiej,aktPlanowaniaprzestrzennego',
            'BBOX': f"{miny},{minx},{maxy},{maxx}",
            'CRS': 'EPSG:2180',
            'WIDTH': '101',
            'HEIGHT': '101',
            'I': '50',
            'J': '50',
            'INFO_FORMAT': 'application/vnd.ogc.gml'
        }
        try:
            resp = requests.get(POG_WFS_URL, params=wms_params, headers=headers, timeout=12)
            if resp.status_code == 200:
                response_text = resp.text
        except Exception:
            pass

    if response_text:
        try:
            root = ET.fromstring(response_text)
            for elem in root.iter():
                tag_clean = elem.tag.split('}')[-1].lower()
                text_val = (elem.text or "").strip()
                if not text_val:
                    continue

                pog_data["raw_attributes"][tag_clean] = text_val

                if tag_clean in ["symbol", "strefasymbol", "oznaczenie", "symbolstrefy"]:
                    pog_data["strefa_symbol"] = text_val
                elif tag_clean in ["nazwa", "strefanazwa", "nazwastrefy"]:
                    pog_data["strefa_nazwa"] = text_val
                elif "wysokosc" in tag_clean or "height" in tag_clean:
                    pog_data["max_wysokosc_m"] = text_val
                elif "biologicz" in tag_clean:
                    pog_data["min_biologicznie_czynna_pct"] = text_val
                elif "intensywnosc" in tag_clean:
                    pog_data["max_intensywnosc_zabudowy"] = text_val
                elif "powierzchniazabudowy" in tag_clean or "udzialzabudowy" in tag_clean:
                    pog_data["max_powierzchnia_zabudowy_pct"] = text_val
                elif "ouz" in tag_clean or "uzupelnieni" in tag_clean:
                    pog_data["obszar_uzupelnienia_zabudowy_ouz"] = text_val
                elif "ozs" in tag_clean or "srodmiejsk" in tag_clean:
                    pog_data["obszar_zabudowy_srodmiejskiej_ozs"] = text_val
                elif "uchwala" in tag_clean or "numeruchwaly" in tag_clean:
                    pog_data["akt_planowania_uchwala"] = text_val
                elif "gmina" in tag_clean:
                    pog_data["gmina"] = text_val
        except Exception:
            pass

    if not pog_data["strefa_symbol"]:
        pog_data["strefa_symbol"] = "Brak jednoznacznego oznaczenia WFS (wymaga weryfikacji w urzędzie gminy)"
    if not pog_data["strefa_nazwa"]:
        pog_data["strefa_nazwa"] = "Strefa planistyczna POG"

    return pog_data


def analyze_pog_with_ai(pog_data_dict, lang="PL"):
    global client
    if client is None:
        init_ai()

    gmina = pog_data_dict.get("gmina") or "Brak danych"
    uchwala = pog_data_dict.get("akt_planowania_uchwala") or "Brak danych"
    symbol = pog_data_dict.get("strefa_symbol") or "Brak symbolu"
    nazwa = pog_data_dict.get("strefa_nazwa") or "Strefa planistyczna POG"
    ouz = pog_data_dict.get("obszar_uzupelnienia_zabudowy_ouz") or "Brak / Nie dotyczy"
    ozs = pog_data_dict.get("obszar_zabudowy_srodmiejskiej_ozs") or "Brak / Nie dotyczy"
    wysokosc = pog_data_dict.get("max_wysokosc_m") or "Brak ustalenia w POG"
    biologiczna = pog_data_dict.get("min_biologicznie_czynna_pct") or "Brak ustalenia"
    intensywnosc = pog_data_dict.get("max_intensywnosc_zabudowy") or "Brak ustalenia"
    pow_zabudowy = pog_data_dict.get("max_powierzchnia_zabudowy_pct") or "Brak ustalenia"
    raw_attrs = pog_data_dict.get("raw_attributes", {})

    system_instruction = """
Jesteś doświadczonym urbanistą i architektem. 
Twoim zadaniem jest sporządzenie zwięzłej, czytelnej Karty Planistycznej Planu Ogólnego Gminy (POG) w formacie Markdown na podstawie dostarczonych danych przestrzennych GML.
Formatuj odpowiedź w przejrzystym języku Markdown z użyciem nagłówków, czytelnych punktów i wyróżnień. Nie używaj emoji.
"""

    prompt = f"""
DANE Z PARSERA GML:
- Gmina: {gmina}
- Uchwała/Akt: {uchwala}
- Strefa Planistyczna: {symbol} ({nazwa})
- Obszar Uzupełnienia Zabudowy (OUZ): {ouz}
- Obszar Zabudowy Śródmiejskiej (OZS): {ozs}
- Maksymalna wysokość (m): {wysokosc}
- Min. pow. biologicznie czynna (%): {biologiczna}
- Maks. intensywność zabudowy: {intensywnosc}
- Maks. pow. zabudowy (%): {pow_zabudowy}
- Wszystkie atrybuty surowe: {raw_attrs}

ZASADY GENEROWANIA KARTY:
1. Skonsoliduj dane w przejrzystą tabelę Markdown.
2. Klasyfikacja Strefy:
   - Jeśli strefa ma charakter NIEOBJĘTY ZABUDOWĄ KUBATUROWĄ (np. symbol SN - zieleń, SP - rola, SOK - ochrona krajobrazu): W punkcie dotyczącym wysokości i intensywności napisz wprost: "Teren wyłączony z intensywnej zabudowy kubaturowej". NIE generuj wymijających tekstów "wymaga weryfikacji w MPZP".
3. Ocena OUZ (Obszar Uzupełnienia Zabudowy):
   - Jeśli OUZ przyjmuje wartość "Brak", "Nie dotyczy", "NIE" lub "False", dodaj jasną informację w sekcji wniosków: "Działka znajduje się poza OUZ – brak możliwości wydania decyzji o Warunkach Zabudowy (WZ)".

WYMAGANY FORMAT ODPOWIEDZI (Markdown):

### Karta Planistyczna POG

| Parametr | Ustalenie POG |
| :--- | :--- |
| **Gmina / Akt Prawny** | {gmina} ({uchwala}) |
| **Strefa Planistyczna** | **{symbol}** - {nazwa} |
| **Obszar Uzupełnienia Zabudowy (OUZ)** | {ouz} |
| **Obszar Zabudowy Śródmiejskiej (OZS)** | {ozs} |
| **Min. Pow. Biologicznie Czynna** | **{biologiczna}** |
| **Maks. Wysokość Zabudowy** | {wysokosc} |
| **Maks. Intensywność Zabudowy** | {intensywnosc} |

#### Wnioski i Wytyczne Architektoniczne
- **Potencjał Inwestycyjny:** [2-3 zwięzłe zdania określające czy i co można tu wybudować na podstawie strefy]
- **Kluczowe Ograniczenia:** [Główne wymogi wynikające ze strefy oraz statusu OUZ/OZS]
"""

    config_params = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.2,
    )

    response = client.models.generate_content(
        model=config.MODEL_NAME,
        contents=prompt,
        config=config_params
    )

    return response.text

def run_pog_analysis_flow(parcel_gdf, status_callback=None, lang="PL"):

    if status_callback:
        status_callback("info", "Pobieranie danych WFS z Geoportalu (Plan Ogólny Gminy)...")

    try:
        pog_data = fetch_pog_data_for_parcel(parcel_gdf)
    except Exception as e:
        if status_callback:
            status_callback("error", f"Błąd pobierania danych POG: {e}")
        return {
            "status": "error",
            "error": str(e),
            "raw_data": None,
            "analysis": None
        }

    if status_callback:
        status_callback("info", "Przetwarzanie parametrów i generowanie Karty Planistycznej przez AI...")

    try:
        ai_card_markdown = analyze_pog_with_ai(pog_data, lang=lang)
    except Exception as e:
        ai_card_markdown = f"### Karta Planistyczna POG (Surowe dane)\n\n- **Symbol Strefy:** {pog_data.get('strefa_symbol')}\n- **Nazwa Strefy:** {pog_data.get('strefa_nazwa')}\n- **Wysokość zabudowy:** {pog_data.get('max_wysokosc_m') or 'Brak'}\n- **Powierzchnia biologicznie czynna:** {pog_data.get('min_biologicznie_czynna_pct') or 'Brak'}\n\n*Uwaga: Generowanie opisu AI nie powiodło się: {e}*"

    if status_callback:
        status_callback("success", "Analiza POG zakończona pomyślnie.")

    return {
        "status": "success",
        "raw_data": pog_data,
        "analysis": ai_card_markdown
    }
