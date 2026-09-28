import requests
import json
import xml.etree.ElementTree as ET
import io
import os
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import config
from google import genai
from google.genai import types

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, Point, box, shape
    from shapely import wkt
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

POG_WFS_URL = "https://mapy.geoportal.gov.pl/wss/ext/PlanyOgolneGmin"
KIMPZP_URLS = [
    "https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaMiejscowychPlanowZagospodarowaniaPrzestrzennego",
    "https://mapy.geoportal.gov.pl/wss/ext/KrajowaIntegracjaMiejscowychPlanowZagospodarowaniaPrzestrzennego"
]

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
        if "Geometria" in parcel_gdf and HAS_GEOPANDAS:
            try:
                geom = wkt.loads(parcel_gdf["Geometria"])
            except Exception:
                pass
        if geom is None:
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
            bounds = parcel_gdf.total_bounds
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


def fix_polish_encoding(text: str) -> str:
    """Repairs double-encoded or Latin1-decoded UTF-8 Polish text strings."""
    if not text or not isinstance(text, str):
        return text
    try:
        if any(c in text for c in ['Ä', 'Ã', 'Å']):
            return text.encode('latin1').decode('utf-8')
    except Exception:
        pass
    return text.strip()


def parse_geoserver_feature_info_html(html_text: str):
    """
    Ekstrahuje atrybuty tabelaryczne oraz linki URL z odpowiedzi GeoServer GetFeatureInfo (HTML).
    """
    attributes = {}
    links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html_text, re.IGNORECASE)

    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html_text, re.DOTALL | re.IGNORECASE)
    if len(rows) >= 2:
        headers = [re.sub(r'<[^>]+>', '', c).strip().lower() for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', rows[0], re.DOTALL | re.IGNORECASE)]
        for row in rows[1:]:
            values = [fix_polish_encoding(re.sub(r'<[^>]+>', '', c).strip()) for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', row, re.DOTALL | re.IGNORECASE)]
            for h, v in zip(headers, values):
                if h and v and h not in attributes:
                    attributes[h] = v
    elif len(rows) == 1:
        cols = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', rows[0], re.DOTALL | re.IGNORECASE)
        for i in range(0, len(cols) - 1, 2):
            k = re.sub(r'<[^>]+>', '', cols[i]).strip().lower()
            v = fix_polish_encoding(re.sub(r'<[^>]+>', '', cols[i+1]).strip())
            if k and v and k not in attributes:
                attributes[k] = v

    th_td_pairs = re.findall(r'<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>', html_text, re.DOTALL | re.IGNORECASE)
    for th, td in th_td_pairs:
        k = re.sub(r'<[^>]+>', '', th).strip().lower()
        v = fix_polish_encoding(re.sub(r'<[^>]+>', '', td).strip())
        if k and v and k not in attributes:
            attributes[k] = v

    if not attributes:
        clean_text = re.sub(r'<[^>]+>', ' ', html_text)
        for line in clean_text.split('\n'):
            line_str = line.strip()
            if ':' in line_str:
                parts = line_str.split(':', 1)
                k = parts[0].strip().lower()
                v = fix_polish_encoding(parts[1].strip())
                if len(k) < 40 and v:
                    attributes[k] = v

    return attributes, links


def fetch_mpzp_kimpzp(parcel_gdf) -> dict:
    """
    Pobiera dane o MPZP z oficjalnej usługi GUGiK KIMPZP.
    Używa wyłącznie sprawdzonych warstw roboczych: 'plany,granice' (bez app:PrzeznaczenieTerenu).
    Filtruje i odrzuca wszelkie linki prowadzące do plików legendy (legenda / _legenda).
    """
    minx, miny, maxx, maxy, geom = _extract_bbox_and_poly(parcel_gdf)

    mpzp_data = {
        "has_mpzp": False,
        "nazwa_planu": None,
        "numer_uchwaly": None,
        "data_uchwaly": None,
        "link_uchwala_tekst": None,
        "link_rysunek": None,
        "gmina": None,
        "raw_attributes": {}
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }

    wms_queries = [
        {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetFeatureInfo',
            'LAYERS': 'plany,granice',
            'QUERY_LAYERS': 'plany,granice',
            'BBOX': f"{miny},{minx},{maxy},{maxx}",
            'CRS': 'EPSG:2180',
            'WIDTH': '101',
            'HEIGHT': '101',
            'I': '50',
            'J': '50',
            'INFO_FORMAT': 'text/html'
        },
        {
            'SERVICE': 'WMS',
            'VERSION': '1.1.1',
            'REQUEST': 'GetFeatureInfo',
            'LAYERS': 'plany,granice',
            'QUERY_LAYERS': 'plany,granice',
            'BBOX': f"{minx},{miny},{maxx},{maxy}",
            'SRS': 'EPSG:2180',
            'WIDTH': '101',
            'HEIGHT': '101',
            'X': '50',
            'Y': '50',
            'INFO_FORMAT': 'text/html'
        },
        {
            'SERVICE': 'WFS',
            'VERSION': '1.1.0',
            'REQUEST': 'GetFeature',
            'TYPENAME': 'plany',
            'BBOX': f"{minx},{miny},{maxx},{maxy},EPSG:2180",
            'SRSNAME': 'EPSG:2180'
        }
    ]

    response_text = None
    for endpoint in KIMPZP_URLS:
        for params in wms_queries:
            try:
                resp = requests.get(endpoint, params=params, headers=headers, timeout=15, verify=False)
                resp.encoding = 'utf-8'
                raw_text = resp.content.decode('utf-8', errors='replace')
                if resp.status_code == 200 and len(raw_text.strip()) > 30:
                    txt_lower = raw_text.lower()
                    if not any(ign in txt_lower for ign in ["brak danych", "nie znaleziono", "empty", "invalid layer"]):
                        response_text = raw_text
                        break
            except Exception:
                continue
        if response_text:
            break

    if response_text:
        if "<html" in response_text.lower() or "<table" in response_text.lower() or "<body" in response_text.lower():
            parsed_attrs, parsed_links = parse_geoserver_feature_info_html(response_text)
            mpzp_data["raw_attributes"] = parsed_attrs

            if parsed_attrs:
                mpzp_data["nazwa_planu"] = fix_polish_encoding(parsed_attrs.get("nazwa_plan") or parsed_attrs.get("nazwa") or parsed_attrs.get("plan"))
                mpzp_data["numer_uchwaly"] = fix_polish_encoding(parsed_attrs.get("nr_uch_uch") or parsed_attrs.get("nr_uch_wsz") or parsed_attrs.get("uchwala") or parsed_attrs.get("numer"))
                mpzp_data["data_uchwaly"] = fix_polish_encoding(parsed_attrs.get("data_uch_u") or parsed_attrs.get("data_uch_w") or parsed_attrs.get("data"))
                mpzp_data["gmina"] = fix_polish_encoding(parsed_attrs.get("gmina") or parsed_attrs.get("miejscowosc"))
                mpzp_data["link_uchwala_tekst"] = parsed_attrs.get("link2") or parsed_attrs.get("dzu_link") or parsed_attrs.get("link_do_bi") or parsed_attrs.get("link3")
                
                raw_rysunek = parsed_attrs.get("link_rysunek") or parsed_attrs.get("legenda1")
                if raw_rysunek and not any(bad in str(raw_rysunek).lower() for bad in ["legenda", "_legenda", "legend"]):
                    mpzp_data["link_rysunek"] = raw_rysunek
                else:
                    mpzp_data["link_rysunek"] = None

                if parsed_attrs.get("id_planu") or mpzp_data["nazwa_planu"] or mpzp_data["numer_uchwaly"]:
                    mpzp_data["has_mpzp"] = True

            all_candidate_urls = list(parsed_links)
            for v in parsed_attrs.values():
                if isinstance(v, str) and v.startswith("http"):
                    all_candidate_urls.append(v)
                elif isinstance(v, str) and any(ext in v.lower() for ext in [".pl", ".gov.pl", ".eu", ".net"]):
                    m = re.search(r'https?://[^\s\'"<>]+', v)
                    if m:
                        all_candidate_urls.append(m.group(0))

            for u in all_candidate_urls:
                u_lower = u.lower()
                if any(bad in u_lower for bad in ["legenda", "_legenda", "legend"]):
                    continue

                if any(ext in u_lower for ext in ["rysunek", "zalacznik", ".tif", ".tiff", "geotiff", "arkusz", "mapa"]):
                    if not mpzp_data["link_rysunek"]:
                        mpzp_data["link_rysunek"] = u
                elif any(w in u_lower for w in ["bip", "dzu", "edziennik", "duwo", "tekst", "uchwal", ".pdf", "wykazplanow"]):
                    if not mpzp_data["link_uchwala_tekst"]:
                        mpzp_data["link_uchwala_tekst"] = u

            if not mpzp_data["numer_uchwaly"]:
                m_nr = re.search(r'(?:uchwa[łl][ay]?|nr|numer)\s*(?:nr\s*)?[:\s]+([IVXLCDM0-9\/\-_]+(?:\/[0-9]+)?)', response_text, re.IGNORECASE)
                if m_nr:
                    mpzp_data["numer_uchwaly"] = fix_polish_encoding(m_nr.group(1).strip())

            if not mpzp_data["data_uchwaly"]:
                m_dt = re.search(r'(?:z\s+dnia|data)[:\s]+(\d{1,2}[\.\-\/]\d{1,2}[\.\-\/]\d{4}|\d{4}[\.\-\/]\d{1,2}[\.\-\/]\d{1,2})', response_text, re.IGNORECASE)
                if m_dt:
                    mpzp_data["data_uchwaly"] = fix_polish_encoding(m_dt.group(1).strip())

            if parsed_attrs.get("id_planu") or parsed_attrs.get("nazwa_plan") or parsed_attrs.get("nr_uch_uch") or mpzp_data["nazwa_planu"] or mpzp_data["numer_uchwaly"] or mpzp_data["link_uchwala_tekst"] or mpzp_data["link_rysunek"]:
                mpzp_data["has_mpzp"] = True
            if mpzp_data["has_mpzp"] and not mpzp_data["nazwa_planu"]:
                mpzp_data["nazwa_planu"] = "Obowiązujący Miejscowy Plan Zagospodarowania Przestrzennego"
        else:
            try:
                root = ET.fromstring(response_text)
                for elem in root.iter():
                    tag_clean = elem.tag.split('}')[-1].lower()
                    text_val = fix_polish_encoding((elem.text or "").strip())
                    if not text_val:
                        continue
                    mpzp_data["raw_attributes"][tag_clean] = text_val
                    if any(k in tag_clean for k in ["nazwa", "tytul", "name"]) and "id" not in tag_clean and not mpzp_data["nazwa_planu"]:
                        mpzp_data["nazwa_planu"] = text_val
                    elif any(k in tag_clean for k in ["uchwala", "numer", "nruchwaly"]) and not mpzp_data["numer_uchwaly"]:
                        mpzp_data["numer_uchwaly"] = text_val
                    elif any(k in tag_clean for k in ["data", "datauchwaly"]) and not mpzp_data["data_uchwaly"]:
                        mpzp_data["data_uchwaly"] = text_val
                    elif any(k in tag_clean for k in ["tekst", "bip", "dziennik", "link_tekst", "url"]) and not mpzp_data["link_uchwala_tekst"]:
                        mpzp_data["link_uchwala_tekst"] = text_val
                    elif any(k in tag_clean for k in ["rysunek", "geotiff", "zalacznik", "link_rysunek"]) and not mpzp_data["link_rysunek"]:
                        if not any(bad in text_val.lower() for bad in ["legenda", "_legenda", "legend"]):
                            mpzp_data["link_rysunek"] = text_val
                    elif "gmina" in tag_clean and not mpzp_data["gmina"]:
                        mpzp_data["gmina"] = text_val

                if mpzp_data["nazwa_planu"] or mpzp_data["numer_uchwaly"] or mpzp_data["link_uchwala_tekst"]:
                    mpzp_data["has_mpzp"] = True
            except Exception:
                pass

    if mpzp_data.get("link_rysunek"):
        if any(bad in str(mpzp_data["link_rysunek"]).lower() for bad in ["legenda", "_legenda", "legend"]):
            mpzp_data["link_rysunek"] = None

    if mpzp_data.get("nazwa_planu"):
        mpzp_data["nazwa_planu"] = fix_polish_encoding(mpzp_data["nazwa_planu"])
    if mpzp_data.get("numer_uchwaly"):
        mpzp_data["numer_uchwaly"] = fix_polish_encoding(mpzp_data["numer_uchwaly"])
    if mpzp_data.get("gmina"):
        mpzp_data["gmina"] = fix_polish_encoding(mpzp_data["gmina"])

    if not mpzp_data["has_mpzp"]:
        mpzp_data["nazwa_planu"] = "Brak obowiązującego planu miejscowego"

    return mpzp_data


def fetch_pog_data_for_parcel(parcel_gdf):
    """
    Pobiera dane o Planie Ogólnym Gminy (POG) z oficjalnych usług WFS/WMS Geoportalu.
    Wykorzystuje WMS GetFeatureInfo jako metodę główną (GML) z zachowaniem pełnej zgodności parametrów.
    """
    minx, miny, maxx, maxy, geom = _extract_bbox_and_poly(parcel_gdf)
    bbox_str = f"{minx},{miny},{maxx},{maxy}"
    bbox_crs_str = f"{minx},{miny},{maxx},{maxy},EPSG:2180"

    pog_data = {
        "has_pog": False,
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

    response_text = None

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
        resp.encoding = 'utf-8'
        raw_text = resp.content.decode('utf-8', errors='replace')
        if resp.status_code == 200 and len(raw_text.strip()) > 50:
            if not ("<msGMLOutput" in raw_text and "</msGMLOutput>" in raw_text and len(raw_text.strip()) < 150):
                response_text = raw_text
    except Exception:
        pass

    if not response_text:
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
        for params in wfs_params_options:
            try:
                resp = requests.get(POG_WFS_URL, params=params, headers=headers, timeout=6)
                resp.encoding = 'utf-8'
                raw_text = resp.content.decode('utf-8', errors='replace')
                if resp.status_code == 200 and ("Feature" in raw_text or "strefa" in raw_text.lower() or "msGMLOutput" in raw_text):
                    response_text = raw_text
                    break
            except Exception:
                continue

    if response_text:
        try:
            root = ET.fromstring(response_text)
            for elem in root.iter():
                tag_clean = elem.tag.split('}')[-1].lower()
                text_val = fix_polish_encoding((elem.text or "").strip())
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

    if pog_data.get("strefa_nazwa"):
        pog_data["strefa_nazwa"] = fix_polish_encoding(pog_data["strefa_nazwa"])
    if pog_data.get("gmina"):
        pog_data["gmina"] = fix_polish_encoding(pog_data["gmina"])
    if pog_data.get("akt_planowania_uchwala"):
        pog_data["akt_planowania_uchwala"] = fix_polish_encoding(pog_data["akt_planowania_uchwala"])

    has_real_symbol = bool(pog_data["strefa_symbol"] and "brak" not in pog_data["strefa_symbol"].lower())
    has_metrics = bool(pog_data["max_wysokosc_m"] or pog_data["min_biologicznie_czynna_pct"] or pog_data["max_intensywnosc_zabudowy"])
    pog_data["has_pog"] = has_real_symbol or has_metrics

    if not pog_data["strefa_symbol"]:
        pog_data["strefa_symbol"] = "Brak jednoznacznego oznaczenia WFS (wymaga weryfikacji w urzędzie gminy)" if pog_data["has_pog"] else "Brak w WFS"
    if not pog_data["strefa_nazwa"]:
        pog_data["strefa_nazwa"] = "Strefa planistyczna POG" if pog_data["has_pog"] else "Brak opublikowanego POG"

    return pog_data


def analyze_pog_with_ai(pog_data_dict, lang="PL"):
    """
    Generuje pełną Kartę Planistyczną POG na podstawie pobranych parametrów GML.
    """
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


def analyze_planning_documents_with_ai(pog_data_dict: dict, mpzp_data_dict: dict, lang="PL") -> str:
    """
    Zintegrowana prezentacja stanu planistycznego:
    1. Prezentuje status MPZP (obowiązuje z nazwą i uchwałą lub brak planu).
    2. Jeśli POG istnieje w WFS (np. Koszalin) - prezentuje PEŁNĄ Kartę Planistyczną POG.
    3. Jeśli brak POG w WFS - jasno informuje o braku opublikowanego POG w WFS.
    """
    has_mpzp = mpzp_data_dict.get("has_mpzp", False) if isinstance(mpzp_data_dict, dict) else False
    nazwa_planu = mpzp_data_dict.get("nazwa_planu") if has_mpzp else None
    numer_uchwaly = mpzp_data_dict.get("numer_uchwaly") if has_mpzp else None

    if has_mpzp:
        uchwala_str = f", Uchwała nr {numer_uchwaly}" if numer_uchwaly else ""
        nazwa_str = nazwa_planu or "Miejscowy Plan Zagospodarowania Przestrzennego"
        mpzp_section = f"""### Miejscowy Plan Zagospodarowania Przestrzennego (MPZP)
- **Status:** Obowiązuje na działce
- **Nazwa planu:** {nazwa_str}
- **Uchwała:** {numer_uchwaly or 'Obowiązująca'}

*Dla działki obowiązuje MPZP ({nazwa_str}{uchwala_str}). Stanowi on bezpośrednią podstawę do wydania pozwolenia na budowę.*"""
    else:
        mpzp_section = """### Miejscowy Plan Zagospodarowania Przestrzennego (MPZP)
- **Status:** Brak obowiązującego planu miejscowego."""

    has_pog = False
    if isinstance(pog_data_dict, dict):
        has_pog = pog_data_dict.get("has_pog", False)
        if not has_pog:
            sym = str(pog_data_dict.get("strefa_symbol") or "")
            if sym and not any(bad in sym.lower() for bad in ["brak", "nie znaleziono", "błąd"]):
                has_pog = True
            elif pog_data_dict.get("max_wysokosc_m") or pog_data_dict.get("min_biologicznie_czynna_pct"):
                has_pog = True

    if has_pog:
        try:
            pog_section = analyze_pog_with_ai(pog_data_dict, lang=lang)
        except Exception:
            sym = pog_data_dict.get("strefa_symbol") or "Brak"
            nazwa = pog_data_dict.get("strefa_nazwa") or "Strefa POG"
            wys = pog_data_dict.get("max_wysokosc_m") or "Brak ustalenia"
            bio = pog_data_dict.get("min_biologicznie_czynna_pct") or "Brak ustalenia"
            intens = pog_data_dict.get("max_intensywnosc_zabudowy") or "Brak ustalenia"
            gmina = pog_data_dict.get("gmina") or "Brak danych"
            uchwala = pog_data_dict.get("akt_planowania_uchwala") or ""
            pog_section = f"""### Karta Planistyczna POG

| Parametr | Ustalenie POG |
| :--- | :--- |
| **Gmina / Akt Prawny** | {gmina} ({uchwala}) |
| **Strefa Planistyczna** | **{sym}** - {nazwa} |
| **Obszar Uzupełnienia Zabudowy (OUZ)** | {pog_data_dict.get("obszar_uzupelnienia_zabudowy_ouz") or "Nie dotyczy"} |
| **Obszar Zabudowy Śródmiejskiej (OZS)** | {pog_data_dict.get("obszar_zabudowy_srodmiejskiej_ozs") or "Nie dotyczy"} |
| **Min. Pow. Biologicznie Czynna** | **{bio}** |
| **Maks. Wysokość Zabudowy** | {wys} |
| **Maks. Intensywność Zabudowy** | {intens} |"""
    else:
        pog_section = """### Plan Ogólny Gminy (POG)
- **Status:** Brak opublikowanego POG w WFS (procedura w toku)."""

    return f"{mpzp_section}\n\n---\n\n{pog_section}"


def run_pog_analysis_flow(parcel_gdf, status_callback=None, lang="PL"):
    """
    Główny zintegrowany przepływ analizy planistycznej (MPZP + POG).
    """
    if status_callback:
        status_callback("info", "Weryfikacja MPZP (KIMPZP) oraz Planu Ogólnego (WMS Geoportal)...")

    try:
        mpzp_data = fetch_mpzp_kimpzp(parcel_gdf)
    except Exception as e:
        mpzp_data = {
            "has_mpzp": False,
            "nazwa_planu": None,
            "numer_uchwaly": None,
            "data_uchwaly": None,
            "link_uchwala_tekst": None,
            "link_rysunek": None,
            "gmina": None,
            "raw_attributes": {},
            "error": str(e)
        }

    try:
        pog_data = fetch_pog_data_for_parcel(parcel_gdf)
    except Exception as e:
        pog_data = {
            "has_pog": False,
            "strefa_symbol": "Brak danych",
            "strefa_nazwa": "Błąd pobierania",
            "error": str(e),
            "raw_attributes": {}
        }

    if status_callback:
        status_callback("info", "Przetwarzanie parametrów i generowanie Karty Planistycznej przez AI...")

    try:
        ai_card_markdown = analyze_planning_documents_with_ai(pog_data, mpzp_data, lang=lang)
    except Exception as e:
        ai_card_markdown = f"### Karta Planistyczna\n\n- **MPZP:** {'Obowiązuje: ' + str(mpzp_data.get('nazwa_planu')) if mpzp_data.get('has_mpzp') else 'Brak obowiązującego planu miejscowego'}\n- **POG:** {pog_data.get('strefa_symbol')}\n\n*Błąd generowania analizy AI: {e}*"

    if status_callback:
        status_callback("success", "Analiza planistyczna zakończona pomyślnie.")

    return {
        "status": "success",
        "raw_data": pog_data,
        "pog_data": pog_data,
        "mpzp_data": mpzp_data,
        "analysis": ai_card_markdown
    }

run_integrated_planning_flow = run_pog_analysis_flow
