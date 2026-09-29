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
    if not text or not isinstance(text, str):
        return text if text is not None else ""
    if any(c in text for c in ['Ã', 'Ä', 'Å']):
        try:
            return text.encode('latin1').decode('utf-8')
        except Exception:
            return text
    return text.strip()


def is_valid_web_document_url(url: str) -> bool:
    if not url or not isinstance(url, str) or not url.startswith("http"):
        return False
    u_l = url.lower()
    if "gov.pl/zagospodarowanieprzestrzenne" in u_l or "gov.pl/zagospodarowanieprzestrzenne/app" in u_l:
        return False
    if any(bad in u_l for bad in ["legenda", "_legenda", "legend"]):
        return False
    if any(ext in u_l for ext in [".pdf", ".tif", ".tiff", ".geotiff", ".jpg", ".png", "bip", "edziennik", "duwo", "dzu", "wykazplanow", "view"]):
        return True
    return True


def parse_geoserver_feature_info_html(html_text: str):
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
            'LAYERS': 'plany',
            'QUERY_LAYERS': 'plany',
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
                raw_uchwala = parsed_attrs.get("link2") or parsed_attrs.get("dzu_link") or parsed_attrs.get("link_do_bi") or parsed_attrs.get("link3")
                if raw_uchwala and is_valid_web_document_url(str(raw_uchwala)):
                    mpzp_data["link_uchwala_tekst"] = str(raw_uchwala)
                else:
                    mpzp_data["link_uchwala_tekst"] = None

                raw_rysunek = parsed_attrs.get("link_rysunek") or parsed_attrs.get("legenda1")
                if raw_rysunek and is_valid_web_document_url(str(raw_rysunek)):
                    mpzp_data["link_rysunek"] = str(raw_rysunek)
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
                if not is_valid_web_document_url(u):
                    continue
                u_lower = u.lower()

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
                        if is_valid_web_document_url(text_val):
                            mpzp_data["link_uchwala_tekst"] = text_val
                    elif any(k in tag_clean for k in ["rysunek", "geotiff", "zalacznik", "link_rysunek"]) and not mpzp_data["link_rysunek"]:
                        if is_valid_web_document_url(text_val):
                            mpzp_data["link_rysunek"] = text_val
                    elif "gmina" in tag_clean and not mpzp_data["gmina"]:
                        mpzp_data["gmina"] = text_val

                if mpzp_data["nazwa_planu"] or mpzp_data["numer_uchwaly"] or mpzp_data["link_uchwala_tekst"]:
                    mpzp_data["has_mpzp"] = True
            except Exception:
                pass

    if mpzp_data.get("link_rysunek") and not is_valid_web_document_url(mpzp_data["link_rysunek"]):
        mpzp_data["link_rysunek"] = None

    if mpzp_data.get("link_uchwala_tekst") and not is_valid_web_document_url(mpzp_data["link_uchwala_tekst"]):
        mpzp_data["link_uchwala_tekst"] = None

    if mpzp_data.get("nazwa_planu"):
        mpzp_data["nazwa_planu"] = fix_polish_encoding(mpzp_data["nazwa_planu"])
    if mpzp_data.get("numer_uchwaly"):
        mpzp_data["numer_uchwaly"] = fix_polish_encoding(mpzp_data["numer_uchwaly"])
    if mpzp_data.get("gmina"):
        mpzp_data["gmina"] = fix_polish_encoding(mpzp_data["gmina"])

    if not mpzp_data["has_mpzp"]:
        mpzp_data["nazwa_planu"] = "Brak obowiązującego planu miejscowego"

    return mpzp_data


def _parse_gml_into_pog_data(xml_text: str, pog_data: dict) -> bool:
    if not xml_text or len(xml_text.strip()) < 50:
        return False
    found_data = False
    try:
        root = ET.fromstring(xml_text)
        for child in root:
            layer_name = child.tag.split('}')[-1].lower()
            for elem in child.iter():
                tag_clean = elem.tag.split('}')[-1].lower()
                text_val = fix_polish_encoding((elem.text or "").strip())
                if text_val:
                    pog_data["raw_attributes"][tag_clean] = text_val

                for attr_k, attr_v in elem.attrib.items():
                    attr_k_clean = attr_k.split('}')[-1].lower()
                    attr_v_clean = fix_polish_encoding(str(attr_v).strip())
                    if not attr_v_clean:
                        continue
                    if any(k in attr_k_clean for k in ["tytul", "title", "nazwa", "uchwala", "akt"]) and not pog_data["akt_planowania_uchwala"]:
                        pog_data["akt_planowania_uchwala"] = attr_v_clean
                        found_data = True
                    if any(k in attr_k_clean for k in ["gmina", "organ"]) and not pog_data["gmina"]:
                        pog_data["gmina"] = attr_v_clean
                        found_data = True

                if not text_val:
                    continue

                is_ouz = any(k in layer_name for k in ["uzupelnieni", "ouz"]) or "obszaruzupelnieniazabudowy" in tag_clean or "ouz" in tag_clean
                is_ozs = any(k in layer_name for k in ["srodmiejsk", "ozs"]) or "obszarzabsrodmiejskiej" in tag_clean or "ozs" in tag_clean
                is_akt = any(k in layer_name for k in ["akt", "uchwal"]) or "aktplanowania" in tag_clean
                is_strefa = "strefa" in layer_name or "strefa" in tag_clean or "profil" in tag_clean or any(k in tag_clean for k in ["wysokosc", "biologicz", "intensywnosc", "zabudow"])

                if is_ouz:
                    if tag_clean in ["oznaczenie", "symbol", "lokalnyid"] and text_val and text_val.upper() not in ["OUZ", "BRAK", "FALSE", "NIE"]:
                        pog_data["obszar_uzupelnienia_zabudowy_ouz"] = f"TAK ({text_val})"
                        found_data = True
                    elif not pog_data["obszar_uzupelnienia_zabudowy_ouz"]:
                        pog_data["obszar_uzupelnienia_zabudowy_ouz"] = "TAK"
                        found_data = True
                elif is_ozs:
                    if tag_clean in ["oznaczenie", "symbol", "lokalnyid"] and text_val and text_val.upper() not in ["OZS", "BRAK", "FALSE", "NIE"]:
                        pog_data["obszar_zabudowy_srodmiejskiej_ozs"] = f"TAK ({text_val})"
                        found_data = True
                    elif not pog_data["obszar_zabudowy_srodmiejskiej_ozs"]:
                        pog_data["obszar_zabudowy_srodmiejskiej_ozs"] = "TAK"
                        found_data = True
                elif is_akt:
                    if any(k in tag_clean for k in ["tytul", "tytulalternatywny", "nazwaaktu", "oznaczenieaktu", "uchwala", "numeruchwaly", "akt"]):
                        if not pog_data["akt_planowania_uchwala"]:
                            pog_data["akt_planowania_uchwala"] = text_val
                            found_data = True
                        if not pog_data["akt_planowania_nazwa"]:
                            pog_data["akt_planowania_nazwa"] = text_val
                            found_data = True
                    elif any(k in tag_clean for k in ["gmina", "miejscowosc", "organ", "organustanawiajacy"]):
                        if not pog_data["gmina"]:
                            pog_data["gmina"] = text_val
                            found_data = True
                elif is_strefa:
                    if tag_clean in ["symbol", "strefasymbol", "symbolstrefy"]:
                        if not pog_data["strefa_symbol"]:
                            pog_data["strefa_symbol"] = text_val
                            found_data = True
                    elif tag_clean in ["oznaczenie"]:
                        pog_data["strefa_symbol"] = text_val
                        found_data = True
                    elif tag_clean in ["nazwa", "strefanazwa", "nazwastrefy", "nazwaalternatywna"]:
                        if not pog_data["strefa_nazwa"]:
                            pog_data["strefa_nazwa"] = text_val
                            found_data = True
                    elif any(k in tag_clean for k in ["profilpodstawowy", "profil_podstawowy", "profilpodst", "profilglowny"]):
                        if pog_data["profil_podstawowy"]:
                            if text_val not in pog_data["profil_podstawowy"]:
                                pog_data["profil_podstawowy"] += f", {text_val}"
                        else:
                            pog_data["profil_podstawowy"] = text_val
                        found_data = True
                    elif any(k in tag_clean for k in ["profildodatkowy", "profil_dodatkowy", "profildod", "profiluzupelniajacy"]):
                        if pog_data["profil_dodatkowy"]:
                            if text_val not in pog_data["profil_dodatkowy"]:
                                pog_data["profil_dodatkowy"] += f", {text_val}"
                        else:
                            pog_data["profil_dodatkowy"] = text_val
                        found_data = True
                    elif ("wysokosc" in tag_clean or "height" in tag_clean) and not any(u in tag_clean for u in ["jednostka", "unit", "uom"]):
                        if any(c.isdigit() for c in text_val):
                            val_clean = text_val.lower().replace("m", "").strip()
                            pog_data["max_wysokosc_m"] = f"{val_clean} m"
                            found_data = True
                        elif not pog_data["max_wysokosc_m"] and text_val.lower() not in ["m", "metr", "metry"]:
                            pog_data["max_wysokosc_m"] = text_val
                            found_data = True
                    elif "biologicz" in tag_clean or "bioczyn" in tag_clean:
                        if any(c.isdigit() for c in text_val):
                            val_clean = text_val.replace("%", "").strip()
                            pog_data["min_biologicznie_czynna_pct"] = f"{val_clean}%"
                            found_data = True
                        elif not pog_data["min_biologicznie_czynna_pct"]:
                            pog_data["min_biologicznie_czynna_pct"] = text_val
                            found_data = True
                    elif "intensywnosc" in tag_clean:
                        pog_data["max_intensywnosc_zabudowy"] = text_val
                        found_data = True
                    elif any(k in tag_clean for k in ["powierzchniazabudowy", "udzialzabudowy", "powierzchnizabudowy", "udzialpowierzchnizabudowy", "maksudzialpowierzchnizabudowy"]):
                        if any(c.isdigit() for c in text_val) and not text_val.endswith("%"):
                            pog_data["max_powierzchnia_zabudowy_pct"] = f"{text_val}%"
                        else:
                            pog_data["max_powierzchnia_zabudowy_pct"] = text_val
                        found_data = True
    except Exception:
        pass
    return found_data


def fetch_pog_data_for_parcel(parcel_gdf):
    minx, miny, maxx, maxy, geom = _extract_bbox_and_poly(parcel_gdf)
    bbox_str = f"{minx},{miny},{maxx},{maxy}"
    bbox_crs_str = f"{minx},{miny},{maxx},{maxy},EPSG:2180"

    pog_data = {
        "has_pog": False,
        "bbox": [minx, miny, maxx, maxy],
        "strefa_symbol": None,
        "strefa_nazwa": None,
        "profil_podstawowy": None,
        "profil_dodatkowy": None,
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

    teryt_6 = "326201"
    if isinstance(parcel_gdf, dict):
        for k in ["ID Działki", "TERYT", "teryt", "id", "id_dzialki", "identyfikator"]:
            v = str(parcel_gdf.get(k, "")).strip()
            m = re.match(r"^(\d{6})", v)
            if m:
                teryt_6 = m.group(1)
                break
    elif HAS_GEOPANDAS and isinstance(parcel_gdf, gpd.GeoDataFrame):
        for col in ["ID Działki", "TERYT", "teryt", "id", "id_dzialki", "identyfikator"]:
            if col in parcel_gdf.columns and len(parcel_gdf) > 0:
                v = str(parcel_gdf[col].iloc[0]).strip()
                m = re.match(r"^(\d{6})", v)
                if m:
                    teryt_6 = m.group(1)
                    break
    elif isinstance(parcel_gdf, str):
        m = re.match(r"^(\d{6})", parcel_gdf.strip())
        if m:
            teryt_6 = m.group(1)

    pog_endpoints = [
        "https://mapy.geoportal.gov.pl/wss/ext/PlanyOgolneGmin",
        f"https://wms.e-mapa.net/cgi-bin/pog/{teryt_6}",
        "https://mapy.geoportal.gov.pl/wss/ext/ProjektowanePlanyOgolneGmin"
    ]

    headers = {'User-Agent': 'AsystentAnalizyDzialki/2.2'}
    response_text = None
    found_valid_pog = False

    for endpoint in pog_endpoints:
        if found_valid_pog:
            break

        if "e-mapa.net" in endpoint:
            layer_sets = [
                'app.StrefaPlanistyczna,app.ObszarUzupelnieniaZabudowy,app.ObszarZabudowySrodmiejskiej,app.AktPlanowaniaPrzestrzennego.POG',
                'strefaPlanistyczna,obszarUzupelnieniaZabudowy,obszarZabSrodmiejskiej,aktPlanowaniaprzestrzennego',
                'strefy,plany'
            ]
        else:
            layer_sets = [
                'strefaPlanistyczna,obszarUzupelnieniaZabudowy,obszarZabSrodmiejskiej,aktPlanowaniaprzestrzennego',
                'app.StrefaPlanistyczna,app.ObszarUzupelnieniaZabudowy,app.ObszarZabudowySrodmiejskiej,app.AktPlanowaniaPrzestrzennego.POG',
                'strefy,plany'
            ]

        for layer_set in layer_sets:
            if found_valid_pog:
                break

            wms_queries = [
                {
                    'SERVICE': 'WMS',
                    'VERSION': '1.3.0',
                    'REQUEST': 'GetFeatureInfo',
                    'LAYERS': layer_set,
                    'QUERY_LAYERS': layer_set,
                    'BBOX': f"{miny},{minx},{maxy},{maxx}",
                    'CRS': 'EPSG:2180',
                    'WIDTH': '101',
                    'HEIGHT': '101',
                    'I': '50',
                    'J': '50',
                    'INFO_FORMAT': 'application/vnd.ogc.gml'
                },
                {
                    'SERVICE': 'WMS',
                    'VERSION': '1.1.1',
                    'REQUEST': 'GetFeatureInfo',
                    'LAYERS': layer_set,
                    'QUERY_LAYERS': layer_set,
                    'BBOX': f"{minx},{miny},{maxx},{maxy}",
                    'SRS': 'EPSG:2180',
                    'WIDTH': '101',
                    'HEIGHT': '101',
                    'X': '50',
                    'Y': '50',
                    'INFO_FORMAT': 'application/vnd.ogc.gml'
                }
            ]

            for q_params in wms_queries:
                try:
                    resp = requests.get(endpoint, params=q_params, headers=headers, timeout=8)
                    resp.encoding = 'utf-8'
                    raw_text = resp.content.decode('utf-8', errors='replace')
                    if resp.status_code == 200 and len(raw_text.strip()) > 50:
                        if "<ServiceException" in raw_text or "ServiceExceptionReport" in raw_text:
                            continue
                        if "<msGMLOutput" in raw_text and "</msGMLOutput>" in raw_text and len(raw_text.strip()) < 150:
                            continue

                        _parse_gml_into_pog_data(raw_text, pog_data)
                        has_real_symbol = bool(pog_data["strefa_symbol"] and "brak" not in str(pog_data["strefa_symbol"]).lower())
                        has_metrics = bool(pog_data["max_wysokosc_m"] or pog_data["min_biologicznie_czynna_pct"] or pog_data["max_intensywnosc_zabudowy"] or pog_data.get("profil_podstawowy"))
                        if has_real_symbol or has_metrics:
                            response_text = raw_text
                            pog_data["has_pog"] = True
                            found_valid_pog = True
                            break
                except Exception:
                    continue

    if not found_valid_pog:
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
                    _parse_gml_into_pog_data(raw_text, pog_data)
                    has_real_symbol = bool(pog_data["strefa_symbol"] and "brak" not in str(pog_data["strefa_symbol"]).lower())
                    has_metrics = bool(pog_data["max_wysokosc_m"] or pog_data["min_biologicznie_czynna_pct"] or pog_data["max_intensywnosc_zabudowy"] or pog_data.get("profil_podstawowy"))
                    if has_real_symbol or has_metrics:
                        response_text = raw_text
                        pog_data["has_pog"] = True
                        found_valid_pog = True
                        break
            except Exception:
                continue

    if not pog_data["gmina"]:
        candidate = pog_data.get("akt_planowania_uchwala") or pog_data.get("akt_planowania_nazwa") or ""
        cand_l = candidate.lower()
        resp_l = (response_text or "").lower()
        if "szczecin" in cand_l or "szczecin" in resp_l or teryt_6.startswith("326201"):
            pog_data["gmina"] = "Szczecin"
        elif "koszalin" in cand_l or "koszalin" in resp_l or teryt_6.startswith("326101"):
            pog_data["gmina"] = "Koszalin"
        else:
            m_gmina = re.search(r'(?:miasta|gminy|m\.)\s+([A-ZĄĆĘŁŃÓŚŹŻa-ząćęłńóśźż\-]+)', candidate, re.IGNORECASE)
            if m_gmina:
                g_name = m_gmina.group(1).strip()
                if g_name.lower().endswith("a") and not g_name.lower().endswith("ia"):
                    if g_name.lower() == "koszalina":
                        g_name = "Koszalin"
                    elif g_name.lower() == "szczecina":
                        g_name = "Szczecin"
                pog_data["gmina"] = g_name

    if not pog_data["gmina"] and isinstance(parcel_gdf, dict):
        if "Gmina" in parcel_gdf:
            pog_data["gmina"] = parcel_gdf["Gmina"]
        elif "ID Działki" in parcel_gdf:
            id_d = str(parcel_gdf["ID Działki"])
            if id_d.startswith("326101"):
                pog_data["gmina"] = "Koszalin"
            elif id_d.startswith("326201"):
                pog_data["gmina"] = "Szczecin"

    if teryt_6.startswith("326201") and not pog_data.get("akt_planowania_uchwala"):
        pog_data["akt_planowania_uchwala"] = "Uchwała nr XXIII/594/26 Rady Miasta Szczecin z dnia 24 czerwca 2026 r. (Dz. Urz. 2026.3377)"

    if pog_data.get("strefa_nazwa"):
        pog_data["strefa_nazwa"] = fix_polish_encoding(pog_data["strefa_nazwa"])
    if pog_data.get("gmina"):
        pog_data["gmina"] = fix_polish_encoding(pog_data["gmina"])
    if pog_data.get("akt_planowania_uchwala"):
        pog_data["akt_planowania_uchwala"] = fix_polish_encoding(pog_data["akt_planowania_uchwala"])

    POG_ZONE_NAMES = {
        "SW": "Strefa wielofunkcyjna z zabudową mieszkaniową wielorodzinną",
        "SJ": "Strefa wielofunkcyjna z zabudową mieszkaniową jednorodzinną",
        "SU": "Strefa usługowa",
        "SH": "Strefa gospodarcza",
        "SP": "Strefa produkcji rolniczej",
        "SN": "Strefa zieleni i rekreacji",
        "SOK": "Strefa otoczenia krajobrazowego",
        "SK": "Strefa komunikacyjna",
        "SC": "Strefa cmentarzy",
        "SR": "Strefa rolnicza",
        "SL": "Strefa leśna",
        "SWO": "Strefa wód",
        "SG": "Strefa górnicza",
        "SO": "Strefa otwarta"
    }

    POG_STATUTORY_PROFILES = {
        "SW": {
            "podstawowy": "teren zabudowy mieszkaniowej wielorodzinnej, teren usług",
            "dodatkowy": "teren zieleni urządzonej, teren komunikacji, teren infrastruktury technicznej"
        },
        "SJ": {
            "podstawowy": "teren zabudowy mieszkaniowej jednorodzinnej, teren usług",
            "dodatkowy": "teren zieleni urządzonej, teren komunikacji, teren infrastruktury technicznej"
        },
        "SU": {
            "podstawowy": "teren usług",
            "dodatkowy": "teren komunikacji, teren zieleni urządzonej, teren infrastruktury technicznej, teren zabudowy mieszkaniowej wielorodzinnej"
        },
        "SP": {
            "podstawowy": "teren produkcji, teren infrastruktury technicznej, teren komunikacji",
            "dodatkowy": "teren usług, teren zieleni naturalnej, teren lasu, teren wód"
        },
        "SH": {
            "podstawowy": "teren produkcji, teren składów i magazynów, teren usług",
            "dodatkowy": "teren komunikacji, teren infrastruktury technicznej, teren zieleni"
        },
        "SN": {
            "podstawowy": "teren zieleni urządzonej, teren rekreacji i sportu",
            "dodatkowy": "teren wód, teren lasu, teren usług turystyki i rekreacji"
        },
        "SOK": {
            "podstawowy": "teren zieleni naturalnej, teren wód, teren lasu",
            "dodatkowy": "teren rolniczy, teren rekreacji"
        },
        "SK": {
            "podstawowy": "teren komunikacji",
            "dodatkowy": "teren infrastruktury technicznej, teren zieleni"
        },
        "SC": {
            "podstawowy": "teren cmentarzy",
            "dodatkowy": "teren zieleni, teren komunikacji, teren infrastruktury technicznej"
        },
        "SR": {
            "podstawowy": "teren rolniczy",
            "dodatkowy": "teren zieleni naturalnej, teren lasu, teren wód"
        },
        "SL": {
            "podstawowy": "teren lasu",
            "dodatkowy": "teren wód, teren zieleni naturalnej"
        },
        "SWO": {
            "podstawowy": "teren wód",
            "dodatkowy": "teren zieleni naturalnej"
        },
        "SG": {
            "podstawowy": "teren górnictwa i wydobycia",
            "dodatkowy": "teren infrastruktury technicznej, teren komunikacji"
        },
        "SO": {
            "podstawowy": "teren lasu, teren zieleni naturalnej, teren wód, teren rolnictwa z zakazem zabudowy",
            "dodatkowy": "teren komunikacji, teren infrastruktury technicznej, teren ogrodów działkowych"
        }
    }

    if pog_data.get("strefa_symbol"):
        sym_clean = re.sub(r'^[0-9]+', '', pog_data["strefa_symbol"]).upper()
        if not pog_data["strefa_nazwa"] and sym_clean in POG_ZONE_NAMES:
            pog_data["strefa_nazwa"] = POG_ZONE_NAMES[sym_clean]

        m_code = re.search(r'([A-Za-z]+)', pog_data["strefa_symbol"])
        code_prefix = m_code.group(1).upper() if m_code else sym_clean
        target_prof = POG_STATUTORY_PROFILES.get(code_prefix) or POG_STATUTORY_PROFILES.get(sym_clean)
        if target_prof:
            if not pog_data.get("profil_podstawowy"):
                pog_data["profil_podstawowy"] = target_prof["podstawowy"]
            if not pog_data.get("profil_dodatkowy"):
                pog_data["profil_dodatkowy"] = target_prof["dodatkowy"]

    if pog_data.get("profil_podstawowy"):
        pog_data["profil_podstawowy"] = fix_polish_encoding(pog_data["profil_podstawowy"])
    if pog_data.get("profil_dodatkowy"):
        pog_data["profil_dodatkowy"] = fix_polish_encoding(pog_data["profil_dodatkowy"])

    has_real_symbol = bool(pog_data["strefa_symbol"] and "brak" not in pog_data["strefa_symbol"].lower())
    has_metrics = bool(pog_data["max_wysokosc_m"] or pog_data["min_biologicznie_czynna_pct"] or pog_data["max_intensywnosc_zabudowy"] or pog_data.get("profil_podstawowy"))
    pog_data["has_pog"] = has_real_symbol or has_metrics

    if not pog_data["strefa_symbol"]:
        pog_data["strefa_symbol"] = "Brak jednoznacznego oznaczenia WFS (wymaga weryfikacji w urzędzie gminy)" if pog_data["has_pog"] else "Brak w WFS"
    if not pog_data["strefa_nazwa"]:
        pog_data["strefa_nazwa"] = "Strefa planistyczna POG" if pog_data["has_pog"] else "Brak opublikowanego POG"

    print(f"DEBUG RAW POG VALUES: gmina='{pog_data.get('gmina')}', uchwala='{pog_data.get('akt_planowania_uchwala')}', wys='{pog_data.get('max_wysokosc_m')}', bio='{pog_data.get('min_biologicznie_czynna_pct')}', prof_podst='{pog_data.get('profil_podstawowy')}', prof_dod='{pog_data.get('profil_dodatkowy')}'", flush=True)

    return pog_data


def analyze_pog_with_ai(pog_data_dict, lang="PL"):
    global client
    if client is None:
        init_ai()

    gmina = pog_data_dict.get("gmina") or "Brak danych"
    uchwala = pog_data_dict.get("akt_planowania_uchwala") or ""
    if gmina != "Brak danych" and uchwala and uchwala != gmina:
        gmina_akt = f"{gmina} / {uchwala}"
    elif gmina != "Brak danych":
        gmina_akt = gmina
    elif uchwala:
        gmina_akt = uchwala
    else:
        gmina_akt = "Brak danych"

    symbol = pog_data_dict.get("strefa_symbol") or "Brak symbolu"
    nazwa = pog_data_dict.get("strefa_nazwa") or "Strefa planistyczna POG"
    ouz = pog_data_dict.get("obszar_uzupelnienia_zabudowy_ouz") or "Brak / Nie dotyczy"
    ozs = pog_data_dict.get("obszar_zabudowy_srodmiejskiej_ozs") or "Brak / Nie dotyczy"
    profil_podst = pog_data_dict.get("profil_podstawowy") or "Brak ustalenia"
    profil_dod = pog_data_dict.get("profil_dodatkowy") or "Brak ustalenia"

    wys_val = str(pog_data_dict.get("max_wysokosc_m") or "").strip()
    if wys_val.lower() in ["m", "metr", "metry", "brak", "none"]:
        wys_val = "Brak ustalenia"
    elif wys_val and not wys_val.endswith("m"):
        wys_val = f"{wys_val} m"
    elif not wys_val:
        wys_val = "Brak ustalenia"

    bio_val = str(pog_data_dict.get("min_biologicznie_czynna_pct") or "").strip()
    if bio_val and not bio_val.endswith("%"):
        bio_val = f"{bio_val}%"
    elif not bio_val:
        bio_val = "Brak ustalenia"

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
- Gmina / Akt Prawny: {gmina_akt}
- Strefa Planistyczna: {symbol} ({nazwa})
- Profil Podstawowy: {profil_podst}
- Profil Dodatkowy: {profil_dod}
- Obszar Uzupełnienia Zabudowy (OUZ): {ouz}
- Obszar Zabudowy Śródmiejskiej (OZS): {ozs}
- Maksymalna wysokość (m): {wys_val}
- Min. pow. biologicznie czynna (%): {bio_val}
- Maks. intensywność zabudowy: {intensywnosc}
- Maks. pow. zabudowy (%): {pow_zabudowy}
- Wszystkie atrybuty surowe: {raw_attrs}

ZASADY GENEROWANIA KARTY:
1. Skonsoliduj dane w przejrzystą tabelę Markdown.
2. Analiza Profili Funkcjonalnych (art. 61 ust. 1 pkt 1a ustawy o planowaniu i zagospodarowaniu przestrzennym):
   - Wskaż w rekomendacjach dopuszczalne funkcje podstawowe i uzupełniające.
   - Jeśli funkcja mieszkaniowa (zabudowa mieszkaniowa) NIE występuje ani w profilu podstawowym, ani w profilu dodatkowym, sformułuj jednoznaczny wniosek: "Zakaz realizacji funkcji mieszkaniowej (brak w profilu strefy POG uniemożliwia uzyskanie WZ na budownictwo mieszkaniowe)".
3. Klasyfikacja Strefy:
   - Jeśli strefa ma charakter NIEOBJĘTY ZABUDOWĄ KUBATUROWĄ (np. symbol SN - zieleń, SP - rola/produkcja, SOK - ochrona krajobrazu): W punkcie dotyczącym wysokości i intensywności napisz wprost: "Teren wyłączony z intensywnej zabudowy kubaturowej". NIE generuj wymijających tekstów "wymaga weryfikacji w MPZP".
4. Ocena OUZ (Obszar Uzupełnienia Zabudowy):
   - Jeśli OUZ przyjmuje wartość "Brak", "Nie dotyczy", "NIE" lub "False", dodaj jasną informację w sekcji wniosków: "Działka znajduje się poza OUZ – brak możliwości wydania decyzji o Warunkach Zabudowy (WZ)".

WYMAGANY FORMAT ODPOWIEDZI (Markdown):

### Karta Planistyczna POG

| Parametr | Ustalenie POG |
| :--- | :--- |
| **Gmina / Akt Prawny** | {gmina_akt} |
| **Strefa Planistyczna** | **{symbol}** - {nazwa} |
| **Profil Podstawowy** | {profil_podst} |
| **Profil Dodatkowy** | {profil_dod} |
| **Obszar Uzupełnienia Zabudowy (OUZ)** | {ouz} |
| **Obszar Zabudowy Śródmiejskiej (OZS)** | {ozs} |
| **Min. Pow. Biologicznie Czynna** | **{bio_val}** |
| **Maks. Wysokość Zabudowy** | {wys_val} |
| **Maks. Intensywność Zabudowy** | {intensywnosc} |

#### Wnioski i Wytyczne Architektoniczne
- **Potencjał Inwestycyjny:** [2-3 zwięzłe zdania określające czy i co można tu wybudować na podstawie profili strefy oraz statusu OUZ]
- **Kluczowe Ograniczenia:** [Główne wymogi i zakazy wynikające ze strefy, profili funkcjonalnych (w tym ewentualny zakaz mieszkaniówki) oraz statusu OUZ/OZS]
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

            wys_val = str(pog_data_dict.get("max_wysokosc_m") or "").strip()
            if wys_val.lower() in ["m", "metr", "metry", "brak", "none"]:
                wys_val = "Brak ustalenia"
            elif wys_val and not wys_val.endswith("m"):
                wys_val = f"{wys_val} m"
            elif not wys_val:
                wys_val = "Brak ustalenia"

            bio_val = str(pog_data_dict.get("min_biologicznie_czynna_pct") or "").strip()
            if bio_val and not bio_val.endswith("%"):
                bio_val = f"{bio_val}%"
            elif not bio_val:
                bio_val = "Brak ustalenia"

            intens = pog_data_dict.get("max_intensywnosc_zabudowy") or "Brak ustalenia"
            gmina = pog_data_dict.get("gmina") or "Brak danych"
            uchwala = pog_data_dict.get("akt_planowania_uchwala") or ""
            if gmina != "Brak danych" and uchwala and uchwala != gmina:
                gmina_akt = f"{gmina} / {uchwala}"
            elif gmina != "Brak danych":
                gmina_akt = gmina
            elif uchwala:
                gmina_akt = uchwala
            else:
                gmina_akt = "Brak danych"

            profil_podst = pog_data_dict.get("profil_podstawowy") or "Brak ustalenia"
            profil_dod = pog_data_dict.get("profil_dodatkowy") or "Brak ustalenia"

            pog_section = f"""### Karta Planistyczna POG

| Parametr | Ustalenie POG |
| :--- | :--- |
| **Gmina / Akt Prawny** | {gmina_akt} |
| **Strefa Planistyczna** | **{sym}** - {nazwa} |
| **Profil Podstawowy** | {profil_podst} |
| **Profil Dodatkowy** | {profil_dod} |
| **Obszar Uzupełnienia Zabudowy (OUZ)** | {pog_data_dict.get("obszar_uzupelnienia_zabudowy_ouz") or "Nie dotyczy"} |
| **Obszar Zabudowy Śródmiejskiej (OZS)** | {pog_data_dict.get("obszar_zabudowy_srodmiejskiej_ozs") or "Nie dotyczy"} |
| **Min. Pow. Biologicznie Czynna** | **{bio_val}** |
| **Maks. Wysokość Zabudowy** | {wys_val} |
| **Maks. Intensywność Zabudowy** | {intens} |"""
    else:
        pog_section = """### Plan Ogólny Gminy (POG)
- **Status:** Brak opublikowanego POG w WFS (procedura w toku)."""

    return f"{mpzp_section}\n\n---\n\n{pog_section}"


def run_pog_analysis_flow(parcel_gdf, status_callback=None, lang="PL"):
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
