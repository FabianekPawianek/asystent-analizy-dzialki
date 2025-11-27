import requests
from pyproj import Transformer
from urllib.parse import quote_plus

def geocode_address_to_coords(address):
    nominatim_url = f"https://nominatim.openstreetmap.org/search?q={quote_plus(address)}&format=json&limit=1&countrycodes=pl"
    headers = {'User-Agent': 'AsystentAnalizyDzialki/2.2'}
    try:
        response = requests.get(nominatim_url, headers=headers, timeout=15)
        response.raise_for_status()
        geodata = response.json()
        if not geodata:
            return None, "Nie znaleziono współrzędnych dla podanego adresu."
        return (float(geodata[0]['lat']), float(geodata[0]['lon'])), None
    except Exception as e:
        return None, f"Błąd geokodowania: {str(e)}"

def get_parcel_by_id(parcel_id):
    uldk_url = f"https://uldk.gugik.gov.pl/?request=GetParcelById&id={parcel_id}&result=geom_wkt"
    try:
        response = requests.get(uldk_url, timeout=15)
        response.raise_for_status()
        responseText = response.text.strip()
        if responseText.startswith('-1'):
            return None, f"Błąd wewnętrzny: Nie znaleziono danych dla działki o ID: {parcel_id}."
        
        wkt_geom_raw = responseText.split('\n')[1].strip()
        if 'SRID=' in wkt_geom_raw:
            wkt_geom_raw = wkt_geom_raw.split(';', 1)[1]
        
        coords_str = wkt_geom_raw.replace('POLYGON((', '').replace('))', '')
        coords_pairs = [pair.split() for pair in coords_str.split(',')]
        coords_2180 = [[float(x), float(y)] for x, y in coords_pairs]
        
        return {"ID Działki": parcel_id, "Współrzędne EPSG:2180": coords_2180}, None
    except Exception as e:
        return None, f"Błąd pobierania danych działki: {str(e)}"

def get_parcel_from_coords(lat, lon):
    try:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
        x, y = transformer.transform(lon, lat)
        identify_url = f"https://uldk.gugik.gov.pl/?request=GetParcelByXY&xy={x},{y}&result=id"
        
        response_id = requests.get(identify_url, timeout=15)
        response_id.raise_for_status()
        id_text = response_id.text.strip()
        
        if id_text.startswith('-1') or len(id_text.split('\n')) < 2:
            return None, "W tym miejscu nie zidentyfikowano działki. Spróbuj kliknąć precyzyjniej."
        
        parcel_id = id_text.split('\n')[1].strip()
        return get_parcel_by_id(parcel_id)
    except Exception as e:
        return None, f"Błąd identyfikacji działki: {e}"

def transform_coordinates_to_wgs84(coords_2180):
    transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    return [[transformer.transform(x, y)[1], transformer.transform(x, y)[0]] for x, y in coords_2180]

def transform_single_coord(x, y, source_crs, target_crs):
    transformer = Transformer.from_crs(f"EPSG:{source_crs}", f"EPSG:{target_crs}", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat
