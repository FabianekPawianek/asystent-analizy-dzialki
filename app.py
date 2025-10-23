import streamlit as st
import time
import json
import requests
import fitz  # PyMuPDF
import folium
import osmnx as ox
import pandas as pd
import pydeck as pdk
import numpy as np
import pvlib
import trimesh
import open3d as o3d
from datetime import datetime
from shapely.geometry import (Polygon)
from shapely.ops import transform
from streamlit_folium import st_folium
from pyproj import Transformer
from urllib.parse import quote_plus
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI

# --- KONFIGURACJA (bez zmian) ---
# ... (cały blok konfiguracyjny jest identyczny)
PROJECT_ID = "***REMOVED***"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-pro"
EMBEDDING_MODEL_NAME = "text-embedding-004"
vertexai.init(project=PROJECT_ID, location=LOCATION)
generative_model = GenerativeModel(MODEL_NAME)
embeddings_model = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)
llm = VertexAI(model_name=MODEL_NAME)


# --- FUNKCJE GEOPRZESTRZENNE (bez zmian) ---
# ... (wszystkie funkcje geo są identyczne)
def geocode_address_to_coords(address):
    nominatim_url = f"https://nominatim.openstreetmap.org/search?q={quote_plus(address)}&format=json&limit=1&countrycodes=pl"
    headers = {'User-Agent': 'AsystentAnalizyDzialki/2.2'}
    response = requests.get(nominatim_url, headers=headers, timeout=15)
    response.raise_for_status()
    geodata = response.json()
    if not geodata: return None, "Nie znaleziono współrzędnych dla podanego adresu."
    return (float(geodata[0]['lat']), float(geodata[0]['lon'])), None


def get_parcel_by_id(parcel_id):
    uldk_url = f"https://uldk.gugik.gov.pl/?request=GetParcelById&id={parcel_id}&result=geom_wkt"
    response = requests.get(uldk_url, timeout=15)
    response.raise_for_status()
    responseText = response.text.strip()
    if responseText.startswith(
        '-1'): return None, f"Błąd wewnętrzny: Nie znaleziono danych dla działki o ID: {parcel_id}."
    wkt_geom_raw = responseText.split('\n')[1].strip()
    if 'SRID=' in wkt_geom_raw: wkt_geom_raw = wkt_geom_raw.split(';', 1)[1]
    coords_str = wkt_geom_raw.replace('POLYGON((', '').replace('))', '')
    coords_pairs = [pair.split() for pair in coords_str.split(',')]
    coords_2180 = [[float(x), float(y)] for x, y in coords_pairs]
    return {"ID Działki": parcel_id, "Współrzędne EPSG:2180": coords_2180}, None


def get_parcel_from_coords(lat, lon):
    try:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
        x, y = transformer.transform(lon, lat)
        identify_url = f"https://uldk.gugik.gov.pl/?request=GetParcelByXY&xy={x},{y}&result=id"
        response_id = requests.get(identify_url, timeout=15)
        response_id.raise_for_status()
        id_text = response_id.text.strip()
        if id_text.startswith('-1') or len(id_text.split(
            '\n')) < 2: return None, "W tym miejscu nie zidentyfikowano działki. Spróbuj kliknąć precyzyjniej."
        parcel_id = id_text.split('\n')[1].strip()
        return get_parcel_by_id(parcel_id)
    except Exception as e:
        return None, f"Błąd identyfikacji działki: {e}"


def transform_coordinates_to_wgs84(coords_2180):
    transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    return [[transformer.transform(x, y)[1], transformer.transform(x, y)[0]] for x, y in coords_2180]


# Zmieniamy dekorator na st.cache_data, aby był zgodny z nowymi typami danych
def generate_3d_context_view(parcel_coords_wgs_84, map_center_wgs_84, map_style: str):
    """
    Generuje interaktywny widok 3D, akceptując styl mapy jako argument.
    """
    try:
        # --- Cała logika pobierania i przetwarzania danych pozostaje BEZ ZMIAN ---
        tags = {"building": True}
        gdf_buildings = ox.features_from_point(
            (map_center_wgs_84[0], map_center_wgs_84[1]), tags, dist=300
        )
        # ... (reszta kodu aż do momentu tworzenia obiektu Deck) ...
        buildings_data_for_pydeck = []
        if not gdf_buildings.empty:
            def estimate_height(row):
                try:
                    if 'height' in row and row['height'] and str(row['height']).strip(): return float(str(row['height']).split(';')[0])
                    if 'building:levels' in row and row['building:levels'] and str(row['building:levels']).strip(): return float(str(row['building:levels']).split(';')[0]) * 3.5 + 2
                except (ValueError, TypeError): pass
                return 10.0
            gdf_buildings['height'] = pd.to_numeric(gdf_buildings.apply(estimate_height, axis=1), errors='coerce').fillna(10.0)
            for _, building in gdf_buildings.iterrows():
                if building.geometry and building.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                    polygons = [building.geometry] if building.geometry.geom_type == 'Polygon' else building.geometry.geoms
                    for poly in polygons: buildings_data_for_pydeck.append({"polygon": [list(poly.exterior.coords)], "height": float(building.height)})
        else:
             st.info("Nie znaleziono danych o budynkach w okolicy w bazie OpenStreetMap.")
        layer_buildings = pdk.Layer("PolygonLayer", data=buildings_data_for_pydeck, get_polygon="polygon", extruded=True, wireframe=True, get_elevation="height", get_fill_color=[180, 180, 180, 200], get_line_color=[100, 100, 100])
        parcel_polygon_coords = [list(Polygon(parcel_coords_wgs_84).exterior.coords)]
        parcel_data_for_pydeck = [{"polygon": parcel_polygon_coords, "height": 1.0}]
        layer_parcel = pdk.Layer("PolygonLayer", data=parcel_data_for_pydeck, get_polygon="polygon", extruded=False, get_elevation="height", filled=False, get_line_color=[255, 0, 0, 255], get_line_width=1, line_width_min_pixels=2)
        view_state = pdk.ViewState(latitude=map_center_wgs_84[0], longitude=map_center_wgs_84[1], zoom=17.5, pitch=50, bearing=0)
        layers_to_render = [layer_parcel]
        if buildings_data_for_pydeck: layers_to_render.append(layer_buildings)

        # --- JEDYNA KLUCZOWA ZMIANA W TEJ FUNKCJI ---
        return pdk.Deck(
            layers=layers_to_render,
            initial_view_state=view_state,
            map_style=map_style # Używamy przekazanego argumentu
        )

    except Exception as e:
        st.error(f"Wystąpił krytyczny błąd podczas generowania modelu 3D: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None


# --- FUNKCJE ANALIZY NASŁONECZNIENIA ---

def create_trimesh_scene(buildings_data_metric: list) -> trimesh.Scene:
    """
    OSTATECZNA WERSJA: Buduje scenę 3D za pomocą fundamentalnej, ręcznej
    konstrukcji brył w bibliotece trimesh, omijając wszystkie problematyczne
    i niepewne funkcje wysokopoziomowe. Ta metoda gwarantuje stabilność.
    """
    scene = trimesh.Scene()
    polygons_added = 0
    failed_reasons = {"invalid_poly": 0, "triangulation_failed": 0, "validation_failed": 0, "exception": 0}


    for building_dict in buildings_data_metric:
        try:
            # CRITICAL FIX: Handle coordinate lists properly
            coords = building_dict['polygon']
            # Remove duplicate closing point if it exists
            if len(coords) > 1:
                # Handle both list and numpy array comparison
                first = np.array(coords[0]) if not isinstance(coords[0], np.ndarray) else coords[0]
                last = np.array(coords[-1]) if not isinstance(coords[-1], np.ndarray) else coords[-1]
                if np.allclose(first, last, rtol=1e-9):
                    coords = coords[:-1]

            # Ensure we have at least 3 points for a valid polygon
            if len(coords) < 3:
                failed_reasons["invalid_poly"] += 1
                continue

            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or not poly.is_valid or poly.area < 1.0:
                failed_reasons["invalid_poly"] += 1
                continue

            height = building_dict['height']

            # --- UPROSZCZONA STRATEGIA - używamy trimesh.creation.extrude_polygon ---
            # Jest to o wiele prostsze i bardziej niezawodne niż ręczna konstrukcja

            try:
                # Tworzymy mesh przez ekstruzję poligonu
                mesh = trimesh.creation.extrude_polygon(poly, height=height)

                if mesh is None or len(mesh.faces) == 0:
                    failed_reasons["triangulation_failed"] += 1
                    continue

            except Exception as extrude_error:
                failed_reasons["triangulation_failed"] += 1
                continue

            # FIX 4: Better mesh validation - try to repair if not watertight
            if not mesh.is_watertight:
                try:
                    trimesh.repair.fix_normals(mesh)
                    trimesh.repair.fill_holes(mesh)
                except Exception:
                    pass  # If repair fails, continue anyway

            # CRITICAL FIX: Remove strict validation - add ALL valid meshes
            # The mesh has vertices and faces, so it can cast shadows
            if len(mesh.faces) > 0 and len(mesh.vertices) > 0:
                scene.add_geometry(mesh)
                polygons_added += 1
            else:
                failed_reasons["validation_failed"] += 1

        except Exception as e:
            failed_reasons["exception"] += 1
            continue

    # Pokaż tylko jeśli są problemy
    if polygons_added == 0:
        st.warning(f"⚠️ Nie udało się dodać budynków do sceny 3D.")

    return scene

def value_to_rgb(value, min_val, max_val, colormap='plasma'):
    """Mapuje wartość liczbową na kolor RGB używając skali z Matplotlib.
    plasma: niebieski -> fioletowy -> czerwony -> pomarańczowy -> żółty"""
    # Zabezpieczenie przed dzieleniem przez zero
    if max_val == min_val:
        norm_value = 0.5
    else:
        norm_value = (value - min_val) / (max_val - min_val)

    from matplotlib import cm
    # Pobranie mapy kolorów
    rgba = cm.get_cmap(colormap)(norm_value)
    # Konwersja na format [R, G, B, A] dla Pydeck
    return [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 200]


def create_discrete_legend_html(min_val, max_val, colormap='plasma', steps=7):
    """Tworzy dyskretną, horyzontalną legendę w HTML, która dostosowuje się do zakresu danych.
    plasma: niebieski -> fioletowy -> czerwony -> pomarańczowy -> żółty
    Używa przyciętego zakresu colormap (0.0-0.92) aby uniknąć żółto-zielonego na końcu"""
    from matplotlib import cm

    # ZMIANA: Obsługa przypadku z jedną wartością
    if min_val == max_val:
        rgba = cm.get_cmap(colormap)(0.5)  # Środkowy kolor z palety
        rgb = f"rgb({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)})"
        label = f"{min_val:.1f}h"
        header = "<div style='font-family: sans-serif; font-size: 13px; background: rgba(40,40,40,0.85); color: white; padding: 10px; border-radius: 5px; border: 1px solid #555;'>"
        title = "<div style='margin-bottom: 8px;'><b>Śr. dzienne nasłonecznienie</b></div>"
        content = f"<div style='text-align: center; margin: 0 4px;'><div style='width: 35px; height: 20px; background: {rgb};'></div><div>{label}</div></div>"
        return f"{header}{title}{content}</div>"

    # ZMIANA: Przycięty zakres colormap (0.0 do 0.92) dla ładniejszego żółtego
    values = np.linspace(min_val, max_val, steps)
    colors = cm.get_cmap(colormap)(np.linspace(0, 0.92, steps))
    header = "<div style='font-family: sans-serif; font-size: 13px; background: rgba(40,40,40,0.85); color: white; padding: 10px; border-radius: 5px; border: 1px solid #555;'>"
    title = "<div style='margin-bottom: 8px;'><b>Śr. dzienne nasłonecznienie</b></div>"
    content = "<div style='display: flex; flex-direction: row; align-items: center; justify-content: space-between;'>"

    for i in range(steps):
        rgb = f"rgb({int(colors[i][0] * 255)}, {int(colors[i][1] * 255)}, {int(colors[i][2] * 255)})"
        label = f"{values[i]:.1f}h"
        content += f"<div style='text-align: center; margin: 0 4px;'><div style='width: 35px; height: 20px; background: {rgb};'></div><div>{label}</div></div>"

    return f"{header}{title}{content}</div></div>"


def generate_sun_path_data(lat: float, lon: float, analysis_date: datetime.date, hour_range: tuple,
                           map_center_metric: tuple):
    """Generuje dane ścieżki słońca przy użyciu precyzyjnej biblioteki pvlib."""
    path_radius = 300
    start_hour, end_hour = hour_range
    center_x, center_y = map_center_metric[0], map_center_metric[1]
    tz = 'Europe/Warsaw'

    # Tworzymy zakres czasu dla analizy co 15 minut
    times = pd.date_range(
        start=f"{analysis_date} {start_hour:02d}:00",
        end=f"{analysis_date} {end_hour:02d}:00",
        freq="15min",
        tz=tz
    )

    # Używamy pvlib do precyzyjnego obliczenia pozycji słońca
    location = pvlib.location.Location(lat, lon, tz=tz)
    solar_position = location.get_solarposition(times)
    solar_position = solar_position[solar_position['apparent_elevation'] > 0] # Bierzemy tylko pozycje nad horyzontem

    sun_path_line = []
    for index, sun in solar_position.iterrows():
        alt_rad = np.deg2rad(sun['apparent_elevation'])
        az_rad = np.deg2rad(sun['azimuth'])

        x_offset = path_radius * np.cos(alt_rad) * np.sin(az_rad)
        y_offset = path_radius * np.cos(alt_rad) * np.cos(az_rad)
        z = path_radius * np.sin(alt_rad)
        sun_path_line.append([center_x + x_offset, center_y + y_offset, z])

    # Generowanie znaczników godzinowych
    hourly_times = pd.date_range(
        start=f"{analysis_date} {start_hour:02d}:00",
        end=f"{analysis_date} {end_hour-1:02d}:00", # do przedostatniej pełnej godziny
        freq="H",
        tz=tz
    )
    hourly_position = location.get_solarposition(hourly_times)
    hourly_position = hourly_position[hourly_position['apparent_elevation'] > 5]

    sun_hour_markers = []
    for index, sun in hourly_position.iterrows():
        alt_rad = np.deg2rad(sun['apparent_elevation'])
        az_rad = np.deg2rad(sun['azimuth'])
        x_offset = path_radius * np.cos(alt_rad) * np.sin(az_rad)
        y_offset = path_radius * np.cos(alt_rad) * np.cos(az_rad)
        z = path_radius * np.sin(alt_rad)
        sun_hour_markers.append({
            "position": [center_x + x_offset, center_y + y_offset, z],
            "hour": f"{index.hour}:00"
        })

    return sun_path_line, sun_hour_markers


def generate_complete_sun_path_diagram(lat: float, lon: float, year: int, map_center_metric: tuple):
    """
    Generuje kompletny diagram ścieżki słońca bez obrotu - proste współrzędne.
    """
    path_radius = 300
    center_x, center_y = map_center_metric[0], map_center_metric[1]
    tz = 'Europe/Warsaw'
    location = pvlib.location.Location(lat, lon, tz=tz)

    key_dates = {
        'winter_solstice': datetime(year, 12, 21).date(),
        'spring_equinox': datetime(year, 3, 20).date(),
        'summer_solstice': datetime(year, 6, 21).date()
    }

    sun_paths = []
    for date_name, date in key_dates.items():
        times = pd.date_range(start=f"{date} 00:00", end=f"{date} 23:45", freq="15min", tz=tz)
        solar_position = location.get_solarposition(times)
        solar_position = solar_position[solar_position['apparent_elevation'] > 0]

        path_coords = []
        for _, sun in solar_position.iterrows():
            alt_rad = np.deg2rad(sun['apparent_elevation'])
            az_rad = np.deg2rad(sun['azimuth'])
            x_offset = path_radius * np.cos(alt_rad) * np.sin(az_rad)
            y_offset = path_radius * np.cos(alt_rad) * np.cos(az_rad)
            z = path_radius * np.sin(alt_rad)
            path_coords.append([center_x + x_offset, center_y + y_offset, z])

        if path_coords:
            sun_paths.append({'path': path_coords, 'name': date_name})

    # Analemmy - 24 godziny
    analemmas = {}
    for hour in range(0, 24):
        year_data = []
        for day_of_year in range(1, 366):
            try:
                date = datetime(year, 1, 1).date() + pd.Timedelta(days=day_of_year-1)
                time = pd.Timestamp(f"{date} {hour:02d}:00", tz=tz)
                solar_pos = location.get_solarposition(time)
                elevation = solar_pos['apparent_elevation'].values[0]
                azimuth = solar_pos['azimuth'].values[0]
                alt_rad = np.deg2rad(elevation)
                az_rad = np.deg2rad(azimuth)
                x_offset = path_radius * np.cos(alt_rad) * np.sin(az_rad)
                y_offset = path_radius * np.cos(alt_rad) * np.cos(az_rad)
                z = path_radius * np.sin(alt_rad)
                year_data.append({
                    'day': day_of_year,
                    'coords': [center_x + x_offset, center_y + y_offset, z],
                    'elevation': elevation
                })
            except:
                continue
        if len(year_data) > 10:
            analemmas[hour] = sorted(year_data, key=lambda x: x['day'])

    # Kompas
    azimuth_markers = []
    azimuth_lines = []
    cardinal_directions = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
    main_cardinals = [0, 90, 180, 270]

    for azimuth_deg in range(0, 360, 30):
        az_rad = np.deg2rad(azimuth_deg)
        extension = 40 if azimuth_deg in main_cardinals else 25
        x_label = (path_radius + extension + 15) * np.sin(az_rad)
        y_label = (path_radius + extension + 15) * np.cos(az_rad)

        if azimuth_deg in cardinal_directions:
            label = f"{cardinal_directions[azimuth_deg]} ({azimuth_deg}°)"
        else:
            label = f"{azimuth_deg}°"

        azimuth_markers.append({
            'position': [center_x + x_label, center_y + y_label, 0.5],
            'label': label, 'azimuth': azimuth_deg
        })

        x_inner = path_radius * np.sin(az_rad)
        y_inner = path_radius * np.cos(az_rad)
        x_outer = (path_radius + extension) * np.sin(az_rad)
        y_outer = (path_radius + extension) * np.cos(az_rad)
        azimuth_lines.append({
            'path': [[center_x + x_inner, center_y + y_inner, 0.5], [center_x + x_outer, center_y + y_outer, 1.0]],
            'azimuth': azimuth_deg, 'is_main': azimuth_deg in main_cardinals
        })

    return sun_paths, analemmas, azimuth_markers, azimuth_lines


def create_analysis_grid(parcel_polygon: Polygon, density: float = 1.0) -> np.ndarray:
    """Tworzy gęstą siatkę punktów 3D na powierzchni działki."""
    bounds = parcel_polygon.bounds
    min_x, min_y, max_x, max_y = bounds
    x_coords = np.arange(min_x, max_x, density); y_coords = np.arange(min_y, max_y, density)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    points = np.vstack([mesh_x.ravel(), mesh_y.ravel()]).T
    from shapely.geometry import Point; from shapely.prepared import prep
    prepared_polygon = prep(parcel_polygon)
    contained_mask = [prepared_polygon.contains(Point(p)) for p in points]
    final_points = points[contained_mask]
    # Wysokość 0.1m (tuż nad gruntem) - zgodnie z praktyką Ladybug/Grasshopper
    return np.hstack([final_points, np.full((len(final_points), 1), 0.1)])


@st.cache_data
def run_solar_simulation(
        _buildings_data_metric_tuple: tuple,
        grid_points_metric: np.ndarray,
        lat: float, lon: float,
        analysis_date: datetime.date,
        hour_range: tuple
) -> np.ndarray:
    """
    Przeprowadza symulację dla JEDNEGO DNIA z użyciem pvlib i POPRAWIONĄ logiką dla pustej sceny.
    """
    buildings_data_metric = [
        {'polygon': b_tuple[0], 'height': b_tuple[1]} for b_tuple in _buildings_data_metric_tuple
    ]

    scene = create_trimesh_scene(buildings_data_metric)
    start_hour, end_hour = hour_range
    tz = 'Europe/Warsaw'

    # Inicjalizacja tablicy na wyniki (w godzinach)
    sunlit_hours = np.zeros(len(grid_points_metric))

    # Przygotowanie zakresu czasu zgodnego z wyborem użytkownika
    times = pd.date_range(
        start=f"{analysis_date} {start_hour:02d}:00",
        end=f"{analysis_date} {end_hour:02d}:00",
        freq="15min", tz=tz
    )
    location = pvlib.location.Location(lat, lon, tz=tz)
    solar_position = location.get_solarposition(times)
    solar_position_above_horizon = solar_position[solar_position['apparent_elevation'] > 0]

    # --- KLUCZOWA POPRAWKA LOGICZNA ---
    if scene.is_empty:
        # Jeśli nie ma budynków, nasłonecznienie jest równe liczbie okresów, w których słońce jest nad horyzontem
        # w WYBRANYM przez użytkownika zakresie.
        total_sun_periods_in_range = len(solar_position_above_horizon)
        st.warning(
            "⚠️ Nie udało się wygenerować geometrii 3D otoczenia. Analiza pokazuje nasłonecznienie bez uwzględnienia cieni.")
        return np.full(len(grid_points_metric), total_sun_periods_in_range * 0.25)

    combined_mesh = scene.dump(concatenate=True)
    if not isinstance(combined_mesh, trimesh.Trimesh):
        st.warning("⚠️ Błąd podczas łączenia geometrii 3D. Analiza pokazuje nasłonecznienie bez uwzględnienia cieni.")
        total_sun_periods_in_range = len(solar_position_above_horizon)
        return np.full(len(grid_points_metric), total_sun_periods_in_range * 0.25)


    # Pętla po precyzyjnie obliczonych pozycjach (bez zmian)
    for _, sun_pos in solar_position_above_horizon.iterrows():
        alt_rad = np.deg2rad(sun_pos['apparent_elevation'])
        az_rad = np.deg2rad(sun_pos['azimuth'])

        sun_direction = np.array([
            np.cos(alt_rad) * np.sin(az_rad),
            np.cos(alt_rad) * np.cos(az_rad),
            np.sin(alt_rad)
        ])

        ray_origins = grid_points_metric
        ray_directions = np.tile(sun_direction, (len(ray_origins), 1))

        # FIX 3: Add ray distance limit to only check nearby buildings (500m max)
        max_ray_distance = 500.0
        locations, index_ray, _ = combined_mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )

        # Filter intersections by distance
        is_lit = np.ones(len(ray_origins), dtype=bool)
        if len(locations) > 0:
            distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
            valid_hits = distances < max_ray_distance
            shadowed_ray_indices = np.unique(index_ray[valid_hits])
            is_lit[shadowed_ray_indices] = False

        sunlit_hours += is_lit * 0.25

    return sunlit_hours


        # --- FUNKCJE AGENTA AI (z kluczową poprawką wydajności) ---

# ... (perform_ai_step bez zmian)...
def perform_ai_step(driver, model, goal_prompt):
    st.info(f"🎯 **Cel:** {goal_prompt}")
    screenshot_bytes = driver.get_screenshot_as_png()
    prompt = f"Cel: '{goal_prompt}'. Odpowiedz w JSON, podając `element_text` do kliknięcia."
    response = model.generate_content([Part.from_data(screenshot_bytes, mime_type="image/png"), prompt])
    try:
        ai_response_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(ai_response_text).get("element_text"), None
    except Exception as e:
        return None, f"Błąd przetwarzania AI: {e}. Odpowiedź: {response.text}"


# --- ZOPTYMALIZOWANY EKSTRAKTOR LINKÓW ---
def extract_links_by_clicking(driver, wait):
    st.info("🎯 **Cel:** Błyskawiczna ekstrakcja linków.")
    extracted_links = {}
    links_to_find = ["Ustalenia ogólne", "Ustalenia morfoplastyczne", "Ustalenia szczegółowe", "Ustalenia końcowe"]
    original_window = driver.current_window_handle

    for label in links_to_find:
        # --- KLUCZOWA ZMIANA WYDAJNOŚCIOWA ---
        # Używamy find_elements (l. mnoga), która nie czeka i zwraca pustą listę, jeśli nic nie znajdzie
        link_locator = (By.XPATH, f"//td/div[text()='{label}']/parent::td/following-sibling::td//a")
        found_links = driver.find_elements(*link_locator)  # Gwiazdka (*) rozpakowuje krotkę (By.XPATH, '...')

        if found_links:
            # Link istnieje, więc go przetwarzamy
            link_to_click = found_links[0]
            try:
                driver.execute_script("arguments[0].click();", link_to_click)
                wait.until(EC.number_of_windows_to_be(2))
                new_window = [w for w in driver.window_handles if w != original_window][0]
                driver.switch_to.window(new_window)
                extracted_links[label] = driver.current_url
                st.success(f"Pobrano link: {label}")
                driver.close()
                driver.switch_to.window(original_window)
                time.sleep(1)
            except Exception as e:
                st.warning(f"Błąd podczas klikania w link dla '{label}': {e}")
        else:
            # Link nie istnieje - informujemy i natychmiast przechodzimy dalej
            st.write(f"ℹ️ Link dla '{label}' nie istnieje na stronie. Pomijam.")

    return extracted_links


# ... (reszta funkcji analitycznych i run_ai_agent_flow bez zmian)...
@st.cache_data
def analyze_documents_with_ai(_links_tuple, parcel_id):  # Zmieniono nazwę argumentu
    links_dict = dict(_links_tuple)
    results = {'ogolne': {}, 'szczegolowe': {}}
    docs_content = {}
    for label, url in links_dict.items():
        try:
            response = requests.get(url);
            response.raise_for_status()
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                docs_content[label] = "".join(page.get_text() for page in doc)
        except Exception:
            continue
    if "Ustalenia ogólne" in docs_content:
        prompt = f"Na podstawie tego dokumentu, jaki jest ogólny cel i charakter obszaru objętego tym planem?\n\nDokument:\n---\n{docs_content['Ustalenia ogólne']}"
        results['ogolne']['Cel Planu'] = llm.invoke(prompt)
    if "Ustalenia szczegółowe" in docs_content:
        doc_szczegolowe = docs_content["Ustalenia szczegółowe"]
        id_prompt = f"Na podstawie poniższego tekstu, jaki jest symbol/oznaczenie terenu elementarnego? (np. 'S.N.9006.MC')\n\nTekst:\n---\n{doc_szczegolowe[:1000]}"
        results['szczegolowe']['Oznaczenie Terenu'] = llm.invoke(id_prompt)
        detail_questions = {
            "Przeznaczenie terenu": "Jakie jest szczegółowe przeznaczenie terenu (podstawowe i dopuszczalne) oraz jakie są zakazy?",
            "Wysokość zabudowy": "Jakie są szczegółowe ustalenia dotyczące wysokości zabudowy w metrach?",
            "Wskaźniki zabudowy": "Jakie są szczegółowe wskaźniki, takie jak maksymalna powierzchnia zabudowy i minimalna powierzchnia biologicznie czynna?",
            "Geometria dachu": "Jakie są szczegółowe wymagania dotyczące geometrii dachu i jego pokrycia?",
        }
        for key, question in detail_questions.items():
            prompt = f"Na podstawie TYLKO i WYŁĄCZNIE poniższego dokumentu 'Ustalenia szczegółowe', odpowiedz na pytanie: {question}\n\nDokument:\n---\n{doc_szczegolowe}"
            results['szczegolowe'][key] = llm.invoke(prompt)
    return results


def run_ai_agent_flow(parcel_id):
    service = Service()
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=service, options=options)
    final_results = {}
    try:
        with st.expander("Postęp misji agenta nawigacyjnego", expanded=True):
            driver.get("https://mapa.szczecin.eu/gpt4/?permalink=56520129")
            time.sleep(5)
            wait = WebDriverWait(driver, 20)

            # Krok 1: Wyszukanie działki
            search_box = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Szukaj...']")))
            search_box.send_keys(parcel_id)
            wait.until(EC.visibility_of_element_located((By.XPATH, "//li[contains(@class, 'x-boundlist-item')]")))
            time.sleep(1); search_box.send_keys(Keys.RETURN)
            time.sleep(1); search_box.send_keys(Keys.RETURN)
            st.success("✅ Krok 1/3: Działka zlokalizowana.")
            time.sleep(4)

            # Krok 2: Otwarcie menu i kliknięcie 'Informacje o obiekcie'
            ActionChains(driver).move_by_offset(driver.get_window_size()['width'] / 2, driver.get_window_size()['height'] / 2).context_click().perform()
            st.success("✅ Krok 2/3: Menu kontekstowe otwarte.")
            time.sleep(1)
            try:
                # W tym przypadku AI nie jest konieczne, można klikać stały tekst
                wait.until(EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Informacje o obiekcie')]"))).click()
                st.success("✅ Akcja 'Informacje o obiekcie' wykonana.")
                time.sleep(3)  # Czas na załadowanie danych w oknie
            except Exception as e:
                st.error(f"⚠️ Nie udało się otworzyć okna 'Informacje o obiekcie': {e}")
                raise e

            # --- POPRAWIONA I PRECYZYJNA LOGIKA SPRAWDZANIA STANU MPZP ---
            st.info("🔎 Krok 3/3: Sprawdzanie statusu MPZP w dedykowanym oknie...")
            time.sleep(2)

            # Definiujemy precyzyjny kontekst - okno "Informacje o obiekcie"
            info_window_context_xpath = "//div[contains(@class, 'x-window') and .//span[text()='Informacje o obiekcie']]"

            # Scenariusz 1: Sprawdzamy czy W OKNIE istnieje UCHWALONY MPZP
            mpzp_uchwalony_locator = (By.XPATH, info_window_context_xpath + "//*[contains(text(), 'MPZP - Tereny elementarne')]")
            # Scenariusz 2: Sprawdzamy czy W OKNIE istnieje WSZCZĘTY MPZP
            mpzp_wszczety_locator = (By.XPATH, info_window_context_xpath + "//*[contains(text(), 'MPZP - plany wszczęte')]")

            if driver.find_elements(*mpzp_uchwalony_locator):
                st.success("✅ Znaleziono UCHWALONY MPZP dla tej działki. Kontynuuję analizę...")
                try:
                    # Klikamy w ten konkretny element, który znaleźliśmy
                    driver.find_element(*mpzp_uchwalony_locator).click()
                    time.sleep(2)

                    with st.spinner("Nawigacja zakończona. Ekstrakcja linków..."):
                        final_links = extract_links_by_clicking(driver, wait)

                    if final_links:
                        final_results['links'] = final_links
                        st.subheader("Pobrane Dokumenty:"); st.toast("✅ Linki pobrane!")
                        for label, link in final_links.items(): st.markdown(f"**{label}:** [Otwórz]({link})")
                        with st.spinner("Uruchamiam Agenta Analityka AI..."):
                            analysis = analyze_documents_with_ai(tuple(sorted(final_links.items())), parcel_id)
                        if analysis: final_results['analysis'] = analysis
                    else:
                        st.error("Nie udało się wyodrębnić żadnych linków, mimo że MPZP został zidentyfikowany.")
                except Exception as e:
                    st.error(f"Wystąpił błąd na etapie interakcji z istniejącym MPZP: {e}")
                    return {}

            elif driver.find_elements(*mpzp_wszczety_locator):
                st.warning("🔵 Dla tej działki procedura sporządzenia MPZP została wszczęta, ale plan nie jest jeszcze uchwalony.")
                st.info("Agent kończy pracę, ponieważ nie ma jeszcze finalnych dokumentów do analizy.")
                return {"status": "wszczęty"}

            else:
                st.error("❌ Dla wybranej działki w oknie informacyjnym nie znaleziono żadnych danych o MPZP.")
                st.info("Agent kończy pracę.")
                return {"status": "brak"}

    finally:
        st.write("Zamykam przeglądarkę.")
        driver.quit()

    return final_results


# --- GŁÓWNY INTERFEJS UŻYTKOWNIKA (ZMIANY W GUI MAPY) ---
st.set_page_config(layout="wide");

# Custom CSS - Light Immersive Theme
st.markdown("""
<style>
    /* === GLOBAL IMMERSIVE STYLING === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* IMMERSIVE: Strong scroll snap - infinite scroll style */
    html {
        scroll-behavior: smooth;
        scroll-snap-type: y mandatory;
        scroll-padding: 0;
    }

    body {
        scroll-snap-type: y mandatory;
        overflow-y: scroll;
    }

    /* Hide Deploy button but keep menu and spinner */
    .stDeployButton {
        display: none !important;
    }

    /* Hide Streamlit branding footer */
    footer {
        display: none !important;
    }

    /* Prevent gray overlay during rerun - ULTRA AGGRESSIVE FIX */
    .main, .stApp, [data-testid="stAppViewContainer"], .element-container {
        transition: none !important;
        filter: none !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }

    /* Override ALL Streamlit overlays */
    .stApp::before, .stApp::after {
        display: none !important;
    }

    /* Force disable the dimming effect */
    div[data-baseweb="modal"] {
        background: none !important;
    }

    /* Main container with blue→green gradient background */
    .main {
        background: linear-gradient(135deg, #e3f2fd 0%, #e8f5e9 50%, #f1f8e9 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide scrollbar but keep functionality */
    ::-webkit-scrollbar {
        width: 0px;
        background: transparent;
    }

    /* For Firefox */
    * {
        scrollbar-width: none;
    }

    /* Section containers - FULL SCREEN with snap points */
    .stApp > div > div {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 3rem;
        margin: 0;
        min-height: 100vh;
        border: 1px solid rgba(40, 167, 69, 0.2);
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        scroll-snap-align: start;
        scroll-snap-stop: always;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .stApp > div > div:hover {
        border: 1px solid rgba(40, 167, 69, 0.4);
        box-shadow: 0 6px 24px 0 rgba(40, 167, 69, 0.15);
    }

    /* Block-level elements */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Headers with clean styling */
    h1 {
        color: #1a237e !important;
        font-weight: 700;
        letter-spacing: -0.5px;
        animation: fadeInDown 0.6s ease;
        text-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
    }

    h2, h3 {
        color: #2e7d32 !important;
        font-weight: 600;
        letter-spacing: -0.3px;
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Input fields styling */
    input, textarea {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(40, 167, 69, 0.3) !important;
        border-radius: 12px !important;
        color: #1a237e !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }

    input:focus, textarea:focus {
        border-color: #28a745 !important;
        box-shadow: 0 0 12px rgba(40, 167, 69, 0.3) !important;
        background: #ffffff !important;
    }

    /* Buttons with hover effects */
    .stButton button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3) !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(40, 167, 69, 0.5) !important;
    }

    /* FORCE GREEN COLOR ON ALL PRIMARY ELEMENTS - Override Streamlit cache */
    /* Slider styling */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #28a745 !important;
        box-shadow: 0 0 15px rgba(40, 167, 69, 0.5) !important;
    }

    div[data-baseweb="slider"] div[data-testid="stSliderTickBar"] {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%) !important;
    }

    /* Slider track fill */
    div[data-baseweb="slider"] div[data-testid="stSliderTickBarMin"] {
        background-color: #28a745 !important;
    }

    /* Slider thumb */
    div[data-baseweb="slider"] div[role="slider"]::before {
        background-color: #28a745 !important;
    }

    /* Radio buttons & checkboxes */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        background-color: #28a745 !important;
        box-shadow: 0 0 10px rgba(40, 167, 69, 0.3) !important;
    }

    /* Success/Info/Warning boxes */
    .stAlert {
        background: rgba(40, 167, 69, 0.1) !important;
        border: 1px solid rgba(40, 167, 69, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Map containers */
    iframe {
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* PyDeck charts */
    .deckgl-wrapper {
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    }

    /* Code blocks */
    code {
        background: rgba(40, 167, 69, 0.2) !important;
        color: #28a745 !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        font-family: 'Fira Code', monospace !important;
    }

    /* Scrollbar is now hidden (see above) */

    /* Fade-in animations for content */
    .element-container {
        animation: fadeIn 0.6s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Title glow effect */
    .stTitle {
        text-shadow: 0 0 30px rgba(40, 167, 69, 0.5);
    }

    /* IMMERSIVE: Large 3D button styling */
    button[key="generate_3d_button"] {
        background: linear-gradient(135deg, #42a5f5 0%, #66bb6a 100%) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 1rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(66, 165, 245, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    button[key="generate_3d_button"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(66, 165, 245, 0.6) !important;
    }

    /* Better button contrast */
    .stButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }

    /* IMMERSIVE: Gradient for ALL headers */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        background: linear-gradient(135deg, #42a5f5 0%, #66bb6a 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 600 !important;
    }
</style>

<script>
// IMMERSIVE: Auto-scroll to bottom after Streamlit reruns
const autoScrollToBottom = () => {
    setTimeout(() => {
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: 'smooth'
        });
    }, 300);
};

// Listen for Streamlit script finished event
window.addEventListener('load', () => {
    const observer = new MutationObserver(() => {
        // Check if new content was added (indicates rerun completed)
        autoScrollToBottom();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
</script>
""", unsafe_allow_html=True)

# Initialize session state
for key in ['map_center', 'parcel_data', 'analysis_results', 'show_search']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'show_search' else False

# IMMERSIVE: Landing page - hero only, clickable anywhere to start
if not st.session_state.show_search and not st.session_state.map_center:
    # Simple approach: show hero and button below
    st.markdown("""
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 85vh; text-align: center;">
        <h1 style="font-size: 4rem; margin-bottom: 1rem; background: linear-gradient(135deg, #42a5f5 0%, #66bb6a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; line-height: 1.2;">
            Asystent Analizy Działki
        </h1>
        <p style="font-size: 1.3rem; color: #424242; margin-bottom: 0.5rem; font-weight: 500;">Szczecin • Wersja Beta 0.2</p>
        <p style="font-size: 1rem; color: #616161; margin-bottom: 3rem;">
            Autor: Fabian Korycki | Powered by <span style="color: #28a745; font-weight: 600;">Google Gemini AI</span>
        </p>
    </div>

    <style>
    @keyframes pulse {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 0.8; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Centered button to start
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Rozpocznij", key="start_button", use_container_width=True):
            st.session_state.show_search = True
            st.rerun()

# Search section (appears after landing)
if st.session_state.show_search or st.session_state.map_center:
    # Header at top (same size as landing page)
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size: 4rem; margin: 0; background: linear-gradient(135deg, #42a5f5 0%, #66bb6a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; line-height: 1.2;">
            Asystent Analizy Działki
        </h1>
        <p style="font-size: 1rem; color: #616161; margin: 0.5rem 0 0 0;">Szczecin • Beta 0.2</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form(key="address_form"):
        address_input = st.text_input("Wpisz adres lub współrzędne:", "", label_visibility="collapsed", placeholder="Wpisz adres, np. Kolumba 64, Szczecin")
        submitted = st.form_submit_button("Wyszukaj działkę", use_container_width=True)

    if submitted:
        st.session_state.parcel_data = None;
        st.session_state.analysis_results = None
        with st.spinner("Pobieram współrzędne..."):
            coords, error = geocode_address_to_coords(address_input)
            if error:
                st.error(error); st.session_state.map_center = None
            else:
                st.session_state.map_center = coords

    if st.session_state.map_center and not st.session_state.parcel_data:
        # IMMERSIVE: Fullscreen centered map section
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h2 style="background: linear-gradient(135deg, #42a5f5 0%, #66bb6a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;">
                Wybierz działkę na mapie
            </h2>
            <p style="color: #616161; font-size: 1rem;">Kliknij na interesującą Cię działkę, aby ją zidentyfikować</p>
        </div>
        """, unsafe_allow_html=True)

        m = folium.Map(location=st.session_state.map_center, zoom_start=18)
        folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                         attr='Esri', name='Satelita', overlay=True).add_to(m)
        folium.WmsTileLayer(url="https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow",
                            layers="dzialki,numery_dzialek", transparent=True, fmt="image/png",
                            name="Działki Ewidencyjne").add_to(m)
        folium.LayerControl().add_to(m)

        # IMMERSIVE: Fullscreen map (80vh for visibility with header)
        map_data = st_folium(m, use_container_width=True, height=700)

        if map_data and map_data.get("last_clicked"):
            lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            with st.spinner(f"Identyfikuję działkę..."):
                parcel_data, error = get_parcel_from_coords(lat, lon)
                if error:
                    st.error(error)
                else:
                    st.session_state.parcel_data = parcel_data; st.rerun()



    if st.session_state.parcel_data:
        # IMMERSIVE: Fullscreen confirmation section with map, data, and 3D button
        coords_wgs84 = transform_coordinates_to_wgs84(st.session_state.parcel_data["Współrzędne EPSG:2180"])
        map_center = [sum(p[0] for p in coords_wgs84) / len(coords_wgs84),
                      sum(p[1] for p in coords_wgs84) / len(coords_wgs84)]

        # Large centered confirmation map (no header)
        m_confirm = folium.Map(location=map_center, zoom_start=19)
        folium.TileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri').add_to(m_confirm)
        folium.Polygon(locations=coords_wgs84, color='#28a745', fill=True, fillColor='#28a745',
                       fill_opacity=0.3, weight=3, tooltip=st.session_state.parcel_data['ID Działki']).add_to(m_confirm)
        st_folium(m_confirm, use_container_width=True, height=550)

        # Parcel ID displayed below map - centered
        st.markdown(f"""
        <div style="text-align: center; margin: 1.5rem 0; padding: 1rem; background: rgba(40, 167, 69, 0.1); border-radius: 12px; border: 2px solid rgba(40, 167, 69, 0.3);">
            <p style="color: #616161; font-size: 0.9rem; margin: 0;">Numer działki ewidencyjnej</p>
            <p style="font-size: 1.3rem; font-weight: 600; color: #28a745; margin: 0.3rem 0 0 0; font-family: monospace;">{st.session_state.parcel_data['ID Działki']}</p>
        </div>
        """, unsafe_allow_html=True)

        # IMMERSIVE: 3D button (no header, fits in viewport)
        show_3d_context = st.button(
            "Wygeneruj widok 3D otoczenia",
            key="generate_3d_button",
            use_container_width=True,
            help="Generowanie widoku 3D może zająć kilka sekund"
        )

        # Store 3D state in session
        if 'show_3d' not in st.session_state:
            st.session_state.show_3d = False

        if show_3d_context:
            st.session_state.show_3d = not st.session_state.show_3d

        # IMMERSIVE: 3D Fullscreen Takeover (separate section)
        if st.session_state.show_3d:
            st.markdown("""<div style="height: 2px; background: linear-gradient(90deg, transparent, #42a5f5, transparent); margin: 3rem 0 2rem 0; opacity: 0.5;"></div>""", unsafe_allow_html=True)

            if 'map_theme' not in st.session_state: st.session_state.map_theme = "Jasny"
            THEME_MAPPING = {"Jasny": "light", "Ciemny": "dark"}

            # Radio buttons aligned to left (no columns)
            st.session_state.map_theme = st.radio(
                "Wybierz motyw mapy:", options=["Jasny", "Ciemny"],
                horizontal=True, key="map_theme_selector"
            )

            selected_map_style = THEME_MAPPING[st.session_state.map_theme]
            with st.spinner("Generuję model 3D otoczenia..."):
                parcel_poly_coords_lon_lat = [(p[1], p[0]) for p in coords_wgs84]
                deck_3d_view = generate_3d_context_view(
                    parcel_poly_coords_lon_lat, map_center, map_style=selected_map_style
                )
                if deck_3d_view:
                    # IMMERSIVE: Fullscreen 3D view with controls info
                    st.markdown("""
                    <div style="background: rgba(66, 165, 245, 0.1); border-left: 4px solid #42a5f5; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <p style="margin: 0; color: #424242; font-size: 0.95rem;">
                            <strong>Sterowanie kamerą:</strong>
                            <strong>Obrót:</strong> Shift + przeciągnij |
                            <strong>Przesuwanie:</strong> Przeciągnij |
                            <strong>Zoom:</strong> Kółko myszy
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.pydeck_chart(deck_3d_view, use_container_width=True, height=750)

            st.markdown("""<div style="height: 2px; background: linear-gradient(90deg, transparent, #42a5f5, transparent); margin: 2rem 0; opacity: 0.5;"></div>""", unsafe_allow_html=True)

        # IMMERSIVE: Split-screen analysis selector (fullscreen section with more spacing)
        st.markdown("""
        <div style="text-align: center; margin: 10rem 0 3rem 0;">
            <h2 style="font-size: 2.2rem; margin-bottom: 0.5rem;">Wybierz typ analizy</h2>
            <p style="color: #616161; font-size: 1.05rem;">Kliknij jedną z opcji, aby rozpocząć szczegółową analizę</p>
        </div>
        """, unsafe_allow_html=True)

        # IMMERSIVE: Big split-screen clickable analysis blocks
        analysis_col1, analysis_col2 = st.columns(2, gap="large")

        # Initialize analysis states
        if 'selected_analysis' not in st.session_state:
            st.session_state.selected_analysis = None

        # --- LEFT HALF: Solar Analysis (block + button) ---
        with analysis_col1:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, rgba(255,193,7,0.08) 0%, rgba(255,152,0,0.08) 100%); border-radius: 20px; border: 2px solid rgba(255,193,7,0.25); min-height: 350px; display: flex; flex-direction: column; justify-content: center; transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 16px rgba(255,193,7,0.15)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                <h3 style="font-size: 1.8rem; margin-bottom: 1.5rem; color: #424242;">Analiza Nasłonecznienia</h3>
                <p style="color: #616161; font-size: 1rem; margin-bottom: 0; line-height: 1.6;">Oblicza średnią dzienną liczbę godzin słońca dla każdego punktu działki, uwzględniając cienie sąsiednich budynków</p>
            </div>
            """, unsafe_allow_html=True)

            # Add spacing before button
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

            # Centered button below block
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("Wybierz", key="select_solar", use_container_width=True):
                    st.session_state.selected_analysis = "solar"

        # --- RIGHT HALF: MPZP Analysis (block + button) ---
        with analysis_col2:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, rgba(33,150,243,0.08) 0%, rgba(25,118,210,0.08) 100%); border-radius: 20px; border: 2px solid rgba(33,150,243,0.25); min-height: 350px; display: flex; flex-direction: column; justify-content: center; transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 16px rgba(33,150,243,0.15)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                <h3 style="font-size: 1.8rem; margin-bottom: 1.5rem; color: #424242;">Analiza MPZP</h3>
                <p style="color: #616161; font-size: 1rem; margin-bottom: 0; line-height: 1.6;">Inteligentna analiza dokumentów planistycznych z wykorzystaniem AI (Gemini 2.5 Pro)</p>
            </div>
            """, unsafe_allow_html=True)

            # Add spacing before button
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

            # Centered button below block
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("Wybierz", key="select_mpzp", use_container_width=True):
                    st.session_state.selected_analysis = "mpzp"

        # IMMERSIVE: Fullscreen analysis sections
        if st.session_state.selected_analysis == "solar":
            st.markdown("""<div style="height: 2px; background: linear-gradient(90deg, transparent, #FFC107, transparent); margin: 3rem 0 2rem 0; opacity: 0.6;"></div>""", unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="font-size: 2rem;">Analiza Nasłonecznienia</h2>
                <p style="color: #616161; font-size: 1rem;">Skonfiguruj parametry analizy</p>
            </div>
            """, unsafe_allow_html=True)

            today = datetime(2025, 1, 1).date()
            selected_date_range = st.date_input(
                "Wybierz dzień lub zakres dni analizy:",
                value=(today.replace(month=3, day=20), today.replace(month=3, day=20)),
            )
            hour_range = st.slider(
                "Wybierz zakres godzin analizy:",
                min_value=0, max_value=23, value=(6, 20), step=1
            )

            if st.button("Uruchom analizę nasłonecznienia", key="run_solar_analysis", use_container_width=True):
                start_date, end_date = selected_date_range
                if start_date is None or end_date is None or start_date > end_date:
                    st.error("Proszę wybrać poprawny zakres dat.")
                else:
                    num_days = (end_date - start_date).days + 1
                    num_hours = hour_range[1] - hour_range[0] + 1
                    spinner_text = f"Przeprowadzam symulację dla {num_days} {'dzień' if num_days == 1 else 'dni'}, {num_hours} {'godzina' if num_hours == 1 else 'godzin'} (godz. {hour_range[0]}:00-{hour_range[1]}:00)..."
                    with st.spinner(spinner_text):

                        # Krok 1: Pobieramy dane o budynkach z OpenStreetMap
                        gdf_buildings_wgs84 = ox.features_from_point((map_center[0], map_center[1]), {"building": True},
                                                                     dist=350)
                        gdf_buildings_metric = gdf_buildings_wgs84.to_crs("epsg:2180")

                        buildings_data_metric = []


                        # Funkcja do szacowania wysokości
                        def est_h(r):
                            try:
                                if 'height' in r and r['height'] and str(r['height']).strip(): return float(
                                    str(r['height']).split(';')[0])
                                if 'building:levels' in r and r['building:levels'] and str(
                                    r['building:levels']).strip(): return float(
                                    str(r['building:levels']).split(';')[0]) * 3.5 + 2
                            except (ValueError, TypeError):
                                pass
                            return 10.0


                        gdf_buildings_metric['height'] = gdf_buildings_metric.apply(est_h, axis=1)
                        for _, building in gdf_buildings_metric.iterrows():
                            if building.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                                polys = [
                                    building.geometry] if building.geometry.geom_type == 'Polygon' else building.geometry.geoms
                                for p in polys: buildings_data_metric.append(
                                    {"polygon": list(p.exterior.coords), "height": building.height})

                        # Krok 2: Tworzymy siatkę analityczną na działce
                        coords_2180 = st.session_state.parcel_data["Współrzędne EPSG:2180"]
                        parcel_poly_2180 = Polygon(coords_2180)
                        grid_points_2180 = create_analysis_grid(parcel_poly_2180, density=1.0)

                        if grid_points_2180.size > 0:
                            # Krok 3: Uruchamiamy symulację
                            buildings_data_for_cache = tuple(
                                (tuple(b['polygon']), b['height']) for b in buildings_data_metric
                            )

                            total_sunlit_hours = np.zeros(len(grid_points_2180))
                            date_range = pd.date_range(start_date, end_date)

                            for single_date in date_range:
                                total_sunlit_hours += run_solar_simulation(
                                    buildings_data_for_cache,
                                    grid_points_2180,
                                    map_center[0], map_center[1], single_date,
                                    hour_range
                                )

                            # Dalsza część do obsługi wyników (bez zmian)
                            average_sunlit_hours = total_sunlit_hours / len(date_range)
                            transformer_to_wgs = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
                            grid_points_wgs84 = np.array(
                                [transformer_to_wgs.transform(p[0], p[1]) for p in grid_points_2180])
                            results_df = pd.DataFrame(grid_points_wgs84, columns=['lon', 'lat'])
                            results_df['sun_hours'] = average_sunlit_hours
                            viz_date = date_range[len(date_range) // 2]
                            map_center_metric = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True).transform(
                                map_center[1], map_center[0])

                            # Generujemy kompletny sun path diagram (styl Ladybug)
                            sun_paths, analemmas, azimuth_markers, azimuth_lines = generate_complete_sun_path_diagram(
                                map_center[0], map_center[1], viz_date.year, map_center_metric
                            )

                            # Generujemy pozycję słońca dla wybranej daty/zakresu
                            sun_position_markers = []
                            location = pvlib.location.Location(map_center[0], map_center[1], tz='Europe/Warsaw')

                            for single_date in date_range:
                                for hour in range(hour_range[0], hour_range[1] + 1):
                                    try:
                                        time_str = f"{single_date} {hour:02d}:00"
                                        time = pd.Timestamp(time_str, tz='Europe/Warsaw')
                                        solar_pos = location.get_solarposition(time)
                                        elevation = solar_pos['apparent_elevation'].values[0]
                                        azimuth_val = solar_pos['azimuth'].values[0]

                                        if elevation > 0:
                                            alt_rad = np.deg2rad(elevation)
                                            az_rad = np.deg2rad(azimuth_val)
                                            x_offset = 300 * np.cos(alt_rad) * np.sin(az_rad)
                                            y_offset = 300 * np.cos(alt_rad) * np.cos(az_rad)
                                            z = 300 * np.sin(alt_rad)
                                            sun_position_markers.append({
                                                'position': [map_center_metric[0] + x_offset, map_center_metric[1] + y_offset, z],
                                                'date': str(single_date),
                                                'hour': hour
                                            })
                                    except:
                                        continue

                            st.session_state.solar_analysis_results = {"results_df": results_df,
                                                                       "buildings_metric": buildings_data_metric,
                                                                       "sun_paths": sun_paths,
                                                                       "analemmas": analemmas,
                                                                       "azimuth_markers": azimuth_markers,
                                                                       "azimuth_lines": azimuth_lines,
                                                                       "sun_position_markers": sun_position_markers}
                        else:
                            st.session_state.solar_analysis_results = None
                st.rerun()

            # --- NOWA, POPRAWIONA SEKCJ_A WIZUALIZACJI WYNIKÓW ---
            if 'solar_analysis_results' in st.session_state and st.session_state.solar_analysis_results:
                data = st.session_state.solar_analysis_results
                if not data["results_df"].empty:
                    results_df = data["results_df"]
                    min_h, max_h = results_df['sun_hours'].min(), results_df['sun_hours'].max()

                    # ZMIANA 3: Zabezpieczenie przed "płaską" legendą
                    if max_h == min_h:
                        max_h += 1.0

                    results_df['color'] = results_df['sun_hours'].apply(lambda x: value_to_rgb(x, min_h, max_h))

                    transformer_to_wgs = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)

                    buildings_data_wgs84 = []
                    for b in data['buildings_metric']:
                        poly_metric = Polygon(b['polygon'])
                        poly_wgs84 = transform(transformer_to_wgs.transform, poly_metric)
                        buildings_data_wgs84.append({
                            "polygon": [list(poly_wgs84.exterior.coords)],
                            "height": b['height']
                        })

                    # Konwertujemy sun paths do WGS84 (SZARE)
                    sun_paths_wgs84 = []
                    for sp in data['sun_paths']:
                        path_wgs = [transformer_to_wgs.transform(p[0], p[1]) + (p[2],) for p in sp['path']]
                        sun_paths_wgs84.append({"path": path_wgs})

                    # Analemmy jako segmenty - FILTROWANIE PO DŁUGOŚCI
                    # Punkty są teraz równomiernie rozłożone (co 1 dzień), więc przeskoki
                    # będą wyraźnie dłuższe od normalnych segmentów
                    analemmas_segments = {}

                    # Najpierw zbierz WSZYSTKIE segmenty ze WSZYSTKICH analemm
                    all_segments = []
                    for hour, ana_data in data['analemmas'].items():
                        # Konwertujemy współrzędne do WGS84
                        ana_wgs = []
                        for point_data in ana_data:
                            coords = point_data['coords']
                            wgs_coords = transformer_to_wgs.transform(coords[0], coords[1]) + (coords[2],)
                            ana_wgs.append(wgs_coords)

                        # Tworzymy segmenty z obliczoną długością
                        for i in range(len(ana_wgs) - 1):
                            source = np.array(ana_wgs[i])
                            target = np.array(ana_wgs[i + 1])
                            length = np.linalg.norm(target - source)

                            all_segments.append({
                                "source": ana_wgs[i],
                                "target": ana_wgs[i + 1],
                                "length": length,
                                "hour": hour
                            })

                    # Oblicz statystyki długości dla WSZYSTKICH segmentów
                    if len(all_segments) > 0:
                        all_lengths = [s['length'] for s in all_segments]
                        median_length = np.median(all_lengths)
                        min_length = np.min(all_lengths)

                        # Odrzuć segmenty dłuższe niż 2x mediana
                        max_allowed_length = median_length * 2

                        # Filtruj według godziny + odetnij linie pod horyzontem (Z < 0)
                        for hour in data['analemmas'].keys():
                            filtered = [
                                {"source": s["source"], "target": s["target"], "hour": s["hour"]}
                                for s in all_segments
                                if s['hour'] == hour
                                and s['length'] <= max_allowed_length
                                and s['source'][2] >= 0  # source Z >= 0
                                and s['target'][2] >= 0  # target Z >= 0
                            ]
                            analemmas_segments[hour] = filtered
                    else:
                        analemmas_segments = {hour: [] for hour in data['analemmas'].keys()}

                    # Konwertujemy azimuth markers do WGS84
                    azimuth_markers_wgs84 = []
                    for am in data['azimuth_markers']:
                        pos_wgs = list(transformer_to_wgs.transform(am['position'][0], am['position'][1]))
                        azimuth_markers_wgs84.append({
                            "position": [pos_wgs[0], pos_wgs[1], am['position'][2]],
                            "label": am['label']
                        })

                    # Konwertujemy azimuth lines do WGS84 (linie kompasu) - BEZ kropek
                    azimuth_lines_main_wgs84 = []  # Główne kierunki (N, E, S, W)
                    azimuth_lines_secondary_wgs84 = []  # Pozostałe kierunki

                    for al in data['azimuth_lines']:
                        line_wgs = [transformer_to_wgs.transform(p[0], p[1]) + (p[2],) for p in al['path']]
                        if al['is_main']:
                            azimuth_lines_main_wgs84.append({"path": line_wgs})
                        else:
                            azimuth_lines_secondary_wgs84.append({"path": line_wgs})

                    # Konwertujemy sun position markers do WGS84 (ŻÓŁTE ikony słońca)
                    sun_positions_wgs84 = []
                    for sp in data['sun_position_markers']:
                        pos_wgs = list(transformer_to_wgs.transform(sp['position'][0], sp['position'][1]))
                        sun_positions_wgs84.append({
                            "position": [pos_wgs[0], pos_wgs[1], sp['position'][2]]
                        })

                    # Definicje warstw wizualizacji
                    # GridCellLayer - cell_size musi być równy density (1.0m) dla idealnego pokrycia
                    heatmap_layer = pdk.Layer("GridCellLayer", data=results_df, get_position=['lon', 'lat'],
                                              get_fill_color='color', cell_size=1.0, extruded=False,
                                              coverage=1.0)

                    building_layer = pdk.Layer("PolygonLayer", data=buildings_data_wgs84, get_polygon="polygon",
                                               extruded=True,
                                               get_elevation="height", get_fill_color=[180, 180, 180, 80], wireframe=True)

                    # KOMPLETNY SUN PATH DIAGRAM (styl Ladybug)
                    # 1. Ścieżki słońca dla kluczowych dat (SZARE) z efektem billboard
                    sun_path_layer = pdk.Layer("PathLayer", data=sun_paths_wgs84, get_path="path",
                                              get_color=[140, 140, 140, 160], get_width=1,
                                              width_min_pixels=1, billboard=True)

                    # 2. OSTATECZNE ROZWIĄZANIE: 13 OSOBNYCH warstw LineLayer - bez segmentu zamykającego
                    # Każda analemma jako osobna warstwa z własnymi segmentami
                    analemma_layers = []


                    # Renderujemy analemmy jako szare, minimalne linie
                    for hour in sorted(analemmas_segments.keys()):
                        layer = pdk.Layer(
                            "LineLayer",
                            id=f"analemma_segments_{hour}",
                            data=analemmas_segments[hour],
                            get_source_position="source",
                            get_target_position="target",
                            get_color=[100, 100, 100, 180],
                            get_width=1,
                            width_min_pixels=1,
                            pickable=False,
                            auto_highlight=False
                        )
                        analemma_layers.append(layer)


                    # 3. Linie kompasu - główne kierunki z efektem 3D
                    compass_main_layer = pdk.Layer("PathLayer", data=azimuth_lines_main_wgs84, get_path="path",
                                                   get_color=[90, 90, 90, 150], get_width=1.5,
                                                   width_min_pixels=1, billboard=True)

                    # 4. Linie kompasu - pozostałe kierunki z efektem 3D
                    compass_secondary_layer = pdk.Layer("PathLayer", data=azimuth_lines_secondary_wgs84, get_path="path",
                                                       get_color=[120, 120, 120, 120], get_width=1,
                                                       width_min_pixels=1, billboard=True)

                    # 5. Markery azymutu na poziomie gruntu (etykiety kierunków)
                    azimuth_text_layer = pdk.Layer("TextLayer", data=azimuth_markers_wgs84,
                                                  get_position="position",
                                                  get_text="label",
                                                  get_size=14,
                                                  get_color=[80, 80, 80, 255],
                                                  get_angle=0,
                                                  get_text_anchor="'middle'",
                                                  get_alignment_baseline="'center'",
                                                  billboard=True)

                    # 6. Pozycje słońca dla wybranej daty (ŻÓŁTE kule BEZ obwódki)
                    sun_markers_layer = pdk.Layer("ScatterplotLayer", data=sun_positions_wgs84,
                                                 get_position="position",
                                                 get_radius=12, filled=True,
                                                 get_fill_color=[255, 223, 0, 255],
                                                 stroked=False, billboard=True)

                    # Wyświetlanie legendy i mapy 3D
                    st.markdown(create_discrete_legend_html(min_h, max_h, colormap='plasma'), unsafe_allow_html=True)

                    # Składamy wszystkie warstwy razem - analemmy z usuniętymi najdłuższymi segmentami
                    all_layers = [
                        building_layer,           # Budynki
                        heatmap_layer,            # Mapa nasłonecznienia
                        compass_main_layer,       # Linie kompasu - główne (N, E, S, W)
                        compass_secondary_layer,  # Linie kompasu - pozostałe
                        sun_path_layer,           # Ścieżki słońca (szare łuki)
                    ] + analemma_layers + [       # 13 osobnych warstw LineLayer
                        azimuth_text_layer,       # Etykiety kierunków
                        sun_markers_layer,        # Żółte ikony słońca dla wybranej daty
                    ]

                    r = pdk.Deck(layers=all_layers,
                                 initial_view_state=pdk.ViewState(latitude=map_center[0], longitude=map_center[1],
                                                                  zoom=17.5, pitch=50, bearing=0),
                                 map_style=None)

                    st.pydeck_chart(r, use_container_width=True, height=600)
                else:
                    st.warning("Nie udało się stworzyć siatki analitycznej dla tej działki.")



        # IMMERSIVE: MPZP Fullscreen Section
        elif st.session_state.selected_analysis == "mpzp":
            st.markdown("""<div style="height: 2px; background: linear-gradient(90deg, transparent, #2196F3, transparent); margin: 3rem 0 2rem 0; opacity: 0.6;"></div>""", unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="font-size: 2rem;">Analiza MPZP (Agent AI)</h2>
                <p style="color: #616161; font-size: 1rem;">Agent nawiguje po geoportalu i analizuje dokumenty planistyczne</p>
            </div>
            """, unsafe_allow_html=True)

            # Initialize analysis state
            if 'mpzp_analysis_started' not in st.session_state:
                st.session_state.mpzp_analysis_started = False

            # Show button only if not started
            if not st.session_state.mpzp_analysis_started:
                start_btn = st.button("Rozpocznij pełną analizę AI", key="run_mpzp_analysis", use_container_width=True)
                if start_btn:
                    st.session_state.mpzp_analysis_started = True

            # Run analysis if started but no results yet
            if st.session_state.mpzp_analysis_started and not st.session_state.get('analysis_results'):
                st.info("🤖 Agent AI uruchomiony - pobieram dokumenty i analizuję...")
                try:
                    results = run_ai_agent_flow(st.session_state.parcel_data['ID Działki'])
                    if results:
                        st.session_state.analysis_results = results
                        st.success("✅ Analiza zakończona!")
                    else:
                        st.error("Nie udało się pobrać wyników analizy.")
                        st.session_state.mpzp_analysis_started = False
                except Exception as e:
                    st.error(f"❌ Błąd podczas analizy: {str(e)}")
                    st.session_state.mpzp_analysis_started = False

            # Wyświetlanie wyników analizy AI (jeśli istnieją)
            if st.session_state.get('analysis_results'):
                results = st.session_state.analysis_results
                if 'analysis' in results and results['analysis']:
                    st.success("🎉 Misja Agenta Analityka zakończona!")
                    if 'ogolne' in results['analysis'] and results['analysis']['ogolne']:
                        st.markdown(f"**Cel Planu:**");
                        st.info(f"{results['analysis']['ogolne'].get('Cel Planu', 'Brak danych.')}")
                    if 'szczegolowe' in results['analysis'] and results['analysis']['szczegolowe']:
                        st.subheader(f"Teren: {results['analysis']['szczegolowe'].get('Oznaczenie Terenu', 'N/A')}")
                        for key, value in results['analysis']['szczegolowe'].items():
                            if key != 'Oznaczenie Terenu': st.markdown(f"**{key}:**"); st.info(f"{value}")