import streamlit as st
import time
import json
import requests
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
from shapely import wkt
from shapely.ops import transform
from streamlit_folium import st_folium
from pyproj import Transformer
from urllib.parse import quote_plus
import platform

st.cache_data.clear()
import os
import config
import modules.geospatial as geospatial
import modules.solar as solar
import modules.visualization as visualization
import modules.mpzp_agent as mpzp_agent

config.setup_tesseract()

try:
    PROJECT_ID = config.setup_gcp_credentials(st.secrets)
    mpzp_agent.init_ai(PROJECT_ID)
except Exception as e:
    st.error(str(e))
    st.info("See Settings → Variables and secrets → Add a new secret")
    st.stop()





def generate_3d_context_view_multiple_parcels(all_parcel_coords_list, map_center_wgs_84, map_style: str):
    try:

        layers, buildings_data = visualization.create_solar_analysis_layers(
            parcel_coords_wgs_84=all_parcel_coords_list,
            map_center_wgs_84=map_center_wgs_84
        )
        
        view_state = pdk.ViewState(
            latitude=map_center_wgs_84[0],
            longitude=map_center_wgs_84[1],
            zoom=17.5,
            pitch=50,
            bearing=0
        )

        return pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style=map_style
        )

    except Exception as e:
        st.error(f"Wystąpił krytyczny błąd podczas generowania modelu 3D: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None



def create_3d_view_with_filled_parcels(combined_polygon, map_center_wgs_84, map_style: str):
    try:
        tags = {"building": True}
        gdf_buildings = ox.features_from_point(
            (map_center_wgs_84[0], map_center_wgs_84[1]), tags, dist=300
        )
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
        
        parcel_data_for_pydeck = []
        
        if hasattr(combined_polygon, 'geoms'):
            for poly in combined_polygon.geoms:
                if poly.geom_type == 'Polygon':
                    exterior_coords = list(poly.exterior.coords)
                    parcel_data_for_pydeck.append({"polygon": [exterior_coords], "height": 1.0})
        elif combined_polygon.geom_type == 'Polygon':
            exterior_coords = list(combined_polygon.exterior.coords)
            parcel_data_for_pydeck.append({"polygon": [exterior_coords], "height": 1.0})
        
        layer_parcel = pdk.Layer("PolygonLayer", data=parcel_data_for_pydeck, get_polygon="polygon",
                               extruded=False, get_elevation="height", filled=True, 
                               get_fill_color=[40, 167, 69, 180],
                               get_line_color=[40, 167, 69, 0], get_line_width=0)
        view_state = pdk.ViewState(latitude=map_center_wgs_84[0], longitude=map_center_wgs_84[1], zoom=17.5, pitch=50, bearing=0)
        layers_to_render = [layer_parcel, layer_buildings] if buildings_data_for_pydeck else [layer_parcel]

        return pdk.Deck(
            layers=layers_to_render,
            initial_view_state=view_state,
            map_style=map_style
        )

    except Exception as e:
        st.error(f"Wystąpił krytyczny błąd podczas generowania modelu 3D: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None


def generate_3d_context_view(parcel_coords_wgs_84, map_center_wgs_84, map_style: str):
    try:
        tags = {"building": True}
        gdf_buildings = ox.features_from_point(
            (map_center_wgs_84[0], map_center_wgs_84[1]), tags, dist=300
        )
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

        return pdk.Deck(
            layers=layers_to_render,
            initial_view_state=view_state,
            map_style=map_style
        )

    except Exception as e:
        st.error(f"Wystąpił krytyczny błąd podczas generowania modelu 3D: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None


def create_trimesh_scene(buildings_data_metric: list) -> trimesh.Scene:
    return solar.create_trimesh_scene(buildings_data_metric)



def generate_sun_path_data(lat: float, lon: float, analysis_date: datetime.date, hour_range: tuple,
                           map_center_metric: tuple):
    return solar.generate_sun_path_geometry(lat, lon, analysis_date, hour_range, map_center_metric)


def generate_complete_sun_path_diagram(lat: float, lon: float, year: int, map_center_metric: tuple):
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

    analemmas = {}
    for hour in range(0, 24):
        year_data = []
        for day_of_year in range(1, 366):
            try:
                date = datetime(year, 1, 1).date() + pd.Timedelta(days=day_of_year-1)
                
                B = 2 * np.pi * (day_of_year - 1) / 365
                declination = 23.45 * np.sin(B)
                declination_rad = np.deg2rad(declination)
                
                equation_of_time = 9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
                
                solar_time = hour + (equation_of_time / 60)
                
                hour_angle = 15 * (solar_time - 12)
                hour_angle_rad = np.deg2rad(hour_angle)
                
                lat_rad = np.deg2rad(lat)
                
                sin_alt = (np.sin(lat_rad) * np.sin(declination_rad) +
                          np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad))
                elevation = np.arcsin(sin_alt)
                elevation_deg = np.rad2deg(elevation)

                cos_az = (np.sin(declination_rad) - np.sin(lat_rad) * sin_alt) / (np.cos(lat_rad) * np.cos(elevation))
                cos_az = np.clip(cos_az, -1, 1)
                azimuth_rad = np.arccos(cos_az)
                
                if np.sin(hour_angle_rad) >= 0:
                    azimuth = 360 - np.rad2deg(azimuth_rad)
                else:
                    azimuth = np.rad2deg(azimuth_rad)
                
                if elevation_deg > 0:
                    alt_rad = elevation
                    az_rad = np.deg2rad(azimuth)
                    x_offset = path_radius * np.cos(alt_rad) * np.sin(az_rad)
                    y_offset = path_radius * np.cos(alt_rad) * np.cos(az_rad)
                    z = path_radius * np.sin(alt_rad)
                    year_data.append({
                        'day': day_of_year,
                        'coords': [center_x + x_offset, center_y + y_offset, z],
                        'elevation': elevation_deg
                    })
            except:
                continue
        if len(year_data) > 10:
            analemmas[hour] = sorted(year_data, key=lambda x: x['day'])

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
            label = f"{cardinal_directions[azimuth_deg]} ({azimuth_deg} deg)"
        else:
            label = f"{azimuth_deg} deg"

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
    return solar.create_analysis_grid(parcel_polygon, density)


import modules.solar as solar
from modules.lidar_service import LidarService

# @st.cache_data
def run_solar_simulation(
        _buildings_data_metric_tuple: tuple,
        grid_points_metric: np.ndarray,
        lat: float, lon: float,
        analysis_date: datetime.date,
        hour_range: tuple,
        freq: str = "1H",
        use_lidar: bool = False,
        lidar_bbox: tuple = None,
        target_parcel_geometry = None
) -> np.ndarray:
    
    print(f"DEBUG TRACER: Wszedłem do run_solar_simulation. use_lidar={use_lidar}, freq={freq}", flush=True)

    if freq == "15min":
        time_step_weight = 0.25
    elif freq == "30min":
        time_step_weight = 0.5
    else:
        time_step_weight = 1.0

    if use_lidar and lidar_bbox:
        print("DEBUG TRACER: Jestem w bloku LiDAR", flush=True)
        width_m = lidar_bbox[2] - lidar_bbox[0]
        height_m = lidar_bbox[3] - lidar_bbox[1]
        print(f"DEBUG: BBOX Dimensions: {width_m:.2f}m x {height_m:.2f}m")
        
        if width_m > 2000 or height_m > 2000:
            st.error(f"Zbyt duży obszar analizy: {width_m:.0f}x{height_m:.0f}m. Zmniejsz bufor.")
            return None

        try:
            lidar_service = LidarService()
            dsm_data, transform = lidar_service.get_dsm_data(lidar_bbox)
            
            dtm_data, dtm_transform = lidar_service.get_dtm_data(lidar_bbox)

            min_elevation = np.nanmin(dtm_data)
            dsm_data = dsm_data - min_elevation
            dtm_data = dtm_data - min_elevation
            print(f"DEBUG: Normalizacja terenu. Odejmuję offset wysokościowy: {min_elevation:.2f}m", flush=True)
            
            parcel_geoms = []
            if target_parcel_geometry:
                 parcel_geoms.append(target_parcel_geometry)
            elif st.session_state.selected_parcels:
                for p_data in st.session_state.selected_parcels:
                    if 'Geometria' in p_data:
                        try:
                            geom = wkt.loads(p_data['Geometria'])
                            parcel_geoms.append(geom)
                        except Exception:
                            pass
            
            print(f"DEBUG TRACER: Wywołuję flatten. Geometria dostępna? {bool(parcel_geoms)}", flush=True)
            
            dsm_for_calc = dsm_data.copy()
            dsm_for_viz = dsm_data.copy()
            dtm_for_viz = dtm_data.copy()

            if parcel_geoms:
                dsm_for_calc = lidar_service.flatten_dsm_on_parcel(dsm_for_calc, dtm_data, transform, parcel_geoms, fill_with_nan=False)
                
                dsm_for_viz = lidar_service.flatten_dsm_on_parcel(dsm_for_viz, dtm_data, transform, parcel_geoms, fill_with_nan=True)
                dtm_for_viz = lidar_service.flatten_dsm_on_parcel(dtm_for_viz, dtm_data, transform, parcel_geoms, fill_with_nan=True)
            
            lidar_layers = []
            
            lines_layer = visualization.create_lidar_lines_layer(dsm_for_viz, dtm_for_viz, transform, subsample=3)
            if lines_layer:
                lidar_layers.append(lines_layer)

            point_cloud_layer = visualization.create_lidar_point_cloud_layer(dsm_for_viz, transform, subsample=2)
            if point_cloud_layer:
                lidar_layers.append(point_cloud_layer)
                
            st.session_state['lidar_point_cloud_layer'] = lidar_layers
            
            scene = lidar_service.convert_dsm_to_trimesh(dsm_for_calc, transform)
            
            z_values = lidar_service.sample_height_for_points(dtm_data, dtm_transform, grid_points_metric[:, :2])
            
            grid_points_metric[:, 2] = z_values + 0.5
            
        except Exception as e:
            st.error(f"Błąd pobierania danych LiDAR: {e}. Przełączam na tryb OSM.")
            buildings_data_metric = [
                {'polygon': b_tuple[0], 'height': b_tuple[1]} for b_tuple in _buildings_data_metric_tuple
            ]
            scene = solar.create_trimesh_scene(buildings_data_metric)
    else:
        buildings_data_metric = [
            {'polygon': b_tuple[0], 'height': b_tuple[1]} for b_tuple in _buildings_data_metric_tuple
        ]
        scene = solar.create_trimesh_scene(buildings_data_metric)
    
    if scene.is_empty:
        st.warning("Nie udało się wygenerować geometrii 3D otoczenia. Analiza pokazuje nasłonecznienie bez uwzględnienia cieni.")
    
    sun_positions = solar.calculate_sun_positions(lat, lon, analysis_date, hour_range, freq=freq)
    
    progress_placeholder = st.empty()
    
    def update_progress(current_step, total_steps):
        dots_html = ""
        for i in range(total_steps):
            color = "#FFD700" if i < current_step else "#BDB76B"
            box_shadow = "0 0 5px #FFD700" if i < current_step else "none"
            dots_html += f'<div style="width: 10px; height: 10px; background-color: {color}; border-radius: 50%; box-shadow: {box_shadow}; transition: all 0.3s ease;"></div>'
        
        container_html = f'<div style="display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 10px 0; margin-bottom: 20px;">{dots_html}</div>'
        progress_placeholder.markdown(container_html, unsafe_allow_html=True)

    update_progress(0, len(sun_positions))

    sunlit_hours = solar.calculate_shadows(
        scene, 
        grid_points_metric, 
        sun_positions, 
        time_step_weight=time_step_weight,
        progress_callback=update_progress
    )
    
    progress_placeholder.empty()
    
    return sunlit_hours






st.set_page_config(layout="wide");

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html {
        scroll-behavior: smooth;
        scroll-snap-type: y mandatory;
        scroll-padding: 0;
    }

    body {
        scroll-snap-type: y mandatory;
        overflow-y: scroll;
    }

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

    /* Responsive h1 styling for mobile devices */
    @media (max-width: 768px) {
        h1 {
            font-size: 2.5rem !important;
        }
    }

    @media (max-width: 480px) {
        h1 {
            font-size: 2rem !important;
        }
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

// CUSTOM IMMERSIVE CURSOR - Enhanced for Streamlit compatibility
function initCustomCursor() {
    // Remove any existing cursor if present
    const existingCursor = document.getElementById('custom-cursor');
    if (existingCursor) {
        existingCursor.remove();
    }
    
    // Create custom cursor element
    const cursor = document.createElement('div');
    cursor.id = 'custom-cursor';
    cursor.style.cssText = `
        position: fixed;
        width: 20px;
        height: 20px;
        border: 2px solid #42a5f5;
        border-radius: 50%;
        pointer-events: none;
        transform: translate(-50%, -50%);
        z-index: 999999 !important;
        transition: width 0.1s ease, height 0.1s ease, border-color 0.3s ease;
        mix-blend-mode: difference;
        background-color: transparent;
    `;
    document.body.appendChild(cursor);
    
    // Update cursor position
    document.addEventListener('mousemove', (e) => {
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
        
        // Dynamic color based on position (blue-green gradient)
        const x = e.clientX / window.innerWidth;
        const r = Math.floor(66 + (106 - 66) * x); // 42a5f5 blue component range
        const g = Math.floor(165 + (186 - 165) * x); // 42a5f5 to 66bb6a green component range
        const b = Math.floor(245 + (106 - 245) * x); // 42a5f5 to 66bb6a blue component range
        cursor.style.borderColor = `rgb(${r}, ${g}, ${b})`;
    });
    
    // Click effect - shrink the cursor
    document.addEventListener('mousedown', () => {
        cursor.style.width = '12px';
        cursor.style.height = '12px';
    });
    
    document.addEventListener('mouseup', () => {
        cursor.style.width = '20px';
        cursor.style.height = '20px';
    });
    
    // Hide default cursor using more specific selectors for Streamlit
    const hideCursorStyle = document.createElement('style');
    hideCursorStyle.innerHTML = `
        body, body *, button, button *, input, input *, select, select *, textarea, textarea *, 
        div, div *, span, span *, a, a *, label, label *, p, p *, h1, h2, h3, h4, h5, h6, 
        .stButton, .stButton button, .stSelectbox, .stTextInput, .stNumberInput, 
        .stSlider, .stCheckbox, .stRadio, .stTextArea, .stDataFrame, 
        [data-testid="stElementToolbar"], .element-container, .main, .block-container {
            cursor: none !important;
        }
    `;
    document.head.appendChild(hideCursorStyle);
}

// Initialize cursor after page is fully loaded
document.addEventListener('DOMContentLoaded', initCustomCursor);
// Also try to initialize after a short delay for Streamlit components
setTimeout(initCustomCursor, 1000);
// And again after another delay to ensure all components are loaded
setTimeout(initCustomCursor, 2000);
</script>
""", unsafe_allow_html=True)

for key in ['map_center', 'parcel_data', 'selected_parcels', 'analysis_results', 'show_search', 'confirming_selection', 'show_3d', 'selected_analysis']:
    if key not in st.session_state:
        if key == 'selected_parcels':
            st.session_state[key] = []
        elif key in ['confirming_selection', 'show_3d']:
            st.session_state[key] = False
        elif key == 'selected_analysis':
            st.session_state[key] = None
        else:
            st.session_state[key] = None if key != 'show_search' else False

if not st.session_state.show_search and not st.session_state.map_center:
    st.markdown("""
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 85vh; text-align: center;">
        <h1 style="font-size: 4rem; margin-bottom: 1rem; background: linear-gradient(135deg, #42a5f5 0%, #66bb6a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; line-height: 1.2;">
            Asystent Analizy Działki
        </h1>
        <p style="font-size: 1.3rem; color: #424242; margin-bottom: 0.5rem; font-weight: 500;">Szczecin/Polska • Wersja Beta 0.2.2</p>
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

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Rozpocznij", key="start_button", use_container_width=True):
            st.session_state.show_search = True
            st.rerun()

if st.session_state.show_search or st.session_state.map_center:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size: 4rem; margin: 0; background: linear-gradient(135deg, #42a5f5 0%, #66bb6a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; line-height: 1.2;">
            Asystent Analizy Działki
        </h1>
        <p style="font-size: 1rem; color: #616161; margin: 0.5rem 0 0 0;">Szczecin • Wersja Beta 0.2.2</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form(key="address_form"):
        address_input = st.text_input("Wpisz adres lub współrzędne:", "", label_visibility="collapsed", placeholder="Wpisz adres, np. Kolumba 64, Szczecin")
        submitted = st.form_submit_button("Wyszukaj działkę", use_container_width=True)

    if submitted:
        st.session_state.parcel_data = None;
        st.session_state.analysis_results = None
        with st.spinner("Pobieram współrzędne..."):
            coords, error = geospatial.geocode_address_to_coords(address_input)
            if error:
                st.error(error); st.session_state.map_center = None
            else:
                st.session_state.map_center = coords

    if st.session_state.map_center:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h2 style="background: linear-gradient(135deg, #42a5f5 0%, #66bb6a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;">
                Wybierz działki na mapie
            </h2>
            <p style="color: #616161; font-size: 1rem;">Kliknij na działki, aby je zaznaczyć/odznaczyć. Wybierz analizę z dołu strony.</p>
        </div>
        """, unsafe_allow_html=True)

        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=st.session_state.map_center, zoom_start=18)
        folium.TileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Satelita', overlay=True).add_to(m)
        folium.WmsTileLayer(url="https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow",
                            layers="dzialki,numery_dzialek", transparent=True, fmt="image/png",
                            name="Działki Ewidencyjne").add_to(m)
        folium.LayerControl().add_to(m)

        m = folium.Map(location=st.session_state.map_center, zoom_start=18)
        folium.TileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Satelita', overlay=True).add_to(m)
        folium.WmsTileLayer(url="https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow",
                            layers="dzialki,numery_dzialek", transparent=True, fmt="image/png",
                            name="Działki Ewidencyjne").add_to(m)
        folium.LayerControl().add_to(m)

        for parcel in st.session_state.selected_parcels:
            coords_wgs84 = geospatial.transform_coordinates_to_wgs84(parcel["Współrzędne EPSG:2180"])
            folium.Polygon(locations=coords_wgs84, color='#28a745', fill=True, fillColor='#28a745',
                           fill_opacity=0.5, weight=3, tooltip=parcel['ID Działki']).add_to(m)

        map_key = f"parcel_map_{len(st.session_state.selected_parcels)}"
        map_data = st_folium(m, use_container_width=True, height=700, key=map_key)

        if map_data and map_data.get("last_clicked"):
            lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            with st.spinner(f"Identyfikuję działkę..."):
                parcel_data, error = geospatial.get_parcel_from_coords(lat, lon)
                if error:
                    st.error(error)
                else:
                    existing_index = None
                    for i, selected_parcel in enumerate(st.session_state.selected_parcels):
                        if selected_parcel['ID Działki'] == parcel_data['ID Działki']:
                            existing_index = i
                            break

                    if existing_index is not None:
                        st.session_state.selected_parcels.pop(existing_index)
                        st.info(f"Działka {parcel_data['ID Działki']} odznaczona")
                    else:
                        st.session_state.selected_parcels.append(parcel_data)
                        st.success(f"Działka {parcel_data['ID Działki']} zaznaczona")
                    
                    coords_2180 = parcel_data["Współrzędne EPSG:2180"]
                    if coords_2180 and len(coords_2180) > 0:
                        avg_x = sum(p[0] for p in coords_2180) / len(coords_2180)
                        avg_y = sum(p[1] for p in coords_2180) / len(coords_2180)
                        center_lon, center_lat = geospatial.transform_single_coord(avg_x, avg_y, "2180", "4326")
                        st.session_state.map_center = (center_lat, center_lon)
                    
                    st.rerun()

        if st.session_state.selected_parcels:
            parcel_ids = [parcel['ID Działki'] for parcel in st.session_state.selected_parcels]
            parcel_list_str = ", ".join(parcel_ids)
            
            st.markdown(f"""
            <div style="text-align: center; margin: 1.5rem 0; padding: 1rem; background: rgba(40, 167, 69, 0.15); border-radius: 12px; border: 2px solid rgba(40, 167, 69, 0.3);">
                <p style="font-size: 1.1rem; font-weight: 600; color: #28a745; margin: 0; font-family: monospace;">{parcel_list_str}</p>
            </div>
            """, unsafe_allow_html=True)


    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    if st.session_state.selected_parcels:
        show_3d_context = st.button(
            "Wygeneruj widok 3D otoczenia",
            key="generate_3d_button",
            use_container_width=True,
            help="Generowanie widoku 3D może zająć kilka sekund"
        )

        if 'show_3d' not in st.session_state:
            st.session_state.show_3d = False

        if show_3d_context:
            st.session_state.show_3d = not st.session_state.show_3d

        if st.session_state.show_3d:
            st.markdown("""<div style="height: 2px; background: linear-gradient(90deg, transparent, #42a5f5, transparent); margin: 3rem 0 2rem 0; opacity: 0.5;"></div>""", unsafe_allow_html=True)

            all_coords = []
            all_coords_wgs84 = []
            for parcel in st.session_state.selected_parcels:
                coords_2180 = parcel["Współrzędne EPSG:2180"]
                coords_wgs84_single = geospatial.transform_coordinates_to_wgs84(coords_2180)
                all_coords.extend(coords_2180)
                all_coords_wgs84.extend(coords_wgs84_single)
            
            avg_x = sum(p[0] for p in all_coords) / len(all_coords)
            avg_y = sum(p[1] for p in all_coords) / len(all_coords)
            map_center_lon, map_center_lat = geospatial.transform_single_coord(avg_x, avg_y, "2180", "4326")
            map_center = (map_center_lat, map_center_lon)

            selected_map_style = "light"
            with st.spinner("Generuję model 3D otoczenia..."):
                all_parcel_coords_list = []
                for parcel in st.session_state.selected_parcels:
                    coords_wgs84_single = geospatial.transform_coordinates_to_wgs84(parcel["Współrzędne EPSG:2180"])
                    if len(coords_wgs84_single) > 0:
                        first_point = coords_wgs84_single[0]
                        last_point = coords_wgs84_single[-1]
                        if first_point != last_point:
                            coords_closed = coords_wgs84_single + [first_point]
                        else:
                            coords_closed = coords_wgs84_single
                        single_parcel_coords = [(p[1], p[0]) for p in coords_closed]
                        all_parcel_coords_list.append(single_parcel_coords)
                
                if all_parcel_coords_list:
                    deck_3d_view = generate_3d_context_view_multiple_parcels(
                        all_parcel_coords_list, map_center, map_style=selected_map_style
                    )
                else:
                    deck_3d_view = generate_3d_context_view(
                        [], map_center, map_style=selected_map_style
                    )
                if deck_3d_view:
                    st.markdown("""
                    <div style="background: rgba(66, 165, 245, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
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

        if st.session_state.selected_parcels:
            st.markdown("""
            <div style="text-align: center; margin: 10rem 0 3rem 0;">
                <h2 style="font-size: 2.2rem; margin-bottom: 0.5rem;">Wybierz typ analizy</h2>
                <p style="color: #616161; font-size: 1.05rem;">Kliknij jedną z opcji, aby rozpocząć szczegółową analizę</p>
            </div>
            """, unsafe_allow_html=True)
            
            analysis_col1, analysis_col2 = st.columns(2, gap="large")

            if 'selected_analysis' not in st.session_state:
                st.session_state.selected_analysis = None

            with analysis_col1:
                st.markdown("""
                <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, rgba(255,193,7,0.08) 0%, rgba(255,152,0,0.08) 100%); border-radius: 20px; border: 2px solid rgba(255,193,7,0.25); min-height: 350px; display: flex; flex-direction: column; justify-content: center; transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 16px rgba(255,193,7,0.15)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                    <h3 style="font-size: 1.8rem; margin-bottom: 1.5rem; color: #424242;">Analiza Nasłonecznienia</h3>
                    <p style="color: #616161; font-size: 1rem; margin-bottom: 0; line-height: 1.6;">Oblicza średnią dzienną liczbę godzin słońca dla każdego punktu działki, uwzględniając cienie sąsiednich budynków</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    if st.button("Wybierz", key="select_solar", use_container_width=True):
                        st.session_state.selected_analysis = "solar"

            with analysis_col2:
                st.markdown("""
                <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, rgba(33,150,243,0.08) 0%, rgba(25,118,210,0.08) 100%); border-radius: 20px; border: 2px solid rgba(33,150,243,0.25); min-height: 350px; display: flex; flex-direction: column; justify-content: center; transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 16px rgba(33,150,243,0.15)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                    <h3 style="font-size: 1.8rem; margin-bottom: 1.5rem; color: #424242;">Analiza MPZP</h3>
                    <p style="color: #616161; font-size: 1rem; margin-bottom: 0; line-height: 1.6;">Inteligentna analiza dokumentów planistycznych z wykorzystaniem AI (Gemini 2.5 Pro)</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    if st.button("Wybierz", key="select_mpzp", use_container_width=True):
                        st.session_state.selected_analysis = "mpzp"

        if st.session_state.selected_analysis == "solar":
            st.markdown("""<div style="height: 2px; background: linear-gradient(90deg, transparent, #FFC107, transparent); margin: 3rem 0 2rem 0; opacity: 0.6;"></div>""", unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="font-size: 2rem;">Analiza Nasłonecznienia</h2>
                <p style="color: #616161; font-size: 1rem;">Skonfiguruj parametry analizy</p>
            </div>
            """, unsafe_allow_html=True)

            data_source = st.radio(
                "Źródło danych 3D:",
                options=["OSM (Budynki)", "LiDAR (Geoportal)"],
                index=0,
                horizontal=True,
                help="Wybierz źródło danych do analizy cienia. OSM jest szybsze, ale mniej dokładne. LiDAR uwzględnia teren, drzewa i kształty dachów."
            )
            
            today = datetime(2025, 1, 1).date()
            selected_date_range = st.date_input(
                "Wybierz dzień lub zakres dni analizy:",
                value=(today.replace(month=3, day=20), today.replace(month=3, day=20)),
            )
            hour_range = st.slider(
                "Wybierz zakres godzin analizy:",
                min_value=0, max_value=23, value=(6, 20), step=1
            )

            sampling_freq_label = st.radio(
                "Dokładność próbkowania (Interwał):",
                options=["1 godzina", "30 min", "15 min"],
                index=0,
                horizontal=True
            )
            
            freq_map = {
                "1 godzina": "1H",
                "30 min": "30min",
                "15 min": "15min"
            }
            sampling_freq = freq_map[sampling_freq_label]

            if st.button("Uruchom analizę nasłonecznienia", key="run_solar_analysis", use_container_width=True):
                start_date, end_date = selected_date_range
                if start_date is None or end_date is None or start_date > end_date:
                    st.error("Proszę wybrać poprawny zakres dat.")
                else:
                    if not st.session_state.selected_parcels:
                        st.error("Nie wybrano działek do analizy.")
                        st.rerun()
                    
                    from shapely.geometry import Polygon as ShapelyPolygon
                    from shapely.ops import unary_union
                    import numpy as np
                    
                    shapely_polygons = []
                    for parcel in st.session_state.selected_parcels:
                        coords_2180 = parcel["Współrzędne EPSG:2180"]
                        if len(coords_2180) >= 3:
                            try:
                                poly = ShapelyPolygon(coords_2180)
                                if poly.is_valid:
                                    shapely_polygons.append(poly)
                                else:
                                    fixed_poly = poly.buffer(0)
                                    if not fixed_poly.is_empty:
                                        shapely_polygons.append(fixed_poly)
                            except:
                                continue
                    
                    if shapely_polygons:
                        if len(shapely_polygons) == 1:
                            combined_parcel_polygon = shapely_polygons[0]
                        else:
                            combined_parcel_polygon = unary_union(shapely_polygons)
                        
                        if hasattr(combined_parcel_polygon, 'exterior'):
                            combined_coords = list(combined_parcel_polygon.exterior.coords)
                        else:
                            if hasattr(combined_parcel_polygon, 'geoms'):
                                largest_poly = max(combined_parcel_polygon.geoms, key=lambda p: p.area)
                                combined_coords = list(largest_poly.exterior.coords)
                            else:
                                combined_coords = list(shapely_polygons[0].exterior.coords)
                        
                        combined_parcel_data = {
                            "ID Działki": f"Połączone ({len(st.session_state.selected_parcels)} działek)",
                            "Współrzędne EPSG:2180": combined_coords
                        }
                        st.session_state.parcel_data = combined_parcel_data
                    else:
                        primary_parcel = st.session_state.selected_parcels[0]
                        st.session_state.parcel_data = primary_parcel
                    
                    num_days = (end_date - start_date).days + 1
                    num_hours = hour_range[1] - hour_range[0] + 1
                    spinner_text = f"Przeprowadzam symulację dla {num_days} {'dzień' if num_days == 1 else 'dni'}, {num_hours} {'godzina' if num_hours == 1 else 'godzin'} (godz. {hour_range[0]}:00-{hour_range[1]}:00)..."
                    with st.spinner(spinner_text):

                        if st.session_state.parcel_data and "Współrzędne EPSG:2180" in st.session_state.parcel_data:
                            coords_2180 = st.session_state.parcel_data["Współrzędne EPSG:2180"]
                            if coords_2180 and len(coords_2180) > 0:
                                avg_x = sum(p[0] for p in coords_2180) / len(coords_2180)
                                avg_y = sum(p[1] for p in coords_2180) / len(coords_2180)
                                center_lon, center_lat = geospatial.transform_single_coord(avg_x, avg_y, "2180", "4326")
                                analysis_map_center = (center_lat, center_lon)
                            else:
                                if 'map_center' in locals() or 'map_center' in globals():
                                    analysis_map_center = map_center
                                else:
                                    if st.session_state.selected_parcels:
                                        first_parcel_coords = st.session_state.selected_parcels[0]["Współrzędne EPSG:2180"]
                                        avg_x = sum(p[0] for p in first_parcel_coords) / len(first_parcel_coords)
                                        avg_y = sum(p[1] for p in first_parcel_coords) / len(first_parcel_coords)
                                        center_lon, center_lat = geospatial.transform_single_coord(avg_x, avg_y, "2180", "4326")
                                        analysis_map_center = (center_lat, center_lon)
                                    else:
                                        analysis_map_center = (53.4285, 14.5511)
                        else:
                            if st.session_state.selected_parcels:
                                first_parcel_coords = st.session_state.selected_parcels[0]["Współrzędne EPSG:2180"]
                                avg_x = sum(p[0] for p in first_parcel_coords) / len(first_parcel_coords)
                                avg_y = sum(p[1] for p in first_parcel_coords) / len(first_parcel_coords)
                                center_lon, center_lat = geospatial.transform_single_coord(avg_x, avg_y, "2180", "4326")
                                analysis_map_center = (center_lat, center_lon)
                            else:
                                analysis_map_center = (53.4285, 14.5511)

                        gdf_buildings_wgs84 = ox.features_from_point((analysis_map_center[0], analysis_map_center[1]), {"building": True},
                                                                     dist=350)
                        gdf_buildings_metric = gdf_buildings_wgs84.to_crs("epsg:2180")

                        buildings_data_metric = []


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

                        if st.session_state.parcel_data and "Współrzędne EPSG:2180" in st.session_state.parcel_data:
                            coords_2180 = st.session_state.parcel_data["Współrzędne EPSG:2180"]
                            parcel_poly_2180 = Polygon(coords_2180)
                            grid_points_2180 = create_analysis_grid(parcel_poly_2180, density=1.0)
                        else:
                            if st.session_state.selected_parcels:
                                coords_2180 = st.session_state.selected_parcels[0]["Współrzędne EPSG:2180"]
                                parcel_poly_2180 = Polygon(coords_2180)
                                grid_points_2180 = create_analysis_grid(parcel_poly_2180, density=1.0)
                            else:
                                st.error("Nie wybrano działki do analizy.")
                                st.session_state.solar_analysis_results = None
                                st.rerun()

                        if grid_points_2180.size > 0:
                            buildings_data_for_cache = tuple(
                                (tuple(b['polygon']), b['height']) for b in buildings_data_metric
                            )

                            total_sunlit_hours = np.zeros(len(grid_points_2180))
                            date_range = pd.date_range(start_date, end_date)

                            total_sunlit_hours = np.zeros(len(grid_points_2180))
                            date_range = pd.date_range(start_date, end_date)

                            lidar_bbox = None
                            use_lidar = (data_source == "LiDAR (Geoportal)")
                            
                            if use_lidar:
                                minx, miny, maxx, maxy = parcel_poly_2180.bounds
                                buffer = 200
                                lidar_bbox = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

                            for single_date in date_range:
                                total_sunlit_hours += run_solar_simulation(
                                    buildings_data_for_cache,
                                    grid_points_2180,
                                    analysis_map_center[0], analysis_map_center[1], single_date,
                                    hour_range,
                                    freq=sampling_freq,
                                    use_lidar=use_lidar,
                                    lidar_bbox=lidar_bbox,
                                    target_parcel_geometry=parcel_poly_2180 if use_lidar else None
                                )

                            average_sunlit_hours = total_sunlit_hours / len(date_range)
                            transformer_to_wgs = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
                            grid_points_wgs84 = np.array(
                                [(*transformer_to_wgs.transform(p[0], p[1]), p[2]) for p in grid_points_2180])
                            results_df = pd.DataFrame(grid_points_wgs84, columns=['lon', 'lat', 'z'])
                            results_df['sun_hours'] = average_sunlit_hours
                            viz_date = date_range[len(date_range) // 2]
                            map_center_metric = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True).transform(
                                analysis_map_center[1], analysis_map_center[0])

                            sun_paths, analemmas, azimuth_markers, azimuth_lines = generate_complete_sun_path_diagram(
                                analysis_map_center[0], analysis_map_center[1], viz_date.year, map_center_metric
                            )

                            sun_position_markers = []
                            location = pvlib.location.Location(analysis_map_center[0], analysis_map_center[1], tz='Europe/Warsaw')

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
                                                                       "sun_position_markers": sun_position_markers,
                                                                       "analysis_map_center": analysis_map_center,
                                                                       "data_source": data_source}
                        else:
                            st.session_state.solar_analysis_results = None
                st.rerun()

            if 'solar_analysis_results' in st.session_state and st.session_state.solar_analysis_results:
                data = st.session_state.solar_analysis_results
                if not data["results_df"].empty:
                    results_df = data["results_df"]
                    results_df = results_df.rename(columns={'sun_hours': 'value'})
                    
                    min_h, max_h = results_df['value'].min(), results_df['value'].max()
                    if max_h == min_h: max_h += 1.0

                    parcel_coords = []
                    if st.session_state.selected_parcels:

                        for p_data in st.session_state.selected_parcels:
                            coords_2180 = p_data['Współrzędne EPSG:2180']
                            p_coords_wgs = geospatial.transform_coordinates_to_wgs84(coords_2180)
                            parcel_coords.append(p_coords_wgs)
                    
                    is_lidar = data.get('data_source') == "LiDAR (Geoportal)"
                    
                    layers, _ = visualization.create_solar_analysis_layers(
                        parcel_coords_wgs_84=parcel_coords,
                        map_center_wgs_84=data['analysis_map_center'],
                        solar_results=results_df,
                        grid_points_metric=None,
                        sun_path_data=data['sun_paths'],
                        analemma_data=data['analemmas'],
                        azimuth_data=(data['azimuth_markers'], data['azimuth_lines']),
                        show_buildings=not is_lidar
                    )
                    
                    sun_positions_wgs84 = []
                    transformer_to_wgs = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
                    for sp in data['sun_position_markers']:
                        pos_wgs = list(transformer_to_wgs.transform(sp['position'][0], sp['position'][1]))
                        sun_positions_wgs84.append({
                            "position": [pos_wgs[0], pos_wgs[1], sp['position'][2]]
                        })
                    
                    sun_markers_layer = pdk.Layer("ScatterplotLayer", data=sun_positions_wgs84,
                                                 get_position="position",
                                                 get_radius=12, filled=True,
                                                 get_fill_color=[255, 223, 0, 255],
                                                 stroked=False, billboard=True)
                    layers.append(sun_markers_layer)

                    if is_lidar and 'lidar_point_cloud_layer' in st.session_state and st.session_state['lidar_point_cloud_layer']:
                        lidar_content = st.session_state['lidar_point_cloud_layer']
                        if isinstance(lidar_content, list):
                            for layer in reversed(lidar_content):
                                layers.insert(0, layer)
                        else:
                            layers.insert(0, lidar_content)

                    legend_html = visualization.create_discrete_legend_html(min_h, max_h, colormap='plasma')
                    if legend_html:
                        legend_html = legend_html.replace('\n', ' ')
                        st.markdown(legend_html, unsafe_allow_html=True)

                    display_map_center = data.get('analysis_map_center', (53.4285, 14.5511))
                    
                    r = pdk.Deck(layers=layers,
                                 initial_view_state=pdk.ViewState(latitude=display_map_center[0], longitude=display_map_center[1],
                                                                  zoom=17.5, pitch=50, bearing=0),
                                 map_style=None)

                    st.pydeck_chart(r, use_container_width=True, height=600)
                else:
                    st.warning("Nie udało się stworzyć siatki analitycznej dla tej działki.")




        elif st.session_state.selected_analysis == "mpzp":
            st.markdown("""<div style="height: 2px; background: linear-gradient(90deg, transparent, #2196F3, transparent); margin: 3rem 0 2rem 0; opacity: 0.6;"></div>""", unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="font-size: 2rem;">Analiza MPZP (Agent AI)</h2>
                <p style="color: #616161; font-size: 1rem;">Agent nawiguje po geoportalu i analizuje dokumenty planistyczne</p>
            </div>
            """, unsafe_allow_html=True)

            if 'mpzp_analysis_started' not in st.session_state:
                st.session_state.mpzp_analysis_started = False

            if not st.session_state.mpzp_analysis_started:
                start_btn = st.button("Rozpocznij analizę AI", key="run_mpzp_analysis", use_container_width=True)
                if start_btn:
                    st.session_state.mpzp_analysis_started = True

            if st.session_state.mpzp_analysis_started and not st.session_state.get('analysis_results'):
                st.info("Agent AI uruchomiony - pobieram dokumenty i analizuję...")
                try:
                    if st.session_state.selected_parcels:
                        primary_parcel_id = st.session_state.selected_parcels[0]['ID Działki']
                        
                        def status_callback(type, message):
                            if type == "info":
                                st.info(message)
                            elif type == "success":
                                st.success(message)
                            elif type == "warning":
                                st.warning(message)
                            elif type == "error":
                                st.error(message)
                            else:
                                st.write(message)
                        
                        results = mpzp_agent.run_ai_agent_flow(primary_parcel_id, status_callback=status_callback)
                        
                        if results:
                            st.session_state.analysis_results = results
                            st.success("Analiza zakończona!")
                        else:
                            st.error("Nie udało się pobrać wyników analizy.")
                            st.session_state.mpzp_analysis_started = False
                    else:
                        st.error("Nie wybrano żadnych działek do analizy.")
                        st.session_state.mpzp_analysis_started = False
                except Exception as e:
                    st.error(f"Błąd podczas analizy: {str(e)}")
                    st.session_state.mpzp_analysis_started = False

            if st.session_state.get('analysis_results'):
                results = st.session_state.analysis_results
                if 'analysis' in results and results['analysis']:
                    st.success("Misja Agenta Analityka zakończona!")
                    if 'ogolne' in results['analysis'] and results['analysis']['ogolne']:
                        st.markdown(f"**Cel Planu:**");
                        st.info(f"{results['analysis']['ogolne'].get('Cel Planu', 'Brak danych.')}")
                    if 'szczegolowe' in results['analysis'] and results['analysis']['szczegolowe']:
                        st.subheader(f"Teren: {results['analysis']['szczegolowe'].get('Oznaczenie Terenu', 'N/A')}")
                        for key, value in results['analysis']['szczegolowe'].items():
                            if key != 'Oznaczenie Terenu': st.markdown(f"**{key}:**"); st.info(f"{value}")