import numpy as np
import pandas as pd
import pvlib
import trimesh
import psutil
import os
import gc
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from datetime import datetime

def calculate_sun_positions(lat: float, lon: float, date: datetime.date, hour_range: tuple, freq: str = "1H", tz='Europe/Warsaw'):
    start_hour, end_hour = hour_range
    times = pd.date_range(
        start=f"{date} {start_hour:02d}:00",
        end=f"{date} {end_hour:02d}:00",
        freq=freq,
        tz=tz
    )
    location = pvlib.location.Location(lat, lon, tz=tz)
    solar_position = location.get_solarposition(times)
    return solar_position[solar_position['apparent_elevation'] > 0]

def create_trimesh_scene(buildings_data_metric: list) -> trimesh.Scene:
    scene = trimesh.Scene()
    
    for building_dict in buildings_data_metric:
        try:
            coords = building_dict['polygon']
            if len(coords) > 1:
                first = np.array(coords[0]) if not isinstance(coords[0], np.ndarray) else coords[0]
                last = np.array(coords[-1]) if not isinstance(coords[-1], np.ndarray) else coords[-1]
                if np.allclose(first, last, rtol=1e-9):
                    coords = coords[:-1]

            if len(coords) < 3:
                continue

            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or not poly.is_valid or poly.area < 1.0:
                continue

            height = building_dict['height']

            try:
                mesh = trimesh.creation.extrude_polygon(poly, height=height)
                if mesh is None or len(mesh.faces) == 0:
                    continue
            except Exception:
                continue

            if not mesh.is_watertight:
                try:
                    trimesh.repair.fix_normals(mesh)
                    trimesh.repair.fill_holes(mesh)
                except Exception:
                    pass

            if len(mesh.faces) > 0 and len(mesh.vertices) > 0:
                scene.add_geometry(mesh)

        except Exception:
            continue

    return scene

def log_mem(tag):
    gc.collect()
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    print(f"DEBUG_MEM [{tag}]: {mem:.2f} GB", flush=True)

def calculate_shadows(scene: trimesh.Scene, grid_points: np.ndarray, sun_positions: pd.DataFrame, time_step_weight: float = 1.0, progress_container=None) -> np.ndarray:
    log_mem("Start calculate_shadows")

    if scene.is_empty:
        return np.full(len(grid_points), len(sun_positions) * time_step_weight)

    if isinstance(scene, trimesh.Scene):
        combined_mesh = scene.dump(concatenate=True)
    elif isinstance(scene, trimesh.Trimesh):
        combined_mesh = scene
    else:
        raise ValueError(f"Nieobsługiwany typ geometrii: {type(scene)}")

    if not isinstance(combined_mesh, trimesh.Trimesh):
        return np.full(len(grid_points), len(sun_positions) * time_step_weight)

    print(f"DEBUG_STATS: Mesh Faces: {len(combined_mesh.faces)}", flush=True)
    print(f"DEBUG_STATS: Grid Points: {len(grid_points)}", flush=True)
    print(f"DEBUG_STATS: Sun Positions: {len(sun_positions)}", flush=True)
    print(f"DEBUG_STATS: Total Rays: {len(grid_points) * len(sun_positions)}", flush=True)
    
    log_mem("Mesh prepared")

    sunlit_hours = np.zeros(len(grid_points))
    max_ray_distance = 500.0

    intersector = combined_mesh.ray
    log_mem("Intersector initialized")

    batch_size = 5000
    total_points = len(grid_points)
    total_steps = len(sun_positions)
    
    for i, (_, sun_pos) in enumerate(sun_positions.iterrows()):
        if progress_container:
            try:
                dots_html = ""
                for step in range(total_steps):

                    color = "#FFD700" if step <= i else "#BDB76B" # #FFD700 is Gold, #BDB76B is DarkKhaki (greyish)
                    box_shadow = "0 0 15px #FFD700" if step == i else "none"
                    
                    dots_html += f'<div style="width: 12px; height: 12px; background-color: {color}; border-radius: 50%; margin: 0 4px; box-shadow: {box_shadow}; transition: all 0.3s ease;"></div>'
                
                container_html = f'''
                <div style="display: flex; flex-wrap: wrap; justify-content: space-evenly; align-items: center; width: 100%; padding: 10px 0; margin-bottom: 20px; gap: 5px;">
                    {dots_html}
                </div>
                '''
                progress_container.markdown(container_html, unsafe_allow_html=True)
            except Exception as e:
                print(f"Progress bar error: {e}")

        log_mem(f"Step {i} (Sun Position)")

        alt_rad = np.deg2rad(sun_pos['apparent_elevation'])
        az_rad = np.deg2rad(sun_pos['azimuth'])

        sun_direction = np.array([
            np.cos(alt_rad) * np.sin(az_rad),
            np.cos(alt_rad) * np.cos(az_rad),
            np.sin(alt_rad)
        ])

        for start_idx in range(0, total_points, batch_size):
            end_idx = min(start_idx + batch_size, total_points)
            batch_origins = grid_points[start_idx:end_idx]
            
            ray_directions = np.tile(sun_direction, (len(batch_origins), 1))

            locations, index_ray, _ = intersector.intersects_location(
                ray_origins=batch_origins,
                ray_directions=ray_directions,
                multiple_hits=False
            )

            is_lit_batch = np.ones(len(batch_origins), dtype=bool)
            if len(locations) > 0:
                distances = np.linalg.norm(locations - batch_origins[index_ray], axis=1)
                valid_hits = distances < max_ray_distance
                shadowed_ray_indices = np.unique(index_ray[valid_hits])
                is_lit_batch[shadowed_ray_indices] = False

            sunlit_hours[start_idx:end_idx] += is_lit_batch * time_step_weight
            
            del locations, index_ray, ray_directions, is_lit_batch
        
        gc.collect()
        
    log_mem("End calculate_shadows")
    return sunlit_hours

def create_analysis_grid(parcel_polygon: Polygon, density: float = 1.0) -> np.ndarray:
    bounds = parcel_polygon.bounds
    min_x, min_y, max_x, max_y = bounds
    x_coords = np.arange(min_x, max_x, density)
    y_coords = np.arange(min_y, max_y, density)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    points = np.vstack([mesh_x.ravel(), mesh_y.ravel()]).T
    
    prepared_polygon = prep(parcel_polygon)
    contained_mask = [prepared_polygon.contains(Point(p)) for p in points]
    final_points = points[contained_mask]
    
    return np.hstack([final_points, np.full((len(final_points), 1), 0.1)])

def generate_sun_path_geometry(lat: float, lon: float, date: datetime.date, hour_range: tuple, center_metric: tuple, tz='Europe/Warsaw'):
    path_radius = 300
    start_hour, end_hour = hour_range
    center_x, center_y = center_metric
    
    times = pd.date_range(
        start=f"{date} {start_hour:02d}:00",
        end=f"{date} {end_hour:02d}:00",
        freq="15min",
        tz=tz
    )
    
    location = pvlib.location.Location(lat, lon, tz=tz)
    solar_position = location.get_solarposition(times)
    solar_position = solar_position[solar_position['apparent_elevation'] > 0]
    
    sun_path_line = []
    for _, sun in solar_position.iterrows():
        alt_rad = np.deg2rad(sun['apparent_elevation'])
        az_rad = np.deg2rad(sun['azimuth'])
        
        x_offset = path_radius * np.cos(alt_rad) * np.sin(az_rad)
        y_offset = path_radius * np.cos(alt_rad) * np.cos(az_rad)
        z = path_radius * np.sin(alt_rad)
        sun_path_line.append([center_x + x_offset, center_y + y_offset, z])
        
    hourly_times = pd.date_range(
        start=f"{date} {start_hour:02d}:00",
        end=f"{date} {end_hour-1:02d}:00",
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
