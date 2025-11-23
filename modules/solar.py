import numpy as np
import pandas as pd
import pvlib
import trimesh
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from datetime import datetime

def calculate_sun_positions(lat: float, lon: float, date: datetime.date, hour_range: tuple, tz='Europe/Warsaw'):
    """
    Calculates sun positions for a given date and hour range.
    Returns a DataFrame with 'apparent_elevation' and 'azimuth'.
    """
    start_hour, end_hour = hour_range
    times = pd.date_range(
        start=f"{date} {start_hour:02d}:00",
        end=f"{date} {end_hour:02d}:00",
        freq="15min",
        tz=tz
    )
    location = pvlib.location.Location(lat, lon, tz=tz)
    solar_position = location.get_solarposition(times)
    # Filter for sun above horizon
    return solar_position[solar_position['apparent_elevation'] > 0]

def create_trimesh_scene(buildings_data_metric: list) -> trimesh.Scene:
    """
    Converts a list of building dictionaries (polygon coords + height) into a trimesh Scene.
    """
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

def calculate_shadows(scene: trimesh.Scene, grid_points: np.ndarray, sun_positions: pd.DataFrame) -> np.ndarray:
    """
    Performs ray tracing to calculate shadow masks.
    Returns an array of 'sunlit hours' (assuming each sun position is 15 mins = 0.25h).
    """
    if scene.is_empty:
        # No shadows, full sun
        return np.full(len(grid_points), len(sun_positions) * 0.25)

    combined_mesh = scene.dump(concatenate=True)
    if not isinstance(combined_mesh, trimesh.Trimesh):
        # Fallback if mesh generation fails
        return np.full(len(grid_points), len(sun_positions) * 0.25)

    sunlit_hours = np.zeros(len(grid_points))
    max_ray_distance = 500.0

    for _, sun_pos in sun_positions.iterrows():
        alt_rad = np.deg2rad(sun_pos['apparent_elevation'])
        az_rad = np.deg2rad(sun_pos['azimuth'])

        sun_direction = np.array([
            np.cos(alt_rad) * np.sin(az_rad),
            np.cos(alt_rad) * np.cos(az_rad),
            np.sin(alt_rad)
        ])

        ray_origins = grid_points
        ray_directions = np.tile(sun_direction, (len(ray_origins), 1))

        locations, index_ray, _ = combined_mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )

        is_lit = np.ones(len(ray_origins), dtype=bool)
        if len(locations) > 0:
            distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
            valid_hits = distances < max_ray_distance
            shadowed_ray_indices = np.unique(index_ray[valid_hits])
            is_lit[shadowed_ray_indices] = False

        sunlit_hours += is_lit * 0.25

    return sunlit_hours

def create_analysis_grid(parcel_polygon: Polygon, density: float = 1.0) -> np.ndarray:
    """
    Generates a grid of points within a polygon.
    Returns an array of points (x, y, z=0.1).
    """
    bounds = parcel_polygon.bounds
    min_x, min_y, max_x, max_y = bounds
    x_coords = np.arange(min_x, max_x, density)
    y_coords = np.arange(min_y, max_y, density)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    points = np.vstack([mesh_x.ravel(), mesh_y.ravel()]).T
    
    prepared_polygon = prep(parcel_polygon)
    contained_mask = [prepared_polygon.contains(Point(p)) for p in points]
    final_points = points[contained_mask]
    
    # Add Z coordinate (0.1m above ground)
    return np.hstack([final_points, np.full((len(final_points), 1), 0.1)])

def generate_sun_path_geometry(lat: float, lon: float, date: datetime.date, hour_range: tuple, center_metric: tuple, tz='Europe/Warsaw'):
    """
    Calculates the 3D coordinates for the sun path line and hour markers.
    Returns: (sun_path_line_coords, sun_hour_markers_list)
    """
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
