import pydeck as pdk
import pandas as pd
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon
from matplotlib import cm
import modules.geospatial as geospatial

def value_to_rgb(value, min_val, max_val, colormap='plasma'):
    """Maps a numerical value to an RGB color list [r, g, b, a]."""
    if max_val == min_val:
        norm_value = 0.5
    else:
        norm_value = (value - min_val) / (max_val - min_val)

    rgba = cm.get_cmap(colormap)(norm_value)
    return [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 200]

def create_discrete_legend_html(min_val, max_val, colormap='plasma', steps=7):
    """Generates HTML for a discrete color legend."""
    if min_val == max_val:
        rgba = cm.get_cmap(colormap)(0.5)
        rgb = f"rgb({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)})"
        label = f"{min_val:.1f}h"
        header = "<div style='font-family: sans-serif; font-size: 13px; background: rgba(40,40,40,0.85); color: white; padding: 10px; border-radius: 5px; border: 1px solid #555;'>"
        title = "<div style='margin-bottom: 8px;'><b>Śr. dzienne nasłonecznienie</b></div>"
        content = f"<div style='text-align: center; margin: 0 4px;'><div style='width: 35px; height: 20px; background: {rgb};'></div><div>{label}</div></div>"
        return f"{header}{title}{content}</div>"

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

def get_buildings_layer(map_center_wgs_84):
    """Fetches buildings from OSM and creates a Pydeck PolygonLayer."""
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
        
        if not buildings_data_for_pydeck:
            return None, []

        layer = pdk.Layer(
            "PolygonLayer",
            data=buildings_data_for_pydeck,
            get_polygon="polygon",
            extruded=True,
            wireframe=True,
            get_elevation="height",
            get_fill_color=[180, 180, 180, 200],
            get_line_color=[100, 100, 100]
        )
        return layer, buildings_data_for_pydeck
    except Exception:
        return None, []

def create_solar_analysis_layers(
    parcel_coords_wgs_84,
    map_center_wgs_84,
    solar_results=None,
    grid_points_metric=None,
    sun_path_data=None,
    analemma_data=None,
    azimuth_data=None
):
    """
    Creates all Pydeck layers for the solar analysis visualization.
    """
    layers = []
    
    # 1. Buildings Layer
    buildings_layer, buildings_data = get_buildings_layer(map_center_wgs_84)
    if buildings_layer:
        layers.append(buildings_layer)

    # 2. Parcel Layer
    # Check if parcel_coords_wgs_84 is a list of polygons (list of lists of coords) or a single polygon (list of coords)
    # A single polygon is a list of [lon, lat] points. So it's a list of lists.
    # A list of polygons is a list of list of lists.
    
    parcels_data = []
    if parcel_coords_wgs_84 and len(parcel_coords_wgs_84) > 0:
        # Heuristic to check depth
        # Case 1: Single polygon [[lon, lat], [lon, lat], ...]
        # parcel_coords_wgs_84[0] is [lon, lat] -> list/tuple
        # parcel_coords_wgs_84[0][0] is lon -> float/int
        
        # Case 2: List of polygons [[[lon, lat], ...], [[lon, lat], ...]]
        # parcel_coords_wgs_84[0] is [[lon, lat], ...] -> list/tuple
        # parcel_coords_wgs_84[0][0] is [lon, lat] -> list/tuple
        
        first_point = parcel_coords_wgs_84[0]
        if isinstance(first_point, (list, tuple)) and len(first_point) > 0:
            if isinstance(first_point[0], (int, float)):
                # Case 1: Single polygon
                polygons = [parcel_coords_wgs_84]
            elif isinstance(first_point[0], (list, tuple)):
                # Case 2: List of polygons
                polygons = parcel_coords_wgs_84
            else:
                polygons = []
        else:
            polygons = []

        for poly_coords in polygons:
            try:
                # Ensure closed polygon for Shapely if needed, but Pydeck just needs list of points
                # Shapely Polygon expects list of tuples/lists
                # We use Shapely to ensure it's a valid polygon structure if needed, or just pass coords
                # The original code used: [list(Polygon(parcel_coords_wgs_84).exterior.coords)]
                # This ensures the polygon is closed and valid.
                
                p = Polygon(poly_coords)
                parcels_data.append({
                    "polygon": [list(p.exterior.coords)],
                    "height": 1.0
                })
            except Exception:
                continue

    layer_parcel = pdk.Layer(
        "PolygonLayer",
        data=parcels_data,
        get_polygon="polygon",
        extruded=False,
        get_elevation="height",
        filled=False,
        get_line_color=[255, 0, 0, 255],
        get_line_width=1,
        line_width_min_pixels=2
    )
    layers.append(layer_parcel)

    # 3. Solar Heatmap Layer (if results exist)
    if solar_results is not None:
        # solar_results is expected to be a list of dicts with {lon, lat, value}
        # or a DataFrame with columns [lon, lat, value]
        
        if isinstance(solar_results, pd.DataFrame):
            min_h, max_h = solar_results['value'].min(), solar_results['value'].max()
            if min_h == max_h: max_h += 1.0
            
            results_data = []
            for _, row in solar_results.iterrows():
                color = value_to_rgb(row['value'], min_h, max_h)
                results_data.append({
                    'lon': row['lon'],
                    'lat': row['lat'],
                    'color': color,
                    'value': row['value']
                })
        else:
            # Assume it's already formatted or handle other formats
            results_data = solar_results

        heatmap_layer = pdk.Layer(
            "GridCellLayer",
            data=results_data,
            get_position=['lon', 'lat'],
            get_fill_color='color',
            cell_size=1.0,
            extruded=False,
            coverage=1.0
        )
        layers.append(heatmap_layer)

    # 4. Sun Path Layers (if data exists)
    if sun_path_data:
        # sun_path_data can be a list of dicts (multiple paths) or a tuple (single path + markers)
        
        if isinstance(sun_path_data, list):
            # Multiple paths (e.g. solstices)
            sun_paths_wgs84 = []
            for sp in sun_path_data:
                # sp is {'path': [[x,y,z], ...], 'name': ...}
                # coords are metric, need transform
                path_wgs = []
                for p in sp['path']:
                    p_wgs = geospatial.transform_single_coord(p[0], p[1], "2180", "4326")
                    path_wgs.append([p_wgs[0], p_wgs[1], p[2]])
                sun_paths_wgs84.append({"path": path_wgs, "name": sp.get('name', '')})
            
            sun_path_layer = pdk.Layer(
                "PathLayer",
                data=sun_paths_wgs84,
                get_path="path",
                get_color=[140, 140, 140, 160],
                get_width=1,
                width_min_pixels=1,
                billboard=True
            )
            layers.append(sun_path_layer)
            
        elif isinstance(sun_path_data, tuple):
            # Single path + markers
            sun_path_line, sun_hour_markers = sun_path_data
            
            # Transform sun path to WGS84
            sun_path_wgs84 = geospatial.transform_coordinates_to_wgs84([p[:2] for p in sun_path_line])
            sun_path_wgs84_3d = [[p[0], p[1], h[2]] for p, h in zip(sun_path_wgs84, sun_path_line)]
            
            sun_path_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": sun_path_wgs84_3d}],
                get_path="path",
                get_color=[140, 140, 140, 160],
                get_width=1,
                width_min_pixels=1,
                billboard=True
            )
            layers.append(sun_path_layer)
            
            # Sun markers
            sun_markers_wgs84 = []
            for marker in sun_hour_markers:
                pos_metric = marker['position']
                pos_wgs = geospatial.transform_single_coord(pos_metric[0], pos_metric[1], "2180", "4326")
                sun_markers_wgs84.append({
                    "position": [pos_wgs[0], pos_wgs[1], pos_metric[2]],
                    "hour": marker['hour']
                })
                
            sun_markers_layer = pdk.Layer(
                "ScatterplotLayer",
                data=sun_markers_wgs84,
                get_position="position",
                get_radius=12,
                filled=True,
                get_fill_color=[255, 223, 0, 255],
                stroked=False,
                billboard=True
            )
            layers.append(sun_markers_layer)

    # 5. Analemma Layers
    if analemma_data:
        analemma_layers = []
        for hour, ana_content in analemma_data.items():
            # ana_content can be a list of segments (dicts with source/target)
            # or a list of points (dicts with coords)
            
            segments = []
            if isinstance(ana_content, list) and len(ana_content) > 0:
                first_item = ana_content[0]
                if 'source' in first_item and 'target' in first_item:
                    # Already segments
                    segments = ana_content
                elif 'coords' in first_item:
                    # List of points, need to create segments
                    sorted_points = sorted(ana_content, key=lambda x: x.get('day', 0))
                    points_wgs = []
                    for p in sorted_points:
                        c = p['coords']
                        wgs = geospatial.transform_single_coord(c[0], c[1], "2180", "4326")
                        points_wgs.append([wgs[0], wgs[1], c[2]])
                    
                    for i in range(len(points_wgs) - 1):
                        segments.append({
                            "source": points_wgs[i],
                            "target": points_wgs[i+1]
                        })
                    if len(points_wgs) > 1:
                        segments.append({
                            "source": points_wgs[-1],
                            "target": points_wgs[0]
                        })
            
            # If segments are already in WGS84 (from the block above), use them
            # If they came in as segments (metric), transform them
            
            wgs84_segments = []
            for seg in segments:
                if 'source' in seg: # It's a segment
                    src = seg['source']
                    tgt = seg['target']
                    # Check if already WGS84 (lon < 180) or Metric (x > 1000)
                    # Simple heuristic
                    if src[0] > 180: 
                        src_wgs = geospatial.transform_single_coord(src[0], src[1], "2180", "4326")
                        src = [src_wgs[0], src_wgs[1], src[2]]
                    if tgt[0] > 180:
                        tgt_wgs = geospatial.transform_single_coord(tgt[0], tgt[1], "2180", "4326")
                        tgt = [tgt_wgs[0], tgt_wgs[1], tgt[2]]
                    
                    wgs84_segments.append({
                        "source": src,
                        "target": tgt
                    })

            layer = pdk.Layer(
                "LineLayer",
                id=f"analemma_segments_{hour}",
                data=wgs84_segments,
                get_source_position="source",
                get_target_position="target",
                get_color=[100, 100, 100, 180],
                get_width=1,
                width_min_pixels=1,
                pickable=False,
                auto_highlight=False
            )
            analemma_layers.append(layer)
        layers.extend(analemma_layers)

    # 6. Compass/Azimuth Layers
    if azimuth_data:
        markers, lines = azimuth_data
        
        # Transform markers
        markers_wgs84 = []
        for m in markers:
            pos = m['position']
            pos_wgs = geospatial.transform_single_coord(pos[0], pos[1], "2180", "4326")
            markers_wgs84.append({
                "position": [pos_wgs[0], pos_wgs[1], pos[2]],
                "label": m['label']
            })
            
        azimuth_text_layer = pdk.Layer(
            "TextLayer",
            data=markers_wgs84,
            get_position="position",
            get_text="label",
            get_size=14,
            get_color=[80, 80, 80, 255],
            get_angle=0,
            get_text_anchor="'middle'",
            get_alignment_baseline="'center'",
            billboard=True
        )
        layers.append(azimuth_text_layer)
        
        # Transform lines
        lines_wgs84 = []
        for l in lines:
            path = l['path']
            p1 = path[0]
            p2 = path[1]
            p1_wgs = geospatial.transform_single_coord(p1[0], p1[1], "2180", "4326")
            p2_wgs = geospatial.transform_single_coord(p2[0], p2[1], "2180", "4326")
            lines_wgs84.append({
                "path": [[p1_wgs[0], p1_wgs[1], p1[2]], [p2_wgs[0], p2_wgs[1], p2[2]]],
                "is_main": l['is_main']
            })
            
        compass_main_layer = pdk.Layer(
            "PathLayer",
            data=[l for l in lines_wgs84 if l['is_main']],
            get_path="path",
            get_color=[90, 90, 90, 150],
            get_width=1.5,
            width_min_pixels=1,
            billboard=True
        )
        compass_secondary_layer = pdk.Layer(
            "PathLayer",
            data=[l for l in lines_wgs84 if not l['is_main']],
            get_path="path",
            get_color=[120, 120, 120, 120],
            get_width=1,
            width_min_pixels=1,
            billboard=True
        )
        layers.extend([compass_main_layer, compass_secondary_layer])

    return layers, buildings_data
