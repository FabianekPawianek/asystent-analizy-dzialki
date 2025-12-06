import pydeck as pdk
import pandas as pd
import numpy as np
import osmnx as ox
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon
from matplotlib import cm
import modules.geospatial as geospatial

def value_to_rgb(value, min_val, max_val, colormap='plasma'):
    if max_val == min_val:
        norm_value = 0.5
    else:
        norm_value = (value - min_val) / (max_val - min_val)

    rgba = cm.get_cmap(colormap)(norm_value)
    return [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 200]

def create_discrete_legend_html(min_val, max_val, colormap='plasma', steps=7):
    if min_val == max_val:
        rgba = cm.get_cmap(colormap)(0.5)
        rgb = f"rgb({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)})"
        label = f"{min_val:.1f}h"
        header = "<div style='font-family: sans-serif; font-size: 13px; background: rgba(40,40,40,0.85); color: white; padding: 10px; border-radius: 5px; border: 1px solid #555;'>"
        title = "<div style='margin-bottom: 8px;'><b>Śr. dzienne nasłonecznienie</b></div>"
        content = f"<div style='text-align: center; margin: 0 4px;'><div style='width: 35px; height: 35px; background: {rgb};'></div><div>{label}</div></div>"
        return f"{header}{title}{content}</div>"

    values = np.linspace(min_val, max_val, steps)
    colors = cm.get_cmap(colormap)(np.linspace(0, 0.90, steps))
    header = "<div style='font-family: sans-serif; font-size: 13px; background: rgba(40,40,40,0.85); color: white; padding: 10px; border-radius: 5px; border: 1px solid #555;'>"
    title = "<div style='margin-bottom: 8px;'><b>Śr. dzienne nasłonecznienie</b></div>"
    content = "<div style='display: flex; flex-direction: row; align-items: center; justify-content: space-between;'>"

    for i in range(steps):
        rgb = f"rgb({int(colors[i][0] * 255)}, {int(colors[i][1] * 255)}, {int(colors[i][2] * 255)})"
        label = f"{values[i]:.1f}h"
        content += f"<div style='text-align: center; margin: 0 4px;'><div style='width: 35px; height: 35px; background: {rgb};'></div><div>{label}</div></div>"

    return f"{header}{title}{content}</div></div>"

def get_buildings_layer(map_center_wgs_84):
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
            wirefame=True,
            get_elevation="height",
            get_fill_color=[180, 180, 180, 200],
            get_line_color=[100, 100, 100]
        )
        return layer, buildings_data_for_pydeck
    except Exception:
        return None, []

def create_lidar_point_cloud_layer(dsm_data, transform, subsample=2, parcel_polygons_2180=None):
    """
    Tworzy warstwę chmury punktów LiDAR.
    
    Args:
        dsm_data: Dane DSM
        transform: Transformacja rastrowa
        subsample: Współczynnik podpróbkowania
        parcel_polygons_2180: Lista poligonów działek w EPSG:2180 (shapely Polygon).
                              Punkty wewnątrz będą kolorowane na biało.
    """
    from matplotlib.path import Path
    
    MAX_POINTS = 150000
    total_pixels = dsm_data.size
    step = int(np.ceil(np.sqrt(total_pixels / MAX_POINTS)))
    step = max(step, 1)
    
    rows, cols = dsm_data.shape
    dsm_sub = dsm_data[::step, ::step]
    
    r_idx = np.arange(0, rows, step)
    c_idx = np.arange(0, cols, step)
    c_grid, r_grid = np.meshgrid(c_idx, r_idx)
    
    xs, ys = rasterio.transform.xy(transform, r_grid.flatten(), c_grid.flatten())
    z_vals = dsm_sub.flatten()
    
    valid_mask = ~np.isnan(z_vals)
    xs = np.array(xs)[valid_mask]
    ys = np.array(ys)[valid_mask]
    z_vals = z_vals[valid_mask]
    
    n_points = len(xs)
    colors = np.tile([[154, 202, 165, 50]], (n_points, 1))
    
    if parcel_polygons_2180:
        points_2180 = np.column_stack((xs, ys))
        inside_any = np.zeros(n_points, dtype=bool)
        
        for poly in parcel_polygons_2180:
            if poly is not None and poly.is_valid:
                poly_coords = np.array(poly.exterior.coords)
                path = Path(poly_coords)
                inside_mask = path.contains_points(points_2180)
                inside_any |= inside_mask
        
        colors[inside_any] = [255, 255, 255, 180]

    transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)
    
    data_stack = np.column_stack((lons, lats, z_vals))
    
    df = pd.DataFrame({
        'position': data_stack.tolist(),
        'color': colors.tolist()
    })

    return pdk.Layer(
        "PointCloudLayer",
        data=df,
        get_position="position",
        get_color="color",
        get_normal=[0, 0, 15],
        point_size=3,
        pickable=False,
    )

def create_lidar_lines_layer(dsm_data, dtm_data, transform, subsample=3):
    import pandas as pd
    import numpy as np
    import pydeck as pdk
    import rasterio
    from pyproj import Transformer

    rows, cols = dsm_data.shape
    
    r_idx = np.arange(0, rows, subsample)
    c_idx = np.arange(0, cols, subsample)
    c_grid, r_grid = np.meshgrid(c_idx, r_idx)
    
    dsm_sub = dsm_data[::subsample, ::subsample]
    dtm_sub = dtm_data[::subsample, ::subsample]
    
    height_diff = dsm_sub - dtm_sub
    mask = (height_diff > 2.0) & (~np.isnan(dsm_sub)) & (~np.isnan(dtm_sub))
    
    if np.sum(mask) == 0:
        return None

    r_flat = r_grid[mask]
    c_flat = c_grid[mask]
    z_top = dsm_sub[mask]
    z_bottom = dtm_sub[mask]

    xs, ys = rasterio.transform.xy(transform, r_flat, c_flat)
    xs = np.array(xs)
    ys = np.array(ys)

    transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)
    
    source_arr = np.column_stack((lons, lats, z_bottom))
    target_arr = np.column_stack((lons, lats, z_top))
    
    df = pd.DataFrame({
        'source_position': source_arr.tolist(),
        'target_position': target_arr.tolist(),
        'color': [[172, 202, 179, 50]] * len(source_arr)
    })

    return pdk.Layer(
        "LineLayer",
        data=df,
        get_source_position="source_position",
        get_target_position="target_position",
        get_color="color",
        get_width=10,
        width_min_pixels=1,
        pickable=False
    )

def create_solar_analysis_layers(
    parcel_coords_wgs_84,
    map_center_wgs_84,
    solar_results=None,
    grid_points_metric=None,
    sun_path_data=None,
    analemma_data=None,
    azimuth_data=None,
    show_buildings=True
):

    layers = []
    
    buildings_data = []
    if show_buildings:
        buildings_layer, buildings_data = get_buildings_layer(map_center_wgs_84)
        if buildings_layer:
            layers.append(buildings_layer)

    parcels_data = []
    if parcel_coords_wgs_84 and len(parcel_coords_wgs_84) > 0:
        first_point = parcel_coords_wgs_84[0]
        if isinstance(first_point, (list, tuple)) and len(first_point) > 0:
            if isinstance(first_point[0], (int, float)):
                polygons = [parcel_coords_wgs_84]
            elif isinstance(first_point[0], (list, tuple)):
                polygons = parcel_coords_wgs_84
            else:
                polygons = []
        else:
            polygons = []

        for poly_coords in polygons:
            try:
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

    if solar_results is not None:
        if isinstance(solar_results, pd.DataFrame):
            min_h, max_h = solar_results['value'].min(), solar_results['value'].max()
            if min_h == max_h: max_h += 1.0
            
            results_data = []
            has_z = 'z' in solar_results.columns
            
            meters_per_deg_lat = 111132.954
            
            for _, row in solar_results.iterrows():
                val = row['value']
                raw_rgb = value_to_rgb(val, min_h, max_h)
                color = [int(c) for c in raw_rgb[:3]] + [230]
                
                lon, lat = row['lon'], row['lat']
                
                if has_z:
                    z = row['z']
                    lat_offset = 0.5 / meters_per_deg_lat
                    lon_offset = 0.5 / (meters_per_deg_lat * np.cos(np.deg2rad(lat)))
                    
                    polygon = [
                        [lon - lon_offset, lat - lat_offset, z],
                        [lon + lon_offset, lat - lat_offset, z],
                        [lon + lon_offset, lat + lat_offset, z],
                        [lon - lon_offset, lat + lat_offset, z]
                    ]
                    
                    results_data.append({
                        'polygon': polygon,
                        'color': color,
                        'value': row['value']
                    })
                else:
                    results_data.append({
                        'lon': lon,
                        'lat': lat,
                        'color': color,
                        'value': row['value']
                    })
            
            if has_z and results_data:
                heatmap_layer = pdk.Layer(
                    "PolygonLayer",
                    data=results_data,
                    get_polygon="polygon",
                    get_fill_color="color",
                    filled=True,
                    extruded=False,
                    stroked=False,
                    opacity=1.0,
                    pickable=True,
                    auto_highlight=True
                )
            else:
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
        else:
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

    if sun_path_data:

        if isinstance(sun_path_data, list):
            sun_paths_wgs84 = []
            for sp in sun_path_data:
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
            sun_path_line, sun_hour_markers = sun_path_data
            
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

    if analemma_data:
        analemma_layers = []
        for hour, ana_content in analemma_data.items():

            segments = []
            if isinstance(ana_content, list) and len(ana_content) > 0:
                first_item = ana_content[0]
                if 'source' in first_item and 'target' in first_item:
                    segments = ana_content
                elif 'coords' in first_item:
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

            wgs84_segments = []
            for seg in segments:
                if 'source' in seg:
                    src = seg['source']
                    tgt = seg['target']
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

    if azimuth_data:
        markers, lines = azimuth_data
        
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
