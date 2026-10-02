import pydeck as pdk
import pandas as pd
import numpy as np
import osmnx as ox
import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon
import matplotlib as mpl
import modules.geospatial as geospatial

MAX_VISUALIZATION_POINTS = 500000


def calculate_visualization_step(total_pixels: int, user_subsample: int = 1) -> int:
    if total_pixels > MAX_VISUALIZATION_POINTS:
        auto_step = int(np.ceil(np.sqrt(total_pixels / MAX_VISUALIZATION_POINTS)))
    else:
        auto_step = 1
    
    final_step = max(user_subsample, auto_step)
    
    print(f"DEBUG_VIZ: total_pixels={total_pixels}, auto_step={auto_step}, "
          f"user_subsample={user_subsample}, final_step={final_step}", flush=True)
    
    return final_step

def value_to_rgb(value, min_val, max_val, colormap='plasma', alpha=255):
    if max_val <= min_val:
        norm_value = 0.5
    else:
        norm_value = (value - min_val) / (max_val - min_val)
    norm_value = float(np.clip(norm_value, 0.0, 1.0))

    try:
        cmap = mpl.colormaps[colormap]
    except (AttributeError, KeyError):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)

    rgba = cmap(norm_value)
    return [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), int(alpha)]

def map_sunlit_hours_to_rgba(sunlit_hours, min_val=None, max_val=None, colormap='plasma', alpha=255):
    sunlit_hours = np.asarray(sunlit_hours, dtype=np.float32)
    if len(sunlit_hours) == 0:
        return np.empty((0, 4), dtype=np.uint8)

    if min_val is None:
        min_val = float(np.nanmin(sunlit_hours))
    if max_val is None:
        max_val = float(np.nanmax(sunlit_hours))

    if max_val <= min_val:
        norm_values = np.full_like(sunlit_hours, 0.5)
    else:
        norm_values = np.clip((sunlit_hours - min_val) / (max_val - min_val), 0.0, 1.0)

    try:
        cmap = mpl.colormaps[colormap]
    except AttributeError:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)

    rgba_float = cmap(norm_values)
    rgba_uint8 = (rgba_float * 255).astype(np.uint8)
    rgba_uint8[:, 3] = alpha
    return rgba_uint8

def create_discrete_legend_html(min_val, max_val, colormap='plasma', steps=7):
    if min_val >= max_val:
        color = value_to_rgb(min_val, min_val, max_val, colormap=colormap)
        rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
        label = f"{min_val:.1f}h"
        header = "<div class='solar-legend-container'>"
        title = "<div class='solar-legend-title'>Czas nasłonecznienia [h]</div>"
        content = f"<div class='solar-legend-items'><div class='solar-legend-item'><div class='solar-legend-color' style='background: {rgb};'></div><div class='solar-legend-label'>{label}</div></div></div>"
        return f"{header}{title}{content}</div>"

    values = np.linspace(min_val, max_val, steps)
    header = "<div class='solar-legend-container'>"
    title = "<div class='solar-legend-title'>Czas nasłonecznienia [h]</div>"
    content = "<div class='solar-legend-items'>"

    for i in range(steps):
        color = value_to_rgb(values[i], min_val, max_val, colormap=colormap)
        rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
        label = f"{values[i]:.1f}h"
        content += f"<div class='solar-legend-item'><div class='solar-legend-color' style='background: {rgb};'></div><div class='solar-legend-label'>{label}</div></div>"

    return f"{header}{title}{content}</div></div>"

def get_buildings_layer(map_center_wgs_84, osm_radius: int = 300):
    try:
        tags = {"building": True}
        gdf_buildings = ox.features_from_point(
            (map_center_wgs_84[0], map_center_wgs_84[1]), tags, dist=osm_radius
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
    from matplotlib.path import Path
    
    rows, cols = dsm_data.shape
    total_pixels = rows * cols
    
    final_step = calculate_visualization_step(total_pixels, subsample)
    
    dsm_sub = dsm_data[::final_step, ::final_step]
    
    r_idx = np.arange(0, rows, final_step)
    c_idx = np.arange(0, cols, final_step)
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


def create_lidar_square_pillars_layer(dsm_data, dtm_data, transform, subsample=1, custom_colors=None, is_building_mask=None, parcel_polygons_2180=None, layer_id="lidar_square_pillars"):
    from matplotlib.path import Path
    
    pixel_size = abs(transform.a)
    rows, cols = dsm_data.shape
    total_pixels = rows * cols
    
    final_step = calculate_visualization_step(total_pixels, subsample)
    
    step_meters = pixel_size * final_step
    half_size = step_meters / 2.0
    
    r_idx = np.arange(0, rows, final_step)
    c_idx = np.arange(0, cols, final_step)
    c_grid, r_grid = np.meshgrid(c_idx, r_idx)
    
    dsm_sub = dsm_data[::final_step, ::final_step]
    dtm_sub = dtm_data[::final_step, ::final_step]

    height_diff = dsm_sub - dtm_sub
    mask = (height_diff > 2.0) & (~np.isnan(dsm_sub)) & (~np.isnan(dtm_sub))
    
    if np.sum(mask) == 0:
        return None, None

    r_flat = r_grid[mask]
    c_flat = c_grid[mask]
    z_dsm = dsm_sub[mask]
    z_dtm = dtm_sub[mask]
    heights = z_dsm - z_dtm

    xs, ys = rasterio.transform.xy(transform, r_flat, c_flat)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    n = len(xs)

    b_mask_sub = None
    if is_building_mask is not None and is_building_mask.shape == dsm_data.shape:
        b_mask_sub = is_building_mask[::final_step, ::final_step][mask]

    inside_parcel = np.zeros(n, dtype=bool)
    if parcel_polygons_2180:
        points_2180 = np.column_stack((xs, ys))
        for poly in parcel_polygons_2180:
            if poly is not None and not poly.is_empty:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.geom_type == 'MultiPolygon':
                    polys_to_check = list(poly.geoms)
                elif poly.geom_type == 'Polygon':
                    polys_to_check = [poly]
                else:
                    polys_to_check = []
                for p_item in polys_to_check:
                    if p_item.is_valid and not p_item.is_empty:
                        poly_coords = np.array(p_item.exterior.coords)
                        path = Path(poly_coords)
                        inside_parcel |= path.contains_points(points_2180)

    corners_x = np.column_stack([xs - half_size, xs + half_size, xs + half_size, xs - half_size]).ravel()
    corners_y = np.column_stack([ys - half_size, ys - half_size, ys + half_size, ys + half_size]).ravel()

    transformer_obj = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    corners_lon, corners_lat = transformer_obj.transform(corners_x, corners_y)

    polygons = np.zeros((n, 4, 3), dtype=np.float64)
    polygons[:, :, 0] = np.asarray(corners_lon, dtype=np.float64).reshape(n, 4)
    polygons[:, :, 1] = np.asarray(corners_lat, dtype=np.float64).reshape(n, 4)
    polygons[:, :, 2] = np.asarray(z_dtm, dtype=np.float64)[:, None]

    if custom_colors is not None and len(custom_colors) == n:
        colors = [list(c) for c in custom_colors]
    else:
        colors = []
        for i in range(n):
            is_bldg = bool(b_mask_sub is not None and b_mask_sub[i] and heights[i] >= 1.50)
            if is_bldg and inside_parcel[i]:
                colors.append([130, 180, 140])
            elif is_bldg and not inside_parcel[i]:
                colors.append([130, 180, 140])
            elif not is_bldg and inside_parcel[i]:
                colors.append([190, 240, 200])
            else:
                colors.append([160, 210, 170])

    pillar_data = pd.DataFrame({
        'polygon': [p.tolist() for p in polygons],
        'height': heights.tolist(),
        'color': colors
    })
    
    layer = pdk.Layer(
        "PolygonLayer",
        id=layer_id,
        data=pillar_data,
        get_polygon="polygon",
        get_elevation="height",
        extruded=True,
        filled=True,
        flat_shading=True,
        get_fill_color="color",
        get_line_color=[100, 100, 100],
        elevation_scale=1,
        pickable=False,
        material=False,
    )
    
    return layer, mask


def create_lidar_square_surface_layer(dsm_data, transform, subsample=1, parcel_polygons_2180=None, custom_colors=None, is_building_mask=None, dtm_data=None, layer_id="lidar_square_surface", exclude_parcel=False):
    from matplotlib.path import Path
    
    pixel_size = abs(transform.a)
    rows, cols = dsm_data.shape
    total_pixels = rows * cols
    
    final_step = calculate_visualization_step(total_pixels, subsample)
    
    step_meters = pixel_size * final_step
    half_size = step_meters / 2.0
    
    dsm_sub = dsm_data[::final_step, ::final_step]
    
    r_idx = np.arange(0, rows, final_step)
    c_idx = np.arange(0, cols, final_step)
    c_grid, r_grid = np.meshgrid(c_idx, r_idx)
    
    xs_all, ys_all = rasterio.transform.xy(transform, r_grid.flatten(), c_grid.flatten())
    z_vals = dsm_sub.flatten()
    
    dtm_valid = None
    if dtm_data is not None and dtm_data.shape == dsm_data.shape:
        dtm_sub = dtm_data[::final_step, ::final_step]
        dtm_flat = dtm_sub.flatten()
        z_vals = np.where(np.isnan(z_vals), dtm_flat, z_vals)

    valid_mask = ~np.isnan(z_vals)
    xs = np.array(xs_all)[valid_mask]
    ys = np.array(ys_all)[valid_mask]
    z_vals = z_vals[valid_mask]
    if dtm_data is not None and dtm_data.shape == dsm_data.shape:
        dtm_valid = dtm_flat[valid_mask]
    
    n_points = len(xs)
    if n_points == 0:
        return None, None, None

    b_mask_sub = None
    if is_building_mask is not None and is_building_mask.shape == dsm_data.shape:
        b_mask_sub = is_building_mask[::final_step, ::final_step].flatten()[valid_mask]

    inside_parcel = np.zeros(n_points, dtype=bool)
    if parcel_polygons_2180:
        points_2180 = np.column_stack((xs, ys))
        for poly in parcel_polygons_2180:
            if poly is not None and not poly.is_empty:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.geom_type == 'MultiPolygon':
                    polys_to_check = list(poly.geoms)
                elif poly.geom_type == 'Polygon':
                    polys_to_check = [poly]
                else:
                    polys_to_check = []
                for p_item in polys_to_check:
                    if p_item.is_valid and not p_item.is_empty:
                        poly_coords = np.array(p_item.exterior.coords)
                        path = Path(poly_coords)
                        inside_parcel |= path.contains_points(points_2180)

    if exclude_parcel or layer_id == "solar_lidar_surface_layer":
        keep = ~inside_parcel
        xs = xs[keep]
        ys = ys[keep]
        z_vals = z_vals[keep]
        if dtm_valid is not None:
            dtm_valid = dtm_valid[keep]
        if b_mask_sub is not None:
            b_mask_sub = b_mask_sub[keep]
        if custom_colors is not None and len(custom_colors) == n_points:
            custom_colors = np.asarray(custom_colors)[keep]
        inside_parcel = inside_parcel[keep]
        n_points = len(xs)

    if n_points == 0:
        return None, valid_mask, final_step

    corners_x = np.column_stack([xs - half_size, xs + half_size, xs + half_size, xs - half_size]).ravel()
    corners_y = np.column_stack([ys - half_size, ys - half_size, ys + half_size, ys + half_size]).ravel()

    transformer_obj = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    corners_lon, corners_lat = transformer_obj.transform(corners_x, corners_y)

    polygons = np.zeros((n_points, 4, 3), dtype=np.float64)
    polygons[:, :, 0] = np.asarray(corners_lon, dtype=np.float64).reshape(n_points, 4)
    polygons[:, :, 1] = np.asarray(corners_lat, dtype=np.float64).reshape(n_points, 4)
    polygons[:, :, 2] = z_vals[:, None]

    if custom_colors is not None and len(custom_colors) == n_points:
        colors = [list(c) for c in custom_colors]
    else:
        colors = []
        for i in range(n_points):
            is_bldg = bool(b_mask_sub is not None and b_mask_sub[i])
            if is_bldg and dtm_valid is not None and not np.isnan(dtm_valid[i]):
                if (z_vals[i] - dtm_valid[i]) < 1.50:
                    is_bldg = False
            if is_bldg and inside_parcel[i]:
                colors.append([190, 240, 200])
            elif is_bldg and not inside_parcel[i]:
                colors.append([130, 180, 140])
            elif not is_bldg and inside_parcel[i]:
                colors.append([190, 240, 200])
            else:
                colors.append([160, 210, 170])

    surface_data = pd.DataFrame({
        'polygon': [p.tolist() for p in polygons],
        'color': colors
    })

    layer = pdk.Layer(
        "PolygonLayer",
        id=layer_id,
        data=surface_data,
        get_polygon="polygon",
        get_fill_color="color",
        filled=True,
        extruded=False,
        flat_shading=True,
        stroked=False,
        pickable=False,
        material=False,
    )

    return layer, valid_mask, final_step


def create_solar_analysis_layers(
    parcel_coords_wgs_84,
    map_center_wgs_84,
    solar_results=None,
    grid_points_metric=None,
    sun_path_data=None,
    analemma_data=None,
    azimuth_data=None,
    show_buildings=True,
    scale_factor=1.0,
    osm_radius: int = 300
):
    layers = []
    
    buildings_data = []
    if show_buildings:
        buildings_layer, buildings_data = get_buildings_layer(map_center_wgs_84, osm_radius=osm_radius)
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
        id="solar_parcel_layer",
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
            
            has_z = 'z' in solar_results.columns
            
            if has_z:
                if grid_points_metric is not None and len(grid_points_metric) == len(solar_results):
                    xs = np.asarray(grid_points_metric)[:, 0]
                    ys = np.asarray(grid_points_metric)[:, 1]
                elif 'x_2180' in solar_results.columns and 'y_2180' in solar_results.columns:
                    xs = solar_results['x_2180'].to_numpy()
                    ys = solar_results['y_2180'].to_numpy()
                else:
                    transformer_from_wgs = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
                    xs, ys = transformer_from_wgs.transform(solar_results['lon'].to_numpy(), solar_results['lat'].to_numpy())

                n_pts = len(solar_results)
                zs = solar_results['z'].to_numpy()
                vals = solar_results['value'].to_numpy()

                half_size = 0.5
                if len(xs) > 1:
                    diffs_x = np.abs(np.diff(xs))
                    valid_diffs_x = diffs_x[diffs_x > 0.01]
                    if len(valid_diffs_x) > 0:
                        step_x = np.min(valid_diffs_x)
                        if 0.1 <= step_x <= 10.0:
                            half_size = step_x / 2.0

                corners_x = np.column_stack([xs - half_size, xs + half_size, xs + half_size, xs - half_size]).ravel()
                corners_y = np.column_stack([ys - half_size, ys - half_size, ys + half_size, ys + half_size]).ravel()

                transformer_to_wgs = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
                corners_lon, corners_lat = transformer_to_wgs.transform(corners_x, corners_y)

                polygons = np.zeros((n_pts, 4, 3), dtype=np.float64)
                polygons[:, :, 0] = corners_lon.reshape(n_pts, 4)
                polygons[:, :, 1] = corners_lat.reshape(n_pts, 4)
                polygons[:, :, 2] = zs[:, None]
                
                if max_h <= min_h:
                    norm_vals = np.full(n_pts, 0.5)
                else:
                    norm_vals = np.clip((vals - min_h) / (max_h - min_h), 0.0, 1.0)
                try:
                    cmap = mpl.colormaps['plasma']
                except (AttributeError, KeyError):
                    import matplotlib.pyplot as plt
                    cmap = plt.get_cmap('plasma')
                rgba_array = (cmap(norm_vals) * 255).astype(int)
                rgba_array[:, 3] = 255
                colors_list = rgba_array.tolist()
                polygons_list = [p.tolist() for p in polygons]
                
                heatmap_df = pd.DataFrame({
                    'polygon': polygons_list,
                    'color': colors_list,
                    'value': vals
                })
                
                heatmap_layer = pdk.Layer(
                    "PolygonLayer",
                    id="solar_heatmap_layer",
                    data=heatmap_df,
                    get_polygon="polygon",
                    get_fill_color="color",
                    filled=True,
                    extruded=False,
                    flat_shading=True,
                    stroked=False,
                    opacity=1.0,
                    pickable=True,
                    auto_highlight=True,
                    material=False
                )
                layers.append(heatmap_layer)
            else:
                results_data = []
                for _, row in solar_results.iterrows():
                    val = row['value']
                    raw_rgb = value_to_rgb(val, min_h, max_h)
                    color = [int(c) for c in raw_rgb[:3]] + [255]
                    results_data.append({
                        'lon': row['lon'],
                        'lat': row['lat'],
                        'color': color,
                        'value': row['value']
                    })
                heatmap_layer = pdk.Layer(
                    "GridCellLayer",
                    id="solar_heatmap_layer",
                    data=results_data,
                    get_position=['lon', 'lat'],
                    get_fill_color='color',
                    cell_size=1.0,
                    extruded=False,
                    coverage=1.0,
                    pickable=True,
                    auto_highlight=True
                )
                layers.append(heatmap_layer)
        else:
            results_data = solar_results
            heatmap_layer = pdk.Layer(
                "GridCellLayer",
                id="solar_heatmap_layer",
                data=results_data,
                get_position=['lon', 'lat'],
                get_fill_color='color',
                cell_size=1.0,
                extruded=False,
                coverage=1.0,
                pickable=True,
                auto_highlight=True
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
            
            sun_path_width = 1
            
            sun_path_layer = pdk.Layer(
                "PathLayer",
                data=sun_paths_wgs84,
                get_path="path",
                get_color=[140, 140, 140, 160],
                get_width=sun_path_width,
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
                get_width=1 * scale_factor,
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
            
            sun_marker_radius = 12 * scale_factor
                
            sun_markers_layer = pdk.Layer(
                "ScatterplotLayer",
                data=sun_markers_wgs84,
                get_position="position",
                get_radius=sun_marker_radius,
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

            analemma_width = 1
            
            layer = pdk.Layer(
                "LineLayer",
                id=f"analemma_segments_{hour}",
                data=wgs84_segments,
                get_source_position="source",
                get_target_position="target",
                get_color=[100, 100, 100, 180],
                get_width=analemma_width,
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
            
        text_size = 14
        
        azimuth_text_layer = pdk.Layer(
            "TextLayer",
            data=markers_wgs84,
            get_position="position",
            get_text="label",
            get_size=text_size,
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
            
        main_width = 1.5 * scale_factor
        secondary_width = 1.0 * scale_factor
        
        compass_main_layer = pdk.Layer(
            "PathLayer",
            data=[l for l in lines_wgs84 if l['is_main']],
            get_path="path",
            get_color=[90, 90, 90, 150],
            get_width=main_width,
            width_min_pixels=1,
            billboard=True
        )
        compass_secondary_layer = pdk.Layer(
            "PathLayer",
            data=[l for l in lines_wgs84 if not l['is_main']],
            get_path="path",
            get_color=[120, 120, 120, 120],
            get_width=secondary_width,
            width_min_pixels=1,
            billboard=True
        )
        layers.extend([compass_main_layer, compass_secondary_layer])

    return layers, buildings_data


def create_generative_volume_layer(massing_points, voxel_size_m=2.0):
    if not massing_points:
        return None

    df = pd.DataFrame(massing_points)
    if df.empty or 'position' not in df.columns or 'height' not in df.columns:
        return None

    import math
    radius_meters = (voxel_size_m / math.sqrt(2)) * 0.95

    return pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position="position",
        get_elevation="height",
        radius=radius_meters,
        disk_resolution=4,
        angle=45,
        extruded=True,
        flat_shading=True,
        get_fill_color="color",
        get_line_color=[160, 210, 170],
        elevation_scale=1,
        pickable=True,
        auto_highlight=True,
        material=False,
    )

