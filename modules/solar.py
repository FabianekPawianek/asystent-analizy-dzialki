import numpy as np
import pandas as pd
import pvlib
import trimesh
import psutil
import os
import gc
import requests
import rasterio.features
import scipy.ndimage
from pyproj import Transformer
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from datetime import datetime


import io
import geopandas as gpd
import xml.etree.ElementTree as ET


def fetch_building_polygons(bbox_epsg2180: tuple, radius_m: int = 1000) -> list:
    minx, miny, maxx, maxy = bbox_epsg2180
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    
    transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    center_lon, center_lat = transformer.transform(center_x, center_y)
    
    if radius_m is None:
        radius_m = 1000
    
    query = f"""
    [out:json][timeout:12];
    (
      way["building"](around:{radius_m}, {center_lat:.6f}, {center_lon:.6f});
      relation["building"](around:{radius_m}, {center_lat:.6f}, {center_lon:.6f});
    );
    out geom qt;
    """
    headers = {
        "User-Agent": "AAD_SolarAnalysis/2.1 (contact: project_aad_local@domain.local)",
        "Accept": "application/json",
    }
    endpoints = [
        "https://overpass.openstreetmap.fr/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]
    
    building_polygons_2180 = []
    transformer_to_2180 = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    
    for url in endpoints:
        try:
            resp = requests.post(url, data={'data': query}, headers=headers, timeout=(3.5, 9.0))
            if resp.status_code == 200:
                data = resp.json()
                elements = data.get("elements", [])
                nodes = {elem["id"]: (elem["lon"], elem["lat"]) for elem in elements if elem.get("type") == "node"}
                
                for elem in elements:
                    elem_type = elem.get("type")
                    if elem_type == "way":
                        coords_4326 = []
                        if "geometry" in elem and elem["geometry"]:
                            coords_4326 = [(pt["lon"], pt["lat"]) for pt in elem["geometry"] if "lon" in pt and "lat" in pt]
                        elif "nodes" in elem and nodes:
                            coords_4326 = [nodes[nid] for nid in elem["nodes"] if nid in nodes]
                        if len(coords_4326) >= 3:
                            lons, lats = zip(*coords_4326)
                            xs, ys = transformer_to_2180.transform(lons, lats)
                            poly = Polygon(list(zip(xs, ys)))
                            if not poly.is_valid:
                                poly = poly.buffer(0)
                            if not poly.is_empty and poly.area > 5.0:
                                building_polygons_2180.append(poly)
                    elif elem_type == "relation":
                        for member in elem.get("members", []):
                            if member.get("role") in ["outer", ""] and "geometry" in member and member["geometry"]:
                                coords_4326 = [(pt["lon"], pt["lat"]) for pt in member["geometry"] if "lon" in pt and "lat" in pt]
                                if len(coords_4326) >= 3:
                                    lons, lats = zip(*coords_4326)
                                    xs, ys = transformer_to_2180.transform(lons, lats)
                                    poly = Polygon(list(zip(xs, ys)))
                                    if not poly.is_valid:
                                        poly = poly.buffer(0)
                                    if not poly.is_empty and poly.area > 5.0:
                                        building_polygons_2180.append(poly)
                if building_polygons_2180:
                    print(f"DEBUG BUILDINGS OVERPASS [{url}]: Successfully fetched {len(building_polygons_2180)} polygons.", flush=True)
                    return building_polygons_2180
        except Exception as e:
            print(f"DEBUG Overpass endpoint {url} failed: {e}", flush=True)
            continue
            
    print(f"DEBUG BUILDINGS: All Overpass mirrors failed. Returning empty list for DSM-DTM fallback.", flush=True)
    return building_polygons_2180


fetch_osm_building_polygons = fetch_building_polygons


def compute_planar_roughness(z, mask):
    m_float = mask.astype(np.float64)
    zm = np.where(mask, z, 0.0)

    k_box = np.ones((3, 3))
    k_u = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    k_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)

    n = scipy.ndimage.convolve(m_float, k_box, mode='constant', cval=0.0)
    sum_u = scipy.ndimage.convolve(m_float, -k_u, mode='constant', cval=0.0)
    sum_v = scipy.ndimage.convolve(m_float, -k_v, mode='constant', cval=0.0)
    sum_u2 = scipy.ndimage.convolve(m_float, k_u**2, mode='constant', cval=0.0)
    sum_v2 = scipy.ndimage.convolve(m_float, k_v**2, mode='constant', cval=0.0)
    sum_uv = scipy.ndimage.convolve(m_float, k_u*k_v, mode='constant', cval=0.0)

    sum_z = scipy.ndimage.convolve(zm, k_box, mode='constant', cval=0.0)
    sum_uz = scipy.ndimage.convolve(zm, -k_u, mode='constant', cval=0.0)
    sum_vz = scipy.ndimage.convolve(zm, -k_v, mode='constant', cval=0.0)
    sum_z2 = scipy.ndimage.convolve(zm**2, k_box, mode='constant', cval=0.0)

    valid = (n >= 4) & mask

    c00 = sum_u2 * sum_v2 - sum_uv * sum_uv
    c01 = -(sum_u * sum_v2 - sum_uv * sum_v)
    c02 = sum_u * sum_uv - sum_u2 * sum_v
    c11 = n * sum_v2 - sum_v * sum_v
    c12 = -(n * sum_uv - sum_u * sum_v)
    c22 = n * sum_u2 - sum_u * sum_u

    det = n * c00 + sum_u * c01 + sum_v * c02
    det_safe = np.where(valid & (np.abs(det) > 1e-4), det, 1.0)

    inv00 = c00 / det_safe
    inv01 = c01 / det_safe
    inv02 = c02 / det_safe
    inv11 = c11 / det_safe
    inv12 = c12 / det_safe
    inv22 = c22 / det_safe

    z0 = inv00 * sum_z + inv01 * sum_uz + inv02 * sum_vz
    p = inv01 * sum_z + inv11 * sum_uz + inv12 * sum_vz
    q = inv02 * sum_z + inv12 * sum_uz + inv22 * sum_vz

    rss = sum_z2 - (z0 * sum_z + p * sum_uz + q * sum_vz)
    rss = np.maximum(0.0, rss)
    n_safe = np.maximum(1.0, n)
    rmse = np.sqrt(rss / n_safe)
    return np.where(valid, rmse, 0.0)


def repair_tree_overhangs(dsm_data, dtm_data, building_mask, pixel_size=1.0, building_labeled=None):
    rows, cols = dsm_data.shape
    dsm_repaired = dsm_data.copy()
    tree_overhang_mask = np.zeros((rows, cols), dtype=bool)

    if not np.any(building_mask):
        return dsm_repaired, tree_overhang_mask

    h_diff = np.where(np.isnan(dsm_data - dtm_data), 0.0, dsm_data - dtm_data)
    external_veg = (~building_mask) & (h_diff >= 2.5)

    if not np.any(external_veg):
        return dsm_repaired, tree_overhang_mask

    struct_8 = np.ones((3, 3), dtype=bool)
    touching_trees_mask = external_veg & scipy.ndimage.binary_dilation(building_mask, structure=struct_8, iterations=2)
    if not np.any(touching_trees_mask):
        return dsm_repaired, tree_overhang_mask

    roughness = compute_planar_roughness(dsm_data, building_mask)
    is_planar_roof = building_mask & (roughness <= 0.22)
    clean_roof_base = scipy.ndimage.binary_closing(is_planar_roof, structure=struct_8, iterations=1) & building_mask

    dist_from_trees = scipy.ndimage.distance_transform_edt(~touching_trees_mask)
    candidate_zone = building_mask & (dist_from_trees <= 6.0)
    if not np.any(candidate_zone):
        return dsm_repaired, tree_overhang_mask

    if building_labeled is None:
        labeled_bld, num_bld = scipy.ndimage.label(building_mask, structure=struct_8)
    else:
        labeled_bld = building_labeled
        num_bld = int(np.max(labeled_bld))

    for b_idx in range(1, num_bld + 1):
        comp = (labeled_bld == b_idx)
        comp_candidates = comp & candidate_zone
        if not np.any(comp_candidates):
            continue

        comp_touching_trees = touching_trees_mask & scipy.ndimage.binary_dilation(comp, structure=struct_8, iterations=2)
        if not np.any(comp_touching_trees):
            continue

        r_indices, c_indices = np.where(comp)
        margin = 12
        r_min, r_max = max(0, r_indices.min() - margin), min(rows, r_indices.max() + margin + 1)
        c_min, c_max = max(0, c_indices.min() - margin), min(cols, c_indices.max() + margin + 1)

        sub_comp = comp[r_min:r_max, c_min:c_max]
        sub_dsm = dsm_data[r_min:r_max, c_min:c_max]
        sub_dtm = dtm_data[r_min:r_max, c_min:c_max]
        sub_roughness = roughness[r_min:r_max, c_min:c_max]
        sub_clean_base = clean_roof_base[r_min:r_max, c_min:c_max]
        sub_dist_trees = dist_from_trees[r_min:r_max, c_min:c_max]
        sub_touching_trees = comp_touching_trees[r_min:r_max, c_min:c_max]
        sub_ext_veg = external_veg[r_min:r_max, c_min:c_max]

        tree_heights = sub_dsm[sub_touching_trees]
        if len(tree_heights) == 0:
            continue
        tree_min_h = float(np.percentile(tree_heights, 10))
        tree_max_h = float(np.percentile(tree_heights, 95))

        suspect_canopy = sub_comp & (sub_dist_trees <= 6.0) & (sub_dsm >= tree_min_h - 1.5) & (sub_dsm <= tree_max_h + 1.0)

        sub_clean_roof = sub_comp & sub_clean_base & (~suspect_canopy)

        if np.sum(sub_clean_roof) < 4:
            sub_clean_roof = sub_comp & sub_clean_base & (sub_dist_trees > 3.0)
        if np.sum(sub_clean_roof) < 4:
            sub_clean_roof = sub_comp & (sub_dsm < tree_min_h - 2.0)
        if np.sum(sub_clean_roof) < 4:
            sub_clean_roof = sub_comp & (sub_roughness <= 0.12)

        if not np.any(sub_clean_roof):
            if np.sum(sub_comp) <= 30 and np.all(sub_dsm[sub_comp] - sub_dtm[sub_comp] >= 5.0):
                dsm_repaired[r_min:r_max, c_min:c_max][sub_comp] = sub_dtm[sub_comp] + 2.8
                tree_overhang_mask[r_min:r_max, c_min:c_max][sub_comp] = True
            continue

        elevated_mask = sub_comp & (sub_dsm >= tree_min_h - 2.0) & (sub_dsm <= tree_max_h + 0.8)
        propagation_domain = sub_ext_veg | elevated_mask
        canopy_connected = scipy.ndimage.binary_propagation(sub_touching_trees, mask=propagation_domain) & sub_comp

        sub_r_grid, sub_c_grid = np.indices((r_max - r_min, c_max - c_min))
        sub_cand_r, sub_cand_c = np.where(sub_comp & canopy_connected & (sub_dist_trees <= 6.0))

        for cr, cc in zip(sub_cand_r, sub_cand_c):
            cz = sub_dsm[cr, cc]
            if cz < tree_min_h - 2.0 or cz > tree_max_h + 0.8:
                continue

            dist_to_cand = np.sqrt((sub_r_grid - cr)**2 + (sub_c_grid - cc)**2)
            local_roof_mask = sub_clean_roof & (dist_to_cand <= 7.0)

            if np.sum(local_roof_mask) < 3:
                local_roof_mask = sub_clean_roof & (dist_to_cand <= 12.0)
            if np.sum(local_roof_mask) < 3:
                local_roof_mask = sub_clean_roof

            lr = sub_r_grid[local_roof_mask]
            lc = sub_c_grid[local_roof_mask]
            lz = sub_dsm[local_roof_mask]
            dists = np.sqrt((lr - cr)**2 + (lc - cc)**2)

            nearest_idx = np.argmin(dists)
            ref_h = lz[nearest_idx]

            max_allowed_diff = dists * 0.70 + 0.6
            consistent = np.abs(lz - ref_h) <= max_allowed_diff

            if np.sum(consistent) >= 3:
                lr = lr[consistent]
                lc = lc[consistent]
                lz = lz[consistent]
                dists = dists[consistent]
            else:
                fallback_consistent = np.abs(lz - ref_h) <= 2.0
                if np.sum(fallback_consistent) >= 3:
                    lr = lr[fallback_consistent]
                    lc = lc[fallback_consistent]
                    lz = lz[fallback_consistent]
                    dists = dists[fallback_consistent]

            weights = 1.0 / (1.0 + dists)

            if len(lz) >= 3 and (np.max(lr) > np.min(lr) or np.max(lc) > np.min(lc)):
                A_mat = np.column_stack([lr - cr, lc - cc, np.ones_like(lr)]) * weights[:, None]
                coeff, _, _, _ = np.linalg.lstsq(A_mat, lz * weights, rcond=None)
                slope = np.sqrt(coeff[0]**2 + coeff[1]**2)
                if slope > 0.85:
                    pred_z = float(ref_h)
                else:
                    pred_z = float(coeff[2])
                    max_dev = float(np.min(dists)) * 0.70 + 0.6
                    pred_z = np.clip(pred_z, ref_h - max_dev, ref_h + max_dev)
            else:
                pred_z = float(np.median(lz))

            height_jump = cz - pred_z

            if height_jump >= 2.0:
                tree_overhang_mask[r_min + cr, c_min + cc] = True
                pred_z = max(pred_z, float(sub_dtm[cr, cc] + 2.0))
                pred_z = min(pred_z, cz)
                dsm_repaired[r_min + cr, c_min + cc] = pred_z

    return dsm_repaired, tree_overhang_mask


detect_tree_overhangs = repair_tree_overhangs


def create_building_mask(dsm_shape: tuple, transform, building_polygons_2180: list, dsm_data=None, dtm_data=None, filter_overhangs: bool = False) -> np.ndarray:
    rows, cols = dsm_shape
    pixel_size = abs(transform.a) if hasattr(transform, 'a') and transform.a != 0 else 1.0
    buffer_dist = 1.0

    if not building_polygons_2180:
        if dsm_data is not None and dtm_data is not None:
            diff = dsm_data - dtm_data
            mask = (diff >= 1.80) & (~np.isnan(diff))
            return mask
        else:
            return np.zeros((rows, cols), dtype=bool)

    try:
        valid_polys = []
        buffered_polys = []
        for poly in building_polygons_2180:
            if poly and not poly.is_empty:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty and poly.is_valid and poly.area > 2.0:
                    valid_polys.append(poly)
                    buffered_polys.append(poly.buffer(buffer_dist))

        if not valid_polys:
            if dsm_data is not None and dtm_data is not None:
                diff = dsm_data - dtm_data
                return (diff >= 1.80) & (~np.isnan(diff))
            return np.zeros((rows, cols), dtype=bool)

        buffered_shapes = [(poly, 1) for poly in buffered_polys]
        buffered_mask = rasterio.features.rasterize(
            shapes=buffered_shapes,
            out_shape=(rows, cols),
            transform=transform,
            fill=0,
            default_value=1,
            dtype='uint8',
            all_touched=False
        ).astype(bool)

        if dsm_data is not None and dtm_data is not None:
            diff = dsm_data - dtm_data
            valid_height = ~np.isnan(diff)
            elevated = (diff >= 1.80) & valid_height
            candidate_mask = buffered_mask & elevated

            struct_8 = np.ones((3, 3), dtype=bool)
            mask = scipy.ndimage.binary_closing(candidate_mask, structure=struct_8, iterations=1) & buffered_mask
            mask = mask & ((diff >= 1.50) & valid_height)
            return mask
        else:
            return buffered_mask

    except Exception as e:
        print(f"Error creating building mask: {e}", flush=True)
        if dsm_data is not None and dtm_data is not None:
            diff = dsm_data - dtm_data
            return (diff >= 1.80) & (~np.isnan(diff))
        return np.zeros((rows, cols), dtype=bool)


def prepare_building_dsm(dsm_data, dtm_data, transform, building_polygons_2180, is_building_mask=None):
    rows, cols = dsm_data.shape

    if is_building_mask is None:
        building_mask = create_building_mask(
            (rows, cols), transform, building_polygons_2180,
            dsm_data=dsm_data, dtm_data=dtm_data
        )
    else:
        diff_orig = dsm_data - dtm_data
        building_mask = is_building_mask & (diff_orig >= 1.50) & (~np.isnan(diff_orig))

    building_labeled = None
    if building_polygons_2180:
        valid_polys = []
        for poly in building_polygons_2180:
            if poly and not poly.is_empty:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty and poly.is_valid and poly.area > 2.0:
                    valid_polys.append(poly)

        if valid_polys:
            core_shapes = [(poly, idx + 1) for idx, poly in enumerate(valid_polys)]
            core_labeled = rasterio.features.rasterize(
                shapes=core_shapes,
                out_shape=(rows, cols),
                transform=transform,
                fill=0,
                dtype='int32',
                all_touched=False
            )
            if np.any(core_labeled):
                dist_core, (nr, nc) = scipy.ndimage.distance_transform_edt(core_labeled == 0, return_indices=True)
                building_labeled = np.where(building_mask, core_labeled[nr, nc], 0)

    pixel_size = abs(transform.a) if hasattr(transform, 'a') and transform.a != 0 else 1.0

    dsm_repaired, tree_overhang_mask = repair_tree_overhangs(
        dsm_data, dtm_data, building_mask, pixel_size=pixel_size, building_labeled=building_labeled
    )

    dsm_cleaned = np.where(building_mask, dsm_repaired, dtm_data)
    building_mask = building_mask & ((dsm_cleaned - dtm_data) >= 1.50)

    return dsm_cleaned, building_mask


def process_solar_parcel_dsm(
    dsm_data: np.ndarray,
    dtm_data: np.ndarray,
    transform,
    parcel_geoms: list,
    is_building_mask: np.ndarray = None,
    building_polygons: list = None,
    include_parcel_buildings: bool = True
):
    from shapely.ops import unary_union
    from shapely.geometry.base import BaseGeometry
    from shapely import wkt

    rows, cols = dsm_data.shape
    dsm_for_calc = dsm_data.copy()

    if not parcel_geoms:
        return dsm_for_calc, dsm_for_calc.copy(), dtm_data.copy(), np.zeros((rows, cols), dtype=bool), np.zeros((rows, cols), dtype=bool)

    parsed_parcel_geoms = []
    for g in parcel_geoms:
        if isinstance(g, str):
            try:
                parsed_g = wkt.loads(g)
                if parsed_g.is_valid and not parsed_g.is_empty:
                    parsed_parcel_geoms.append(parsed_g)
                elif not parsed_g.is_empty:
                    fixed_g = parsed_g.buffer(0)
                    if not fixed_g.is_empty:
                        parsed_parcel_geoms.append(fixed_g)
            except Exception:
                pass
        elif isinstance(g, BaseGeometry) and not g.is_empty:
            if g.is_valid:
                parsed_parcel_geoms.append(g)
            else:
                fixed_g = g.buffer(0)
                if not fixed_g.is_empty:
                    parsed_parcel_geoms.append(fixed_g)

    if not parsed_parcel_geoms:
        return dsm_for_calc, dsm_for_calc.copy(), dtm_data.copy(), np.zeros((rows, cols), dtype=bool), np.zeros((rows, cols), dtype=bool)

    parcel_mask = rasterio.features.geometry_mask(
        parsed_parcel_geoms,
        transform=transform,
        out_shape=(rows, cols),
        invert=True,
        all_touched=False
    )
    parcel_mask_touch = rasterio.features.geometry_mask(
        parsed_parcel_geoms,
        transform=transform,
        out_shape=(rows, cols),
        invert=True,
        all_touched=True
    )

    if not np.any(parcel_mask_touch):
        return dsm_for_calc, dsm_for_calc.copy(), dtm_data.copy(), parcel_mask, np.zeros((rows, cols), dtype=bool)

    if is_building_mask is None:
        diff = dsm_data - dtm_data
        is_building_mask = (diff >= 1.50) & (~np.isnan(diff))

    parcel_union = unary_union(parsed_parcel_geoms)
    parcel_building_mask = np.zeros((rows, cols), dtype=bool)
    bldg_geom_mask = np.zeros((rows, cols), dtype=bool)

    parcel_building_polys = []
    other_building_polys = []
    if parcel_union is not None and not parcel_union.is_empty and building_polygons:
        for b_poly in building_polygons:
            if b_poly and not b_poly.is_empty:
                if b_poly.intersects(parcel_union):
                    inter_area = b_poly.intersection(parcel_union).area
                    b_area = b_poly.area
                    ratio = (inter_area / b_area) if b_area > 0 else 0.0
                    if ratio >= 0.15 or inter_area >= 10.0:
                        parcel_building_polys.append(b_poly)
                    else:
                        other_building_polys.append(b_poly)
                else:
                    other_building_polys.append(b_poly)

    if parcel_building_polys:
        other_union = unary_union(other_building_polys) if other_building_polys else None
        bldg_buffered = []
        for p in parcel_building_polys:
            buffered = p.buffer(0.8)
            if other_union is not None and not other_union.is_empty:
                buffered = buffered.difference(other_union)
            if not buffered.is_empty:
                bldg_buffered.append(buffered)

        if bldg_buffered:
            bldg_geom_mask = rasterio.features.geometry_mask(
                bldg_buffered,
                transform=transform,
                out_shape=(rows, cols),
                invert=True,
                all_touched=True
            )
            parcel_building_mask = is_building_mask & bldg_geom_mask
        else:
            parcel_building_mask = is_building_mask & parcel_mask_touch
    elif np.any(parcel_mask_touch & is_building_mask):
        struct_8 = np.ones((3, 3), dtype=bool)
        labeled_bldg, _ = scipy.ndimage.label(is_building_mask, structure=struct_8)
        labels_on_parcel = np.unique(labeled_bldg[parcel_mask_touch & is_building_mask])
        labels_on_parcel = labels_on_parcel[labels_on_parcel > 0]
        parcel_building_mask = np.isin(labeled_bldg, labels_on_parcel)
        parcel_dilated = scipy.ndimage.binary_dilation(parcel_mask_touch, iterations=2)
        bldg_geom_mask = parcel_building_mask & parcel_dilated

    if include_parcel_buildings:
        non_building_parcel = parcel_mask_touch & (~parcel_building_mask)
        dsm_for_calc[non_building_parcel] = dtm_data[non_building_parcel]
    else:
        flatten_mask = parcel_mask_touch | parcel_building_mask | bldg_geom_mask
        dsm_for_calc[flatten_mask] = dtm_data[flatten_mask]

    dsm_for_surface = dsm_for_calc.copy()
    dtm_for_surface = dtm_data.copy()
    dsm_for_surface[parcel_mask_touch] = np.nan
    dtm_for_surface[parcel_mask_touch] = np.nan

    return dsm_for_calc, dsm_for_surface, dtm_for_surface, parcel_mask, parcel_building_mask


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
        return np.full(len(grid_points), len(sun_positions) * time_step_weight, dtype=np.float32)

    if isinstance(scene, trimesh.Scene):
        combined_mesh = scene.dump(concatenate=True)
    elif isinstance(scene, trimesh.Trimesh):
        combined_mesh = scene
    else:
        raise ValueError(f"Nieobsługiwany typ geometrii: {type(scene)}")

    if not isinstance(combined_mesh, trimesh.Trimesh):
        return np.full(len(grid_points), len(sun_positions) * time_step_weight, dtype=np.float32)

    print(f"DEBUG_STATS: Mesh Faces: {len(combined_mesh.faces)}", flush=True)
    print(f"DEBUG_STATS: Grid Points: {len(grid_points)}", flush=True)
    print(f"DEBUG_STATS: Sun Positions: {len(sun_positions)}", flush=True)
    print(f"DEBUG_STATS: Total Rays: {len(grid_points) * len(sun_positions)}", flush=True)

    log_mem("Mesh prepared")

    grid_points = grid_points.astype(np.float32, copy=False)

    sunlit_hours = np.zeros(len(grid_points), dtype=np.float32)
    max_ray_distance = 500.0

    intersector = combined_mesh.ray
    log_mem("Intersector initialized")

    batch_size = 1000
    total_points = len(grid_points)
    total_steps = len(sun_positions)

    for i, (_, sun_pos) in enumerate(sun_positions.iterrows()):
        mem_percent = psutil.virtual_memory().percent
        if mem_percent > 85:
            print(f"INFO: High system memory usage detected ({mem_percent:.1f}%), but continuing analysis (container environment)...", flush=True)

        if progress_container:
            try:
                dots_html = ""
                for step in range(total_steps):

                    color = "#FFD700" if step <= i else "#BDB76B"
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
        ], dtype=np.float32)

        for start_idx in range(0, total_points, batch_size):
            end_idx = min(start_idx + batch_size, total_points)
            batch_origins = grid_points[start_idx:end_idx].copy()
            batch_origins[:, 2] += 0.50

            ray_directions = np.tile(sun_direction, (len(batch_origins), 1)).astype(np.float32)

            locations, index_ray, _ = intersector.intersects_location(
                ray_origins=batch_origins,
                ray_directions=ray_directions,
                multiple_hits=False
            )

            is_lit_batch = np.ones(len(batch_origins), dtype=bool)
            if len(locations) > 0:
                distances = np.linalg.norm(locations - batch_origins[index_ray], axis=1)
                valid_hits = (distances > 0.005) & (distances < max_ray_distance)
                shadowed_ray_indices = np.unique(index_ray[valid_hits])
                is_lit_batch[shadowed_ray_indices] = False

            sunlit_hours[start_idx:end_idx] += is_lit_batch * time_step_weight

            del locations, index_ray, ray_directions, is_lit_batch, batch_origins
            gc.collect()

        gc.collect()

    log_mem("End calculate_shadows")
    return sunlit_hours

def create_analysis_grid(parcel_polygon: Polygon, density: float = 1.0) -> np.ndarray:
    bounds = parcel_polygon.bounds
    min_x, min_y, max_x, max_y = bounds
    start_x = np.floor(min_x / density) * density + 0.5 * density
    end_x = np.ceil(max_x / density) * density
    x_coords = np.arange(start_x, end_x, density)
    start_y = np.floor(min_y / density) * density + 0.5 * density
    end_y = np.ceil(max_y / density) * density
    y_coords = np.arange(start_y, end_y, density)
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
