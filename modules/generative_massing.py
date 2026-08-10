import numpy as np
import pandas as pd
import logging
import json
import re
from shapely.geometry import Polygon, MultiPolygon, Point
from pyproj import Transformer
import config

logger = logging.getLogger(__name__)

def get_genai_client(api_key=None):

    try:
        from google import genai
        if not api_key:
            try:
                import streamlit as st
                import config
                api_key = config.get_google_api_key(st.secrets)
            except Exception:
                try:
                    import config
                    api_key = config.get_google_api_key()
                except Exception:
                    pass
        if api_key:
            return genai.Client(api_key=api_key)
    except Exception as e:
        logger.warning(f"Could not initialize google-genai client in generative_massing: {e}")
    return None


def extract_pog_legal_envelope(pog_data_dict: dict, parcel_area_m2: float) -> dict:

    min_bio_pct = 30.0
    max_footprint_pct = 70.0
    max_height_m = 12.0
    is_ozs = False
    
    pog_text_combined = ""

    if pog_data_dict and isinstance(pog_data_dict, dict):
        analysis = pog_data_dict.get("analysis", {})
        if isinstance(analysis, str):
            pog_text_combined = analysis
        elif isinstance(analysis, dict):
            pog_text_combined = json.dumps(analysis, ensure_ascii=False)
            szczegolowe = analysis.get("szczegolowe", {})
            if isinstance(szczegolowe, dict):
                for k, v in szczegolowe.items():
                    v_str = str(v).lower()
                    k_str = str(k).lower()
                    
                    if "biologicznie" in k_str or "biologicznie" in v_str:
                        nums = re.findall(r'\d+(?:\.\d+)?', v_str)
                        if nums:
                            min_bio_pct = float(nums[0])
                    
                    if "zabudowy" in k_str or "powierzchnia zabudowy" in v_str:
                        nums = re.findall(r'\d+(?:\.\d+)?', v_str)
                        if nums:
                            max_footprint_pct = float(nums[0])

                    if "wysokość" in k_str or "wysokosc" in k_str or "wysokość" in v_str:
                        nums = re.findall(r'\d+(?:\.\d+)?', v_str)
                        if nums:
                            max_height_m = float(nums[0])

    pog_text_lower = pog_text_combined.lower()
    if any(term in pog_text_lower for term in ["ozs", "obszar zabudowy śródmiejskiej", "obszar zabudowy srodmiejskiej", "ouz"]):
        is_ozs = True

    calculated_max_footprint_pct = min(max_footprint_pct, max(10.0, 100.0 - min_bio_pct))
    max_allowed_footprint_m2 = parcel_area_m2 * (calculated_max_footprint_pct / 100.0)
    setback_m = 3.0 if is_ozs else 4.0

    return {
        "min_bio_pct": min_bio_pct,
        "max_footprint_pct": calculated_max_footprint_pct,
        "max_allowed_footprint_m2": max_allowed_footprint_m2,
        "max_allowed_height_m": max_height_m,
        "is_ozs": is_ozs,
        "setback_m": setback_m
    }


def interpret_typology_with_ai(user_prompt: str, pog_envelope: dict, parcel_area_m2: float, api_key=None) -> dict:

    default_config = {
        "typology_name": "Koncepcja zrównoważona",
        "target_footprint_m2": round(min(160.0, pog_envelope["max_allowed_footprint_m2"]), 1),
        "target_height_m": round(min(8.0, pog_envelope["max_allowed_height_m"]), 1),
        "target_stories": 2,
        "sun_preference": "morning",
        "design_intent_summary": "Zrównoważona koncepcja w granicach POG",
        "anti_max_pum_triggered": False,
        "warning_message": None
    }

    client = get_genai_client(api_key)
    if not client:
        return default_config

    prompt_text = user_prompt if user_prompt else "Brak dodatkowych życzeń - zaprojektuj optymalny budynek."

    sys_prompt = f"""You are an expert computational architect evaluating user design prompts against municipal zoning (POG - Plan Ogólny Gminy) constraints.

Parcel Area: {parcel_area_m2:.1f} m²
POG Legal Limits:
- Max Allowed Footprint Area: {pog_envelope['max_allowed_footprint_m2']:.1f} m²
- Max Building Height: {pog_envelope['max_allowed_height_m']:.1f} m
- Min Biologically Active Area: {pog_envelope['min_bio_pct']:.1f}%
- Is Downtown / OZS Active: {pog_envelope['is_ozs']}

User Prompt: "{prompt_text}"

Instructions:
1. Evaluate the user's intent. If the user asks for excessive building area ("max PUM", "wyciśnij 100%", etc.), prioritize daylight, greenery, and biological area while capping values strictly within POG limits.
2. Return ONLY a raw valid JSON object (no markdown, no ```json ``` fences) matching this schema:
{{
  "typology_name": "string (e.g. Dom jednorodzinny, Zabudowa szeregowa, Budynek wielorodzinny)",
  "target_footprint_m2": float (MUST BE <= {pog_envelope['max_allowed_footprint_m2']:.1f}),
  "target_height_m": float (MUST BE <= {pog_envelope['max_allowed_height_m']:.1f}),
  "target_stories": int,
  "sun_preference": "morning" or "afternoon" or "balanced",
  "design_intent_summary": "string summary of design decision",
  "anti_max_pum_triggered": boolean
}}"""

    try:
        response = client.models.generate_content(
            model=config.MODEL_NAME,
            contents=sys_prompt
        )
        text = response.text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        
        parsed = json.loads(text)
        
        parsed["target_footprint_m2"] = float(min(parsed.get("target_footprint_m2", default_config["target_footprint_m2"]), pog_envelope["max_allowed_footprint_m2"]))
        parsed["target_height_m"] = float(min(parsed.get("target_height_m", default_config["target_height_m"]), pog_envelope["max_allowed_height_m"]))
        parsed["target_stories"] = int(parsed.get("target_stories", max(1, int(parsed["target_height_m"] // 3))))
        
        if parsed.get("anti_max_pum_triggered"):
            parsed["warning_message"] = (
                "Wykryto próbę maksymalizacji PUM. Gemini AI skorygował cel projektowy: "
                f"powierzchnia zabudowy została ograniczona do {parsed['target_footprint_m2']:.1f} m² "
                "zgodnie z limitami POG, z zachowaniem bufora ogrodowego i optymalnym nasłonecznieniem."
            )
        return parsed
    except Exception as e:
        logger.warning(f"Error calling Gemini in interpret_typology_with_ai: {e}")
        return default_config


def sanitize_user_intent(user_prompt: str, pog_data_dict: dict = None, parcel_area_m2: float = 500.0) -> dict:

    pog_envelope = extract_pog_legal_envelope(pog_data_dict, parcel_area_m2)
    ai_config = interpret_typology_with_ai(user_prompt, pog_envelope, parcel_area_m2)
    return ai_config


def generate_massing_volume(
    pog_data_dict,
    solar_grid_points,
    sunlit_hours,
    parcel_geometry,
    dtm_data=None,
    transform=None,
    user_intent: str = ""
) -> list:

    try:
        # 1. Process Parcel Geometry in EPSG:2180
        poly = None
        if isinstance(parcel_geometry, (Polygon, MultiPolygon)):
            poly = parcel_geometry
        elif hasattr(parcel_geometry, 'geometry'):
            poly = parcel_geometry.geometry
        elif isinstance(parcel_geometry, list) and len(parcel_geometry) > 0:
            poly = Polygon(parcel_geometry)

        if poly is None or not poly.is_valid or poly.is_empty:
            logger.error("Invalid parcel geometry provided for massing generation.")
            return []

        parcel_area_m2 = float(poly.area)

        pog_envelope = extract_pog_legal_envelope(pog_data_dict, parcel_area_m2)

        ai_config = interpret_typology_with_ai(user_intent, pog_envelope, parcel_area_m2)

        target_footprint_m2 = ai_config["target_footprint_m2"]
        target_height_m = ai_config["target_height_m"]
        sun_pref = ai_config.get("sun_preference", "balanced")

        setback_m = pog_envelope["setback_m"]
        buildable_polygon = poly.buffer(-setback_m)
        
        if buildable_polygon.is_empty or not buildable_polygon.is_valid:
            buildable_polygon = poly.buffer(-2.0)
            if buildable_polygon.is_empty:
                buildable_polygon = poly.buffer(-1.0)

        voxel_size = 2.0
        minx, miny, maxx, maxy = buildable_polygon.bounds
        x_range = np.arange(minx + voxel_size / 2.0, maxx, voxel_size)
        y_range = np.arange(miny + voxel_size / 2.0, maxy, voxel_size)
        grid_x, grid_y = np.meshgrid(x_range, y_range)
        pts_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        from shapely.prepared import prep
        prep_buildable = prep(buildable_polygon)
        inside_mask = np.array([prep_buildable.contains(Point(p[0], p[1])) for p in pts_xy])
        valid_pts = pts_xy[inside_mask]

        if len(valid_pts) == 0:
            return []

        target_voxels_count = int(np.round(target_footprint_m2 / (voxel_size ** 2)))
        target_voxels_count = max(1, min(len(valid_pts), target_voxels_count))

        from scipy.spatial import cKDTree
        
        has_solar = (solar_grid_points is not None and len(solar_grid_points) > 0 and 
                     sunlit_hours is not None and len(sunlit_hours) > 0)
        
        if has_solar:
            solar_pts_2d = np.asarray(solar_grid_points)[:, :2]
            solar_tree = cKDTree(solar_pts_2d)
            dists, solar_idx = solar_tree.query(valid_pts)
            pts_sunlit = np.asarray(sunlit_hours)[solar_idx]
        else:
            northing_norm = (valid_pts[:, 1] - miny) / (maxy - miny + 1e-5)
            pts_sunlit = 8.0 - (northing_norm * 4.0)

        if sun_pref == "morning":
            northing_factor = (valid_pts[:, 1] - miny) / (maxy - miny + 1e-5)
            easting_factor = 1.0 - ((valid_pts[:, 0] - minx) / (maxx - minx + 1e-5))
            placement_score = northing_factor * 0.6 + easting_factor * 0.4
            
            top_candidate_idx = np.argsort(-placement_score)[:max(1, target_voxels_count * 2)]
            best_seed = valid_pts[top_candidate_idx[0]]
            dists_to_seed = np.linalg.norm(valid_pts - best_seed, axis=1)
            selected_indices = np.argsort(dists_to_seed)[:target_voxels_count]
        else:
            centroid_x, centroid_y = buildable_polygon.centroid.x, buildable_polygon.centroid.y
            dist_to_center = np.linalg.norm(valid_pts - np.array([centroid_x, centroid_y]), axis=1)
            selected_indices = np.argsort(dist_to_center)[:target_voxels_count]

        building_pts = valid_pts[selected_indices]

        voxel_heights = np.full(len(building_pts), target_height_m, dtype=np.float32)

        transformer = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(building_pts[:, 0], building_pts[:, 1])

        z_dtm = np.zeros(len(building_pts), dtype=np.float32)
        min_elevation = 0.0
        
        if dtm_data is not None and transform is not None:
            import rasterio.transform
            if not np.all(np.isnan(dtm_data)):
                min_elevation = float(np.nanmin(dtm_data))
                
            rows, cols = dtm_data.shape
            for i, (px, py) in enumerate(building_pts):
                try:
                    r, c = rasterio.transform.rowcol(transform, px, py)
                    if 0 <= r < rows and 0 <= c < cols:
                        val = dtm_data[r, c]
                        z_dtm[i] = val if not np.isnan(val) else min_elevation
                except Exception:
                    z_dtm[i] = min_elevation

        z_base_normalized = z_dtm - min_elevation

        terracotta_color = [224, 109, 83, 230]
        
        voxel_list = []
        for i in range(len(building_pts)):
            voxel_list.append({
                'position': [float(lons[i]), float(lats[i]), float(z_base_normalized[i])],
                'height': float(voxel_heights[i]),
                'color': terracotta_color
            })

        return voxel_list
    except Exception as e:
        logger.error(f"Error in generate_massing_volume: {e}", exc_info=True)
        return []
