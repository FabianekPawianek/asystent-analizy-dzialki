import os
import time
import tempfile
import logging
import numpy as np
import rasterio
from rasterio.features import geometry_mask
import pyvista as pv
import trimesh
import requests
import http.client
from pyproj import Transformer
from shapely.ops import transform as shapely_transform
import io

logger = logging.getLogger(__name__)

class LidarService:
    WCS_DSM_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMPT/GRID1/WCS/DigitalSurfaceModel"
    WCS_DTM_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/GRID1/WCS/DigitalTerrainModel"
    DSM_LAYER_ID = "DSM_PL-EVRF2007-NH"
    DTM_LAYER_ID = "DTM_PL-EVRF2007-NH"
    DEFAULT_COVERAGE = DSM_LAYER_ID
    DEFAULT_DTM_COVERAGE = DTM_LAYER_ID
    DEFAULT_FORMAT = "image/x-aaigrid"

    def __init__(self):
        pass

    def _fetch_with_retry(self, base_url, params=None, timeout=300):
        max_retries = 3
        retry_delay = 5
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; SolarAnalysisBot/1.0)'}

        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, params=params, stream=True, timeout=timeout, headers=headers)
                if response.status_code != 200:
                    try:
                        error_content = response.content.decode('utf-8', errors='ignore')[:500]
                    except:
                        error_content = "Could not read error content"
                    logger.error(f"HTTP Error {response.status_code}: {error_content}")
                    response.raise_for_status()

                return response
            except (requests.exceptions.RequestException, http.client.RemoteDisconnected) as e:
                logger.warning(f"Network error during WCS fetch (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Giving up.")
                    raise
            except Exception as e:
                raise e

    def _apply_circular_mask(self, data, transform=None):
        rows, cols = data.shape
        center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
        
        dx = 1.0
        dy = 1.0
        if transform is not None:
            if hasattr(transform, 'a') and transform.a != 0:
                dx = abs(transform.a)
            if hasattr(transform, 'e') and transform.e != 0:
                dy = abs(transform.e)
            else:
                dy = dx

        radius_m = min(rows * dy, cols * dx) / 2.0
        
        y, x = np.ogrid[:rows, :cols]
        x_m = (x - center_c) * dx
        y_m = (y - center_r) * dy
        
        dist_from_center_m_sq = x_m**2 + y_m**2
        mask_outside = dist_from_center_m_sq > radius_m**2
        
        if np.issubdtype(data.dtype, np.floating):
            data[mask_outside] = np.nan
        
        return data

    def apply_circular_mask(self, data: np.ndarray, transform=None) -> np.ndarray:
        return self._apply_circular_mask(data.copy(), transform=transform)


    def get_dsm_data(self, bbox, crs="EPSG:2180", width=None, height=None, res_x=None, res_y=None, coverage_id=None, apply_circular_mask: bool = True):
        try:
            if coverage_id is None:
                coverage_id = self.DEFAULT_COVERAGE

            logger.info(f"Fetching coverage: {coverage_id} for bbox: {bbox}")

            if width is None and height is None and res_x is None and res_y is None:
                width = int(round(bbox[2] - bbox[0]))
                height = int(round(bbox[3] - bbox[1]))
                logger.info(f"Calculated raster dimensions: width={width}, height={height} (1m resolution)")

            params = {
                "SERVICE": "WCS",
                "VERSION": "1.0.0",
                "REQUEST": "GetCoverage",
                "COVERAGE": coverage_id,
                "CRS": crs,
                "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "WIDTH": str(width),
                "HEIGHT": str(height),
                "FORMAT": "image/x-aaigrid",
                "RESX": str(res_x) if res_x else "",
                "RESY": str(res_y) if res_y else ""
            }

            params = {k: v for k, v in params.items() if v}

            logger.info(f"Manual Request Params: {params}")

            http_response = self._fetch_with_retry(self.WCS_DSM_URL, params=params, timeout=300)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.asc') as tmp_file:
                for chunk in http_response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name

            with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                content_preview = f.read(500)
            print(f"DEBUG FILE CONTENT: {content_preview}")
            
            if content_preview.strip().startswith('<'):
                raise ValueError(f"Server returned XML error instead of data: {content_preview}")

            try:
                try:
                    with rasterio.open(tmp_path) as src:
                        data = src.read(1)
                        transform = src.transform
                        nodata = src.nodata
                except Exception as rasterio_error:
                    with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content_preview = f.read(500)

                    error_msg = f"Rasterio failed to open file. Content preview:\n{content_preview}"
                    logger.error(error_msg)
                    print(f"DEBUG: {error_msg}")
                    raise Exception(f"WCS Error or Invalid Format: {rasterio_error}. Server response: {content_preview}")

                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)

                if apply_circular_mask:
                    data = self._apply_circular_mask(data, transform)
                return data, transform

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            logger.error(f"Critical error in get_dsm_data: {e}")
            raise

    def get_dtm_data(self, bbox, crs="EPSG:2180", width=None, height=None, res_x=None, res_y=None, coverage_id=None, apply_circular_mask: bool = True):
        try:
            if coverage_id is None:
                coverage_id = self.DEFAULT_DTM_COVERAGE

            logger.info(f"Fetching DTM coverage: {coverage_id} for bbox: {bbox}")

            if width is None and height is None and res_x is None and res_y is None:
                width = int(round(bbox[2] - bbox[0]))
                height = int(round(bbox[3] - bbox[1]))
                logger.info(f"Calculated DTM raster dimensions: width={width}, height={height} (1m resolution)")

            params = {
                "SERVICE": "WCS",
                "VERSION": "1.0.0",
                "REQUEST": "GetCoverage",
                "COVERAGE": coverage_id,
                "CRS": crs,
                "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "WIDTH": str(width),
                "HEIGHT": str(height),
                "FORMAT": "image/x-aaigrid",
                "RESX": str(res_x) if res_x else "",
                "RESY": str(res_y) if res_y else ""
            }

            params = {k: v for k, v in params.items() if v}

            logger.info(f"Manual DTM Request Params: {params}")

            http_response = self._fetch_with_retry(self.WCS_DTM_URL, params=params, timeout=300)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.asc') as tmp_file:
                for chunk in http_response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name

            with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                content_preview = f.read(500)
            print(f"DEBUG FILE CONTENT: {content_preview}")
            
            if content_preview.strip().startswith('<'):
                raise ValueError(f"Server returned XML error instead of data: {content_preview}")

            try:
                try:
                    with rasterio.open(tmp_path) as src:
                        data = src.read(1)
                        transform = src.transform
                        nodata = src.nodata
                except Exception as rasterio_error:
                    with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content_preview = f.read(500)

                    error_msg = f"Rasterio failed to open DTM file. Content preview:\n{content_preview}"
                    logger.error(error_msg)
                    print(f"DEBUG: {error_msg}")
                    raise Exception(f"WCS DTM Error or Invalid Format: {rasterio_error}. Server response: {content_preview}")

                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)

                if apply_circular_mask:
                    data = self._apply_circular_mask(data, transform)
                return data, transform

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            logger.error(f"Error fetching LiDAR DTM data: {e}")
            raise

    def convert_dsm_to_trimesh(self, data, transform, downsample_factor=4):
        try:
            if np.isnan(data).any():
                min_val = np.nanmin(data)
                data = np.nan_to_num(data, nan=min_val)

            if downsample_factor > 1:
                print(f"DEBUG: Downsampling mesh by factor {downsample_factor} (16x triangle reduction)...")
                data = data[::downsample_factor, ::downsample_factor]
                
                from affine import Affine
                transform = transform * Affine.scale(downsample_factor, downsample_factor)

            rows, cols = data.shape

            c, r = np.meshgrid(np.arange(cols), np.arange(rows))

            xs = transform.c + c * transform.a + r * transform.b
            ys = transform.f + c * transform.d + r * transform.e

            grid = pv.StructuredGrid(xs, ys, data)

            surface = grid.extract_surface().triangulate()

            pv_faces = surface.faces.reshape(-1, 4)
            faces = pv_faces[:, 1:]

            vertices = surface.points

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            return mesh

        except Exception as e:
            logger.error(f"Error converting DSM to trimesh: {e}")
            raise

    def sample_height_for_points(self, raster_data, transform, points_xy):
        try:
            rows, cols = raster_data.shape
            z_values = []

            for x, y in points_xy:
                r, c = rasterio.transform.rowcol(transform, x, y)

                if 0 <= r < rows and 0 <= c < cols:
                    val = raster_data[r, c]
                    if np.isnan(val):
                        z_values.append(np.nanmin(raster_data))
                    else:
                        z_values.append(val)
                else:
                    z_values.append(np.nanmin(raster_data) if not np.isnan(raster_data).all() else 0)

            return np.array(z_values)

        except Exception as e:
            logger.error(f"Error sampling heights: {e}")
            return np.zeros(len(points_xy))
    def flatten_dsm_on_parcel(self, dsm_data, dtm_data, transform, parcel_geom_list, fill_with_nan=False):
        print("ROZPOCZYNAM KARCZOWANIE")
        import gc
        from rasterio.features import geometry_mask
        from pyproj import Transformer
        from shapely.ops import transform as shapely_transform
        import shapely

        if dsm_data.shape != dtm_data.shape:
            min_rows = min(dsm_data.shape[0], dtm_data.shape[0])
            min_cols = min(dsm_data.shape[1], dtm_data.shape[1])
            dsm_data = dsm_data[:min_rows, :min_cols]
            dtm_data = dtm_data[:min_rows, :min_cols]

        dsm_data = dsm_data.astype(np.float32, copy=False)
        dtm_data = dtm_data.astype(np.float32, copy=False)

        dsm_modified = dsm_data.copy()

        r_min_x = transform.c
        r_max_y = transform.f
        r_max_x = r_min_x + (dsm_data.shape[1] * transform.a)
        r_min_y = r_max_y + (dsm_data.shape[0] * transform.e)
        
        raster_box = shapely.geometry.box(
            min(r_min_x, r_max_x), min(r_min_y, r_max_y),
            max(r_min_x, r_max_x), max(r_min_y, r_max_y)
        )
        print(f"DEBUG: Raster Box: {raster_box.bounds}")

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)

        final_geoms = []
        
        for i, geom in enumerate(parcel_geom_list):
            print(f"DEBUG: Input Geom [{i}] Bounds: {geom.bounds}")
            
            if geom.intersects(raster_box):
                print("SUKCES: Wykryto geometrię w układzie natywnym (EPSG:2180).")
                final_geoms.append(geom)
                continue

            try:
                proj_geom = shapely_transform(transformer.transform, geom)
                if proj_geom.intersects(raster_box):
                    print("SUKCES: Transformacja standardowa (WGS84 -> EPSG:2180) OK.")
                    final_geoms.append(proj_geom)
                    continue
            except Exception as e:
                print(f"WARN: Błąd transformacji standardowej: {e}")

            try:
                print("WARN: Geometria nie trafia. Próba zamiany współrzędnych (Lat/Lon flip)...")
                geom_swapped = shapely.ops.transform(lambda x, y: (y, x), geom)
                proj_geom_swapped = shapely_transform(transformer.transform, geom_swapped)
                
                if proj_geom_swapped.intersects(raster_box):
                    print("SUKCES: Autokorekta (Swap) zadziałała.")
                    final_geoms.append(proj_geom_swapped)
                    continue
            except Exception as e:
                print(f"WARN: Błąd transformacji swap: {e}")

            print("ERROR: Geometria nie pasuje do rastra w żadnym wariancie.")

        if not final_geoms:
            print("ERROR: Żadna geometria nie trafiła w raster. Karczowanie anulowane.")
            return dsm_data

        try:
            mask = geometry_mask(
                final_geoms,
                transform=transform,
                out_shape=dsm_modified.shape,
                invert=True,
                all_touched=True
            )
            
            count = np.sum(mask)
            print(f"DEBUG: Piksele do spłaszczenia: {count}")
            
            if count > 0:
                if fill_with_nan:
                    dsm_modified[mask] = np.nan
                else:
                    dsm_modified[mask] = dtm_data[mask]
                print("SUKCES: Wykonano karczowanie.")
            
            del mask
            gc.collect()
            
        except Exception as e:
            print(f"ERROR masking: {e}")

        return dsm_modified

    def _to_trimesh(self, geometry) -> trimesh.Trimesh:
        if isinstance(geometry, trimesh.Scene):
            return geometry.dump(concatenate=True)
        if isinstance(geometry, trimesh.Trimesh):
            return geometry
        raise TypeError(f"Unsupported geometry type for export: {type(geometry)}")

    def export_trimesh_to_obj(self, geometry) -> bytes:
        mesh = self._to_trimesh(geometry)
        buf = io.BytesIO()
        mesh.export(file_obj=buf, file_type='obj')
        return buf.getvalue()

    def export_trimesh_to_xyz(self, geometry, precision: int = 3) -> bytes:
        mesh = self._to_trimesh(geometry)
        v = mesh.vertices
        fmt = f"{{:.{precision}f}} {{:.{precision}f}} {{:.{precision}f}}"
        lines = (fmt.format(x, y, z) for x, y, z in v)
        text = "\n".join(lines) + "\n"
        return text.encode('utf-8')


    def export_dsm_to_xyz_raw(self, data: np.ndarray, transform, precision: int = 3) -> bytes:
        rows, cols = data.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        xs = transform.c + c * transform.a + r * transform.b
        ys = transform.f + c * transform.d + r * transform.e
        zs = data
        fmt = f"{{:.{precision}f}} {{:.{precision}f}} {{:.{precision}f}}"
        lines = (fmt.format(x, y, z) for x, y, z in zip(xs.ravel(), ys.ravel(), zs.ravel()))
        text = "\n".join(lines) + "\n"
        return text.encode('utf-8')

    def export_solar_trimesh_to_obj(self, dsm_data, dtm_data, transform, parcel_geoms, grid_points_metric, sunlit_hours, max_hours=None, downsample_factor=2) -> bytes:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            logger.error("scipy is required for spatial matching in export_solar_trimesh_to_obj")
            raise

        try:
            from modules.visualization import map_sunlit_hours_to_rgba
        except ImportError:
            from visualization import map_sunlit_hours_to_rgba

        import shapely.wkt
        from shapely.geometry.base import BaseGeometry

        parsed_parcel_geoms = []
        if parcel_geoms:
            if not isinstance(parcel_geoms, (list, tuple)):
                parcel_geoms = [parcel_geoms]
            for g in parcel_geoms:
                if isinstance(g, str):
                    try:
                        parsed_parcel_geoms.append(shapely.wkt.loads(g))
                    except Exception:
                        pass
                elif isinstance(g, BaseGeometry):
                    parsed_parcel_geoms.append(g)

        if dtm_data is not None:
            min_elevation = np.nanmin(dtm_data)
            dsm_norm = dsm_data - min_elevation
            dtm_norm = dtm_data - min_elevation
        else:
            min_elevation = np.nanmin(dsm_data)
            dsm_norm = dsm_data - min_elevation
            dtm_norm = dsm_norm

        if parsed_parcel_geoms:
            dsm_flattened = self.flatten_dsm_on_parcel(dsm_norm, dtm_norm, transform, parsed_parcel_geoms, fill_with_nan=False)
        else:
            dsm_flattened = dsm_norm

        mesh = self.convert_dsm_to_trimesh(dsm_flattened, transform, downsample_factor=downsample_factor)
        
        n_faces = len(mesh.faces)
        face_colors = np.full((n_faces, 4), [210, 210, 210, 255], dtype=np.uint8)

        if grid_points_metric is not None and len(grid_points_metric) > 0 and sunlit_hours is not None and len(sunlit_hours) > 0:
            grid_points_arr = np.asarray(grid_points_metric)
            sunlit_hours_arr = np.asarray(sunlit_hours, dtype=np.float32)
            
            grid_xy = grid_points_arr[:, :2]
            
            face_centers_xy = np.mean(mesh.vertices[mesh.faces, :2], axis=1)
            
            tree = cKDTree(grid_xy)
            distances, indices = tree.query(face_centers_xy)
            
            pixel_size = abs(transform.a) if hasattr(transform, 'a') else 1.0
            max_dist = max(2.5, pixel_size * downsample_factor * 0.8)
            
            mask = distances <= max_dist
            if np.any(mask):
                matched_hours = sunlit_hours_arr[indices[mask]]
                if max_hours is None:
                    max_hours = float(np.nanmax(sunlit_hours_arr)) if len(sunlit_hours_arr) > 0 else 1.0
                
                matched_colors = map_sunlit_hours_to_rgba(matched_hours, min_val=0.0, max_val=max_hours, colormap='plasma', alpha=255)
                face_colors[mask] = matched_colors

        mesh.visual = trimesh.visual.color.ColorVisuals(mesh=mesh, face_colors=face_colors)

        try:
            unmerged = mesh.unmerge_vertices()
            if unmerged is not None:
                mesh = unmerged
        except Exception as e:
            logger.warning(f"Could not unmerge vertices: {e}")

        buf = io.BytesIO()
        mesh.export(file_obj=buf, file_type='obj')
        return buf.getvalue()




