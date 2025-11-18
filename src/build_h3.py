# =====================================================
# src/build_h3.py
# =====================================================

import pandas as pd
import geopandas as gpd
from shapely import wkb
from shapely.geometry import (
    LineString, LinearRing, Polygon,
    MultiLineString, MultiPolygon, GeometryCollection
)
from pathlib import Path
import numpy as np
from src.config import load_config

# -----------------------------
# Compatibilidad H3 v3 / v4
# -----------------------------
try:
    import h3 as _h3
    if hasattr(_h3, "latlng_to_cell"):  # v4
        def h3_index(lat, lon, res):
            return _h3.latlng_to_cell(lat, lon, res)
    else:
        # Algunos builds exponen h3.geo_to_h3 en _h3.h3
        try:
            from h3 import h3 as _h3v3  # v3
        except Exception:
            _h3v3 = _h3
        def h3_index(lat, lon, res):
            return _h3v3.geo_to_h3(lat, lon, res)
except ImportError:
    from h3 import h3 as _h3v3  # fallback a v3
    def h3_index(lat, lon, res):
        return _h3v3.geo_to_h3(lat, lon, res)


# =====================================================
#  WKB HEX → Shapely 
# =====================================================
def safe_wkb_load(x):
    if isinstance(x, str):
        try:
            return wkb.loads(bytes.fromhex(x))
        except Exception:
            return None
    return None


# =====================================================
# Iterar coords de cualquier geometría (lat, lon)
# =====================================================
def iter_coords(geom):
    if geom is None:
        return
    gtype = geom.geom_type

    if gtype in ("LineString", "LinearRing"):
        for x, y in geom.coords:
            yield (y, x)
    elif gtype == "Polygon":
        for x, y in geom.exterior.coords:
            yield (y, x)
        for ring in geom.interiors:
            for x, y in ring.coords:
                yield (y, x)
    elif gtype in ("MultiLineString", "MultiPolygon", "GeometryCollection"):
        for sub in getattr(geom, "geoms", []):
            yield from iter_coords(sub)
    else:
        if hasattr(geom, "coords"):
            for x, y in geom.coords:
                yield (y, x)


# =====================================================
# Geometría → lista de hexágonos H3 (por vértices)
# =====================================================
def geometry_to_h3_list(geometry, resolution):
    if geometry is None or geometry.is_empty:
        return None
    hidx = {h3_index(lat, lon, resolution) for (lat, lon) in iter_coords(geometry)}
    return list(hidx) if hidx else None


# =====================================================
# Construir huella H3 desde catálogo CSV
# =====================================================
def build_h3_from_catalog(config):
    # --- Config con defaults seguros ---
    paths_cfg = config.get("paths", {}) or {}
    inputs_cfg = config.get("inputs", {}) or {}
    spatial_cfg = config.get("spatial", {}) or {}

    processed_dir = Path(paths_cfg.get("processed", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(inputs_cfg.get("catalogo_rutas", "data/raw/catalogo_rutas_cid.csv"))
    crs = spatial_cfg.get("crs_wgs84", "EPSG:4326")
    res = int(spatial_cfg.get("h3_res", 8))

    # --- Leer CSV e interpretar geometría ---
    df = pd.read_csv(csv_path)
    df["geometry"] = df["geom"].apply(safe_wkb_load)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    # --- Limpiar vacíos ---
    before = len(gdf)
    gdf = gdf[gdf["geometry"].notna() & (~gdf["geometry"].is_empty)].copy()
    dropped = before - len(gdf)

    # --- Generar H3 por fila ---
    print(f"🔹 Generando hexágonos H3 a resolución {res}...")
    gdf["h3_list"] = gdf["geometry"].apply(lambda geom: geometry_to_h3_list(geom, res))

    # --- Guardar ---
    out_path = processed_dir / "rutas_h3.parquet"
    gdf.to_parquet(out_path)

    print(f"✅ Guardado: {out_path}")
    print(f"📦 Rutas procesadas: {len(gdf)}  | 🗑️ Sin geometría válida: {dropped}")


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    config = load_config()
    build_h3_from_catalog(config)
