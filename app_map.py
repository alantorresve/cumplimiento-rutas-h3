# app_map_layers.py — control por capas (rutas, puntos dentro, puntos fuera)
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import wkb, wkt
from shapely.geometry import base as shapely_base
from pathlib import Path
import pydeck as pdk
import numpy as np
import json

# ------------------ Paths ------------------
PATH_RUTAS_H3 = Path("data/processed/rutas_h3.parquet")         # geometry (WKB/WKT) de líneas
PATH_POINTS   = Path("data/processed/gps_match_points.parquet")  # latitude, longitude, en_ruta (opcional) o ruta_hex

# ------------------ Config página ------------------
st.set_page_config(page_title="Capas: Rutas / Dentro / Fuera", layout="wide")
st.title("Capas controlables — Rutas / Puntos dentro / Puntos fuera")

# ------------------ Sidebar: controles ------------------
with st.sidebar:
    st.header("Capas")
    show_routes   = st.checkbox("Mostrar Rutas", True)
    show_in       = st.checkbox("Mostrar Puntos DENTRO (verde)", True)
    show_out      = st.checkbox("Mostrar Puntos FUERA (rojo)", True)

    st.header("Límites y estilo")
    max_in   = st.number_input("Máx. puntos dentro", 0, 2_000_000, 150_000, step=50_000)
    max_out  = st.number_input("Máx. puntos fuera", 0, 2_000_000, 150_000, step=50_000)
    point_radius = st.slider("Radio del punto (px)", 1, 20, 4)
    line_width   = st.slider("Grosor líneas (px)", 1, 10, 3)

# ------------------ Utilidades ------------------
BBOX = dict(min_lon=-63.0, max_lon=-54.0, min_lat=-28.0, max_lat=-19.0)  # Paraguay

def in_bbox(df, lat_col="latitude", lon_col="longitude"):
    m = (
        (df[lon_col].astype(float) >= BBOX["min_lon"]) &
        (df[lon_col].astype(float) <= BBOX["max_lon"]) &
        (df[lat_col].astype(float) >= BBOX["min_lat"]) &
        (df[lat_col].astype(float) <= BBOX["max_lat"])
    )
    return df[m]

def to_shapely_geom(val):
    if val is None:
        return None
    if isinstance(val, shapely_base.BaseGeometry):
        return val
    if isinstance(val, (bytes, bytearray)):
        try: return wkb.loads(val)
        except Exception: return None
    if isinstance(val, str):
        s = val.strip()
        try: return wkb.loads(bytes.fromhex(s))  # WKB-hex
        except Exception: pass
        try: return wkt.loads(s)                 # WKT
        except Exception: pass
    return None

@st.cache_data(show_spinner=False)
def load_routes_geojson(path: Path):
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    df = pd.read_parquet(path, engine="pyarrow")
    if "geometry" not in df.columns:
        return {"type":"FeatureCollection","features":[]}
    gser = df["geometry"].apply(to_shapely_geom)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gser, crs="EPSG:4326").dropna(subset=["geometry"])
    # solo líneas
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString","MultiLineString","GeometryCollection"])]
    keep = [c for c in ["ruta_hex","linea","ramal","origen","destino","identificacion"] if c in gdf.columns]

    feats = []
    for _, r in gdf.iterrows():
        feats.append({
            "type": "Feature",
            "geometry": r.geometry.__geo_interface__,
            "properties": {k: (None if pd.isna(r.get(k)) else r.get(k)) for k in keep}
        })
    fc = {"type":"FeatureCollection","features": feats}
    # validar serialización
    json.loads(json.dumps(fc))
    return fc

@st.cache_data(show_spinner=False)
def load_points_split(path: Path, max_in: int, max_out: int):
    """Devuelve dos listas de dicts planos para pydeck: dentro y fuera."""
    if not path.exists():
        empty = []
        return empty, empty

    pts = pd.read_parquet(path, engine="pyarrow")
    if not {"latitude","longitude"}.issubset(pts.columns):
        raise ValueError("gps_match_points.parquet debe tener 'latitude' y 'longitude'.")

    pts = pts.dropna(subset=["latitude","longitude"]).copy()
    pts["latitude"]  = pts["latitude"].astype(float)
    pts["longitude"] = pts["longitude"].astype(float)
    pts = in_bbox(pts, "latitude", "longitude")

    # bandera dentro/fuera por punto
    if "en_ruta" in pts.columns:
        inside = pts["en_ruta"].astype(bool).to_numpy()
    elif "trip_match" in pts.columns:
        # fallback: todo punto de un trip OK va como "dentro"
        inside = pts["trip_match"].astype(bool).to_numpy()
    elif "ruta_hex" in pts.columns:
        inside = pts["ruta_hex"].notna().to_numpy()
    else:
        inside = np.zeros(len(pts), dtype=bool)

    pts_in  = pts[inside]
    pts_out = pts[~inside]

    # muestreo independiente por grupo
    if max_in and len(pts_in) > max_in:
        pts_in  = pts_in.sample(n=max_in, random_state=42)
    if max_out and len(pts_out) > max_out:
        pts_out = pts_out.sample(n=max_out, random_state=42)

    # construir registros planos para pydeck
    rec_in  = [{"longitude": float(lon), "latitude": float(lat), "_color": [0,158,115,200]}
               for lon, lat in zip(pts_in["longitude"],  pts_in["latitude"])]
    rec_out = [{"longitude": float(lon), "latitude": float(lat), "_color": [217,95,2,200]}
               for lon, lat in zip(pts_out["longitude"], pts_out["latitude"])]

    return rec_in, rec_out

# ------------------ Carga ------------------
routes_geojson = load_routes_geojson(PATH_RUTAS_H3)
rec_in, rec_out = load_points_split(PATH_POINTS, int(max_in), int(max_out))

# Centro del mapa: usa todos los puntos disponibles para centrar
all_pts = (rec_in or []) + (rec_out or [])
if all_pts:
    lat0 = float(np.nanmedian([r["latitude"] for r in all_pts]))
    lon0 = float(np.nanmedian([r["longitude"] for r in all_pts]))
else:
    lat0, lon0 = -25.286, -57.647  # Asunción

# ------------------ Capas ------------------
layers = []

if show_routes:
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=routes_geojson,
            stroked=True,
            filled=False,
            get_line_color=[120,120,120,220],
            get_line_width=line_width,
            lineWidthMinPixels=line_width,
            pickable=False,
        )
    )

if show_in and rec_in:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=rec_in,
            get_position=["longitude","latitude"],
            get_fill_color="_color",
            get_radius=point_radius * 3,
            radius_min_pixels=point_radius,
            radius_max_pixels=point_radius*3,
            pickable=False,
        )
    )

if show_out and rec_out:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=rec_out,
            get_position=["longitude","latitude"],
            get_fill_color="_color",
            get_radius=point_radius * 3,
            radius_min_pixels=point_radius,
            radius_max_pixels=point_radius*3,
            pickable=False,
        )
    )

# ------------------ Render ------------------
view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=11, pitch=0, bearing=0)
deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
)

# Validación antes de mostrar
try:
    _ = deck.to_json()
except Exception as e:
    st.error(f"Error de serialización de pydeck: {e}")
    st.stop()

st.pydeck_chart(deck, use_container_width=True)

# ------------------ Resumen ------------------
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Puntos dentro mostrados", f"{len(rec_in):,}")
with col_b:
    st.metric("Puntos fuera mostrados", f"{len(rec_out):,}")
with col_c:
    st.metric("Rutas (features)", f"{len(routes_geojson.get('features', [])):,}")

st.caption("Consejo: si el basemap no se ve, prueba otra red o cambia `map_style` a `None`.")
