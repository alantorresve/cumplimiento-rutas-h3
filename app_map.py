# ===============================================================
# app_map.py — Visualización interactiva (Streamlit + PyDeck)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
from shapely.geometry import Polygon
from pathlib import Path
import h3
import json, ast

st.set_page_config(layout="wide", page_title="Mapa de rutas y puntos")

# Ajusta estos paths a tu proyecto
PATH_RUTAS_H3 = Path("data/processed/rutas_h3.parquet")          # ruta_hex, h3_list
PATH_POINTS   = Path("data/processed/gps_match_points.parquet")  # latitude, longitude, fecha_hora, mean_id, trip_id, ruta_hex, inside_buffer/en_ruta
CRS = "EPSG:4326"

# ----------------- Helpers -----------------
def parse_h3_list(val):
    if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
        return list(val)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []
    try:
        x = json.loads(s)
        return list(x) if isinstance(x, (list, tuple, set)) else [s]
    except Exception:
        pass
    try:
        x = ast.literal_eval(s)
        return list(x) if isinstance(x, (list, tuple, set)) else [s]
    except Exception:
        pass
    return [s]

def boundary_to_polygon(hcell: str) -> Polygon:
    if hasattr(h3, "h3_to_geo_boundary"):          # v3
        coords = h3.h3_to_geo_boundary(hcell, geo_json=True)  # [(lat, lng), ...]
    elif hasattr(h3, "cell_to_boundary"):          # v4
        coords = h3.cell_to_boundary(hcell)                   # [(lat, lng), ...]
    else:
        raise RuntimeError("Librería h3 no tiene funciones de boundary.")
    ring = [(lng, lat) for lat, lng in coords]
    return Polygon(ring)

def polygon_to_coords_list(poly: Polygon):
    if poly is None or poly.is_empty:
        return []
    return [[float(x), float(y)] for x, y in poly.exterior.coords]

# ----------------- Carga de datos (cache) -----------------
@st.cache_data(show_spinner=True)
def load_rutas(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_parquet(path)
    if "ruta_hex" not in df.columns:
        raise ValueError("rutas_h3.parquet debe incluir 'ruta_hex'.")
    if "h3_list" not in df.columns:
        alt = next((c for c in ("h3_cells", "h3_hexes") if c in df.columns), None)
        if not alt:
            raise ValueError("No se encontró 'h3_list' (ni h3_cells/h3_hexes).")
        df = df.rename(columns={alt: "h3_list"})
    df["ruta_hex"] = df["ruta_hex"].astype(str).str.upper().str.strip()

    polys = []
    for _, row in df.iterrows():
        ruta = row["ruta_hex"]
        for cell in parse_h3_list(row["h3_list"]):
            try:
                poly = boundary_to_polygon(str(cell))
                polys.append({"ruta_hex": ruta, "geometry": poly})
            except Exception:
                pass

    if not polys:
        return gpd.GeoDataFrame(columns=["ruta_hex", "geometry"], geometry="geometry", crs=CRS)

    gdf = gpd.GeoDataFrame(polys, geometry="geometry", crs=CRS)
    gdf["coords"] = gdf["geometry"].apply(polygon_to_coords_list)
    return gdf

@st.cache_data(show_spinner=True)
def load_points(path: Path, max_points: int = 300_000) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Normalizaciones de texto/ids
    for c in ("agency_id", "empresa_nombre", "linea", "ramal", "identificacion", "mean_id", "trip_id", "ruta_hex"):
        if c in df.columns:
            df[c] = df[c].astype(str)

    # trip_uid
    if {"mean_id", "trip_id"}.issubset(df.columns):
        df["_trip_uid"] = df["mean_id"].astype(str) + "§" + df["trip_id"].astype(str)
    else:
        df["_trip_uid"] = None

    # hora
    if "fecha_hora" in df.columns:
        try:
            df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce", utc=True)
            df["hora"] = df["fecha_hora"].dt.hour
        except Exception:
            df["hora"] = pd.NA
    else:
        df["hora"] = pd.NA

    # flag de puntos dentro (robusto)
    if "inside_buffer" in df.columns:
        df["_in"] = df["inside_buffer"]
    elif "en_ruta" in df.columns:
        df["_in"] = df["en_ruta"]
    elif "ruta_hex" in df.columns:
        df["_in"] = df["ruta_hex"].notna()
    else:
        df["_in"] = False
    df["_in"] = pd.Series(df["_in"]).fillna(False).astype(bool).values  # bool puro

    # asegurar lat/lon
    if not {"latitude", "longitude"}.issubset(df.columns):
        raise ValueError("Faltan 'latitude' y/o 'longitude' en gps_match_points.parquet.")
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    # limitar volumen
    if len(df) > max_points:
        df = df.sample(n=int(max_points), random_state=42)

    # fallbacks de nombre si faltan
    if "empresa_nombre" not in df.columns:
        df["empresa_nombre"] = df.get("agency_id", "Empresa")
    if "linea" not in df.columns:
        df["linea"] = df.get("ruta_hex", pd.Series(["0000"] * len(df))).str[:4]
    if "ramal" not in df.columns:
        df["ramal"] = "—"
    if "identificacion" not in df.columns:
        df["identificacion"] = df["mean_id"].astype(str)

    # colores robustos (sin broadcasting)
    colors = np.tile([217, 95, 2, 220], (len(df), 1))        # rojo por defecto
    colors[df["_in"]] = [0, 158, 115, 220]                   # verde si _in=True
    df["_color_rgba"] = colors.tolist()

    # columnas “serializables” extra (strings)
    if "trip_id" in df.columns:
        df["trip_id_str"] = df["trip_id"].astype(str)
    else:
        df["trip_id_str"] = ""

    df["hora_str"] = df["hora"].apply(lambda x: "" if pd.isna(x) else str(int(x)))
    if "fecha_hora" in df.columns:
        df["fecha_hora_str"] = df["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    else:
        df["fecha_hora_str"] = ""

    # dataset mínimo para mapas (evita objetos no JSON-serializables)
    safe_cols = [
        "longitude", "latitude", "_color_rgba",
        "empresa_nombre", "linea", "ramal", "identificacion",
        "trip_id_str", "hora_str", "fecha_hora_str", "ruta_hex"
    ]
    # Añadir solo si existen
    existing = [c for c in safe_cols if c in df.columns]
    df_safe = df[existing].copy()

    return df_safe

# ----------------- Sidebar: carga y parámetros -----------------
st.sidebar.title("Datos")
max_points = st.sidebar.number_input(
    "Máx. puntos a cargar", min_value=10_000, max_value=2_000_000, value=200_000, step=50_000
)

st.write("### Cargando datos…")
rutas_gdf = load_rutas(PATH_RUTAS_H3)
pts       = load_points(PATH_POINTS, max_points)
st.success(f"Datos cargados: {len(pts):,} puntos | {len(rutas_gdf):,} celdas H3")

# ----------------- Filtros en cascada -----------------
st.sidebar.header("Filtros (en cascada)")

def _opts(df, col, all_label="(todas)"):
    if col not in df.columns:
        return [all_label]
    vals = sorted([x for x in df[col].dropna().astype(str).unique().tolist() if x])
    return [all_label] + vals

def _opts_any(df, col, all_label="(todos)"):
    if col not in df.columns:
        return [all_label]
    vals = sorted([x for x in df[col].dropna().astype(str).unique().tolist() if x])
    return [all_label] + vals

def _index_or_zero(current, options):
    try:
        return options.index(current) if current in options else 0
    except Exception:
        return 0

# 1) Empresa
emp_options = _opts(pts, "empresa_nombre", "(todas)")
emp_curr    = st.session_state.get("f_emp", "(todas)")
emp_sel     = st.sidebar.selectbox("Empresa", emp_options, index=_index_or_zero(emp_curr, emp_options), key="f_emp")
df1 = pts if emp_sel == "(todas)" else pts[pts["empresa_nombre"] == emp_sel]

# 2) Línea
lin_options = _opts(df1, "linea", "(todas)")
lin_curr    = st.session_state.get("f_lin", "(todas)")
lin_sel     = st.sidebar.selectbox("Línea", lin_options, index=_index_or_zero(lin_curr, lin_options), key="f_lin")
df2 = df1 if lin_sel == "(todas)" else df1[df1["linea"] == lin_sel]

# 3) Ramal
ram_options = _opts_any(df2, "ramal", "(todos)")
ram_curr    = st.session_state.get("f_ram", "(todos)")
ram_sel     = st.sidebar.selectbox("Ramal", ram_options, index=_index_or_zero(ram_curr, ram_options), key="f_ram")
df3 = df2 if ram_sel == "(todos)" else df2[df2["ramal"] == ram_sel]

# 4) Identificación (bus)
idn_options = _opts_any(df3, "identificacion", "(todos)")
idn_curr    = st.session_state.get("f_idn", "(todos)")
idn_sel     = st.sidebar.selectbox("Identificación (bus)", idn_options, index=_index_or_zero(idn_curr, idn_options), key="f_idn")
df4 = df3 if idn_sel == "(todos)" else df3[df3["identificacion"] == idn_sel]

# 5) Trip
trip_options = _opts_any(df4, "trip_id_str", "(todos)")
trip_curr    = st.session_state.get("f_trip", "(todos)")
trip_sel     = st.sidebar.selectbox("Trip", trip_options, index=_index_or_zero(trip_curr, trip_options), key="f_trip")
df5 = df4 if trip_sel == "(todos)" else df4[df4["trip_id_str"] == trip_sel]

# 6) Hora
if "hora_str" in df5.columns:
    horas_vals   = sorted([h for h in df5["hora_str"].dropna().unique().tolist() if h != ""])
    hora_options = ["(todas)"] + horas_vals
else:
    hora_options = ["(todas)"]
hr_curr = st.session_state.get("f_hr", "(todas)")
hr_sel  = st.sidebar.selectbox("Hora", hora_options, index=_index_or_zero(hr_curr, hora_options), key="f_hr")
sel_pts = df5 if hr_sel == "(todas)" else df5[df5["hora_str"] == hr_sel]

# ----------------- Parámetros de capas -----------------
st.sidebar.markdown("---")
st.sidebar.subheader("Capas y estilo")

show_routes      = st.sidebar.checkbox("Mostrar rutas H3 (planas)", True)
show_points_in   = st.sidebar.checkbox("Puntos DENTRO (verde)", True)
show_points_out  = st.sidebar.checkbox("Puntos FUERA (rojo)", True)
show_trip_paths  = st.sidebar.checkbox("Dibujar caminos del trip (si ≤5 trips)", True)

point_radius     = st.sidebar.slider("Radio del punto (px)", 1, 20, 4)
line_width       = st.sidebar.slider("Grosor líneas (px)", 1, 10, 3)
only_selected_routes = st.sidebar.checkbox("Solo rutas de la selección", True)

# ----------------- Construcción de datos mínimos por capa -----------------
# Centro del mapa
center_lat, center_lon = -25.3, -57.6
if not sel_pts.empty:
    center_lat = float(sel_pts["latitude"].mean())
    center_lon = float(sel_pts["longitude"].mean())

layers = []

# Rutas H3 planas
if show_routes and not rutas_gdf.empty:
    if only_selected_routes and "ruta_hex" in sel_pts.columns:
        sel_rutas = sel_pts["ruta_hex"].dropna().astype(str).str.upper().unique().tolist()
        rutas_plot = rutas_gdf[rutas_gdf["ruta_hex"].isin(sel_rutas)]
    else:
        rutas_plot = rutas_gdf

    if not rutas_plot.empty:
        routes_data = rutas_plot[["ruta_hex", "coords"]].rename(columns={"coords": "polygon"}).to_dict("records")
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=routes_data,
                get_polygon="polygon",
                get_fill_color=[200, 200, 200, 40],
                get_line_color=[150, 150, 150, 150],
                line_width_min_pixels=line_width,
                stroked=True,
                filled=True,
                pickable=False,
            )
        )

# Función para construir registros de puntos serializables
def make_point_records(df_in: pd.DataFrame):
    if df_in.empty:
        return []
    cols = ["longitude", "latitude", "_color_rgba",
            "empresa_nombre", "linea", "ramal", "identificacion",
            "trip_id_str", "hora_str", "fecha_hora_str"]
    cols = [c for c in cols if c in df_in.columns]
    recs = df_in[cols].copy()

    # asegurar tipos nativos JSON
    recs["longitude"] = recs["longitude"].astype(float)
    recs["latitude"]  = recs["latitude"].astype(float)
    for c in ("empresa_nombre", "linea", "ramal", "identificacion", "trip_id_str", "hora_str", "fecha_hora_str"):
        if c in recs.columns:
            recs[c] = recs[c].astype(str)
    return recs.to_dict("records")

# Puntos DENTRO (verde)
if show_points_in:
    pts_in = sel_pts[sel_pts["_color_rgba"].apply(lambda c: c[1] == 158 if isinstance(c, list) and len(c) == 4 else False)]
    data_in = make_point_records(pts_in)
    if data_in:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=data_in,
                get_position="[longitude, latitude]",
                get_color="_color_rgba",
                get_radius=point_radius,
                pickable=True,
            )
        )

# Puntos FUERA (rojo)
if show_points_out:
    pts_out = sel_pts[sel_pts["_color_rgba"].apply(lambda c: c[1] != 158 if isinstance(c, list) and len(c) == 4 else True)]
    data_out = make_point_records(pts_out)
    if data_out:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=data_out,
                get_position="[longitude, latitude]",
                get_color="_color_rgba",
                get_radius=point_radius,
                pickable=True,
            )
        )

# Path del/los trip(s) (si pocos seleccionados)
if show_trip_paths and "trip_id_str" in sel_pts.columns:
    unique_trips = list(pd.unique(sel_pts["trip_id_str"]))
    if len(unique_trips) <= 5 and len(unique_trips) > 0:
        for tid in unique_trips:
            grp = sel_pts[sel_pts["trip_id_str"] == tid].copy()
            # ordenar por fecha_hora_str si existe
            if "fecha_hora_str" in grp.columns:
                # convertir a datetime local (seguro) o dejar string sorting
                try:
                    grp["_fh"] = pd.to_datetime(grp["fecha_hora_str"], errors="coerce")
                    grp = grp.sort_values("_fh")
                except Exception:
                    grp = grp.sort_values("fecha_hora_str")
            path = grp[["longitude", "latitude"]].dropna().values.tolist()
            if len(path) > 1:
                layers.append(
                    pdk.Layer(
                        "PathLayer",
                        data=[{"path": path}],
                        get_path="path",
                        get_color=[0, 100, 200, 180],
                        width_scale=10,
                        width_min_pixels=2,
                    )
                )

# Basemap Carto
view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12)
deck = pdk.Deck(
    map_provider="carto",
    map_style="light",
    initial_view_state=view_state,
    layers=layers,
    tooltip={"text": "{empresa_nombre}\nLínea: {linea}\nBus: {identificacion}\nTrip: {trip_id_str}\nHora: {hora_str}"},
)

st.pydeck_chart(deck, use_container_width=True)

# KPIs rápidos
n_trips = sel_pts["trip_id_str"].nunique() if "trip_id_str" in sel_pts.columns else 0

# como proxy de % dentro, contamos por color (verde tiene G=158)
if "_color_rgba" in sel_pts.columns and not sel_pts.empty:
    is_in = sel_pts["_color_rgba"].apply(lambda c: isinstance(c, list) and len(c) == 4 and c[1] == 158)
    pct_in = float(is_in.sum()) / float(len(sel_pts)) * 100.0
else:
    pct_in = 0.0

st.markdown(
    f"""
**Puntos mostrados:** {len(sel_pts):,}  
**Trips únicos:** {n_trips:,}  
**% puntos DENTRO:** {pct_in:.1f}%
"""
)
