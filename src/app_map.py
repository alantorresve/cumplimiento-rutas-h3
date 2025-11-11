# ===============================================================
# app_map.py — Streamlit + PyDeck
# Filtros: Empresa → Línea → Bus → Trip
# Puntos DENTRO (verde) si h3 ∈ hexágonos de rutas activas; FUERA (rojo) si no
# Rutas H3 planas (polígonos), sin PathLayer
# Panel "Peor cumplimiento" (empresa, línea, bus) basado en la SELECCIÓN
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

# --------- Paths (ajústalos a tu proyecto) ----------
PATH_RUTAS_H3 = Path("data/processed/rutas_h3.parquet")          # ruta_hex, h3_list
PATH_POINTS   = Path("data/processed/gps_match_points.parquet")  # latitude, longitude, h3, agency_id, ruta_hex, mean_id/identificacion, trip_id, fecha_hora
PATH_EOTS     = Path("data/raw/eots.csv")                        # catálogo empresas
PATH_RUT_CAT  = Path("data/raw/catalogo_rutas_cid.csv")          # catálogo rutas (ruta_hex, linea, ramal, origen, destino, identificacion)

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
        x = json.loads(s);  return list(x) if isinstance(x, (list, tuple, set)) else [s]
    except Exception:
        pass
    try:
        x = ast.literal_eval(s);  return list(x) if isinstance(x, (list, tuple, set)) else [s]
    except Exception:
        pass
    return [s]

def boundary_to_polygon(hcell: str) -> Polygon:
    if hasattr(h3, "h3_to_geo_boundary"):          # v3
        coords = h3.h3_to_geo_boundary(hcell, geo_json=True)
    elif hasattr(h3, "cell_to_boundary"):          # v4
        coords = h3.cell_to_boundary(hcell)
    else:
        raise RuntimeError("Librería h3 no tiene funciones de boundary.")
    ring = [(lng, lat) for lat, lng in coords]
    return Polygon(ring)

def polygon_to_coords_list(poly: Polygon):
    if poly is None or poly.is_empty:
        return []
    return [[float(x), float(y)] for x, y in poly.exterior.coords]

def norm_emp_id(x):
    if pd.isna(x): return x
    s = str(x).strip()
    return s.zfill(4) if s.isdigit() and len(s) <= 4 else s

# ----------------- Diccionarios (cache) -----------------
@st.cache_data(show_spinner=False)
def load_dim_empresas(path_eots: Path):
    try:
        eots = pd.read_csv(path_eots, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        eots = pd.read_csv(path_eots, dtype=str, encoding="latin-1")
    cand_id  = [c for c in ["eot_id","cod_catalogo","id_eot_vmt_hex","agency_id"] if c in eots.columns]
    cand_nom = [c for c in ["eot_nombre","permisionario","nombre","razon_social"] if c in eots.columns]
    if not cand_id or not cand_nom:
        return {}
    eots["_emp_id"]  = eots[cand_id[0]].apply(norm_emp_id)
    eots["_emp_nom"] = eots[cand_nom[0]].astype(str)
    eots = eots.dropna(subset=["_emp_id"]).drop_duplicates("_emp_id")
    return dict(zip(eots["_emp_id"], eots["_emp_nom"]))

@st.cache_data(show_spinner=False)
def load_dim_rutas(path_cat: Path):
    try:
        cat = pd.read_csv(path_cat, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        cat = pd.read_csv(path_cat, dtype=str, encoding="latin-1")
    cat.columns = [c.strip().lower() for c in cat.columns]
    if "ruta_hex" not in cat.columns:
        return {}
    cat["ruta_hex"] = cat["ruta_hex"].astype(str).str.upper().str.strip()
    for c in ["linea","ramal","origen","destino","identificacion"]:
        if c not in cat.columns:
            cat[c] = pd.NA
    return cat.set_index("ruta_hex")[["linea","ramal","origen","destino","identificacion"]].to_dict(orient="index")

# ----------------- Rutas H3 (cache) -----------------
@st.cache_data(show_spinner=True)
def load_rutas(path: Path):
    df = pd.read_parquet(path)
    if "ruta_hex" not in df.columns:
        raise ValueError("rutas_h3.parquet debe incluir 'ruta_hex'.")
    if "h3_list" not in df.columns:
        alt = next((c for c in ("h3_cells", "h3_hexes") if c in df.columns), None)
        if not alt:
            raise ValueError("No se encontró 'h3_list' (ni h3_cells/h3_hexes).")
        df = df.rename(columns={alt: "h3_list"})
    df["ruta_hex"] = df["ruta_hex"].astype(str).str.upper().str.strip()

    rows, hex_by_route = [], {}
    for _, row in df.iterrows():
        ruta = row["ruta_hex"]
        hlist = [str(x) for x in parse_h3_list(row["h3_list"])]
        hex_by_route[ruta] = set(hlist)
        for cell in hlist:
            try:
                poly = boundary_to_polygon(cell)
                rows.append({"ruta_hex": ruta, "h3": cell, "geometry": poly})
            except Exception:
                pass

    if not rows:
        rutas_gdf = gpd.GeoDataFrame(columns=["ruta_hex","h3","geometry"], geometry="geometry", crs=CRS)
    else:
        rutas_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
        rutas_gdf["coords"] = rutas_gdf["geometry"].apply(polygon_to_coords_list)
    return rutas_gdf, hex_by_route

# ----------------- Puntos (cache) -----------------
@st.cache_data(show_spinner=True)
def load_points(path: Path, max_points: int,
                emp_dict: dict, ruta_dict: dict) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Normalizaciones
    if "agency_id" in df.columns:
        df["agency_id"] = df["agency_id"].apply(norm_emp_id).astype(str)
    if "ruta_hex" in df.columns:
        df["ruta_hex"] = df["ruta_hex"].astype(str).str.upper().str.strip()
    for c in ("empresa_nombre","linea","ramal","origen","destino","identificacion","mean_id","trip_id","h3"):
        if c in df.columns:
            df[c] = df[c].astype(str)

    # hora
    if "fecha_hora" in df.columns:
        try:
            df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce", utc=True)
            df["hora"] = df["fecha_hora"].dt.hour
        except Exception:
            df["hora"] = pd.NA
    else:
        df["hora"] = pd.NA

    # lat/lon
    if not {"latitude","longitude"}.issubset(df.columns):
        raise ValueError("Faltan 'latitude' y/o 'longitude' en gps_match_points.parquet.")
    df["latitude"]  = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    # limitar volumen
    if len(df) > max_points:
        df = df.sample(n=int(max_points), random_state=42)

    # Diccionario empresas
    if "empresa_nombre" not in df.columns or df["empresa_nombre"].isna().all():
        df["empresa_nombre"] = df.get("agency_id", pd.Series([""]*len(df)))
    df["empresa_nombre"] = df.apply(
        lambda r: emp_dict.get(r["agency_id"], r.get("empresa_nombre","")) if pd.notna(r.get("agency_id", None)) else r.get("empresa_nombre",""),
        axis=1
    )

    # Diccionario rutas → relleno de linea/ramal/origen/destino/identificacion
    for c in ["linea","ramal","origen","destino","identificacion"]:
        if c not in df.columns:
            df[c] = pd.NA

    def fill_from_route(row, key):
        rh = row.get("ruta_hex", None)
        if (pd.isna(row.get(key)) or str(row.get(key)).strip() in ("", "nan", "None")) and rh in ruta_dict:
            val = ruta_dict[rh].get(key, None)
            if pd.notna(val) and str(val).strip() not in ("","nan","None"):
                return val
        return row.get(key)

    for key in ["linea","ramal","origen","destino","identificacion"]:
        df[key] = df.apply(lambda r, k=key: fill_from_route(r, k), axis=1)

    # fallbacks
    df["linea"] = df["linea"].fillna(df.get("ruta_hex", pd.Series(["0000"]*len(df)))).astype(str).str[:4]
    df["ramal"] = df["ramal"].fillna("—").astype(str)
    df["identificacion"] = df["identificacion"].fillna(df.get("mean_id","—")).astype(str)

    # serializables
    if "trip_id" in df.columns:
        df["trip_id_str"] = df["trip_id"].astype(str)
    else:
        df["trip_id_str"] = ""
    df["hora_str"] = df["hora"].apply(lambda x: "" if pd.isna(x) else str(int(x)))
    if "fecha_hora" in df.columns:
        df["fecha_hora_str"] = df["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    else:
        df["fecha_hora_str"] = ""

    keep = [
        "longitude","latitude","h3",
        "empresa_nombre","linea","ramal","identificacion",
        "trip_id_str","hora_str","fecha_hora_str","ruta_hex"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

# ----------------- Sidebar: carga y parámetros -----------------
st.sidebar.title("Datos")
max_points = st.sidebar.number_input(
    "Máx. puntos a cargar", min_value=10_000, max_value=2_000_000, value=400_000, step=50_000  # <- 400k por defecto
)

emp_dict  = load_dim_empresas(PATH_EOTS)
ruta_dict = load_dim_rutas(PATH_RUT_CAT)

st.write("### Cargando datos…")
rutas_gdf, hex_by_route = load_rutas(PATH_RUTAS_H3)
pts = load_points(PATH_POINTS, max_points, emp_dict, ruta_dict)
st.success(f"Datos cargados: {len(pts):,} puntos | {len(rutas_gdf):,} celdas H3")

# ----------------- Filtros en cascada: Empresa → Línea → Bus → Trip -----------------
st.sidebar.header("Filtros (en cascada)")

def _opts(df, col, all_label="(todas)"):
    if col not in df.columns: return [all_label]
    vals = sorted([x for x in df[col].dropna().astype(str).unique().tolist() if x])
    return [all_label] + vals

def _opts_any(df, col, all_label="(todos)"):
    if col not in df.columns: return [all_label]
    vals = sorted([x for x in df[col].dropna().astype(str).unique().tolist() if x])
    return [all_label] + vals

def _index_or_zero(curr, opts):
    try:
        return opts.index(curr) if curr in opts else 0
    except Exception:
        return 0

# 1) Empresa
emp_opts = _opts(pts, "empresa_nombre", "(todas)")
emp_sel  = st.sidebar.selectbox("Empresa", emp_opts, index=_index_or_zero(st.session_state.get("f_emp","(todas)"), emp_opts), key="f_emp")
df1 = pts if emp_sel == "(todas)" else pts[pts["empresa_nombre"] == emp_sel]

# 2) Línea
lin_opts = _opts(df1, "linea", "(todas)")
lin_sel  = st.sidebar.selectbox("Línea", lin_opts, index=_index_or_zero(st.session_state.get("f_lin","(todas)"), lin_opts), key="f_lin")
df2 = df1 if lin_sel == "(todas)" else df1[df1["linea"] == lin_sel]

# 3) Bus (identificacion)
bus_opts = _opts_any(df2, "identificacion", "(todos)")
bus_sel  = st.sidebar.selectbox("Bus (identificación)", bus_opts, index=_index_or_zero(st.session_state.get("f_bus","(todos)"), bus_opts), key="f_bus")
df3 = df2 if bus_sel == "(todos)" else df2[df2["identificacion"] == bus_sel]

# 4) Trip
trip_opts = _opts_any(df3, "trip_id_str", "(todos)")
trip_sel  = st.sidebar.selectbox("Trip", trip_opts, index=_index_or_zero(st.session_state.get("f_trip","(todos)"), trip_opts), key="f_trip")
sel_pts   = df3 if trip_sel == "(todos)" else df3[df3["trip_id_str"] == trip_sel]

# ----------------- DENTRO/FUERA por H3 -----------------
# Rutas activas según puntos visibles (ruta_hex)
if "ruta_hex" in sel_pts.columns and len(sel_pts):
    routes_active = sorted(sel_pts["ruta_hex"].dropna().astype(str).str.upper().unique())
else:
    routes_active = []

# Unión de hex de rutas activas; si no hay, usa todas
if routes_active:
    hex_union = set().union(*[hex_by_route.get(r, set()) for r in routes_active])
else:
    hex_union = set().union(*hex_by_route.values()) if hex_by_route else set()

# _in por H3 en la SELECCIÓN
if len(sel_pts) and "h3" in sel_pts.columns and len(hex_union) > 0:
    _in_mask = sel_pts["h3"].isin(hex_union).values
else:
    _in_mask = np.zeros(len(sel_pts), dtype=bool)

# Colores RGBA
colors = np.tile([217, 95, 2, 220], (len(sel_pts), 1))   # rojo
if len(sel_pts):
    colors[_in_mask] = [0, 158, 115, 220]                # verde
sel_pts = sel_pts.copy()
sel_pts["_in"] = _in_mask
sel_pts["_color_rgba"] = colors.tolist()

# ----------------- Tabs -----------------
tab_map, tab_worst = st.tabs(["Mapa", "Peor cumplimiento"])

with tab_map:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Capas y estilo")

    show_routes      = st.sidebar.checkbox("Mostrar rutas H3 (planas)", True)
    show_points_in   = st.sidebar.checkbox("Puntos DENTRO (verde)", True)
    show_points_out  = st.sidebar.checkbox("Puntos FUERA (rojo)", True)
    only_selected_routes = st.sidebar.checkbox("Solo rutas de la selección", True)

    point_radius     = st.sidebar.slider("Radio del punto (px)", 1, 20, 4)
    line_width       = st.sidebar.slider("Grosor borde rutas (px)", 1, 10, 3)

    # Centro del mapa
    center_lat, center_lon = -25.3, -57.6
    if not sel_pts.empty:
        center_lat = float(sel_pts["latitude"].mean())
        center_lon = float(sel_pts["longitude"].mean())

    layers = []

    # Rutas H3 planas
    rutas_plot = rutas_gdf
    if only_selected_routes and routes_active:
        rutas_plot = rutas_gdf[rutas_gdf["ruta_hex"].isin(routes_active)]
    if show_routes and not rutas_plot.empty:
        routes_data = rutas_plot[["ruta_hex","coords"]].rename(columns={"coords":"polygon"}).to_dict("records")
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

    def make_point_records(df_in: pd.DataFrame):
        if df_in.empty: return []
        cols = ["longitude","latitude","_color_rgba",
                "empresa_nombre","linea","ramal","identificacion",
                "trip_id_str","hora_str","fecha_hora_str"]
        cols = [c for c in cols if c in df_in.columns]
        recs = df_in[cols].copy()
        recs["longitude"] = recs["longitude"].astype(float)
        recs["latitude"]  = recs["latitude"].astype(float)
        for c in ("empresa_nombre","linea","ramal","identificacion","trip_id_str","hora_str","fecha_hora_str"):
            if c in recs.columns:
                recs[c] = recs[c].astype(str)
        return recs.to_dict("records")

    # Puntos DENTRO (verde)
    if show_points_in:
        data_in = make_point_records(sel_pts[sel_pts["_in"]])
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
        data_out = make_point_records(sel_pts[~sel_pts["_in"]])
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

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12)
    deck = pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "{empresa_nombre}\nLínea: {linea}\nBus: {identificacion}\nTrip: {trip_id_str}\nHora: {hora_str}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

    n_trips = sel_pts["trip_id_str"].nunique() if "trip_id_str" in sel_pts.columns else 0
    pct_in  = (float(sel_pts["_in"].sum())/len(sel_pts)*100.0) if len(sel_pts) else 0.0
    st.markdown(
        f"""
**Puntos mostrados:** {len(sel_pts):,}  
**Trips únicos (vista):** {n_trips:,}  
**% puntos DENTRO (por H3):** {pct_in:.1f}%
"""
    )

# ----------------- Peor cumplimiento (sobre la SELECCIÓN) -----------------
with tab_worst:
    st.subheader("Identificadores con menor cumplimiento (puntos dentro / total) — según selección actual")
    if sel_pts.empty or "_in" not in sel_pts.columns:
        st.info("No hay puntos en la selección actual.")
    else:
        min_pts = st.number_input("Mínimo de puntos por grupo", min_value=1, max_value=100_000, value=500, step=100)
        top_k   = st.number_input("Top N (peores)", min_value=1, max_value=100, value=10, step=1)

        def compliance_table(df, group_cols):
            if df.empty:
                return pd.DataFrame(columns=group_cols + ["pts_total","pts_dentro","pct_dentro"])
            agg = (
                df.groupby(group_cols)["_in"]
                  .agg(pts_total="count", pts_dentro="sum")
                  .reset_index()
            )
            agg = agg[agg["pts_total"] >= int(min_pts)]
            if agg.empty:
                return agg.assign(pct_dentro=pd.Series(dtype="float"))
            agg["pct_dentro"] = (agg["pts_dentro"] / agg["pts_total"]).astype(float)
            return agg.sort_values("pct_dentro", ascending=True)

        # 1) Empresas
        worst_emp = compliance_table(sel_pts, ["empresa_nombre"]).head(int(top_k))
        # 2) Rutas (línea)
        worst_linea = compliance_table(sel_pts, ["empresa_nombre","linea"]).head(int(top_k))
        # 3) Buses (identificación)
        worst_bus = compliance_table(sel_pts, ["empresa_nombre","linea","identificacion"]).head(int(top_k))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Empresas (peor % dentro)**")
            st.dataframe(worst_emp, use_container_width=True)
        with c2:
            st.markdown("**Rutas / Línea (peor % dentro)**")
            st.dataframe(worst_linea, use_container_width=True)
        with c3:
            st.markdown("**Buses (peor % dentro)**")
            st.dataframe(worst_bus, use_container_width=True)

        # Descargas
        st.download_button("Descargar empresas (CSV)", worst_emp.to_csv(index=False).encode("utf-8"), file_name="worst_empresas.csv")
        st.download_button("Descargar líneas (CSV)", worst_linea.to_csv(index=False).encode("utf-8"), file_name="worst_lineas.csv")
        st.download_button("Descargar buses (CSV)", worst_bus.to_csv(index=False).encode("utf-8"), file_name="worst_buses.csv")
