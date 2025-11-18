# ===============================================================
# app_map.py — Streamlit + PyDeck (todo en uno)
# Filtros: Empresa → Línea → Bus (mean_id) → Trip
# Mapa: Rutas H3 planas + puntos DENTRO (verde) / FUERA (rojo) por H3
# Peor cumplimiento: % de puntos dentro (sobre la selección)
# KPIs (trips): usa gps_match_trips.parquet (ratio ≥ 0.60 = OK) — sin export PNG
# Export PNG (mapa): auto-ajustado (sin bordes), con ejes y encabezado (filtros + KPI)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
from shapely.geometry import Polygon
from pathlib import Path
from io import BytesIO
import h3
import json, ast

st.set_page_config(layout="wide", page_title="Mapa de rutas y KPIs")

# --------- Paths ----------
PATH_RUTAS_H3 = Path("data/processed/rutas_h3.parquet")
PATH_POINTS   = Path("data/processed/gps_match_points.parquet")
PATH_TRIPS    = Path("data/processed/gps_match_trips.parquet")
PATH_EOTS     = Path("data/raw/eots.csv")
PATH_RUT_CAT  = Path("data/raw/catalogo_rutas_cid.csv")
CRS = "EPSG:4326"

# ----------------- Helpers -----------------
def parse_h3_list(val):
    if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
        return list(val)
    if val is None or (isinstance(val, float) and np.isnan(val)): return []
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}: return []
    for loader in (lambda x: json.loads(x), lambda x: ast.literal_eval(x)):
        try:
            x = loader(s)
            return list(x) if isinstance(x, (list, tuple, set)) else [s]
        except Exception:
            pass
    return [s]

def boundary_to_polygon(hcell: str) -> Polygon:
    if hasattr(h3, "h3_to_geo_boundary"):
        coords = h3.h3_to_geo_boundary(hcell, geo_json=True)
    elif hasattr(h3, "cell_to_boundary"):
        coords = h3.cell_to_boundary(hcell)
    else:
        raise RuntimeError("Librería h3 sin funciones de boundary.")
    ring = [(lng, lat) for lat, lng in coords]
    return Polygon(ring)

def polygon_to_coords_list(poly: Polygon):
    if poly is None or poly.is_empty: return []
    return [[float(x), float(y)] for x, y in poly.exterior.coords]

def norm_emp_id(x):
    if pd.isna(x): return x
    s = str(x).strip()
    return s.zfill(4) if s.isdigit() and len(s) <= 4 else s

# ================= util export: MAP PNG (AUTO) =================
def export_map_png(rutas_gdf, sel_pts, routes_active,
                   empresa=None, linea=None, bus=None, trip=None,
                   max_side_px=3000, dpi=300):
    """
    Exporta PNG auto-ajustado sin bordes, con ejes visibles y
    encabezado superpuesto (filtros + KPI). Usa contextily si está disponible.
    El lado mayor se limita a max_side_px a 'dpi' especificados.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    try:
        import contextily as cx
        has_cx = True
    except Exception:
        has_cx = False

    # GDF de puntos dentro/fuera en WGS84
    gdf_in  = gpd.GeoDataFrame(
        sel_pts[sel_pts["_in"]].copy(),
        geometry=gpd.points_from_xy(
            sel_pts.loc[sel_pts["_in"], "longitude"].astype(float),
            sel_pts.loc[sel_pts["_in"], "latitude"].astype(float)
        ),
        crs="EPSG:4326"
    )
    gdf_out = gpd.GeoDataFrame(
        sel_pts[~sel_pts["_in"]].copy(),
        geometry=gpd.points_from_xy(
            sel_pts.loc[~sel_pts["_in"], "longitude"].astype(float),
            sel_pts.loc[~sel_pts["_in"], "latitude"].astype(float)
        ),
        crs="EPSG:4326"
    )

    rutas_plot = rutas_gdf
    if routes_active:
        rutas_plot = rutas_gdf[rutas_gdf["ruta_hex"].isin(routes_active)]

    # Elegir CRS de trabajo para export: 3857 si hay contextily, sino 4326
    target_crs = "EPSG:3857" if has_cx else "EPSG:4326"
    for g in [rutas_plot, gdf_in, gdf_out]:
        if g is not None and not g.empty and g.crs != target_crs:
            g.to_crs(target_crs, inplace=True)

    # Extent global
    all_bounds = []
    for g in [rutas_plot, gdf_in, gdf_out]:
        if g is not None and not g.empty:
            all_bounds.append(g.total_bounds)  # minx, miny, maxx, maxy
    if all_bounds:
        b = np.array(all_bounds)
        minx, miny = np.min(b[:,0]), np.min(b[:,1])
        maxx, maxy = np.max(b[:,2]), np.max(b[:,3])
        # margen 8%
        mx, my = (maxx-minx)*0.08, (maxy-miny)*0.08
        minx, maxx = minx-mx, maxx+mx
        miny, maxy = miny-my, maxy+my
    else:
        # Extensión por defecto (AMA aprox. en EPSG:4326); se transforma si hace falta
        fallback = gpd.GeoDataFrame(geometry=[Polygon([(-57.75,-25.45),(-57.45,-25.45),(-57.45,-25.2),(-57.75,-25.2)])], crs="EPSG:4326")
        if target_crs != "EPSG:4326":
            fallback = fallback.to_crs(target_crs)
        minx, miny, maxx, maxy = fallback.total_bounds

    # Tamaño de figura auto según aspecto del extent
    width_units  = maxx - minx
    height_units = maxy - miny
    aspect = width_units / max(height_units, 1e-9)
    if aspect >= 1.0:
        width_px  = max_side_px
        height_px = int(round(width_px / max(aspect, 1e-9)))
    else:
        height_px = max_side_px
        width_px  = int(round(height_px * aspect))

    fig_w = width_px / dpi
    fig_h = height_px / dpi

    # Crear figura sin bordes
    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Basemap si hay contextily
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    if has_cx:
        try:
            cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, crs=target_crs, attribution_size=5)
        except Exception:
            pass

    # Rutas planas
    if rutas_plot is not None and not rutas_plot.empty:
        rutas_plot.boundary.plot(ax=ax, color=(0.6,0.6,0.6,0.9), linewidth=0.5)
        rutas_plot.plot(ax=ax, facecolor=(0.8,0.8,0.8,0.15), edgecolor=(0.6,0.6,0.6,0.6), linewidth=0.3)

    # Puntos
    if gdf_out is not None and len(gdf_out):
        gdf_out.plot(ax=ax, markersize=1.2, color=(217/255,95/255,2/255,0.9))
    if gdf_in is not None and len(gdf_in):
        gdf_in.plot(ax=ax, markersize=1.2, color=(0/255,158/255,115/255,0.9))

    # Ejes + grilla
    ax.grid(True, color=(0,0,0,0.08), linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color((0,0,0,0.5))
    ax.tick_params(axis='both', which='both', labelsize=6, colors=(0,0,0,0.6), length=2)

    # Leyenda
    patches = [
        Patch(color=(0/255,158/255,115/255,0.9), label='Dentro'),
        Patch(color=(217/255,95/255,2/255,0.9), label='Fuera')
    ]
    leg = ax.legend(handles=patches, loc='lower left', frameon=False, fontsize=7)
    if leg:
        for txt in leg.get_texts():
            txt.set_color((0,0,0,0.8))

    # Encabezado con filtros + KPI
    empresa = empresa or "(todas)"
    linea   = linea   or "(todas)"
    bus     = bus     or "(todos)"
    trip    = trip    or "(todos)"
    pct_in  = (sel_pts["_in"].sum() / len(sel_pts) * 100) if len(sel_pts) else 0.0
    header  = (f"Empresa: {empresa} | Línea: {linea} | Bus (mean_id): {bus} | Trip: {trip}\n"
               f"Puntos: {len(sel_pts):,}  •  % Dentro (H3): {pct_in:.1f}%")
    ax.text(0.01, 0.99, header,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8, color=(0,0,0,0.9),
            bbox=dict(facecolor=(1,1,1,0.75), edgecolor='none', pad=4))

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf

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
    if "ruta_hex" not in cat.columns: return {}
    cat["ruta_hex"] = cat["ruta_hex"].astype(str).str.upper().str.strip()
    for c in ["linea","ramal","origen","destino","identificacion"]:
        if c not in cat.columns: cat[c] = pd.NA
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
        if c in df.columns: df[c] = df[c].astype(str)

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

    # Empresas (nombre)
    if "empresa_nombre" not in df.columns or df["empresa_nombre"].isna().all():
        df["empresa_nombre"] = df.get("agency_id", pd.Series([""]*len(df)))
    df["empresa_nombre"] = df.apply(
        lambda r: emp_dict.get(r["agency_id"], r.get("empresa_nombre","")) if pd.notna(r.get("agency_id", None)) else r.get("empresa_nombre",""),
        axis=1
    )

    # Rutas → completa campos si faltan
    for c in ["linea","ramal","origen","destino","identificacion"]:
        if c not in df.columns: df[c] = pd.NA

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
    if "mean_id" not in df.columns:
        df["mean_id"] = df.get("identificacion", "—")
    df["mean_id"] = df["mean_id"].fillna("—").astype(str)

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
        "empresa_nombre","linea","ramal","mean_id",
        "trip_id_str","hora_str","fecha_hora_str","ruta_hex"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

# ----------------- Trips (cache) -----------------
@st.cache_data(show_spinner=True)
def load_trips(path: Path, emp_dict: dict, ruta_dict: dict):
    df = pd.read_parquet(path)
    for c in ("agency_id","ruta_hex","mean_id"):
        if c in df.columns: df[c] = df[c].astype(str).str.strip()
    if "agency_id" in df.columns:
        df["agency_id"] = df["agency_id"].apply(norm_emp_id)
    df["ratio"] = pd.to_numeric(df.get("ratio", np.nan), errors="coerce")
    if "trip_match" not in df.columns or df["trip_match"].isna().all():
        df["trip_match"] = df["ratio"] >= 0.60
    df["empresa_nombre"] = df["agency_id"].map(emp_dict).fillna(df.get("agency_id",""))
    if "linea" not in df.columns:
        df["linea"] = df.get("ruta_hex","").astype(str).str.upper().map(
            lambda rh: (ruta_dict.get(rh, {}) or {}).get("linea", str(rh)[:4])
        )
    if "trip_id" in df.columns:
        df["trip_id_str"] = df["trip_id"].astype(str)
    else:
        df["trip_id_str"] = ""
    if "hora" in df.columns:
        df["hora_str"] = df["hora"].apply(lambda x: "" if pd.isna(x) else str(int(x)))
    else:
        df["hora_str"] = ""
    return df

# ----------------- Sidebar: carga y parámetros -----------------
st.sidebar.title("Datos")
max_points = st.sidebar.number_input(
    "Máx. puntos a cargar", min_value=10_000, max_value=2_000_000, value=400_000, step=50_000
)

emp_dict  = load_dim_empresas(PATH_EOTS)
ruta_dict = load_dim_rutas(PATH_RUT_CAT)

st.write("### Cargando datos…")
rutas_gdf, hex_by_route = load_rutas(PATH_RUTAS_H3)
pts = load_points(PATH_POINTS, max_points, emp_dict, ruta_dict)
st.success(f"Datos cargados: {len(pts):,} puntos | {len(rutas_gdf):,} celdas H3")

# ----------------- Filtros en cascada -----------------
st.sidebar.header("Filtros")

def _opts(df, col, all_label="(todas)"):
    if col not in df.columns: return [all_label]
    vals = sorted([x for x in df[col].dropna().astype(str).unique().tolist() if x])
    return [all_label] + vals

def _opts_any(df, col, all_label="(todos)"):
    if col not in df.columns: return [all_label]
    vals = sorted([x for x in df[col].dropna().astype(str).unique().tolist() if x])
    return [all_label] + vals

def _index_or_zero(curr, opts):
    try: return opts.index(curr) if curr in opts else 0
    except Exception: return 0

# 1) Empresa
emp_opts = _opts(pts, "empresa_nombre", "(todas)")
emp_sel  = st.sidebar.selectbox("Empresa", emp_opts, index=_index_or_zero(st.session_state.get("f_emp","(todas)"), emp_opts), key="f_emp")
df1 = pts if emp_sel == "(todas)" else pts[pts["empresa_nombre"] == emp_sel]

# 2) Línea
lin_opts = _opts(df1, "linea", "(todas)")
lin_sel  = st.sidebar.selectbox("Línea", lin_opts, index=_index_or_zero(st.session_state.get("f_lin","(todas)"), lin_opts), key="f_lin")
df2 = df1 if lin_sel == "(todas)" else df1[df1["linea"] == lin_sel]

# 3) Bus (mean_id)
bus_opts = _opts_any(df2, "mean_id", "(todos)")
bus_sel  = st.sidebar.selectbox("Bus (mean_id)", bus_opts, index=_index_or_zero(st.session_state.get("f_bus","(todos)"), bus_opts), key="f_bus")
df3 = df2 if bus_sel == "(todos)" else df2[df2["mean_id"] == bus_sel]

# 4) Trip
trip_opts = _opts_any(df3, "trip_id_str", "(todos)")
trip_sel  = st.sidebar.selectbox("Trip", trip_opts, index=_index_or_zero(st.session_state.get("f_trip","(todos)"), trip_opts), key="f_trip")
sel_pts   = df3 if trip_sel == "(todos)" else df3[df3["trip_id_str"] == trip_sel]

# ----------------- DENTRO/FUERA por H3 -----------------
if "ruta_hex" in sel_pts.columns and len(sel_pts):
    routes_active = sorted(sel_pts["ruta_hex"].dropna().astype(str).str.upper().unique())
else:
    routes_active = []

if routes_active:
    hex_union = set().union(*[hex_by_route.get(r, set()) for r in routes_active])
else:
    hex_union = set().union(*hex_by_route.values()) if hex_by_route else set()

if len(sel_pts) and "h3" in sel_pts.columns and len(hex_union) > 0:
    _in_mask = sel_pts["h3"].isin(hex_union).values
else:
    _in_mask = np.zeros(len(sel_pts), dtype=bool)

colors = np.tile([217, 95, 2, 220], (len(sel_pts), 1))
if len(sel_pts):
    colors[_in_mask] = [0, 158, 115, 220]
sel_pts = sel_pts.copy()
sel_pts["_in"] = _in_mask
sel_pts["_color_rgba"] = colors.tolist()

# ----------------- Tabs -----------------
tab_map, tab_worst, tab_kpi = st.tabs(["Mapa", "Peor cumplimiento", "KPIs (trips)"])

# ===================== MAPA =====================
with tab_map:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Capas y estilo")

    show_routes      = st.sidebar.checkbox("Mostrar rutas H3 (planas)", True)
    show_points_in   = st.sidebar.checkbox("Puntos DENTRO (verde)", True)
    show_points_out  = st.sidebar.checkbox("Puntos FUERA (rojo)", True)
    only_selected_routes = st.sidebar.checkbox("Solo rutas de la selección", True)

    point_radius     = st.sidebar.slider("Radio del punto (px)", 1, 20, 4)
    line_width       = st.sidebar.slider("Grosor borde rutas (px)", 1, 10, 3)

    center_lat, center_lon = -25.3, -57.6
    if not sel_pts.empty:
        center_lat = float(sel_pts["latitude"].mean())
        center_lon = float(sel_pts["longitude"].mean())

    layers = []

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
                "empresa_nombre","linea","ramal","mean_id",
                "trip_id_str","hora_str","fecha_hora_str"]
        cols = [c for c in cols if c in df_in.columns]
        recs = df_in[cols].copy()
        recs["longitude"] = recs["longitude"].astype(float)
        recs["latitude"]  = recs["latitude"].astype(float)
        for c in ("empresa_nombre","linea","ramal","mean_id","trip_id_str","hora_str","fecha_hora_str"):
            if c in recs.columns: recs[c] = recs[c].astype(str)
        return recs.to_dict("records")

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
        tooltip={"text": "{empresa_nombre}\nLínea: {linea}\nBus (mean_id): {mean_id}\nTrip: {trip_id_str}\nHora: {hora_str}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

    n_trips = sel_pts["trip_id_str"].nunique() if "trip_id_str" in sel_pts.columns else 0
    pct_in  = (float(sel_pts["_in"].sum())/len(sel_pts)*100.0) if len(sel_pts) else 0.0
    st.markdown(
        f"""
**Puntos mostrados:** {len(sel_pts):,}  
**Trips únicos (vista):** {n_trips:,}  
**% puntos DENTRO (por H3):** {pct_in:.1f}%"""
    )

    st.markdown("---")
    st.subheader("Exportar imagen PNG del mapa (auto)")
    if st.button("Generar PNG (auto)"):
        try:
            buf = export_map_png(
                rutas_gdf, sel_pts, routes_active,
                empresa=emp_sel, linea=lin_sel, bus=bus_sel, trip=trip_sel,
                max_side_px=3000, dpi=300
            )
            st.download_button(
                "Descargar mapa (PNG)",
                data=buf,
                file_name="mapa_seleccion.png",
                mime="image/png"
            )
            st.success("PNG generado.")
        except Exception as e:
            st.error(f"No se pudo exportar PNG: {e}")

# ===================== PEOR CUMPLIMIENTO (puntos) =====================
with tab_worst:
    st.subheader("Identificadores con menor % de puntos dentro — según selección actual")
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

        worst_emp   = compliance_table(sel_pts, ["empresa_nombre"]).head(int(top_k))
        worst_linea = compliance_table(sel_pts, ["empresa_nombre","linea"]).head(int(top_k))
        worst_bus   = compliance_table(sel_pts, ["empresa_nombre","linea","mean_id"]).head(int(top_k))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Empresas**")
            st.dataframe(worst_emp, use_container_width=True)
        with c2:
            st.markdown("**Rutas / Línea**")
            st.dataframe(worst_linea, use_container_width=True)
        with c3:
            st.markdown("**Buses (mean_id)**")
            st.dataframe(worst_bus, use_container_width=True)

        st.download_button("Descargar empresas (CSV)", worst_emp.to_csv(index=False).encode("utf-8"), file_name="worst_empresas.csv")
        st.download_button("Descargar líneas (CSV)",  worst_linea.to_csv(index=False).encode("utf-8"), file_name="worst_lineas.csv")
        st.download_button("Descargar buses (CSV)",   worst_bus.to_csv(index=False).encode("utf-8"), file_name="worst_buses.csv")

# ===================== KPIs (TRIPS) =====================
with tab_kpi:
    st.subheader("KPIs por viaje (usa gps_match_trips.parquet; OK si ratio ≥ 0.60)")
    try:
        trips = load_trips(PATH_TRIPS, emp_dict, ruta_dict)
    except Exception as e:
        st.warning(f"No se pudo leer {PATH_TRIPS}: {e}")
        st.stop()

    dfk = trips.copy()

    # Alinear con filtros de la vista actual
    if emp_sel != "(todas)":
        dfk = dfk[dfk["empresa_nombre"] == emp_sel]
    if lin_sel != "(todas)" and "linea" in dfk.columns:
        dfk = dfk[dfk["linea"] == lin_sel]
    if bus_sel != "(todos)":
        # 'mean_id' es el id de bus en trips
        if "mean_id" in dfk.columns:
            dfk = dfk[dfk["mean_id"] == bus_sel]
    if trip_sel != "(todos)":
        dfk = dfk[dfk["trip_id_str"] == trip_sel]

    if dfk.empty:
        st.info("Sin viajes para la selección.")
    else:
        k = st.number_input("Top N por menor % OK", 1, 100, 10)

        def agg(gcols):
            tmp = (dfk
                   .groupby(gcols, dropna=False)["trip_match"]
                   .agg(total_trips="count", trips_ok="sum")
                   .reset_index())
            tmp["pct_ok"] = tmp["trips_ok"]/tmp["total_trips"]
            return tmp.sort_values("pct_ok").head(int(k))

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Empresas (trips)**")
            emp_tbl = agg(["empresa_nombre"])
            st.dataframe(emp_tbl, use_container_width=True)

        with c2:
            st.markdown("**Empresa + Línea (trips)**")
            cols = ["empresa_nombre","linea"] if "linea" in dfk.columns else ["empresa_nombre","ruta_hex"]
            lin_tbl = agg(cols)
            st.dataframe(lin_tbl, use_container_width=True)

        with c3:
            st.markdown("**Empresa + Línea + Bus (trips)**")
            if "linea" in dfk.columns:
                cols = ["empresa_nombre","linea","mean_id"]
            else:
                cols = ["empresa_nombre","ruta_hex","mean_id"]
            bus_tbl = agg(cols)
            st.dataframe(bus_tbl, use_container_width=True)

        st.markdown("---")
        st.markdown("**Detalle de viajes (selección actual)**")
        cols_show = [c for c in ["empresa_nombre","linea","mean_id","trip_id_str","ruta_hex","ratio","trip_match"] if c in dfk.columns]
        st.dataframe(dfk[cols_show].sort_values(["empresa_nombre","linea","mean_id","trip_id_str"]), use_container_width=True)

        # Descargas
        st.download_button("Descargar resumen empresas (CSV)", emp_tbl.to_csv(index=False).encode("utf-8"), file_name="kpi_empresas_trips.csv")
        st.download_button("Descargar resumen líneas (CSV)",   lin_tbl.to_csv(index=False).encode("utf-8"), file_name="kpi_lineas_trips.csv")
        st.download_button("Descargar resumen buses (CSV)",    bus_tbl.to_csv(index=False).encode("utf-8"), file_name="kpi_buses_trips.csv")
        st.download_button("Descargar detalle (CSV)",          dfk[cols_show].to_csv(index=False).encode("utf-8"), file_name="kpi_trips_detalle.csv")
