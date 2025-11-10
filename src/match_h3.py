# =====================================================
# src/match_h3.py  — Ruta primero, trip después (¡sin duplicados!)
# =====================================================
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config

# --- Compatibilidad H3 v3/v4 ---
try:
    import h3 as _h3
    if hasattr(_h3, "latlng_to_cell"):  # v4
        def h3_cell(lat, lon, res): return _h3.latlng_to_cell(lat, lon, res)
    else:
        from h3 import h3 as _h3v3      # v3
        def h3_cell(lat, lon, res): return _h3v3.geo_to_h3(lat, lon, res)
except ImportError:
    from h3 import h3 as _h3v3
    def h3_cell(lat, lon, res): return _h3v3.geo_to_h3(lat, lon, res)


# ---------------- Utilidades básicas ----------------
def normalize_datetimes(df, col="fecha_hora"):
    """Parsea fechas a UTC, descarta inválidas."""
    dt = pd.to_datetime(df[col], errors="coerce", utc=True)
    df = df.loc[dt.notna()].copy()
    df[col] = dt[dt.notna()]
    return df

def drop_point_duplicates(df):
    """Quita duplicados exactos para no inflar puntos."""
    return df.drop_duplicates(subset=["agency_id","mean_id","fecha_hora","latitude","longitude"]).copy()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def compute_speed_kmh(df):
    """Velocidad aprox entre puntos consecutivos del mismo bus (km/h)."""
    df = df.sort_values(["agency_id","mean_id","fecha_hora"]).copy()
    lat_prev = df.groupby(["agency_id","mean_id"])["latitude"].shift()
    lon_prev = df.groupby(["agency_id","mean_id"])["longitude"].shift()
    t_prev   = df.groupby(["agency_id","mean_id"])["fecha_hora"].shift()

    dist_km = haversine_km(df["latitude"], df["longitude"], lat_prev, lon_prev)
    dt_h = (df["fecha_hora"] - t_prev).dt.total_seconds() / 3600.0
    speed = dist_km / dt_h
    speed[~np.isfinite(speed)] = np.nan
    df["speed_kmh"] = speed
    return df


# ---------------- H3 → rutas (sin duplicar filas) ----------------
def build_h3_routes_map(rutas_df):
    """
    Devuelve un dict: h3_index (str) -> set({ruta_hex, ...})
    a partir de rutas_h3.parquet (col 'h3_list', 'ruta_hex').
    """
    rutas_hex = rutas_df.explode("h3_list")[["ruta_hex","h3_list"]].dropna()
    rutas_hex["h3_list"] = rutas_hex["h3_list"].astype(str)
    mapping = {}
    for h, group in rutas_hex.groupby("h3_list"):
        mapping[h] = set(group["ruta_hex"].astype(str))
    return mapping

def candidate_routes_for_point(h3_idx, h3_to_routes):
    """Conjunto de rutas candidatas que contienen este h3 (puede ser vacío)."""
    return list(h3_to_routes.get(str(h3_idx), []))


# ---------------- Ruta dominante por ventana ----------------
def rolling_route_label(route_candidates_series, window_points, min_persist_points):
    """
    route_candidates_series: Serie de listas (candidatos por punto) ordenada por tiempo.
    Regresa una Serie con una etiqueta de ruta dominante por punto (str o NaN),
    usando ventana de tamaño 'window_points' y confirmación por 'min_persist_points'
    (histeresis) para evitar saltos por ruido.
    """
    n = len(route_candidates_series)
    labels = [None] * n

    # Función para ruta mayoritaria en una ventana de listas
    def window_mode(i):
        a = max(0, i - window_points//2)
        b = min(n, i + window_points//2 + 1)
        counts = {}
        for lst in route_candidates_series.iloc[a:b]:
            if isinstance(lst, list):
                for r in lst:
                    counts[r] = counts.get(r, 0) + 1
        if not counts:
            return None
        # ruta más frecuente en la ventana
        return max(counts, key=counts.get)

    # generar label tentativa por ventana
    tentative = [window_mode(i) for i in range(n)]

    # aplicar histeresis: confirmar cambios solo si se sostienen 'min_persist_points'
    current = tentative[0]
    consec_new = 0
    for i in range(n):
        lab = tentative[i]
        if lab == current or current is None:
            current = lab if current is None else current
            consec_new = 0
        else:
            consec_new += 1
            if consec_new >= min_persist_points:
                current = lab
                consec_new = 0
        labels[i] = current
    return pd.Series(labels, index=route_candidates_series.index, dtype="object")


# ---------------- Trips dentro de bloques de ruta ----------------
def split_trips_within_route_blocks(df, gap_minutes, stationary_kmh, stationary_minutes):
    """
    Dentro de cada (agency_id, mean_id, service_day, ruta_hex) corta el trip cuando:
      - hay gap > gap_minutes, o
      - inactividad sostenida (speed < stationary_kmh) acumulada >= stationary_minutes
    """
    df = df.sort_values(["agency_id","mean_id","service_day","fecha_hora"]).copy()
    df = compute_speed_kmh(df)

    gap = pd.Timedelta(minutes=gap_minutes)
    stationary_thresh_h = stationary_minutes / 60.0
    df["trip_id"] = -1

    for keys, grp in df.groupby(["agency_id","mean_id","service_day","ruta_hex"], sort=False):
        idx = grp.index.to_list()
        if not idx:
            continue
        current_trip = 0
        df.at[idx[0], "trip_id"] = current_trip
        quiet_hours_acc = 0.0

        for k in range(1, len(idx)):
            i_prev, i_cur = idx[k-1], idx[k]
            t_prev = df.at[i_prev, "fecha_hora"]
            t_cur  = df.at[i_cur,  "fecha_hora"]
            dt = t_cur - t_prev

            # gap temporal
            if pd.notna(dt) and dt > gap:
                current_trip += 1
                quiet_hours_acc = 0.0
                df.at[i_cur, "trip_id"] = current_trip
                continue

            # inactividad sostenida
            spd = df.at[i_cur, "speed_kmh"]
            dt_h = (dt.total_seconds()/3600.0) if pd.notna(dt) else 0.0
            if pd.notna(spd) and spd < stationary_kmh:
                quiet_hours_acc += dt_h
            else:
                quiet_hours_acc = 0.0

            if quiet_hours_acc >= stationary_thresh_h:
                current_trip += 1
                quiet_hours_acc = 0.0

            df.at[i_cur, "trip_id"] = current_trip

    return df


# ---------------- Pipeline principal ----------------
def main():
    cfg = load_config()

    # --- parámetros (con defaults mejorados) ---
    # Hexágonos más grandes por defecto (7); puedes subir a 6 si querés aún más grandes.
    res = int(cfg["spatial"].get("h3_res", 7))

    umbral = float(cfg["metrics"].get("umbral_compliance", 0.6))  # 60%
    gap_minutes = int(cfg["metrics"].get("gap_minutes", 30))

    # Inactividad / ruido
    stationary_kmh = float(cfg["metrics"].get("stationary_speed_kmh", 2.0))    # quieto ~2 km/h
    stationary_minutes = int(cfg["metrics"].get("stationary_minutes", 30))     # 30 min

    # Dominancia H3 (ventana y persistencia)
    route_window_points = int(cfg["metrics"].get("route_window_points", 9))        # tamaño ventana
    route_switch_min_points = int(cfg["metrics"].get("route_switch_min_points", 6))# confirmación cambio

    processed = Path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    # 1) Rutas H3
    rutas = pd.read_parquet(processed / "rutas_h3.parquet")
    h3_to_routes = build_h3_routes_map(rutas)

    # 2) GPS
    gps_path = Path(cfg["inputs"]["gps_csv"])
    gps = pd.read_csv(gps_path)

    # Limpieza básica: fechas válidas, sin duplicados, orden
    gps = normalize_datetimes(gps, col="fecha_hora")
    gps = drop_point_duplicates(gps)
    gps = gps.sort_values(["agency_id","mean_id","fecha_hora"]).reset_index(drop=True)

    # 3) H3 por punto (hexágono más grande)
    gps["h3"] = gps.apply(lambda r: h3_cell(r["latitude"], r["longitude"], res), axis=1).astype(str)

    # Día de servicio (para cortar por días)
    gps["service_day"] = gps["fecha_hora"].dt.tz_convert("UTC").dt.date

    # 4) Rutas candidatas por punto (lista) — sin duplicar filas
    gps["route_candidates"] = gps["h3"].apply(lambda h: candidate_routes_for_point(h, h3_to_routes))

    # 5) Ruta dominante por ventana, por (agencia, bus, día)
    gps["ruta_hex"] = None  # etiqueta de ruta real (dominante) por punto
    for keys, grp in gps.groupby(["agency_id","mean_id","service_day"], sort=False):
        idx = grp.index
        labels = rolling_route_label(gps.loc[idx,"route_candidates"], route_window_points, route_switch_min_points)
        gps.loc[idx, "ruta_hex"] = labels

    # 6) Trips dentro de bloques de ruta
    gps = split_trips_within_route_blocks(
        gps,
        gap_minutes=gap_minutes,
        stationary_kmh=stationary_kmh,
        stationary_minutes=stationary_minutes
    )

    # 7) Métricas por trip (sin duplicar puntos)
    grp_keys = ["agency_id","mean_id","trip_id"]

    # puntos cuyo H3 incluye la ruta declarada (cumplimiento vs 'route_id')
    def in_declared(row):
        rid = str(row.get("route_id"))
        cands = row.get("route_candidates") or []
        return rid in [str(x) for x in cands]

    gps["in_declared"] = gps.apply(in_declared, axis=1)

    # totales
    trip_tot = gps.groupby(grp_keys, as_index=False).size().rename(columns={"size":"pts_trip"})
    trip_declared = gps.groupby(grp_keys, as_index=False)["in_declared"].sum().rename(columns={"in_declared":"pts_en_declared"})

    # ruta dominante REAL del trip (modo sobre 'ruta_hex' por punto)
    trip_real = (
        gps.dropna(subset=["ruta_hex"])
           .groupby(grp_keys + ["ruta_hex"], as_index=False)
           .size()
    )
    if len(trip_real):
        dom_idx = trip_real.groupby(grp_keys)["size"].idxmax()
        trip_real_dom = trip_real.loc[dom_idx, grp_keys + ["ruta_hex"]]
    else:
        trip_real_dom = pd.DataFrame(columns=grp_keys + ["ruta_hex"])

    # route_id declarado dominante del trip (por si mezcla etiquetas)
    trip_declared_mode = (
        gps.groupby(grp_keys + ["route_id"], as_index=False).size()
    )
    decl_idx = trip_declared_mode.groupby(grp_keys)["size"].idxmax()
    trip_declared_mode = trip_declared_mode.loc[decl_idx, grp_keys + ["route_id"]]

    # hora de inicio del trip
    trip_start = (
        gps.groupby(grp_keys, as_index=False)["fecha_hora"]
           .min()
           .rename(columns={"fecha_hora":"trip_start"})
    )
    trip_start["hora"] = trip_start["trip_start"].dt.hour.astype("Int64")

    # armar resumen trips
    trips = (trip_tot
             .merge(trip_declared, on=grp_keys, how="left")
             .merge(trip_real_dom, on=grp_keys, how="left")
             .merge(trip_declared_mode, on=grp_keys, how="left")
             .merge(trip_start[grp_keys + ["hora"]], on=grp_keys, how="left"))

    trips["pts_en_declared"] = trips["pts_en_declared"].fillna(0)
    trips["ratio"] = trips["pts_en_declared"] / trips["pts_trip"]
    trips["trip_match"] = trips["ratio"] >= umbral
    trips["route_id_match"] = trips["ruta_hex"].astype(str) == trips["route_id"].astype(str)

    # 8) Exportar — sin duplicar puntos
    out_points = processed / "gps_match_points.parquet"
    cols_points = [
        "agency_id","mean_id","trip_id","service_day","fecha_hora",
        "latitude","longitude","h3","route_id","ruta_hex",
        "route_candidates","in_declared"
    ]
    gps[cols_points].to_parquet(out_points)

    out_trips = processed / "gps_match_trips.parquet"
    cols_trips = [
        "agency_id","mean_id","trip_id","route_id","ruta_hex",
        "pts_en_declared","pts_trip","ratio","trip_match","route_id_match","hora"
    ]
    trips[cols_trips].to_parquet(out_trips)

    print(f"✅ Puntos (sin duplicados) → {out_points}")
    print(f"✅ Resumen de trips      → {out_trips}")
    print("   Trips incluyen: route_id (declarada), ruta_hex (real dominante), route_id_match, hora, ratio (vs declarada), trip_match (≥ umbral).")


if __name__ == "__main__":
    main()
