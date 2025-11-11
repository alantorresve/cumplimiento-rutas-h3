# ================== Utilidades y funciones ==================
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import load_config

# -------- H3 compat (v3/v4) --------
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

# --------- Fechas / limpieza ----------
def normalize_datetimes(df, col="fecha_hora"):
    dt = pd.to_datetime(df[col], errors="coerce")
    df = df.loc[dt.notna()].copy()
    df[col] = dt[dt.notna()]
    return df

def drop_point_duplicates(df):
    return df.drop_duplicates(
        subset=["agency_id","mean_id","fecha_hora","latitude","longitude","route_id"]
    ).copy()

# --------- Velocidad (para cortes) ----------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def compute_speed_kmh(df):
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

# --------- Service day (hora Paraguay) ----------
def add_service_day_py(df, col="fecha_hora"):
    # si ya tiene tz, convertimos; si no, asumimos UTC y convertimos
    if getattr(df[col].dtype, "tz", None) is None:
        dt = pd.to_datetime(df[col], errors="coerce", utc=True)
    else:
        dt = df[col]
    df["service_day"] = dt.dt.tz_convert("America/Asuncion").dt.date
    return df

# --------- Segmentación de trips (route_id, gap, velocidad) ----------
def split_trips_by_gap_and_speed(df, gap_minutes, stationary_kmh, stationary_minutes):
    df = df.sort_values(["agency_id","mean_id","service_day","route_id","fecha_hora"]).copy()
    df = compute_speed_kmh(df)

    gap = pd.Timedelta(minutes=gap_minutes)
    stationary_thresh_h = stationary_minutes / 60.0
    df["trip_id"] = -1

    group_keys = ["agency_id","mean_id","service_day","route_id"]
    for keys, grp in df.groupby(group_keys, sort=False):
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

            # Corte por gap
            if pd.notna(dt) and dt > gap:
                current_trip += 1
                quiet_hours_acc = 0.0
                df.at[i_cur, "trip_id"] = current_trip
                continue

            # Acumulación de inactividad (velocidad baja)
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

# --------- Ruta declarada → camino H3 (ordenado) ----------
def build_routeid_to_h3path(rutas_df, col_route="route_id", col_alt="ruta_hex", col_list="h3_list"):
    # Soporta parquet con 'route_id' o 'ruta_hex' como identificador
    if col_route not in rutas_df.columns and col_alt in rutas_df.columns:
        col_route = col_alt
    out = {}
    tmp = rutas_df[[col_route, col_list]].dropna().copy()
    for _, row in tmp.iterrows():
        r = str(row[col_route]).strip().upper()  # normalizar clave de ruta

        seq = row[col_list]
        # convertir numpy.ndarray a lista
        if isinstance(seq, np.ndarray):
            seq = seq.tolist()
        # asegurar lista de strings (minúsculas para H3)
        if isinstance(seq, (list, tuple)):
            seq = [str(x).lower() for x in seq]

        out[r] = seq
    return out

# --------- Secuencia H3 del trip (comprimida sin repeticiones consecutivas) ----------
def trip_h3_sequence(df_trip, res):
    hseq = []
    prev = None
    for la, lo in zip(df_trip["latitude"], df_trip["longitude"]):
        h = str(h3_cell(la, lo, res)).lower()  # h3 en minúsculas
        if h != prev:
            hseq.append(h)
            prev = h
    return hseq

# --------- Métricas hexagonales por trip (match por conjunto, no orden) ----------
def summarize_trip_hex_metrics(df_trip, ref_seq, trip_seq):
    # Conjunto único de hex del trip
    trip_set = set(trip_seq)
    trip_len = len(trip_set)

    # Conjunto de referencia
    ref_set = set(ref_seq) if isinstance(ref_seq, (list, tuple)) else set()
    inter = trip_set & ref_set
    pts_en_declared = len(inter)

    ratio = (pts_en_declared / trip_len) if trip_len else np.nan

    t0 = df_trip["fecha_hora"].min()
    hora = pd.NaT if pd.isna(t0) else int(pd.to_datetime(t0).hour)

    return {
        "pts_trip": trip_len,             # hex únicos del viaje
        "pts_en_declared": pts_en_declared,
        "ratio": ratio,
        "hora": hora
    }

# --------- Flag por punto: está en los hex de la ruta declarada ----------
def compute_in_declared_per_point(df, routeid_to_path):
    # Precompute sets para velocidad
    ref_sets = {rid: set(seq) for rid, seq in routeid_to_path.items()}
    out = []
    for rid, h in zip(df["route_id"], df["h3"]):
        seq_set = ref_sets.get(str(rid).strip().upper(), None)
        if not seq_set:
            out.append(False)
        else:
            out.append(str(h).lower() in seq_set)
    return pd.Series(out, index=df.index, dtype=bool)

def main():
    # ---- config ----
    cfg = load_config()  # TOML
    res = int(cfg["spatial"].get("h3_res", 8))

    umbral = float(cfg["metrics"].get("umbral_compliance", 0.6))
    gap_minutes = int(cfg["metrics"].get("gap_minutes", 30))
    stationary_kmh = float(cfg["metrics"].get("stationary_speed_kmh", 2.0))
    stationary_minutes = int(cfg["metrics"].get("stationary_minutes", 30))
    min_trip_hex = int(cfg["metrics"].get("min_trip_hex", 3))  # mínimo hex únicos para considerar viaje

    processed = Path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    # ---- rutas (referencia) ----
    rutas = pd.read_parquet(processed / "rutas_h3.parquet")
    rutas["ruta_hex"] = rutas["ruta_hex"].astype(str).str.upper()   # normalizar IDs

    # usar SIEMPRE ruta_hex como clave del mapping
    routeid_to_path = build_routeid_to_h3path(
        rutas, col_route="ruta_hex", col_alt="ruta_hex", col_list="h3_list"
    )

    # detectar resolución H3 del parquet (soporta list o ndarray)
    try:
        sample_seq = next(seq for seq in rutas["h3_list"]
                          if (isinstance(seq, (list, tuple)) and len(seq) > 0)
                          or (isinstance(seq, np.ndarray) and seq.size > 0))
        sample_first = str(sample_seq[0]).lower()
        try:
            import h3 as _h3
            if hasattr(_h3, "get_resolution"):        # h3 v4
                res = _h3.get_resolution(sample_first)
            else:                                      # h3 v3
                from h3 import h3 as _h3v3
                res = _h3v3.h3_get_resolution(sample_first)
        except Exception:
            pass  # si falla, se mantiene el res de config
    except StopIteration:
        pass

    # ---- gps ----
    gps_path = Path(cfg["inputs"]["gps_csv"])
    gps = pd.read_csv(gps_path)

    gps = normalize_datetimes(gps, col="fecha_hora")
    gps = drop_point_duplicates(gps)
    gps = gps.sort_values(["agency_id", "mean_id", "fecha_hora"]).reset_index(drop=True)
    gps = add_service_day_py(gps, col="fecha_hora")
    gps["route_id"] = gps["route_id"].astype(str).str.upper()       # normalizar IDs

    # h3 por punto (misma resolución que rutas)
    gps["h3"] = [str(h3_cell(la, lo, res)).lower() for la, lo in zip(gps["latitude"], gps["longitude"])]

    # in_declared por punto (conjunto de la ruta declarada)
    gps["in_declared"] = compute_in_declared_per_point(gps, routeid_to_path)

    # ---- trips por (agency_id, mean_id, service_day, route_id) ----
    gps = split_trips_by_gap_and_speed(
        gps,
        gap_minutes=gap_minutes,
        stationary_kmh=stationary_kmh,
        stationary_minutes=stationary_minutes,
    )

    # ---- resumen por trip (match hex por conjunto) ----
    grp_keys = ["agency_id", "mean_id", "service_day", "route_id", "trip_id"]
    trip_rows = []
    for keys, grp in gps.groupby(grp_keys, sort=False):
        _, _, _, rid, tid = keys
        ref_seq = routeid_to_path.get(str(rid).strip().upper(), [])
        trip_seq = trip_h3_sequence(grp, res)
        metrics = summarize_trip_hex_metrics(grp, ref_seq, trip_seq)

        # aplicar mínimo de hex únicos para considerar viaje
        effective_match = bool(metrics["ratio"] >= umbral and metrics["pts_trip"] >= min_trip_hex)

        trip_rows.append({
            "agency_id": keys[0],
            "mean_id": keys[1],
            "trip_id": tid,
            "route_id": str(rid),
            "ruta_hex": str(rid),  # compat
            "pts_en_declared": metrics["pts_en_declared"],
            "pts_trip": metrics["pts_trip"],
            "ratio": metrics["ratio"],
            "trip_match": effective_match,
            "route_id_match": effective_match,
            "hora": metrics["hora"],
        })
    trips = pd.DataFrame(trip_rows)

    # ---- exportar (mismo formato) ----
    out_points = processed / "gps_match_points.parquet"
    cols_points = [
        "agency_id", "mean_id", "trip_id", "service_day", "fecha_hora",
        "latitude", "longitude", "h3", "route_id", "ruta_hex",
        "route_candidates", "in_declared",
    ]
    # compat: 'ruta_hex' = route_id; 'route_candidates' vacío
    gps["ruta_hex"] = gps["route_id"].astype(str)
    gps["route_candidates"] = [[] for _ in range(len(gps))]

    gps[cols_points].to_parquet(out_points)

    out_trips = processed / "gps_match_trips.parquet"
    cols_trips = [
        "agency_id", "mean_id", "trip_id", "route_id", "ruta_hex",
        "pts_en_declared", "pts_trip", "ratio", "trip_match", "route_id_match", "hora",
    ]
    trips[cols_trips].to_parquet(out_trips)

    print(f"✅ Puntos (match hex) → {out_points}")
    print(f"✅ Resumen trips (hex) → {out_trips}")
    print("   ratio = |hex_trip ∩ hex_ruta| / |hex_trip_únicos| ; min_trip_hex =", min_trip_hex,
          "; trip_match = ratio ≥ umbral AND pts_trip ≥ min_trip_hex")


if __name__ == "__main__":
    main()
