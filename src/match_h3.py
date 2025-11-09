# =====================================================
# src/match_h3.py
# =====================================================
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from src.config import load_config
from datetime import timedelta

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


# ---------- util ----------
def bearing_deg(lat1, lon1, lat2, lon2):
    """Rumboa (grados) 0-360 usando fórmula de rumbo."""
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    dlon = np.radians(lon2 - lon1)
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r)*np.sin(lat2r) - np.sin(lat1r)*np.cos(lat2r)*np.cos(dlon)
    brng = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return brng

def quadrant_from_bearing(b):
    if pd.isna(b): return None
    # NESW por cuadrante de 90°
    if   45 <= b < 135:  return "E"
    elif 135 <= b < 225: return "S"
    elif 225 <= b < 315: return "W"
    else:                return "N"

def assign_trip_ids(df, min_steps, gap_minutes):
    """
    Asigna trip_id por (agency_id, mean_id), cortando por:
      - cambio sostenido de cuadrante (min_steps)
      - gap de tiempo > gap_minutes
    """
    df = df.sort_values(["agency_id", "mean_id", "fecha_hora"]).copy()
    df["trip_id"] = -1

    gap = pd.Timedelta(minutes=gap_minutes)

    for (ag, bus), grp in df.groupby(["agency_id", "mean_id"], sort=False):
        idx = grp.index.to_list()
        # bearings entre puntos consecutivos
        b = [np.nan]
        for i in range(1, len(idx)):
            p0, p1 = idx[i-1], idx[i]
            b.append(bearing_deg(df.at[p0,"latitude"], df.at[p0,"longitude"],
                                 df.at[p1,"latitude"], df.at[p1,"longitude"]))
        q = [quadrant_from_bearing(x) for x in b]
        # gaps
        t = pd.to_datetime(grp["fecha_hora"])
        gaps = [pd.NaT] + list(t.values[1:] - t.values[:-1])

        # recorre y asigna trips
        current_trip = 0
        pending_dir = None
        consec = 0
        prev_q = q[0]

        for k, irow in enumerate(idx):
            # cortar por gap grande
            if k > 0 and isinstance(gaps[k], pd.Timedelta) and gaps[k] > gap:
                current_trip += 1
                pending_dir = None
                consec = 0
                prev_q = q[k]
            else:
                # evaluar cambio de cuadrante
                if prev_q is None:
                    prev_q = q[k]
                elif q[k] is None:
                    pass
                elif q[k] != prev_q:
                    # contar pasos consistentes para confirmar cambio
                    if pending_dir is None or pending_dir != q[k]:
                        pending_dir = q[k]
                        consec = 1
                    else:
                        consec += 1
                        if consec >= min_steps:
                            current_trip += 1
                            prev_q = q[k]
                            pending_dir = None
                            consec = 0
                else:
                    # misma dirección, limpiar ventana
                    pending_dir = None
                    consec = 0

            df.at[irow, "trip_id"] = current_trip

    return df


def main():
    cfg = load_config()
    res = int(cfg["spatial"]["h3_res"])
    umbral = float(cfg["metrics"]["umbral_compliance"])
    min_steps = int(cfg["metrics"]["min_consistent_steps"])
    gap_minutes = int(cfg["metrics"]["gap_minutes"])

    processed = Path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    # 1) Cargar huella H3 de rutas (salida de build_h3)
    rutas = pd.read_parquet(processed / "rutas_h3.parquet")
    # Explode a formato largo: una fila por (ruta_hex, h3)
    rutas_hex = rutas.explode("h3_list")[["ruta_hex", "h3_list"]].dropna()
    rutas_hex = rutas_hex.rename(columns={"h3_list": "h3"})
    rutas_hex["h3"] = rutas_hex["h3"].astype(str)

    # 2) Cargar GPS
    gps_path = Path(cfg["inputs"]["gps_csv"])
    gps = pd.read_csv(gps_path)
    # parseo y orden
    gps["fecha_hora"] = pd.to_datetime(gps["fecha_hora"], errors="coerce", utc=True)
    gps = gps.sort_values(["agency_id","mean_id","fecha_hora"])

    # 3) H3 para cada punto
    gps["h3"] = gps.apply(lambda r: h3_cell(r["latitude"], r["longitude"], res), axis=1).astype(str)

    # 4) Asignar trip_id por empresa+bus + dirección
    gps = assign_trip_ids(gps, min_steps=min_steps, gap_minutes=gap_minutes)

    # 5) Match punto ↔ ruta por H3 (join)
    #    NOTA: un punto puede caer en varias rutas si comparten hex; mantenemos todas (largo)
    matched = gps.merge(rutas_hex, on="h3", how="left")

    # 6) Determinar ruta dominante por trip (empresa+bus+trip)
    #    (la ruta con más puntos matcheados dentro del trip)
    grp_keys = ["agency_id","mean_id","trip_id"]

    # conteo por ruta dentro de cada trip
    counts = (
        matched.dropna(subset=["ruta_hex"])
               .groupby(grp_keys + ["ruta_hex"], as_index=False)
               .size()
               .rename(columns={"size":"pts_en_ruta"})
    )

    # total de puntos por trip
    tot = matched.groupby(grp_keys, as_index=False).size().rename(columns={"size":"pts_trip"})

    # ruta dominante (argmax)
    dom = counts.loc[counts.groupby(grp_keys)["pts_en_ruta"].idxmax()].reset_index(drop=True)
    dom = dom.merge(tot, on=grp_keys, how="left")
    dom["ratio"] = dom["pts_en_ruta"] / dom["pts_trip"]
    dom["trip_match"] = dom["ratio"] >= umbral

    # 7) Exportar
    # a) puntos matcheados (formato largo: puede tener varias filas por punto si entra en varias rutas)
    out_points = processed / "gps_match_points.parquet"
    matched.to_parquet(out_points)

    # b) resumen por trip (una fila por trip con su ruta dominante y si matchea)
    out_trips = processed / "gps_match_trips.parquet"
    dom.to_parquet(out_trips)

    print(f"✅ Puntos matcheados → {out_points}")
    print(f"✅ Resumen de trips → {out_trips}")
    print("   Columnas clave en trips: agency_id, mean_id, trip_id, ruta_hex_dominante, ratio, trip_match")


if __name__ == "__main__":
    main()
