#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
import os
import time
import subprocess
import webbrowser
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
DATA = ROOT / "data"
PRO  = DATA / "processed"

APP_PATH      = SRC / "app_map.py"
RUTAS_H3_PARQ = PRO / "rutas_h3.parquet"
POINTS_PARQ   = PRO / "gps_match_points.parquet"
TRIPS_PARQ    = PRO / "gps_match_trips.parquet"

STREAMLIT_URL    = "http://localhost:8501"
STREAMLIT_HEALTH = f"{STREAMLIT_URL}/healthz"

def file_exists(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False

def streamlit_is_up(timeout_sec: float = 0.7) -> bool:
    try:
        with urlopen(STREAMLIT_HEALTH, timeout=timeout_sec) as resp:
            body = (resp.read() or b"").decode("utf-8", "ignore").strip().lower()
            return "ok" in body
    except (URLError, HTTPError, TimeoutError):
        return False
    except Exception:
        return False

def touch(path: Path) -> None:
    ts = time.time()
    os.utime(path, (ts, ts))

def run_module(modname: str) -> int:
    # Intenta: python -m src.<module>
    rc = subprocess.run([sys.executable, "-m", modname], cwd=ROOT, check=False).returncode
    if rc == 0:
        return rc
    # Fallback: establece PYTHONPATH=ROOT y ejecuta de nuevo
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    rc2 = subprocess.run([sys.executable, "-m", modname], cwd=ROOT, env=env, check=False).returncode
    return rc2

def start_streamlit() -> None:
    if streamlit_is_up():
        webbrowser.open_new_tab(STREAMLIT_URL)
        return
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    if os.name == "nt":
        subprocess.Popen(cmd, cwd=ROOT, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        subprocess.Popen(cmd, cwd=ROOT)
    for _ in range(20):
        if streamlit_is_up():
            break
        time.sleep(0.3)
    webbrowser.open_new_tab(STREAMLIT_URL)

def reload_streamlit() -> None:
    if streamlit_is_up():
        touch(APP_PATH)
        webbrowser.open_new_tab(STREAMLIT_URL)
    else:
        start_streamlit()

def mode_default() -> None:
    have_points = file_exists(POINTS_PARQ)
    have_trips  = file_exists(TRIPS_PARQ)
    have_rutas  = file_exists(RUTAS_H3_PARQ)

    if not (have_points and have_trips and have_rutas):
        print("[default] Faltan insumos procesados. Archivos requeridos:")
        print(f"  - {RUTAS_H3_PARQ}  -> {'OK' if have_rutas else 'NO'}")
        print(f"  - {POINTS_PARQ}    -> {'OK' if have_points else 'NO'}")
        print(f"  - {TRIPS_PARQ}     -> {'OK' if have_trips else 'NO'}")
        print("Sugerencia: ejecuta --mode new para generarlos.")
    else:
        print("[default] Archivos procesados detectados. Abriendo/recargando mapa...")
    reload_streamlit()

def mode_new() -> None:
    print("[new] Generando rutas H3 (python -m src.build_h3)...")
    rc = run_module("src.build_h3")
    if rc != 0:
        print(f"[new] build_h3 falló con código {rc}. Abortando.")
        sys.exit(rc)

    print("[new] Corriendo matcheo (python -m src.match_h3)...")
    rc = run_module("src.match_h3")
    if rc != 0:
        print(f"[new] match_h3 falló con código {rc}. Abortando.")
        sys.exit(rc)

    ok = all(file_exists(p) for p in (RUTAS_H3_PARQ, POINTS_PARQ, TRIPS_PARQ))
    if not ok:
        print("[new] Advertencia: faltan salidas luego del cálculo:")
        print(f"  - {RUTAS_H3_PARQ} -> {'OK' if file_exists(RUTAS_H3_PARQ) else 'NO'}")
        print(f"  - {POINTS_PARQ}   -> {'OK' if file_exists(POINTS_PARQ) else 'NO'}")
        print(f"  - {TRIPS_PARQ}    -> {'OK' if file_exists(TRIPS_PARQ) else 'NO'}")

    print("[new] Abriendo/recargando mapa…")
    reload_streamlit()

def main():
    parser = argparse.ArgumentParser(description="Pipeline: default (abrir mapa) | new (recalcular + abrir mapa)")
    parser.add_argument("--mode", choices=["default", "new"], default="default")
    args = parser.parse_args()
    if args.mode == "default":
        mode_default()
    else:
        mode_new()

if __name__ == "__main__":
    main()
