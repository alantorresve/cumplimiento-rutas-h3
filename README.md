# Cumplimiento de Rutas con H3 (VMT – AMA)

## Título del Proyecto  
**Identificación de Cumplimiento de Rutas en Base a Puntos GPS de Monitoreo de Buses del Área Metropolitana, Utilizando Hexágonos de Uber (H3)**

## Integrantes  
- Ing. Alan Torres  
- Ing. Tais Machado  

---

## Objetivo General  
Desarrollar un sistema en Python que permita identificar el nivel de cumplimiento de las rutas de transporte público, comparando los puntos GPS reales de los buses con las rutas oficiales del Viceministerio de Transporte (VMT), utilizando una representación geoespacial basada en hexágonos H3.

## Objetivos Específicos  
1. Convertir las rutas oficiales en formato `.csv` a una estructura de hexágonos H3 (nivel h7/h8).  
2. Comparar los puntos GPS reportados por las empresas operadoras con la huella H3 de cada ruta.  
3. Calcular el porcentaje de cumplimiento por bus, ruta y fecha.  
4. Detectar incumplimientos o desvíos prolongados fuera de la ruta.  
5. Generar mapas interactivos y reportes que permitan visualizar los resultados y facilitar la fiscalización.  

---

## Requisitos del Entorno  

**Versión de Python recomendada:** 3.11 o superior  

### Instalación de dependencias  

#### En Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### En Linux / macOS:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Librerías principales  
- pandas y numpy: procesamiento de datos  
- geopandas: soporte geoespacial (polígonos y hexágonos)  
- h3: generación de celdas hexagonales  
- shapely y pyproj: operaciones geoespaciales  
- streamlit + pydeck: visualización de mapas interactivos  

---

## Estructura del Proyecto  

```
cumplimiento-rutas-h3/
├─ README.md
├─ requirements.txt
├─ config.toml
├─ data/
│  ├─ raw/               # Archivos de entrada (rutas oficiales y puntos GPS)
│  └─ processed/         # Resultados procesados (Parquet, CSV)
└─ src/
   ├─ build_h3.py        # Conversión de rutas oficiales CSV → hexágonos H3
   ├─ match_h3.py        # Verificación de cumplimiento GPS vs H3
   ├─ app_map.py         # Interfaz Streamlit con mapa interactivo y KPIs
   ├─ pipeline.py        # Flujo principal de ejecución (pipeline)
   └─ config.py          # Carga de configuración general

```

---

## Ejecución del Sistema

El flujo completo se controla desde el archivo `src/pipeline.py`, que integra los módulos de generación, análisis y visualización.

### Modo por defecto
Ejecuta directamente el mapa interactivo si los archivos `.parquet` ya existen.  
No recalcula las rutas ni los puntos GPS.

```powershell
python src\pipeline.py --mode default
```

### Modo “new”
Reconstruye toda la información: genera las rutas H3 desde los CSV oficiales, realiza el análisis de cumplimiento con los datos GPS y luego lanza la visualización en Streamlit.

```powershell
python src\pipeline.py --mode new
```

### Notas
- Debe ejecutarse desde la **carpeta raíz del proyecto**, con el entorno virtual activado.  
- Si el mapa ya está abierto, puede cerrarse con `Ctrl + C` en la terminal antes de volver a correr el pipeline.  
- No es necesario ejecutar manualmente `build_h3.py` ni `match_h3.py`; el pipeline se encarga de eso.  

---

## Configuración (`config.toml`)
```toml
[spatial]
crs_wgs84 = "EPSG:4326"
h3_res = 7
kring = 0
sampling_meters = 10

[metrics]
umbral_compliance = 0.6
min_run_fuera = 3

[paths]
raw = "data/raw"
processed = "data/processed"
reports = "reports"
```

---

## Descripción Técnica  
El sistema utiliza la indexación espacial jerárquica H3 para representar las rutas oficiales a partir de sus coordenadas contenidas en archivos `.csv`.  
Cada punto GPS se convierte a su celda H3 correspondiente y se verifica si pertenece a la huella oficial de la ruta declarada.  
Se calculan métricas de cumplimiento (porcentaje de puntos dentro de la ruta) y se generan visualizaciones interactivas con filtros por empresa, línea, bus y viaje, implementadas en Streamlit.

---

## Resumen  
Este proyecto automatiza el control del cumplimiento de rutas de transporte público en el Área Metropolitana de Asunción.  
Permite al Viceministerio de Transporte realizar un monitoreo más preciso, detectar desvíos de operación y mejorar la fiscalización mediante un análisis geoespacial reproducible, dinámico y visualmente interpretativo.

---
