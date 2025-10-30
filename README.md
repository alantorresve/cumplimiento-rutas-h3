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
1. Convertir las rutas oficiales en formato `.shp` a una estructura de hexágonos H3 (nivel h7/h8).  
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
copy .env.example .env
```

#### En Linux / macOS:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Librerías principales  
- geopandas: lectura y análisis de archivos .shp  
- pandas y numpy: procesamiento de datos  
- h3: conversión de coordenadas a celdas hexagonales  
- folium: visualización de mapas  
- matplotlib: gráficos estadísticos  
- shapely y pyproj: operaciones geoespaciales  

---

## Estructura del Proyecto  

```
cumplimiento-rutas-h3/
├─ README.md
├─ requirements.txt
├─ config.toml
├─ .env.example
├─ data/
│  ├─ raw/               # Archivos de entrada (.shp y GPS)
│  └─ processed/         # Resultados (Parquet, GeoJSON, CSV)
├─ reports/              # Mapas Folium y resúmenes
├─ src/
│  ├─ build_h3.py        # Conversión de rutas .shp a hexágonos H3
│  ├─ match_h3.py        # Verificación de cumplimiento GPS vs H3
│  ├─ metrics.py         # Cálculo de métricas de cumplimiento
│  ├─ viz_map.py         # Generación del mapa interactivo
│  └─ config.py          # Carga de configuración
└─ scripts/
   └─ run_pipeline.sh    # Ejecución completa del flujo
```

---

## Flujo de Ejecución  

### 1. Construir la huella H3 desde las rutas oficiales  
```bash
python -m src.build_h3 --shp 
```

### 2. Calcular el cumplimiento de los puntos GPS  
```bash
python -m src.match_h3 --gps 
```

### 3. Generar un mapa de resultados  
```bash
python -m src.viz_map --kpis 
```

---

## Configuración (`config.toml`)
```toml
[spatial]
crs_wgs84 = 
h3_res = 
kring = 
sampling_meters = 

[metrics]
umbral_compliance = 
min_run_fuera = 

[paths]
raw = "data/raw"
processed = "data/processed"
reports = "reports"
```

---

## Validación  
```bash
pytest -q
```

---

## Descripción Técnica  
El sistema utiliza la indexación espacial jerárquica H3 para representar las rutas oficiales en forma de celdas hexagonales. Cada punto GPS se convierte a su celda H3 correspondiente y se verifica si pertenece a la huella oficial de la ruta declarada.  
Se calculan métricas de cumplimiento (compliance rate) y se generan visualizaciones interactivas en HTML mediante Folium.  

---


## Resumen  
Este proyecto automatiza el control del cumplimiento de rutas de transporte público en el Área Metropolitana de Asunción.  
Permite al Viceministerio de Transporte realizar un monitoreo más preciso, detectar desvíos de operación y mejorar la fiscalización mediante un análisis geoespacial reproducible y visualmente interpretativo.  
