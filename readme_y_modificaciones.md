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
├─ src/
│  ├─ build_h3.py        # Conversión de rutas oficiales CSV → hexágonos H3
│  ├─ match_h3.py        # Verificación de cumplimiento GPS vs H3
│  ├─ app_map.py         # Interfaz Streamlit con mapa interactivo y KPIs
│  ├─ pipeline.py        # Flujo principal de ejecución (pipeline)
│  └─ config.py          # Carga de configuración general
└─ scripts/
   └─ run_pipeline.sh    # (Opcional, solo para entornos Unix)
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
h3_res = 8
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

# Modificaciones al Anteproyecto – Cumplimiento de Rutas con H3

## 1. Sustitución de archivos `.shp` por `.csv`
**Antes:** el diseño original asumía que las rutas oficiales del VMT estaban en formato shapefile.  
**Ahora:** se confirmó que las rutas se encuentran en archivos `.csv`, con coordenadas de los tramos o puntos de la ruta.  
**Justificación:** simplifica la lectura y evita dependencias de archivos GIS complejos; permite integración directa con `pandas` y `geopandas`.

---

## 2. Implementación de un pipeline automatizado (`pipeline.py`)
**Antes:** el flujo de trabajo se ejecutaba por módulos separados (`build_h3.py`, `match_h3.py`, `viz_map.py`).  
**Ahora:** el archivo `pipeline.py` coordina todas las etapas del procesamiento.  
**Justificación:** unifica la ejecución y evita errores manuales. El usuario puede optar por:
- `--mode default`: abrir directamente la visualización si ya existen los datos procesados.  
- `--mode new`: regenerar todo el proceso desde los datos brutos.

---

## 3. Migración de la visualización a Streamlit + PyDeck
**Antes:** la visualización se realizaba con Folium y QGIS.  
**Ahora:** se reemplazó por una interfaz unificada en **Streamlit**, con renderizado WebGL mediante **PyDeck**.  
**Justificación:**  
- Mucho mejor rendimiento (puede mostrar cientos de miles de puntos).  
- Filtros interactivos jerárquicos (empresa → línea → bus → viaje).  
- Visualización instantánea en navegador, sin necesidad de software GIS externo.

---

## 4. Incorporación de métricas y panel de cumplimiento
**Antes:** los cálculos de cumplimiento se limitaban a promedios generales.  
**Ahora:** el sistema calcula KPIs por empresa, línea y bus, y muestra los casos con menor porcentaje de puntos dentro de la ruta.  
**Justificación:** facilita el análisis de desempeño operativo y la detección rápida de desvíos.

---

## 5. Simplificación de dependencias y flujo de instalación
**Antes:** se incluía un script `.sh` para Linux/macOS.  
**Ahora:** todo el flujo se gestiona con `pipeline.py`, eliminando la necesidad de scripts adicionales.  
**Justificación:** mejora la portabilidad en Windows, que es el entorno principal de ejecución.

---

## 6. Revisión del objetivo técnico
**Antes:** el objetivo principal era generar una herramienta de validación estática.  
**Ahora:** se orienta hacia una **plataforma dinámica e interactiva**, capaz de actualizarse periódicamente y servir como módulo de fiscalización del VMT.  
**Justificación:** la integración de visualización, análisis y KPIs en un solo entorno lo convierte en una herramienta operativa y no solo de investigación.

---

## 7. Estructura final del proyecto
**Archivos clave:**
- `src/build_h3.py`: genera celdas H3 por ruta.  
- `src/match_h3.py`: compara puntos GPS con rutas H3.  
- `src/app_map.py`: visualización con Streamlit y PyDeck.  
- `src/pipeline.py`: pipeline completo de generación y ejecución.  

**Datos principales:**
- `data/raw/`: rutas oficiales y datos GPS (.csv).  
- `data/processed/`: resultados procesados (.parquet, .csv).  

---

## 8. Impacto de las modificaciones
Las modificaciones introducidas:
- Mejoran la velocidad de procesamiento.  
- Eliminaron dependencias innecesarias (como archivos shapefile o Folium).  
- Introducen un flujo reproducible, con opciones claras y controladas.  
- Amplían el alcance del proyecto al permitir un análisis detallado por bus y viaje, visualizable en tiempo real.

---

## 9. Próximos pasos
- Incorporar filtros por fecha o rango horario en el mapa.  
- Permitir actualización incremental (solo nuevos GPS).  
- Conectar los resultados a un panel de reportes automatizado (Power BI o Streamlit Dashboard).

