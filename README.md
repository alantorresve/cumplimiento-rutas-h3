# Cumplimiento de Rutas con H3 (VMT – AMA)

## Título del Proyecto

**Identificación del Cumplimiento de Rutas mediante Análisis Espacial H3 y Visualización Interactiva en Streamlit**

## Integrantes

* Ing. Alan Torres
* Ing. Tais Machado

---

## Objetivo General

Desarrollar un sistema reproducible en Python que permita evaluar el cumplimiento operativo de las rutas de transporte público en el Área Metropolitana de Asunción (AMA), mediante la comparación de puntos GPS reales con la huella oficial de las rutas H3 del Viceministerio de Transporte (VMT).

---

## Objetivos Específicos

1. Convertir las rutas oficiales del VMT a una representación espacial en hexágonos H3.
2. Vincular los puntos GPS de monitoreo con las celdas H3 correspondientes.
3. Calcular métricas de cumplimiento por empresa, línea, bus y viaje (trip).
4. Identificar desvíos o tramos fuera de ruta.
5. Visualizar los resultados de forma interactiva y filtrable en un mapa dinámico (Streamlit + PyDeck).
6. Automatizar el flujo completo de procesamiento y visualización mediante un pipeline ejecutable.

---

## Estructura del Proyecto

```
cumplimiento-rutas-h3/
├─ README.md
├─ config.toml
├─ requirements.txt
│
├─ data/
│  ├─ raw/               # Datos fuente (GPS, catálogos de rutas, EOT)
│  └─ processed/         # Resultados intermedios y finales (.parquet, .csv)
│
├─ src/
│  ├─ build_h3.py        # Convierte rutas oficiales a celdas H3
│  ├─ match_h3.py        # Asocia puntos GPS con las rutas H3
│  ├─ app_map.py         # App Streamlit: filtros, mapa y análisis
│  ├─ config.py          # Carga de parámetros desde config.toml
│  └─ pipeline.py        # Orquestador automático (build + match + mapa)
│
└─ scripts/
   └─ run_pipeline.bat   # Lanzador rápido (Windows)
```

---

## Flujo de Ejecución

### Modo rápido (usar resultados existentes y abrir mapa)

```bash
python src/pipeline.py --mode default
```

* No recalcula nada si ya existen los archivos `.parquet`.
* Solo recarga o abre el mapa interactivo (`http://localhost:8501`).

### Modo completo (regenerar todo y abrir mapa)

```bash
python src/pipeline.py --mode new
```

* Ejecuta `build_h3.py` → `match_h3.py`
* Guarda los resultados en `data/processed/`
* Luego abre automáticamente el mapa Streamlit.

---

## Mapa Interactivo (Streamlit)

**URL local:** [http://localhost:8501](http://localhost:8501)

### Funcionalidades principales:

* Filtros en cascada:
  `Empresa → Línea → Bus → Trip`
* Capas configurables:

  * Rutas H3 planas
  * Puntos dentro (verde) / fuera (rojo)
* Métricas en tiempo real:

  * % de puntos dentro de la huella
  * Número de trips únicos
  * Tabla de “peor cumplimiento” (empresas, líneas y buses)

---

## Tecnologías Principales

| Librería            | Uso principal                                    |
| ------------------- | ------------------------------------------------ |
| pandas / numpy      | Procesamiento tabular eficiente                  |
| geopandas           | Análisis geoespacial                             |
| h3-py               | Conversión de coordenadas a celdas H3            |
| pydeck / Streamlit  | Visualización interactiva y mapas                |
| pyarrow / parquet   | Lectura y escritura de datos de alto rendimiento |

---

## Configuración (`config.toml`)

```toml
[spatial]
crs_wgs84 = "EPSG:4326"
h3_res = 8
kring = 1

[metrics]
umbral_compliance = 0.7
min_run_fuera = 3

[paths]
raw = "data/raw"
processed = "data/processed"
```

---

## Descripción Técnica

El sistema utiliza Uber H3 como índice espacial jerárquico para representar rutas oficiales mediante hexágonos.
Cada punto GPS se asocia a su celda H3 correspondiente, y se determina si pertenece o no a la huella de la ruta.
Los resultados se visualizan en un mapa interactivo con filtros jerárquicos, mostrando el grado de cumplimiento operativo por empresa, línea, bus y viaje.

---

## Resumen

Este proyecto brinda al Viceministerio de Transporte una herramienta práctica y reproducible para monitorear el cumplimiento de rutas, detectar desvíos operativos y fortalecer la gestión basada en datos dentro del sistema de transporte público del AMA.
El flujo completo —desde la lectura de datos hasta la visualización— es automático, modular y escalable, lo que permite su integración en futuros sistemas de monitoreo y fiscalización.
