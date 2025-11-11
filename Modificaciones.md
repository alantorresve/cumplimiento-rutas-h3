# Modificaciones al Anteproyecto – Cumplimiento de Rutas con H3

## 1. Sustitución de archivos `.shp` por `.csv`
**Antes:** el diseño original asumía que las rutas oficiales del VMT estaban en formato shapefile.  
**Ahora:** se confirmó que las rutas se encuentran en archivos `.csv`, con coordenadas de los tramos o puntos de la ruta.  

---

## 2. Migración de la visualización a Streamlit + PyDeck
**Antes:** la visualización se realizaba con Folium y QGIS.  
**Ahora:** se reemplazó por una interfaz unificada en **Streamlit**, con renderizado WebGL mediante **PyDeck**.  
**Justificación:**  
- Mucho mejor rendimiento que folium (puede mostrar cientos de miles de puntos).  
- Filtros interactivos jerárquicos (empresa → línea → bus → viaje).  
- Visualización instantánea en navegador, sin necesidad de software GIS externo.

---

## 4. Incorporación de métricas y panel de cumplimiento
**Antes:** los cálculos de cumplimiento se limitaban a promedios generales.  
**Ahora:** el sistema calcula KPIs por empresa, línea y bus, y muestra los casos con menor porcentaje de puntos dentro de la ruta.  
**Justificación:** facilita el análisis de desempeño operativo y la detección rápida de desvíos.


---

## 5. Revisión del objetivo técnico
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
