# Geospatial Data

Avance reproducible del análisis de cianobacteria en los lagos Atitlán y Amatitlán con Sentinel-2 L2A.

## Funcionalidades

- conexión comprobable al backend openEO de Copernicus Data Space;
- consulta de las 22 fechas oficiales mediante una API STAC de Sentinel-2;
- lectura por rangos de solo `B03`, `B04`, `B05`, `B08` y `SCL` (sin escenas completas);
- GeoTIFF multibanda por lago/fecha con clorofila-a proxy de cianobacteria, NDCI, NDVI y NDWI;
- CSV con promedios, cobertura válida y estadísticos robustos;
- gráfico temporal, fechas críticas, mapas de comprobación e informe PDF dirigido a ambientalistas.

La adquisición ejecutada usa el espejo público Sentinel-2 L2A de Microsoft Planetary Computer para no requerir secretos. El script también verifica el backend oficial openEO de Copernicus y permite autenticarlo por OIDC. Ambos accesos consultan productos Sentinel-2; el origen y los identificadores exactos quedan registrados en `outputs/metadata/catalogo_escenas.csv`.

## Ejecución

En PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

Para probar además la autenticación interactiva de Copernicus:

```powershell
.venv\Scripts\python main.py --authenticate-openeo --only-connection
```

La primera ejecución descarga únicamente ventanas espaciales de las cinco bandas requeridas y guarda una caché local ignorada por Git. Las ejecuciones posteriores reutilizan esa caché. Use `--force-download` para renovarla.

## Metodología resumida

- NDVI = `(B08 - B04) / (B08 + B04)`.
- NDWI = `(B03 - B08) / (B03 + B08)`.
- NDCI = `(B05 - B04) / (B05 + B04)`.
- Clorofila-a proxy = `826.57*NDCI^3 - 176.43*NDCI^2 + 19*NDCI + 4.071`, limitada al intervalo físico del script (0–500 mg/m³).
- Se excluyen nodata, píxeles saturados, sombras, nubes, cirros y nieve mediante SCL.
- Para productos con línea base de procesamiento 04.00 o posterior se convierte DN a reflectancia como `(DN - 1000) / 10000`, de acuerdo con `BOA_ADD_OFFSET=-1000` y `BOA_QUANTIFICATION_VALUE=10000` del metadato Sentinel-2.
- La validez radiométrica se evalúa por índice: NDCI/cianobacteria requiere B04 y B05 positivas; NDVI requiere B04 y B08; NDWI requiere B03 y B08. Así, un NIR negativo sobre agua oscura no elimina una estimación de cianobacteria todavía válida.
- La máscara de cada lago se obtiene por consenso temporal de SCL/NDWI, conserva el componente de agua mayor y erosiona 20 m el borde para reducir píxeles mixtos de costa.

La clorofila-a es un **proxy de floración compatible con cianobacteria**, no un conteo directo de células ni una medición de toxinas. Las fechas críticas requieren confirmación con muestreo de campo.
El algoritmo publicado fue calibrado originalmente para Sentinel-2 L1C; aquí se aplica de forma explícita a reflectancia L2A, como solicita el laboratorio. Esa adaptación mejora la corrección atmosférica operativa, pero no sustituye una calibración local para los lagos guatemaltecos.
