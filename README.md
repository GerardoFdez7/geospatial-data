# Geospatial Data

Análisis de cianobacteria en los lagos Atitlán y Amatitlán con Sentinel-2 L2A. El repositorio cubre las dos partes del Laboratorio 4: el análisis geoespacial (Parte 1) y los modelos de aprendizaje automático construidos sobre esos productos (Parte 2).

## Parte 1 — Análisis geoespacial

- conexión comprobable al backend openEO de Copernicus Data Space;
- consulta de las 22 fechas oficiales mediante una API STAC de Sentinel-2;
- lectura por rangos de solo `B03`, `B04`, `B05`, `B08` y `SCL` (sin escenas completas);
- GeoTIFF multibanda por lago/fecha con clorofila-a proxy de cianobacteria, NDCI, NDVI y NDWI;
- CSV con promedios, cobertura válida, estadísticos robustos y % de extensión de floración (niveles guía OMS);
- gráfico temporal, fechas críticas, mapas comparativos entre fechas y mapas interactivos (folium) por lago;
- correlación de píxeles (Pearson) entre el proxy de cianobacteria y NDVI/NDWI, por lago;
- análisis exploratorio: zonas de acumulación persistente, distribución por fecha (boxplots) y patrón estacional (seca/lluviosa);
- comparación explícita de intensidad y frecuencia de floración entre ambos lagos.

La adquisición ejecutada usa el espejo público Sentinel-2 L2A de Microsoft Planetary Computer para no requerir secretos. El script también verifica el backend oficial openEO de Copernicus y permite autenticarlo por OIDC. Ambos accesos consultan productos Sentinel-2; el origen y los identificadores exactos quedan registrados en `outputs/metadata/catalogo_escenas.csv`.

## Parte 2 — Modelos de Machine Learning

- conjunto de datos píxel-fecha construido desde los rásters de la Parte 1, con coordenadas, fecha, lago, bandas en reflectancia, índices y variables derivadas;
- variable respuesta binaria con corte en 50 mg/m³ (Alert Level 2 de la OMS);
- control explícito de fuga de información: `cyano_chla_proxy`, `ndci`, `B04`, `B05` y `ndvi` quedan fuera del conjunto de predictores;
- ingeniería de características espaciales (textura local, anomalía local, distancia a la orilla) y temporales (codificación cíclica del día del año);
- regresión logística, bosque aleatorio y XGBoost con ajuste de hiperparámetros por PR-AUC;
- validación aleatoria, validación espacial por bloques de 1 km × 1 km (`GroupKFold`) y validación temporal por fechas futuras;
- experimentos de generalización cruzada entre lagos;
- interpretabilidad con importancia de variables y SHAP;
- mapas predictivos de probabilidad por lago y análisis espacial de los errores.

## Ejecución

En PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

La primera ejecución descarga únicamente ventanas espaciales de las cinco bandas requeridas y guarda una caché local ignorada por Git. Las ejecuciones posteriores reutilizan esa caché. Use `--force-download` para renovarla.

Para probar además la autenticación interactiva de Copernicus:

```powershell
.venv\Scripts\python main.py --authenticate-openeo --only-connection
```

Para la Parte 2, el notebook necesita un kernel de Jupyter que apunte al entorno virtual. Se registra una sola vez:

```powershell
.venv\Scripts\python -m ipykernel install --user --name geospatial --display-name "Python (geospatial .venv)"
```

Luego se construye el conjunto de datos de modelado y se ejecuta el notebook:

```powershell
.venv\Scripts\python ml_pipeline.py
.venv\Scripts\python -m jupyter nbconvert --execute --inplace notebooks\laboratorio4_parte2.ipynb
```

El notebook también puede abrirse y ejecutarse de forma interactiva, seleccionando el kernel «Python (geospatial .venv)». `ml_pipeline.py` cachea el conjunto de datos en Parquet, de modo que solo la primera construcción lee los 22 rásters.

## Metodología resumida

- NDVI = `(B08 - B04) / (B08 + B04)`.
- NDWI = `(B03 - B08) / (B03 + B08)`.
- NDCI = `(B05 - B04) / (B05 + B04)`.
- Clorofila-a proxy = `826.57*NDCI^3 - 176.43*NDCI^2 + 19*NDCI + 4.071`, limitada al intervalo físico del script (0–500 mg/m³).
- Se excluyen nodata, píxeles saturados, sombras, nubes, cirros y nieve mediante SCL.
- Para productos con línea base de procesamiento 04.00 o posterior se convierte DN a reflectancia como `(DN - 1000) / 10000`, de acuerdo con `BOA_ADD_OFFSET=-1000` y `BOA_QUANTIFICATION_VALUE=10000` del metadato Sentinel-2.
- La validez radiométrica se evalúa por índice: NDCI/cianobacteria requiere B04 y B05 positivas; NDVI requiere B04 y B08; NDWI requiere B03 y B08. Así, un NIR negativo sobre agua oscura no elimina una estimación de cianobacteria todavía válida.
- La máscara de cada lago se obtiene por consenso temporal de SCL/NDWI, conserva el componente de agua mayor y erosiona 20 m el borde para reducir píxeles mixtos de costa.
- Todos los productos están en EPSG:32615 (WGS 84 / UTM 15N) a 20 m, que es el sistema que la Parte 2 requiere para construir los bloques espaciales.

La clorofila-a es un **proxy de floración compatible con cianobacteria**, no un conteo directo de células ni una medición de toxinas. Las fechas críticas requieren confirmación con muestreo de campo.
El algoritmo publicado fue calibrado originalmente para Sentinel-2 L1C; aquí se aplica de forma explícita a reflectancia L2A, como solicita el laboratorio. Esa adaptación mejora la corrección atmosférica operativa, pero no sustituye una calibración local para los lagos guatemaltecos.

La extensión de floración usa los niveles guía de la OMS para clorofila-a asociada a cianobacteria: alerta moderada ≥10 mg/m³, alerta alta ≥50 mg/m³.

### Escena descartada en la Parte 2

La escena **2025-01-18 de Atitlán** se excluye del conjunto de modelado. Conserva solo 23.8 % de cobertura válida y reporta una media de 21.29 mg/m³ frente a los 2.35–3.96 mg/m³ de las otras diez fechas del mismo lago, con un percentil 90 de 72.78 contra 4.07–5.72. Concentraba el 83 % de los positivos de Atitlán. El patrón —peor calidad de escena junto con mayor intensidad reportada— corresponde a contaminación atmosférica residual que la banda SCL no clasifica, no a un proceso limnológico. Se conserva en el conjunto bruto y en el análisis exploratorio, pero entrenar sobre ella habría enseñado al modelo a reconocer nubes delgadas.

## Salidas

Todo lo que generan los scripts vive en `outputs/`, que está ignorado por Git porque es reproducible a partir del código.

### Parte 1

- `outputs/resumen_temporal.csv`: tabla con todos los estadísticos por lago/fecha.
- `outputs/rasters/<lago>/`: GeoTIFF de índices por fecha y máscara del lago.
- `outputs/mapas_interactivos/<lago>.html`: mapa interactivo (folium) navegable con el proxy de cianobacteria.
- `outputs/figures/`: gráficos del análisis temporal, espacial y exploratorio.
- `outputs/metadata/`: catálogo de escenas usadas, verificación de conexión openEO y validación de entrega.

### Parte 2

- `outputs/ml/dataset_cianobacteria_ml.parquet`: conjunto de datos completo píxel-fecha.
- `outputs/ml/dataset_cianobacteria_ml.csv.gz`: muestra de 200 000 filas en texto plano, para inspección sin `pyarrow`.
- `outputs/ml/resultados.json`: todas las métricas del notebook, reunidas en un solo archivo.
- `outputs/ml/*.csv`: métricas por esquema de validación, generalización entre lagos y resúmenes del conjunto.
- `outputs/ml/figuras/`: figuras del notebook (también quedan embebidas en el `.ipynb`).
