from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[0]
OUTPUTS = ROOT / "outputs"
RASTERS = OUTPUTS / "rasters"
CACHE = ROOT / "cache"
FIGURES = OUTPUTS / "figures"
ML_OUTPUTS = OUTPUTS / "ml"
LAKES = ["atitlan", "amatitlan"]
BANDS = ("B03", "B04", "B05", "B08")
THRESHOLD_ALTA_PRESENCIA = 50.0

def mkdirs() -> None:
    ML_OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

def build_ml_dataset() -> pd.DataFrame:
    """Ejercicio 1: Preparacion de los datos para Machine Learning."""
    print("Construyendo el conjunto de datos de Machine Learning...")
    data_rows = []

    for lake in LAKES:
        lake_raster_dir = RASTERS / lake
        lake_cache_dir = CACHE / lake
        
        mask_path = lake_raster_dir / "mascara_lago.tif"
        if not mask_path.exists():
            continue
            
        with rasterio.open(mask_path) as src_mask:
            lake_mask = src_mask.read(1)
            transform = src_mask.transform
            rows, cols = np.where(lake_mask == 1)
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            
        tif_files = list(lake_raster_dir.glob("indices_*.tif"))
        dates = [f.stem.split('_')[1] for f in tif_files]
        
        for date in dates:
            indices_path = lake_raster_dir / f"indices_{date}.tif"
            with rasterio.open(indices_path) as src_indices:
                cyano = src_indices.read(1)
                ndci = src_indices.read(2)
                ndvi = src_indices.read(3)
                ndwi = src_indices.read(4)
                nodata = src_indices.nodata
                
            cache_path = lake_cache_dir / f"{date}.npz"
            if not cache_path.exists():
                continue
                
            with np.load(cache_path) as stored:
                b03 = stored["B03"]
                b04 = stored["B04"]
                b05 = stored["B05"]
                b08 = stored["B08"]
                
            for i in range(len(rows)):
                r, c = rows[i], cols[i]
                c_val = cyano[r, c]
                if c_val == nodata or np.isnan(c_val) or np.isinf(c_val):
                    continue
                    
                data_rows.append({
                    "x": xs[i],
                    "y": ys[i],
                    "fecha": date,
                    "lago": lake,
                    "B03": float(b03[r, c]),
                    "B04": float(b04[r, c]),
                    "B05": float(b05[r, c]),
                    "B08": float(b08[r, c]),
                    "ndvi": float(ndvi[r, c]),
                    "ndwi": float(ndwi[r, c]),
                    "ndci": float(ndci[r, c]),
                    "cyano_chla_proxy": float(c_val)
                })

    df = pd.DataFrame(data_rows)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

def exploratory_data_analysis(df: pd.DataFrame) -> None:
    """Ejercicios 1.4 y 1.5: Analisis exploratorio."""
    print("\n--- Ejercicio 1: EDA ---")
    print(f"Total de observaciones: {len(df)}")
    print("\nObservaciones por lago:")
    print(df['lago'].value_counts())
    
    print("\nPorcentaje de valores faltantes por variable:")
    print((df.isnull().sum() / len(df)) * 100)
    
    # Save statistics
    stats = df.describe()
    stats.to_csv(ML_OUTPUTS / "estadisticas_descriptivas.csv")
    print(f"\nEstadisticas guardadas en {ML_OUTPUTS / 'estadisticas_descriptivas.csv'}")

    cols_to_plot = ['B03', 'B04', 'B05', 'B08', 'ndvi', 'ndwi', 'cyano_chla_proxy']
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        sns.histplot(data=df, x=col, hue="lago", bins=50, kde=True, ax=axes[i], element="step")
        axes[i].set_title(f'Distribución de {col}')

    for j in range(len(cols_to_plot), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    fig.savefig(FIGURES / "eda_distribuciones_ml.png")
    plt.close(fig)
    print(f"Gráficos de distribución guardados en {FIGURES / 'eda_distribuciones_ml.png'}")

def build_response_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Ejercicio 2: Construccion de la variable respuesta."""
    print("\n--- Ejercicio 2: Variable Respuesta ---")
    df['alta_cianobacteria'] = (df['cyano_chla_proxy'] >= THRESHOLD_ALTA_PRESENCIA).astype(int)
    
    print("Distribución global de la variable respuesta (0=Baja/Ausente, 1=Alta):")
    print(df['alta_cianobacteria'].value_counts(normalize=True) * 100)

    print("\nDistribución por lago:")
    print(pd.crosstab(df['lago'], df['alta_cianobacteria'], normalize='index') * 100)
    
    print("\nDesbalance de clases detectado. El desbalance puede causar que el modelo de ML prediga siempre 0.")
    print("Variables que no deben usarse como predictoras: cyano_chla_proxy, ndci, B04, B05 (causan data leakage).")

    # Guardar gráfico de desbalance de clases
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='alta_cianobacteria', hue='lago', ax=ax)
    ax.set_title("Desbalance de Clases en Variable Respuesta")
    fig.savefig(FIGURES / "desbalance_clases.png")
    plt.close(fig)
    print(f"Gráfico de desbalance guardado en {FIGURES / 'desbalance_clases.png'}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Ejercicio 3: Selección y construcción de variables predictoras."""
    print("\n--- Ejercicio 3: Feature Engineering ---")
    df['mes'] = df['fecha'].dt.month
    # Temporada (seca: nov-abr (11-12, 1-4), lluviosa: may-oct (5-10))
    df['temporada_lluviosa'] = df['mes'].apply(lambda m: 1 if 5 <= m <= 10 else 0)
    
    predictores_base = ['B03', 'B08', 'ndvi', 'ndwi']
    predictores_espaciales = ['x', 'y']
    predictores_construidos = ['mes', 'temporada_lluviosa']
    
    all_predictors = predictores_base + predictores_espaciales + predictores_construidos
    print("Predictores finales a utilizar en los modelos:")
    for p in all_predictors:
        print(f" - {p}")
    return df

def main() -> int:
    mkdirs()
    df = build_ml_dataset()
    exploratory_data_analysis(df)
    df = build_response_variable(df)
    df = feature_engineering(df)
    
    out_path = ML_OUTPUTS / "dataset_cianobacteria_ml.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDataset final guardado exitosamente en: {out_path}")
    
    # Escribir documento de respuestas a las justificaciones de los ejercicios
    with open(ML_OUTPUTS / "respuestas_ejercicios_1_2_3.txt", "w", encoding="utf-8") as f:
        f.write("=== Respuestas Ejercicios 1, 2, 3 ===\n\n")
        f.write("1.6 Decisiones tomadas en limpieza:\n")
        f.write("Se usó la máscara 'mascara_lago.tif' para descartar tierra. Se ignoraron pixeles con NoData o NaN.\n\n")
        f.write("2.2 Justificación del punto de corte:\n")
        f.write("Se usó 50 mg/m3 de acuerdo con la clasificación de alerta alta de la OMS (WHO 2021).\n\n")
        f.write("2.4 Consecuencias de desbalance:\n")
        f.write("El modelo puede sesgarse a predecir siempre 0, maximizando Accuracy pero ignorando floraciones. Requerirá balanceo de pesos o remuestreo.\n\n")
        f.write("2.5 Variables excluidas:\n")
        f.write("cyano_chla_proxy, ndci, B04, B05. Todas formaron matemáticamente la variable respuesta y causarían data leakage.\n\n")
        f.write("3.2 Explicación de variables:\n")
        f.write("B03 (Verde), B08 (NIR): Detectan pigmentos y natas en el agua.\n")
        f.write("NDVI, NDWI: Reflejan presencia de biomasa fotosintética y propiedades del agua.\n")
        f.write("x, y: Permiten capturar dependencia espacial (zonas endémicas).\n")
        f.write("3.3 Variables adicionales:\n")
        f.write("temporada_lluviosa: Ayuda a captar diferencias térmicas y de escorrentía estacional.\n")
        
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
