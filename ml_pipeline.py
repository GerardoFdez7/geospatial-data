"""Preparacion del conjunto de datos de Machine Learning (Laboratorio 4, Parte 2).

Reutiliza los productos de la Parte I (`main.py`): la mascara permanente de cada
lago y el cache local de bandas Sentinel-2. Construye una tabla donde cada fila
es un pixel de agua valido en una fecha concreta, con sus bandas, indices
espectrales y variables espaciales/temporales derivadas.

El modelado (ejercicios 4-10) vive en `notebooks/laboratorio4_parte2.ipynb`;
aqui solo se resuelve la preparacion para que el notebook no repita el trabajo
pesado de leer 22 rasters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rasterio import open as rio_open
from rasterio.transform import xy as transform_xy
from scipy import ndimage

from main import BANDS, LAKES, NODATA, RESOLUTION, derive_indices, cache_path

ROOT = Path(__file__).resolve().parents[0]
OUTPUTS = ROOT / "outputs"
RASTERS = OUTPUTS / "rasters"
FIGURES = OUTPUTS / "figures"
ML_OUTPUTS = OUTPUTS / "ml"

DATASET_PARQUET = ML_OUTPUTS / "dataset_cianobacteria_ml.parquet"
DATASET_CSV = ML_OUTPUTS / "dataset_cianobacteria_ml.csv.gz"

# Ventana de 5x5 pixeles = 100 m x 100 m a 20 m de resolucion.
TEXTURE_WINDOW = 5

# Umbral de alta presencia de cianobacteria (mg/m3 de clorofila-a proxy).
# Nivel de "Alert Level 2" de la OMS (Chorus & Welker, 2021).
THRESHOLD_ALTA_PRESENCIA = 50.0

# Variables que participaron, directa o indirectamente, en la construccion de
# la respuesta y por lo tanto quedan prohibidas como predictoras:
#   cyano_chla_proxy -> es la respuesta antes de binarizar
#   ndci             -> unico insumo del polinomio de clorofila-a
#   B04, B05         -> las dos bandas que forman el NDCI
#   ndvi             -> (B08 - B04) / (B08 + B04); contiene B04
FUGA_RESPUESTA = ("cyano_chla_proxy", "ndci", "B04", "B05", "ndvi")

PREDICTORES = (
    "B03",
    "B08",
    "ndwi",
    "razon_b03_b08",
    "b03_textura",
    "b03_anomalia",
    "b08_textura",
    "dist_orilla_m",
    "doy_sin",
    "doy_cos",
    "temporada_lluviosa",
)


def mkdirs() -> None:
    ML_OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def reflectancia(dn: np.ndarray) -> np.ndarray:
    """DN Sentinel-2 (baseline >= 04.00) a reflectancia BOA."""
    return (dn.astype(np.float32) - 1000.0) * 0.0001


def local_stats(valores: np.ndarray, valido: np.ndarray, size: int = TEXTURE_WINDOW):
    """Media y desviacion locales ignorando los pixeles invalidos de la ventana."""
    peso = valido.astype(np.float32)
    x = np.where(valido, valores, 0.0).astype(np.float32)
    n = ndimage.uniform_filter(peso, size=size, mode="constant")
    s1 = ndimage.uniform_filter(x, size=size, mode="constant")
    s2 = ndimage.uniform_filter(x * x, size=size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        media = np.where(n > 0, s1 / n, np.nan)
        var = np.where(n > 0, s2 / n - media**2, np.nan)
    return media, np.sqrt(np.clip(var, 0.0, None))


def lake_context(lake_key: str) -> dict:
    """Mascara permanente del lago, georreferencia y distancia a la orilla."""
    with rio_open(RASTERS / lake_key / "mascara_lago.tif") as src:
        mask = src.read(1).astype(bool)
        transform = src.transform
        crs = src.crs.to_string()

    # distance_transform_edt mide, para cada pixel de agua, cuantos pixeles hay
    # hasta el no-agua mas cercano. En metros: x resolucion.
    dist_orilla = ndimage.distance_transform_edt(mask) * RESOLUTION

    filas, columnas = np.nonzero(mask)
    xs, ys = transform_xy(transform, filas, columnas)
    return {
        "mask": mask,
        "transform": transform,
        "crs": crs,
        "dist_orilla": dist_orilla.astype(np.float32),
        "filas": filas.astype(np.int32),
        "columnas": columnas.astype(np.int32),
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
    }


def date_frame(lake, date: str, ctx: dict) -> pd.DataFrame:
    """Tabla de un lago en una fecha: un renglon por pixel de agua valido."""
    with np.load(cache_path(lake, date)) as stored:
        arrays = {band: stored[band] for band in BANDS}

    idx = derive_indices(arrays)
    mask = ctx["mask"]

    # Misma regla de validez que la Parte I: pixel dentro del lago, sin nube ni
    # sombra, con B04 y B05 utilizables para el NDCI. Asi el conteo de filas
    # coincide exactamente con `pixeles_validos` de outputs/resumen_temporal.csv.
    valido = mask & idx["cyano_valid"]
    if not valido.any():
        return pd.DataFrame()

    b03 = reflectancia(arrays["B03"])
    b04 = reflectancia(arrays["B04"])
    b05 = reflectancia(arrays["B05"])
    b08 = reflectancia(arrays["B08"])

    media_b03, textura_b03 = local_stats(b03, valido)
    _, textura_b08 = local_stats(b08, valido)

    with np.errstate(invalid="ignore", divide="ignore"):
        razon = np.where(np.abs(b08) > 1e-4, b03 / b08, np.nan)
    razon = np.clip(razon, -50.0, 50.0)

    # NDVI y NDWI conservan su propia regla de validez: sobre agua muy oscura el
    # NIR puede quedar negativo tras la correccion atmosferica, lo que invalida
    # el cociente sin invalidar el NDCI. Esos casos entran como NaN declarado.
    ndvi = np.where(idx["ndvi_valid"], idx["ndvi"], np.nan)
    ndwi = np.where(idx["ndwi_valid"], idx["ndwi"], np.nan)

    fecha = pd.Timestamp(date)
    doy = fecha.day_of_year
    n = int(valido.sum())

    # Indices de los pixeles validos dentro del vector de pixeles del lago.
    sel = valido[ctx["filas"], ctx["columnas"]]

    datos = {
        "lago": np.full(n, lake.label),
        "fecha": np.full(n, fecha),
        "x": ctx["x"][sel],
        "y": ctx["y"][sel],
        "fila": ctx["filas"][sel],
        "columna": ctx["columnas"][sel],
        "B03": b03[valido],
        "B04": b04[valido],
        "B05": b05[valido],
        "B08": b08[valido],
        "ndvi": ndvi[valido],
        "ndwi": ndwi[valido],
        "ndci": idx["ndci"][valido],
        "cyano_chla_proxy": idx["cyano_chla"][valido],
        "razon_b03_b08": razon[valido],
        "b03_textura": textura_b03[valido],
        "b03_anomalia": (b03 - media_b03)[valido],
        "b08_textura": textura_b08[valido],
        "dist_orilla_m": ctx["dist_orilla"][valido],
        "mes": np.full(n, fecha.month, dtype=np.int16),
        "doy_sin": np.full(n, np.sin(2 * np.pi * doy / 365.25), dtype=np.float32),
        "doy_cos": np.full(n, np.cos(2 * np.pi * doy / 365.25), dtype=np.float32),
        "temporada_lluviosa": np.full(n, int(5 <= fecha.month <= 10), dtype=np.int8),
    }
    return pd.DataFrame(datos)


def build_ml_dataset(verbose: bool = True) -> pd.DataFrame:
    """Ejercicio 1.1-1.3: conjunto de datos listo para Machine Learning."""
    partes: list[pd.DataFrame] = []
    for lake in LAKES.values():
        ctx = lake_context(lake.key)
        if verbose:
            area = ctx["mask"].sum() * RESOLUTION**2 / 1e6
            print(f"[{lake.label}] {ctx['mask'].sum():,} pixeles de agua ({area:.1f} km2)")
        for date in lake.dates:
            parte = date_frame(lake, date, ctx)
            partes.append(parte)
            if verbose:
                print(f"  {date}: {len(parte):,} observaciones validas")

    df = pd.concat(partes, ignore_index=True)
    df["lago"] = df["lago"].astype("category")
    df["alta_cianobacteria"] = (
        df["cyano_chla_proxy"] >= THRESHOLD_ALTA_PRESENCIA
    ).astype(np.int8)
    return df


def resumen_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Ejercicio 1.4: conteos, tipos y porcentaje de faltantes."""
    por_lago = df.groupby("lago", observed=True).size().rename("observaciones").to_frame()
    por_lago["porcentaje"] = 100 * por_lago["observaciones"] / len(df)

    por_fecha = (
        df.groupby(["lago", "fecha"], observed=True)
        .size()
        .rename("observaciones")
        .reset_index()
    )

    variables = pd.DataFrame(
        {
            "tipo": df.dtypes.astype(str),
            "faltantes_pct": 100 * df.isna().sum() / len(df),
            "rol": [
                "respuesta"
                if c == "alta_cianobacteria"
                else "excluida (fuga)"
                if c in FUGA_RESPUESTA
                else "predictora"
                if c in PREDICTORES
                else "identificador"
                for c in df.columns
            ],
        }
    )
    return {"por_lago": por_lago, "por_fecha": por_fecha, "variables": variables}


def cargar_dataset(reconstruir: bool = False) -> pd.DataFrame:
    """Lee el dataset cacheado o lo reconstruye desde los rasters."""
    if DATASET_PARQUET.exists() and not reconstruir:
        return pd.read_parquet(DATASET_PARQUET)
    mkdirs()
    df = build_ml_dataset()
    df.to_parquet(DATASET_PARQUET, index=False)
    return df


def main() -> int:
    mkdirs()
    df = build_ml_dataset()

    df.to_parquet(DATASET_PARQUET, index=False)
    print(f"\nDataset guardado en {DATASET_PARQUET} ({len(df):,} filas)")

    # Copia comprimida en texto plano para inspeccion sin pyarrow.
    df.sample(n=min(200_000, len(df)), random_state=42).to_csv(
        DATASET_CSV, index=False, compression="gzip"
    )
    print(f"Muestra de 200k filas en {DATASET_CSV}")

    resumen = resumen_dataset(df)
    for nombre, tabla in resumen.items():
        destino = ML_OUTPUTS / f"resumen_{nombre}.csv"
        tabla.to_csv(destino)
        print(f"  {nombre}: {destino}")

    positivos = int(df["alta_cianobacteria"].sum())
    print(
        f"\nRespuesta (>= {THRESHOLD_ALTA_PRESENCIA:.0f} mg/m3): "
        f"{positivos:,} positivos = {100 * positivos / len(df):.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
