from __future__ import annotations
import argparse
import json
import math
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import matplotlib

matplotlib.use("Agg")
import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from rasterio import Env, open as rio_open
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from scipy import ndimage
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[0]
CACHE = ROOT / "cache"
OUTPUTS = ROOT / "outputs"
RASTERS = OUTPUTS / "rasters"
FIGURES = OUTPUTS / "figures"
METADATA = OUTPUTS / "metadata"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
OPENEO_URL = "https://openeofed.dataspace.copernicus.eu"
COLLECTION = "sentinel-2-l2a"
OPENEO_COLLECTION = "SENTINEL2_L2A"
BANDS = ("B03", "B04", "B05", "B08", "SCL")
TARGET_CRS = "EPSG:32615"
RESOLUTION = 20.0
NODATA = -9999.0
# Niveles guia OMS para clorofila-a asociada a cianobacteria (mg/m3 ~ ug/L).
BLOOM_MODERATE_MG_M3 = 10.0
BLOOM_HIGH_MG_M3 = 50.0


@dataclass(frozen=True)
class Lake:
    key: str
    label: str
    bbox: tuple[float, float, float, float]
    dates: tuple[str, ...]


LAKES = {
    "atitlan": Lake(
        "atitlan",
        "Lago Atitlán",
        (-91.326256, 14.5948, -91.07151, 14.750979),
        (
            "2025-01-18", "2025-04-13", "2025-05-13", "2025-07-17",
            "2025-11-21", "2025-12-29", "2026-02-12", "2026-03-24",
            "2026-04-13", "2026-04-28", "2026-07-22",
        ),
    ),
    "amatitlan": Lake(
        "amatitlan",
        "Lago Amatitlán",
        (-90.638065, 14.412347, -90.512924, 14.493799),
        (
            "2025-01-28", "2025-04-15", "2025-04-28", "2025-11-24",
            "2026-01-08", "2026-02-02", "2026-02-07", "2026-03-29",
            "2026-04-13", "2026-04-28", "2026-06-19",
        ),
    ),
}


def mkdirs() -> None:
    for path in (CACHE, RASTERS, FIGURES, METADATA):
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_openeo(authenticate: bool = False) -> dict:
    """Ejercicio 1: establece y documenta la conexion a openEO/CDSE."""
    result: dict = {
        "checked_at_utc": utc_now(),
        "backend": OPENEO_URL,
        "collection": OPENEO_COLLECTION,
        "authenticated": False,
    }
    try:
        import openeo

        connection = openeo.connect(OPENEO_URL)
        if authenticate:
            connection.authenticate_oidc()
            result["authenticated"] = True
        metadata = connection.describe_collection(OPENEO_COLLECTION)
        raw_bands = metadata.get("cube:dimensions", {}).get("bands", {}).get("values", [])
        band_names = [b.get("name") if isinstance(b, dict) else str(b) for b in raw_bands]
        result.update(
            status="ok",
            api_version=connection.capabilities().api_version(),
            collection_title=metadata.get("title", OPENEO_COLLECTION),
            bands=band_names or list(BANDS),
        )
    except Exception as exc:
        # La conexion HTTP publica sigue dejando evidencia util si el cliente cambia.
        discovery = requests.get(f"{OPENEO_URL}/.well-known/openeo", timeout=60)
        discovery.raise_for_status()
        versions = discovery.json().get("versions", [])
        result.update(
            status="discovery_ok_client_error",
            versions=versions,
            client_error=f"{type(exc).__name__}: {exc}",
        )
    (METADATA / "conexion_openeo.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def target_grid(bbox: tuple[float, float, float, float]) -> tuple:
    left, bottom, right, top = transform_bounds("EPSG:4326", TARGET_CRS, *bbox, densify_pts=21)
    left = math.floor(left / RESOLUTION) * RESOLUTION
    bottom = math.floor(bottom / RESOLUTION) * RESOLUTION
    right = math.ceil(right / RESOLUTION) * RESOLUTION
    top = math.ceil(top / RESOLUTION) * RESOLUTION
    width = int(round((right - left) / RESOLUTION))
    height = int(round((top - bottom) / RESOLUTION))
    return from_origin(left, top, RESOLUTION, RESOLUTION), width, height, (left, bottom, right, top)


def search_items(lake: Lake, date: str) -> list[dict]:
    payload = {
        "collections": [COLLECTION],
        "bbox": list(lake.bbox),
        "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z",
        "limit": 100,
    }
    response = requests.post(f"{STAC_URL}/search", json=payload, timeout=90)
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        raise RuntimeError(f"No se encontraron escenas para {lake.label}, {date}")
    # En traslapes se prioriza el granulo con menor nubosidad y se completa con el vecino.
    return sorted(features, key=lambda x: x["properties"].get("eo:cloud_cover", 100.0))


_SAS_TOKEN: str | None = None


def sas_token() -> str:
    global _SAS_TOKEN
    if _SAS_TOKEN is None:
        response = requests.get(
            "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a",
            timeout=60,
        )
        response.raise_for_status()
        _SAS_TOKEN = response.json()["token"]
    return _SAS_TOKEN


def signed_href(href: str) -> str:
    return f"{href}{'&' if '?' in href else '?'}{sas_token()}"


def read_mosaic(items: list[dict], band: str, grid: tuple) -> np.ndarray:
    transform, width, height, _ = grid
    categorical = band == "SCL"
    dtype = np.uint8 if categorical else np.uint16
    mosaic = np.zeros((height, width), dtype=dtype)
    filled = np.zeros((height, width), dtype=bool)
    resampling = Resampling.nearest if categorical else Resampling.bilinear
    env_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MAX_RETRY": "6",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "GDAL_HTTP_TIMEOUT": "60",
    }
    with Env(**env_options):
        for item in items:
            href = signed_href(item["assets"][band]["href"])
            with rio_open(href) as src:
                with WarpedVRT(
                    src,
                    crs=TARGET_CRS,
                    transform=transform,
                    width=width,
                    height=height,
                    resampling=resampling,
                    nodata=0,
                ) as vrt:
                    arr = vrt.read(1, masked=True)
                    good = (~np.ma.getmaskarray(arr)) & (np.asarray(arr) != 0) & (~filled)
                    mosaic[good] = np.asarray(arr, dtype=dtype)[good]
                    filled[good] = True
    return mosaic


def cache_path(lake: Lake, date: str) -> Path:
    return CACHE / lake.key / f"{date}.npz"


def acquire_date(lake: Lake, date: str, grid: tuple, force: bool, attempts: int = 4) -> tuple[dict, list[dict]]:
    path = cache_path(lake, date)
    items = search_items(lake, date)
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                arrays = {band: read_mosaic(items, band, grid) for band in BANDS}
                np.savez_compressed(path, **arrays)
                last_error = None
                break
            except Exception as exc:  # transient red (curl/HTTP): reintenta con backoff
                last_error = exc
                if attempt < attempts:
                    print(f"  aviso: fallo de red ({exc}); reintento {attempt}/{attempts - 1}...", flush=True)
                    time.sleep(3 * attempt)
        if last_error is not None:
            raise last_error
    with np.load(path) as stored:
        arrays = {band: stored[band] for band in BANDS}
    return arrays, items


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(a.shape, np.nan, dtype=np.float32)
    denom = a + b
    np.divide(a - b, denom, out=out, where=np.abs(denom) > 1e-6)
    return out


def derive_indices(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Baseline Sentinel-2 >= 04.00: reflectancia = (DN + BOA_ADD_OFFSET) /
    # BOA_QUANTIFICATION_VALUE. Las escenas oficiales usan -1000 y 10000.
    green = (arrays["B03"].astype(np.float32) - 1000.0) * 0.0001
    red = (arrays["B04"].astype(np.float32) - 1000.0) * 0.0001
    rededge = (arrays["B05"].astype(np.float32) - 1000.0) * 0.0001
    nir = (arrays["B08"].astype(np.float32) - 1000.0) * 0.0001
    scl = arrays["SCL"]
    ndvi = safe_ratio(nir, red)
    ndwi = safe_ratio(green, nir)
    ndci = safe_ratio(rededge, red)
    raw_chla = 826.57 * ndci**3 - 176.43 * ndci**2 + 19.0 * ndci + 4.071
    cyano_chla = np.clip(raw_chla, 0.0, 500.0).astype(np.float32)
    bad_scl = np.isin(scl, [0, 1, 3, 8, 9, 10, 11])
    has_data = np.logical_and.reduce([arrays[b] > 0 for b in ("B03", "B04", "B05", "B08")])
    clear = (~bad_scl) & has_data
    # Cada cociente usa su propia validez. El NIR puede ser negativo sobre agua
    # oscura tras la correccion atmosferica, sin invalidar B04/B05 para NDCI.
    cyano_valid = clear & (red > 0) & (rededge > 0) & ((red + rededge) > 1e-6)
    ndvi_valid = clear & (red > 0) & (nir > 0) & ((red + nir) > 1e-6)
    ndwi_valid = clear & (green > 0) & (nir > 0) & ((green + nir) > 1e-6)
    water_evidence = clear & ((scl == 6) | (ndwi_valid & (ndwi > 0.05)))
    return {
        "cyano_chla": cyano_chla,
        "ndci": ndci,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "clear": clear,
        "cyano_valid": cyano_valid,
        "ndvi_valid": ndvi_valid,
        "ndwi_valid": ndwi_valid,
        "water_evidence": water_evidence,
    }


def largest_component(mask: np.ndarray) -> np.ndarray:
    cleaned = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
    cleaned = ndimage.binary_fill_holes(cleaned)
    labels, count = ndimage.label(cleaned, structure=np.ones((3, 3)))
    if count == 0:
        raise RuntimeError("No se pudo identificar el cuerpo de agua")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    lake = labels == sizes.argmax()
    # Retira 20 m del borde: reduce mezcla espectral tierra/agua.
    eroded = ndimage.binary_erosion(lake, structure=np.ones((3, 3)), iterations=1)
    return eroded if eroded.sum() > 0.75 * lake.sum() else lake


def write_raster(path: Path, arrays: Iterable[np.ndarray], grid: tuple, descriptions: list[str], dtype="float32") -> None:
    transform, width, height, _ = grid
    profile = {
        "driver": "GTiff", "height": height, "width": width,
        "count": len(descriptions), "dtype": dtype, "crs": TARGET_CRS,
        "transform": transform, "nodata": NODATA if dtype == "float32" else 0,
        "compress": "DEFLATE", "predictor": 3 if dtype == "float32" else 2,
        "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rio_open(path, "w", **profile) as dst:
        for idx, (array, description) in enumerate(zip(arrays, descriptions), start=1):
            dst.write(array.astype(dtype), idx)
            dst.set_band_description(idx, description)


def scene_rows(lake: Lake, date: str, items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        p = item["properties"]
        rows.append({
            "lago": lake.label, "fecha_solicitada": date,
            "fecha_hora_utc": p.get("datetime"), "scene_id": item["id"],
            "satelite": p.get("platform"), "nubosidad_escena_pct": p.get("eo:cloud_cover"),
            "coleccion": COLLECTION, "bandas": ",".join(BANDS),
        })
    return rows


def bloom_extent_pct(cyano_values: np.ndarray) -> tuple[float, float]:
    """Ejercicio 8.1: % de pixeles del lago en alerta moderada/alta (niveles guia OMS)."""
    if cyano_values.size == 0:
        return 0.0, 0.0
    moderate = float(100 * np.mean(cyano_values >= BLOOM_MODERATE_MG_M3))
    high = float(100 * np.mean(cyano_values >= BLOOM_HIGH_MG_M3))
    return moderate, high


def process_lake(lake: Lake, force: bool) -> tuple[pd.DataFrame, dict]:
    grid = target_grid(lake.bbox)
    votes = np.zeros((grid[2], grid[1]), dtype=np.uint16)
    seen = np.zeros_like(votes)
    scenes: list[dict] = []

    for pos, date in enumerate(lake.dates, start=1):
        print(f"[{lake.label}] descarga {pos:02d}/{len(lake.dates)}: {date}", flush=True)
        arrays, items = acquire_date(lake, date, grid, force)
        idx = derive_indices(arrays)
        votes += idx["water_evidence"].astype(np.uint16)
        seen += idx["clear"].astype(np.uint16)
        scenes.extend(scene_rows(lake, date, items))

    consensus = np.divide(votes, seen, out=np.zeros(votes.shape, dtype=np.float32), where=seen > 0)
    permanent = largest_component((seen >= 3) & (consensus >= 0.35))
    mask_path = RASTERS / lake.key / "mascara_lago.tif"
    write_raster(mask_path, [permanent.astype(np.uint8)], grid, ["mascara_lago"], dtype="uint8")

    rows = []
    peak_maps: dict[str, np.ndarray] = {}
    bloom_votes = np.zeros(permanent.shape, dtype=np.uint16)
    bloom_seen = np.zeros(permanent.shape, dtype=np.uint16)
    corr_pixels: dict[str, dict[str, list[np.ndarray]]] = {
        "ndvi": {"cyano": [], "other": []},
        "ndwi": {"cyano": [], "other": []},
    }
    for date in lake.dates:
        with np.load(cache_path(lake, date)) as stored:
            arrays = {band: stored[band] for band in BANDS}
        idx = derive_indices(arrays)
        valid_by_index = {
            "cyano_chla": permanent & idx["cyano_valid"],
            "ndci": permanent & idx["cyano_valid"],
            "ndvi": permanent & idx["ndvi_valid"],
            "ndwi": permanent & idx["ndwi_valid"],
        }
        out_arrays = []
        for name in ("cyano_chla", "ndci", "ndvi", "ndwi"):
            out = np.full(permanent.shape, NODATA, dtype=np.float32)
            out[valid_by_index[name]] = idx[name][valid_by_index[name]]
            out_arrays.append(out)
        write_raster(
            RASTERS / lake.key / f"indices_{date}.tif",
            out_arrays,
            grid,
            ["cyano_chla_proxy_mg_m3", "ndci", "ndvi", "ndwi"],
        )
        valid = valid_by_index["cyano_chla"]
        cya = idx["cyano_chla"][valid]
        floracion_moderada_pct, floracion_alta_pct = bloom_extent_pct(cya)
        row = {
            "lago": lake.label, "fecha": date,
            "cyano_media_mg_m3": float(np.nanmean(cya)),
            "cyano_mediana_mg_m3": float(np.nanmedian(cya)),
            "cyano_p90_mg_m3": float(np.nanpercentile(cya, 90)),
            "cyano_desv_mg_m3": float(np.nanstd(cya)),
            "ndci_medio": float(np.nanmean(idx["ndci"][valid])),
            "ndvi_medio": float(np.nanmean(idx["ndvi"][valid_by_index["ndvi"]])),
            "ndwi_medio": float(np.nanmean(idx["ndwi"][valid_by_index["ndwi"]])),
            "pixeles_validos": int(valid.sum()),
            "pixeles_lago": int(permanent.sum()),
            "cobertura_valida_pct": float(100 * valid.sum() / permanent.sum()),
            "floracion_moderada_pct": floracion_moderada_pct,
            "floracion_alta_pct": floracion_alta_pct,
        }
        rows.append(row)
        peak_maps[date] = out_arrays[0]

        bloom_seen[valid] += 1
        bloom_votes[valid & (idx["cyano_chla"] >= BLOOM_MODERATE_MG_M3)] += 1
        for other_name in ("ndvi", "ndwi"):
            pair_mask = valid_by_index["cyano_chla"] & valid_by_index[other_name]
            if pair_mask.any():
                corr_pixels[other_name]["cyano"].append(idx["cyano_chla"][pair_mask])
                corr_pixels[other_name]["other"].append(idx[other_name][pair_mask])

    bloom_persistence = np.divide(
        bloom_votes, bloom_seen, out=np.zeros(bloom_votes.shape, dtype=np.float32), where=bloom_seen > 0
    )
    corr_arrays = {
        name: (np.concatenate(d["cyano"]), np.concatenate(d["other"]))
        for name, d in corr_pixels.items() if d["cyano"]
    }

    return pd.DataFrame(rows), {
        "grid": grid, "mask": permanent, "scenes": scenes, "maps": peak_maps,
        "bloom_persistence": bloom_persistence, "corr": corr_arrays,
    }


def classify_peaks(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["fecha"] = pd.to_datetime(result["fecha"])
    result["calidad_suficiente"] = result["cobertura_valida_pct"] >= 70.0
    result["es_fecha_critica"] = False
    result["rango_intensidad"] = 0
    for lake, group in result.groupby("lago"):
        eligible = group[group["calidad_suficiente"]]
        values = eligible["cyano_media_mg_m3"]
        threshold = values.mean() + values.std(ddof=0)
        critical = eligible.index[values >= threshold]
        # Siempre se conservan las tres fechas mas intensas como prioridades relativas.
        top = eligible.nlargest(3, "cyano_media_mg_m3").index
        result.loc[critical.union(top), "es_fecha_critica"] = True
        ranks = eligible["cyano_media_mg_m3"].rank(method="first", ascending=False).astype(int)
        result.loc[eligible.index, "rango_intensidad"] = ranks
    return result.sort_values(["lago", "fecha"])


def plot_temporal(summary: pd.DataFrame) -> Path:
    path = FIGURES / "evolucion_temporal_cianobacteria.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=False, constrained_layout=True)
    colors = {"Lago Atitlán": "#176b87", "Lago Amatitlán": "#b44728"}
    for ax, (lake, group) in zip(axes, summary.groupby("lago", sort=False)):
        group = group.sort_values("fecha")
        good = group[group["calidad_suficiente"]]
        bad = group[~group["calidad_suficiente"]]
        ax.plot(good["fecha"], good["cyano_media_mg_m3"], marker="o", lw=2.2, color=colors[lake])
        if not bad.empty:
            ax.scatter(bad["fecha"], bad["cyano_media_mg_m3"], marker="x", s=75, color="#9b2226", label="cobertura < 70 %")
        crit = group[group["es_fecha_critica"]]
        ax.scatter(crit["fecha"], crit["cyano_media_mg_m3"], s=70, facecolor="#ffd166", edgecolor="#7a4d00", zorder=4, label="prioridad relativa")
        for _, row in group[group["rango_intensidad"] == 1].iterrows():
            ax.annotate(row["fecha"].strftime("%d-%m-%Y"), (row["fecha"], row["cyano_media_mg_m3"]), xytext=(8, 8), textcoords="offset points", fontsize=9)
        ax.set_title(lake, loc="left", weight="bold")
        ax.set_ylabel("Clorofila-a proxy (mg/m³)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.suptitle("Evolución temporal del proxy de floración compatible con cianobacteria", fontsize=15, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coverage(summary: pd.DataFrame) -> Path:
    path = FIGURES / "cobertura_valida.png"
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    for lake, group in summary.groupby("lago", sort=False):
        group = group.sort_values("fecha")
        ax.plot(group["fecha"], group["cobertura_valida_pct"], marker="o", label=lake)
    ax.axhline(70, color="#8d2f23", ls="--", lw=1, label="referencia 70 %")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Superficie del lago con dato válido (%)")
    ax.set_title("Control de calidad por fecha", loc="left", weight="bold")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_peak_maps(summary: pd.DataFrame, contexts: dict) -> Path:
    path = FIGURES / "mapas_fechas_mayor_indice.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    maxima = []
    selections = []
    for lake, group in summary.groupby("lago", sort=False):
        peak = group[group["calidad_suficiente"]].nlargest(1, "cyano_media_mg_m3").iloc[0]
        key = "atitlan" if "Atitlán" in lake else "amatitlan"
        arr = contexts[key]["maps"][peak["fecha"].strftime("%Y-%m-%d")]
        vals = arr[arr != NODATA]
        maxima.append(float(np.nanpercentile(vals, 98)))
        selections.append((lake, peak, arr))
    vmax = max(maxima)
    image = None
    for ax, (lake, peak, arr) in zip(axes, selections):
        masked = np.ma.masked_equal(arr, NODATA)
        image = ax.imshow(masked, cmap="turbo", norm=Normalize(0, vmax))
        ax.set_title(f"{lake}\n{peak['fecha'].strftime('%d-%m-%Y')}", weight="bold")
        ax.set_axis_off()
    fig.colorbar(image, ax=axes, shrink=0.82, label="Clorofila-a proxy (mg/m³)")
    fig.suptitle("Distribución en la fecha de mayor promedio (control visual)", fontsize=14, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def select_key_dates(summary: pd.DataFrame, lake_label: str) -> dict[str, pd.Series]:
    group = summary[summary["lago"] == lake_label].sort_values("fecha")
    good = group[group["calidad_suficiente"]]
    return {
        "Primera fecha": good.iloc[0],
        "Fecha pico": good.nlargest(1, "cyano_media_mg_m3").iloc[0],
        "Última fecha": good.iloc[-1],
    }


def plot_date_comparison(lake: Lake, contexts: dict, summary: pd.DataFrame) -> Path:
    """Ejercicio 5.2: mapas comparativos entre distintas fechas para un mismo lago."""
    picks = select_key_dates(summary, lake.label)
    maps = contexts[lake.key]["maps"]
    path = FIGURES / f"comparacion_fechas_{lake.key}.png"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    arrays = {label: maps[row["fecha"].strftime("%Y-%m-%d")] for label, row in picks.items()}
    vals_all = np.concatenate([arr[arr != NODATA] for arr in arrays.values()])
    vmax = float(np.nanpercentile(vals_all, 98)) if vals_all.size else 1.0
    image = None
    for ax, (label, row) in zip(axes, picks.items()):
        masked = np.ma.masked_equal(arrays[label], NODATA)
        image = ax.imshow(masked, cmap="turbo", norm=Normalize(0, vmax))
        ax.set_title(f"{label}\n{row['fecha'].strftime('%d-%m-%Y')}", fontsize=10, weight="bold")
        ax.set_axis_off()
    fig.colorbar(image, ax=axes, shrink=0.8, label="Clorofila-a proxy (mg/m³)")
    fig.suptitle(f"{lake.label}: comparación espacial entre fechas", fontsize=13, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bloom_persistence(contexts: dict) -> Path:
    """Ejercicio 8.2: zonas de acumulación persistente de floración."""
    path = FIGURES / "persistencia_floracion.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    labels = {"atitlan": "Lago Atitlán", "amatitlan": "Lago Amatitlán"}
    image = None
    for ax, key in zip(axes, ("atitlan", "amatitlan")):
        persistence = contexts[key]["bloom_persistence"]
        mask = contexts[key]["mask"]
        masked = np.ma.array(persistence, mask=~mask)
        image = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(labels[key], weight="bold")
        ax.set_axis_off()
    fig.colorbar(image, ax=axes, shrink=0.82, label="Fracción de fechas en alerta ≥10 mg/m³")
    fig.suptitle("Zonas de acumulación persistente de floración", fontsize=14, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_distribution_by_date(contexts: dict) -> Path:
    """Ejercicio 8.3: comparación de la distribución de valores entre fechas (boxplots)."""
    path = FIGURES / "distribucion_por_fecha.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    labels = {"atitlan": "Lago Atitlán", "amatitlan": "Lago Amatitlán"}
    for ax, key in zip(axes, ("atitlan", "amatitlan")):
        maps = contexts[key]["maps"]
        dates_sorted = sorted(maps.keys())
        data = [maps[d][maps[d] != NODATA] for d in dates_sorted]
        labels_x = [datetime.strptime(d, "%Y-%m-%d").strftime("%d-%m-%y") for d in dates_sorted]
        ax.boxplot(data, tick_labels=labels_x, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#a9d6e5"))
        ax.set_title(labels[key], loc="left", weight="bold")
        ax.set_ylabel("Clorofila-a proxy (mg/m³)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Distribución del proxy de cianobacteria por fecha", fontsize=14, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def add_season(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["temporada"] = np.where(
        result["fecha"].dt.month.isin([11, 12, 1, 2, 3, 4]), "seca", "lluviosa"
    )
    return result


def plot_seasonal(summary: pd.DataFrame) -> Path:
    """Ejercicio 8.4: patrón estacional (temporada seca nov-abr vs lluviosa may-oct en Guatemala)."""
    path = FIGURES / "patron_estacional.png"
    grouped = (
        summary[summary["calidad_suficiente"]]
        .groupby(["lago", "temporada"])["cyano_media_mg_m3"]
        .mean()
        .unstack()
        .reindex(columns=["seca", "lluviosa"])
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    grouped.plot(kind="bar", ax=ax, color=["#e9c46a", "#2a9d8f"])
    ax.set_ylabel("Clorofila-a proxy media (mg/m³)")
    ax.set_title("Comparación estacional (seca: nov-abr / lluviosa: may-oct)", loc="left", weight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(title="Temporada")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation(contexts: dict) -> tuple[Path, dict]:
    """Ejercicio 6: correlación entre el proxy de cianobacteria y NDVI/NDWI."""
    path = FIGURES / "correlacion_indices.png"
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    labels = {"atitlan": "Lago Atitlán", "amatitlan": "Lago Amatitlán"}
    colors = {"atitlan": "#176b87", "amatitlan": "#b44728"}
    rng = np.random.default_rng(42)
    stats_summary: dict[tuple[str, str], dict] = {}
    for row_idx, key in enumerate(("atitlan", "amatitlan")):
        corr = contexts[key]["corr"]
        for col_idx, other in enumerate(("ndvi", "ndwi")):
            ax = axes[row_idx, col_idx]
            cyano, other_vals = corr[other]
            n = cyano.size
            r, p = pearsonr(cyano, other_vals)
            stats_summary[(key, other)] = {"r": float(r), "p": float(p), "n": int(n)}
            if n > 20000:
                sample_idx = rng.choice(n, 20000, replace=False)
                cyano_s, other_s = cyano[sample_idx], other_vals[sample_idx]
            else:
                cyano_s, other_s = cyano, other_vals
            ax.scatter(other_s, cyano_s, s=4, alpha=0.25, color=colors[key])
            if n > 1:
                coeffs = np.polyfit(other_vals, cyano, 1)
                xs = np.linspace(float(other_vals.min()), float(other_vals.max()), 50)
                ax.plot(xs, np.polyval(coeffs, xs), color="black", lw=1.5)
            p_txt = "< 0.001" if p < 0.001 else f"= {p:.3f}"
            ax.set_title(f"{labels[key]}: cianobacteria vs {other.upper()}\nr = {r:.2f} (p {p_txt}, n={n})", fontsize=10)
            ax.set_xlabel(other.upper())
            ax.set_ylabel("Clorofila-a proxy (mg/m³)")
            ax.grid(alpha=0.25)
    fig.suptitle("Correlación entre el proxy de cianobacteria y NDVI/NDWI", fontsize=14, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path, stats_summary


def correlation_interpretation(correlation_stats: dict) -> list[str]:
    labels = {"atitlan": "Lago Atitlán", "amatitlan": "Lago Amatitlán"}
    messages = []
    for key, label in labels.items():
        for other in ("ndvi", "ndwi"):
            r = correlation_stats[(key, other)]["r"]
            strength = "fuerte" if abs(r) >= 0.5 else "moderada" if abs(r) >= 0.3 else "débil"
            sign = "positiva" if r > 0 else "negativa" if r < 0 else "nula"
            messages.append(
                f"{label} — cianobacteria vs {other.upper()}: correlación {sign} y {strength} (r = {r:.2f})."
            )
    messages.append(
        "Una correlación positiva con NDVI es consistente con acumulaciones o natas superficiales de "
        "cianobacteria, que reflejan la luz de forma parecida a la vegetación densa; el NDWI, en cambio, "
        "se usó para delimitar el propio cuerpo de agua, por lo que su relación con la cianobacteria refleja "
        "sobre todo condiciones de la superficie del agua (turbidez, oleaje, presencia de nata) más que un "
        "vínculo causal directo."
    )
    return messages


def compare_lakes(summary: pd.DataFrame) -> list[str]:
    """Ejercicio 7: comparación de intensidad y frecuencia de floración entre ambos lagos."""
    good = summary[summary["calidad_suficiente"]]
    stats = good.groupby("lago").agg(
        media=("cyano_media_mg_m3", "mean"),
        maximo=("cyano_media_mg_m3", "max"),
        n_criticas=("es_fecha_critica", "sum"),
        n_fechas=("cyano_media_mg_m3", "count"),
    )
    a = stats.loc["Lago Atitlán"]
    b = stats.loc["Lago Amatitlán"]
    mas_intenso = "Atitlán" if a["media"] > b["media"] else "Amatitlán"
    freq_a = a["n_criticas"] / a["n_fechas"]
    freq_b = b["n_criticas"] / b["n_fechas"]
    mas_frecuente = "Atitlán" if freq_a > freq_b else "Amatitlán"
    return [
        f"En el período analizado, Lago Atitlán registró un proxy medio de {a['media']:.2f} mg/m³ "
        f"(máximo {a['maximo']:.2f} mg/m³; {int(a['n_criticas'])} de {int(a['n_fechas'])} fechas marcadas "
        f"como críticas), mientras que Lago Amatitlán registró {b['media']:.2f} mg/m³ "
        f"(máximo {b['maximo']:.2f} mg/m³; {int(b['n_criticas'])} de {int(b['n_fechas'])} fechas críticas).",
        f"En intensidad promedio, {mas_intenso} presenta los valores más altos del proxy; en frecuencia "
        f"relativa de fechas críticas, {mas_frecuente} muestra mayor proporción de eventos "
        f"({freq_a:.0%} de las fechas en Atitlán frente a {freq_b:.0%} en Amatitlán).",
        "Amatitlán es un lago mucho más pequeño y de cuenca más urbanizada e industrializada, cercano al "
        "área metropolitana de Guatemala, con aportes históricos de aguas residuales y agrícolas; eso "
        "favorece mayor disponibilidad de nutrientes (fósforo y nitrógeno) y temperaturas de agua más "
        "cálidas por su menor profundidad. Atitlán, de origen volcánico y mayor profundidad, con menor "
        "densidad poblacional relativa a su tamaño, tiende a diluir más los aportes de nutrientes, aunque "
        "no está exento de floraciones en bahías cercanas a poblaciones costeras.",
        "Estas diferencias de causas probables —mayor presión urbana y carga de nutrientes en Amatitlán "
        "frente a mayor dilución y profundidad en Atitlán— son consistentes con patrones documentados en "
        "la literatura ambiental de ambos lagos, pero esta comparación satelital por sí sola no los "
        "confirma: se requieren datos de campo (fósforo, temperatura, oxígeno disuelto) para validarlos.",
    ]


def exploratory_interpretation(summary: pd.DataFrame, contexts: dict) -> list[str]:
    """Ejercicio 8.5: interpretación integrada del análisis exploratorio."""
    good = summary[summary["calidad_suficiente"]]
    ext = good.groupby("lago")["floracion_alta_pct"].agg(["mean", "max"])
    persistencia_pct = {}
    for key, label in (("atitlan", "Lago Atitlán"), ("amatitlan", "Lago Amatitlán")):
        persistence = contexts[key]["bloom_persistence"]
        mask = contexts[key]["mask"]
        persistencia_pct[label] = float(np.mean(persistence[mask] >= 0.5) * 100)
    return [
        f"En promedio, {ext.loc['Lago Atitlán', 'mean']:.1f}% de la superficie de Atitlán y "
        f"{ext.loc['Lago Amatitlán', 'mean']:.1f}% de la de Amatitlán muestran niveles de alerta alta "
        f"(≥50 mg/m³, criterio guía OMS) en un momento dado; los máximos puntuales llegan a "
        f"{ext.loc['Lago Atitlán', 'max']:.1f}% y {ext.loc['Lago Amatitlán', 'max']:.1f}% respectivamente.",
        f"Considerando persistencia en el tiempo, un {persistencia_pct['Lago Atitlán']:.1f}% del área de "
        f"Atitlán y un {persistencia_pct['Lago Amatitlán']:.1f}% del área de Amatitlán superó el umbral de "
        "alerta moderada en al menos la mitad de las fechas analizadas. Estas zonas son candidatas a "
        "acumulación crónica: típicamente bahías, zonas costeras de baja circulación o cercanas a "
        "descargas de aguas residuales.",
        "Los boxplots por fecha muestran cómo cambia no solo el promedio sino la dispersión de valores: "
        "fechas con distribuciones muy asimétricas (colas largas hacia valores altos) indican floraciones "
        "localizadas, mientras que distribuciones más compactas y elevadas sugieren eventos generalizados "
        "en todo el lago.",
        "La comparación estacional (seca vs. lluviosa) ayuda a distinguir si las floraciones responden más "
        "a aporte de nutrientes por escorrentía de lluvias o a estancamiento y estratificación térmica en "
        "temporada seca; ambos mecanismos son plausibles en Guatemala y no son excluyentes entre sí.",
    ]


def build_interactive_map(lake: Lake, arr: np.ndarray, date_label: str) -> Path:
    """Ejercicio 5.1 (extra): mapa interactivo folium con el proxy de cianobacteria."""
    maps_dir = OUTPUTS / "mapas_interactivos"
    maps_dir.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_equal(arr, NODATA)
    valid_vals = masked.compressed()
    vmax = float(np.nanpercentile(valid_vals, 98)) if valid_vals.size else 1.0
    norm = Normalize(vmin=0, vmax=max(vmax, 1e-6))
    cmap = matplotlib.colormaps["turbo"]
    rgba = cmap(norm(masked.filled(0)))
    rgba[..., 3] = np.where(np.ma.getmaskarray(masked), 0.0, 0.85)
    img_path = maps_dir / f"{lake.key}_overlay.png"
    plt.imsave(img_path, rgba)
    west, south, east, north = lake.bbox
    center = [(south + north) / 2, (west + east) / 2]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    folium.raster_layers.ImageOverlay(
        image=str(img_path),
        bounds=[[south, west], [north, east]],
        opacity=0.85,
        name=f"Cianobacteria proxy — {date_label}",
    ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    html_path = maps_dir / f"{lake.key}.html"
    fmap.save(str(html_path))
    return html_path


def temporal_interpretation(group: pd.DataFrame) -> list[str]:
    excluded = group[~group["calidad_suficiente"]]
    g = group[group["calidad_suficiente"]].sort_values("fecha").copy()
    first = g.iloc[0]
    last = g.iloc[-1]
    peak = g.nlargest(1, "cyano_media_mg_m3").iloc[0]
    low = g.nsmallest(1, "cyano_media_mg_m3").iloc[0]
    cv = 100 * g["cyano_media_mg_m3"].std(ddof=0) / max(g["cyano_media_mg_m3"].mean(), 1e-6)
    direction = "aumenta" if last["cyano_media_mg_m3"] > first["cyano_media_mg_m3"] * 1.1 else "disminuye" if last["cyano_media_mg_m3"] < first["cyano_media_mg_m3"] * 0.9 else "termina en un nivel parecido al inicial"
    messages = [
        f"La serie fluctúa (variación relativa {cv:.1f} %) y {direction} entre la primera y la última observación.",
        f"El máximo relativo ocurre el {peak['fecha'].strftime('%d-%m-%Y')} ({peak['cyano_media_mg_m3']:.2f} mg/m³ proxy); el mínimo, el {low['fecha'].strftime('%d-%m-%Y')} ({low['cyano_media_mg_m3']:.2f}).",
        "Los cambios pueden relacionarse con aportes de nutrientes, temperatura, mezcla del agua y lluvias, pero estas imágenes por sí solas no prueban una causa.",
    ]
    if not excluded.empty:
        dates = ", ".join(d.strftime("%d-%m-%Y") for d in excluded["fecha"])
        messages.insert(0, f"Se excluye del ranking la fecha {dates} por cobertura radiométrica menor a 70 %, aunque permanece documentada en la tabla completa.")
    return messages


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str], footer: str = "") -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.93, title, fontsize=19, weight="bold", color="#164e63", va="top")
    y = 0.875
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=92)
        fig.text(0.08, y, wrapped, fontsize=10.5, va="top", linespacing=1.45, color="#1f2937")
        y -= 0.035 * (wrapped.count("\n") + 1) + 0.026
    if footer:
        fig.text(0.08, 0.045, footer, fontsize=8, color="#4b5563")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.95, title, fontsize=17, weight="bold", color="#164e63", va="top")
    ax = fig.add_axes([0.06, 0.18, 0.88, 0.70])
    ax.imshow(plt.imread(image_path))
    ax.axis("off")
    fig.text(0.07, 0.12, textwrap.fill(caption, 105), fontsize=9.5, va="top", color="#374151")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_results_table(pdf: PdfPages, summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.95, "Fechas prioritarias y control de calidad", fontsize=17, weight="bold", color="#164e63", va="top")
    ax = fig.add_axes([0.055, 0.16, 0.89, 0.72])
    ax.axis("off")
    selected = summary[summary["rango_intensidad"].between(1, 3)].copy().sort_values(["lago", "rango_intensidad"])
    table_data = []
    for _, row in selected.iterrows():
        table_data.append([
            row["lago"].replace("Lago ", ""), row["fecha"].strftime("%d-%m-%Y"),
            f"{row['cyano_media_mg_m3']:.2f}", f"{row['cyano_p90_mg_m3']:.2f}",
            f"{row['cobertura_valida_pct']:.1f} %", str(int(row["rango_intensidad"])),
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=["Lago", "Fecha", "Media", "P90", "Cobertura", "Rango"],
        cellLoc="center", colLoc="center", loc="upper center",
        colWidths=[0.18, 0.18, 0.14, 0.14, 0.18, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.65)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#164e63")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#e8f1f5")
    fig.text(0.07, 0.095, "P90 resume el extremo alto de la distribución sin depender de un único píxel. Una cobertura baja reduce la comparabilidad y exige cautela.", fontsize=9.5, color="#374151", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_report(
    summary: pd.DataFrame,
    connection: dict,
    contexts: dict,
    figures: dict[str, Path],
    correlation_stats: dict,
) -> Path:
    report_path = OUTPUTS / "informe_lagos_cianobacteria.pdf"
    fecha_min = summary["fecha"].min().strftime("%d-%m-%Y")
    fecha_max = summary["fecha"].max().strftime("%d-%m-%Y")

    with PdfPages(report_path) as pdf:
        add_text_page(
            pdf,
            "Monitoreo satelital de cianobacteria\nen los lagos Atitlán y Amatitlán",
            [
                "Este informe resume un monitoreo de floraciones de cianobacteria en los lagos Atitlán y "
                "Amatitlán (Guatemala), realizado con imágenes del satélite Sentinel-2 (programa Copernicus "
                f"de la Unión Europea), entre el {fecha_min} y el {fecha_max}.",
                "Las cianobacterias son microorganismos que, bajo ciertas condiciones (agua cálida, exceso de "
                "nutrientes, poca circulación), pueden multiplicarse rápidamente y formar 'floraciones' que "
                "afectan la calidad del agua, el turismo, la pesca y, en algunos casos, producen toxinas "
                "dañinas para la salud humana y animal.",
                "Monitorear estas floraciones con muestreos físicos en el terreno es costoso y limitado: solo "
                "cubre algunos puntos del lago y pocas fechas. Los satélites, en cambio, permiten observar la "
                "superficie completa del lago de forma repetida en el tiempo, ayudando a identificar cuándo y "
                "dónde ocurren los eventos más intensos, como una primera alerta que después debe confirmarse "
                "con muestreo de campo.",
                "Este documento está dirigido a personas sin conocimientos de programación: se explican los "
                "métodos en términos simples y todos los hallazgos se acompañan de gráficos y mapas.",
            ],
            footer="Fuente de datos: Sentinel-2 L2A (Copernicus), mosaico público de Microsoft Planetary Computer. "
                   "Script de referencia: Cyanobacteria Chlorophyll-a NDCI (custom-scripts.sentinel-hub.com).",
        )

        add_text_page(
            pdf,
            "¿Cómo se hizo el análisis?",
            [
                "El satélite Sentinel-2 capta la luz reflejada por la superficie terrestre en varias bandas de "
                "color, incluyendo colores que el ojo humano no ve (como el infrarrojo cercano). Combinando "
                "esas bandas se calculan 'índices' que resaltan fenómenos específicos:",
                "• NDVI: sensible a vegetación densa; en el agua, valores altos pueden indicar acumulaciones "
                "superficiales de algas o cianobacteria.",
                "• NDWI: ayuda a identificar y delimitar con precisión la superficie de agua del lago.",
                "• Índice de cianobacteria (NDCI, adaptado del script oficial de detección de cianobacteria "
                "de Sentinel Hub): combina bandas del rojo y del borde rojo (red edge) y se convierte a un "
                "'proxy' de clorofila-a en mg/m³, un indicador reconocido de la biomasa de algas/cianobacteria.",
                "Para cada fecha se descartan automáticamente los píxeles con nubes, sombras, nieve o sin "
                "datos válidos, y solo se analiza el área permanente de agua de cada lago (calculada por "
                "consenso entre todas las fechas). Una fecha se considera de calidad suficiente cuando al "
                "menos el 70 % del lago tiene datos válidos; las fechas con menor cobertura se muestran igual "
                "en las tablas, pero se excluyen de los rankings de intensidad para no distorsionar las "
                "comparaciones.",
                "El proxy de clorofila-a es una estimación indirecta a partir del color del agua: no es un "
                "conteo de células ni una medición de toxinas, y su fórmula original fue calibrada para otro "
                "nivel de procesamiento del satélite, por lo que debe interpretarse como una señal de alerta, "
                "no como una medición de laboratorio.",
            ],
        )

        add_text_page(
            pdf,
            "Acceso a los datos",
            [
                f"Se verificó el acceso al backend oficial openEO de Copernicus Data Space ({connection.get('backend', '')}) "
                f"para la colección {connection.get('collection', '')}. Estado de la verificación: "
                f"{connection.get('status', 'sin datos')}.",
                "La descarga efectiva de las imágenes usadas en este análisis se realizó a través del catálogo "
                "público STAC de Microsoft Planetary Computer (espejo del mismo archivo oficial de Sentinel-2 "
                "L2A), lo que permitió automatizar el proceso sin necesidad de credenciales. El identificador "
                "exacto de cada escena satelital utilizada queda documentado en el archivo "
                "outputs/metadata/catalogo_escenas.csv, para trazabilidad completa.",
                "En cada fecha se descargaron únicamente las cinco bandas necesarias (verde, rojo, borde rojo, "
                "infrarrojo cercano y la máscara de clasificación de escena), evitando descargar la imagen "
                "satelital completa y reduciendo así el tiempo y volumen de datos transferidos.",
            ],
        )

        add_image_page(
            pdf, "Evolución temporal de la cianobacteria", figures["temporal"],
            "Proxy de clorofila-a promedio por fecha en cada lago. Los puntos amarillos marcan las fechas "
            "identificadas como prioritarias (picos relativos o valores muy por encima del promedio); las "
            "equis rojas marcan fechas excluidas del ranking por baja cobertura de datos válidos.",
        )
        temporal_paragraphs = []
        for lake_label, group in summary.groupby("lago", sort=False):
            temporal_paragraphs.append(f"— {lake_label} —")
            temporal_paragraphs.extend(temporal_interpretation(group))
        add_text_page(pdf, "Interpretación de la evolución temporal", temporal_paragraphs)

        add_image_page(
            pdf, "Control de calidad de las imágenes", figures["coverage"],
            "Porcentaje del lago con datos válidos (sin nubes ni sombras) en cada fecha. La línea punteada "
            "marca el umbral mínimo (70 %) usado para considerar una fecha comparable en los rankings.",
        )

        add_results_table(pdf, summary)

        add_image_page(
            pdf, f"Mapas espaciales — {LAKES['atitlan'].label}", figures["comparacion_atitlan"],
            "Distribución espacial del proxy de cianobacteria en tres momentos del período: la primera "
            "fecha, la fecha de mayor promedio (pico) y la última fecha analizada.",
        )
        add_image_page(
            pdf, f"Mapas espaciales — {LAKES['amatitlan'].label}", figures["comparacion_amatitlan"],
            "Distribución espacial del proxy de cianobacteria en tres momentos del período: la primera "
            "fecha, la fecha de mayor promedio (pico) y la última fecha analizada.",
        )
        add_image_page(
            pdf, "Mapas de control en la fecha de mayor índice", figures["maps"],
            "Distribución del proxy de cianobacteria en la fecha de mayor promedio de cada lago, usada como "
            "control visual adicional de los resultados numéricos.",
        )
        add_text_page(
            pdf,
            "Interpretación de los patrones espaciales",
            [
                "En ambos lagos, los valores más altos del proxy tienden a concentrarse en las orillas, "
                "bahías y zonas de baja circulación de agua, donde los nutrientes se acumulan más fácilmente "
                "y el agua se mezcla menos con el resto del cuerpo lacustre.",
                "Al comparar fechas dentro de un mismo lago, se observa que la distribución espacial no es "
                "estática: algunas zonas aparecen afectadas de forma recurrente en varias fechas (ver sección "
                "de persistencia más adelante), mientras que otras solo muestran valores altos en fechas "
                "puntuales, lo que sugiere eventos de floración más transitorios.",
                "Además de los mapas estáticos de este informe, se generaron mapas interactivos (uno por "
                "lago) que permiten hacer zoom y explorar la distribución espacial sobre un mapa base real; "
                "estos se entregan como archivos HTML en outputs/mapas_interactivos/.",
            ],
        )

        add_image_page(
            pdf, "Zonas de acumulación persistente", figures["persistencia"],
            "Fracción de las fechas analizadas en que cada punto del lago superó el nivel de alerta "
            "moderada de cianobacteria (≥10 mg/m³, criterio guía de la OMS). Los tonos más oscuros señalan "
            "zonas afectadas de forma repetida a lo largo del período.",
        )
        add_image_page(
            pdf, "Distribución de valores por fecha", figures["distribucion"],
            "Cada caja resume la distribución del proxy de cianobacteria en el lago para una fecha dada "
            "(línea central = mediana, caja = rango donde se concentra la mitad de los valores).",
        )
        add_image_page(
            pdf, "Patrón estacional", figures["estacional"],
            "Comparación del proxy promedio de cianobacteria entre la temporada seca (noviembre-abril) y "
            "la temporada lluviosa (mayo-octubre) en Guatemala, para cada lago.",
        )
        add_text_page(pdf, "Análisis exploratorio adicional", exploratory_interpretation(summary, contexts))

        add_image_page(
            pdf, "Correlación con NDVI y NDWI", figures["correlacion"],
            "Relación, a nivel de píxel, entre el proxy de cianobacteria y los índices NDVI y NDWI en cada "
            "lago. La línea negra es la tendencia lineal; r es el coeficiente de correlación de Pearson.",
        )
        add_text_page(pdf, "Interpretación de la correlación", correlation_interpretation(correlation_stats))

        add_text_page(pdf, "Comparación entre los dos lagos", compare_lakes(summary))

        add_text_page(
            pdf,
            "Conclusiones y limitaciones",
            [
                "El monitoreo satelital permitió identificar, de forma consistente y con la misma "
                "metodología para ambos lagos, las fechas y zonas con mayor probabilidad de floración de "
                "cianobacteria durante el período analizado, así como diferencias claras de intensidad y "
                "frecuencia entre Atitlán y Amatitlán.",
                "El indicador utilizado es un proxy de clorofila-a, no una medición directa de células de "
                "cianobacteria ni de las toxinas que pueden producir. Las fechas y zonas marcadas como "
                "críticas en este informe deben interpretarse como una alerta que orienta dónde y cuándo "
                "priorizar el muestreo físico de campo, no como una confirmación de riesgo sanitario.",
                "Se recomienda dar seguimiento a las zonas de acumulación persistente identificadas y "
                "contrastar los resultados con mediciones de campo (temperatura, fósforo, nitrógeno, oxígeno "
                "disuelto) para fortalecer la interpretación ambiental de estos hallazgos.",
            ],
        )

    return report_path


def validate_delivery(summary: pd.DataFrame, contexts: dict, report: Path) -> dict:
    expected_dates = {(lake.label, date) for lake in LAKES.values() for date in lake.dates}
    actual_dates = {(row.lago, row.fecha.strftime("%Y-%m-%d")) for row in summary.itertuples()}
    raster_checks = []
    for lake in LAKES.values():
        for date in lake.dates:
            path = RASTERS / lake.key / f"indices_{date}.tif"
            with rio_open(path) as src:
                raster_checks.append(
                    path.exists()
                    and src.count == 4
                    and src.crs.to_string() == TARGET_CRS
                    and abs(src.res[0] - RESOLUTION) < 1e-6
                    and list(src.descriptions) == [
                        "cyano_chla_proxy_mg_m3", "ndci", "ndvi", "ndwi"
                    ]
                )
    areas = {
        key: float(context["mask"].sum() * RESOLUTION * RESOLUTION / 1_000_000)
        for key, context in contexts.items()
    }
    checks = {
        "22_fechas_exactas": len(summary) == 22 and actual_dates == expected_dates,
        "22_rasters_indices_validos": len(raster_checks) == 22 and all(raster_checks),
        "2_mascaras": all((RASTERS / lake.key / "mascara_lago.tif").exists() for lake in LAKES.values()),
        "rangos_cyano_validos": bool(summary["cyano_media_mg_m3"].between(0, 500).all()),
        "cobertura_en_rango": bool(summary["cobertura_valida_pct"].between(0, 100).all()),
        "area_atitlan_plausible": 100 <= areas["atitlan"] <= 150,
        "area_amatitlan_plausible": 10 <= areas["amatitlan"] <= 20,
        "informe_pdf_no_vacio": report.exists() and report.stat().st_size > 100_000,
    }
    result = {
        "checked_at_utc": utc_now(),
        "status": "ok" if all(checks.values()) else "failed",
        "checks": checks,
        "area_mascara_km2": areas,
        "nota_calidad": "Atitlan 2025-01-18 se conserva, pero se excluye del ranking si su cobertura es menor a 70 %.",
    }
    (METADATA / "validacion.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result["status"] != "ok":
        raise RuntimeError(f"Falló la validación de entrega: {checks}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authenticate-openeo", action="store_true", help="Autenticacion OIDC interactiva con Copernicus")
    parser.add_argument("--only-connection", action="store_true", help="Solo verifica openEO")
    parser.add_argument("--force-download", action="store_true", help="Ignora la cache local")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mkdirs()
    print("Verificando conexión openEO/Copernicus...", flush=True)
    connection = verify_openeo(authenticate=args.authenticate_openeo)
    if args.only_connection:
        print(json.dumps(connection, ensure_ascii=False, indent=2))
        return 0

    all_summaries = []
    contexts = {}
    all_scenes = []
    for lake in LAKES.values():
        summary, context = process_lake(lake, force=args.force_download)
        all_summaries.append(summary)
        contexts[lake.key] = context
        all_scenes.extend(context["scenes"])

    summary = add_season(classify_peaks(pd.concat(all_summaries, ignore_index=True)))
    summary.to_csv(OUTPUTS / "resumen_temporal.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    pd.DataFrame(all_scenes).drop_duplicates("scene_id").to_csv(
        METADATA / "catalogo_escenas.csv", index=False, encoding="utf-8-sig"
    )

    print("Generando gráficos y mapas...", flush=True)
    figures = {
        "temporal": plot_temporal(summary),
        "coverage": plot_coverage(summary),
        "maps": plot_peak_maps(summary, contexts),
        "comparacion_atitlan": plot_date_comparison(LAKES["atitlan"], contexts, summary),
        "comparacion_amatitlan": plot_date_comparison(LAKES["amatitlan"], contexts, summary),
        "persistencia": plot_bloom_persistence(contexts),
        "distribucion": plot_distribution_by_date(contexts),
        "estacional": plot_seasonal(summary),
    }
    figures["correlacion"], correlation_stats = plot_correlation(contexts)

    print("Generando mapas interactivos...", flush=True)
    for lake in LAKES.values():
        picks = select_key_dates(summary, lake.label)
        peak_row = picks["Fecha pico"]
        peak_arr = contexts[lake.key]["maps"][peak_row["fecha"].strftime("%Y-%m-%d")]
        build_interactive_map(lake, peak_arr, peak_row["fecha"].strftime("%d-%m-%Y"))

    report = make_report(summary, connection, contexts, figures, correlation_stats)
    validation = validate_delivery(summary, contexts, report)
    print(f"Validación: {validation['status']}", flush=True)
    print(f"Listo: {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
