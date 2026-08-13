from __future__ import annotations
import argparse
import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import matplotlib

matplotlib.use("Agg")
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


ROOT = Path(__file__).resolve().parents[1]
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


def acquire_date(lake: Lake, date: str, grid: tuple, force: bool) -> tuple[dict, list[dict]]:
    path = cache_path(lake, date)
    items = search_items(lake, date)
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {band: read_mosaic(items, band, grid) for band in BANDS}
        np.savez_compressed(path, **arrays)
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
        }
        rows.append(row)
        peak_maps[date] = out_arrays[0]

    return pd.DataFrame(rows), {"grid": grid, "mask": permanent, "scenes": scenes, "maps": peak_maps}


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

    summary = classify_peaks(pd.concat(all_summaries, ignore_index=True))
    summary.to_csv(OUTPUTS / "resumen_temporal.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    pd.DataFrame(all_scenes).drop_duplicates("scene_id").to_csv(
        METADATA / "catalogo_escenas.csv", index=False, encoding="utf-8-sig"
    )
    temporal = plot_temporal(summary)
    coverage = plot_coverage(summary)
    maps = plot_peak_maps(summary, contexts)
    report = make_report(summary, connection, temporal, coverage, maps)
    validation = validate_delivery(summary, contexts, report)
    print(f"Validación: {validation['status']}", flush=True)
    print(f"Listo: {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
