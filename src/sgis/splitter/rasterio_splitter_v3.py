"""
# Remplacement de `sgis.splitter._splitter.Splitter` basé sur Rasterio

## Fonctionnement

- Si un bâtiment s'étend sur plusieurs rasters, le premier raster qui génère son image est prioritaire ; les rasters suivants ignorent ce bâtiment.
- Reprise au niveau des bâtiments : si l'exécution est interrompue ou plante, il suffit de la relancer. Seuls les bâtiments dont le fichier `<ID>.jpg` n'existe pas encore seront traités. Les images sont d'abord écrites dans un fichier temporaire, puis renommées. Ainsi, un fichier `<ID>.jpg` existant est toujours complet : un plantage ne peut jamais laisser une image tronquée qui serait ensuite ignorée à tort.
- Génère un fichier `progress.txt` : il sert uniquement de raccourci pour ignorer les rasters déjà entièrement traités, sans même avoir à les ouvrir.

## Pourquoi cette solution est plus rapide que la version QGIS

- Aucun appel à `native:extractbyexpression` pour chaque bâtiment, ce qui entraînerait un parcours complet de la couche pour chaque bâtiment.
- Aucun appel à `gdal:cliprasterbymasklayer` pour chaque bâtiment.
- Chaque raster est décodé en grandes bandes horizontales, et les bâtiments sont découpés directement à partir des données en mémoire. Les bandes ne contenant aucun bâtiment restant à traiter ne sont même pas décodées.
- Les rasters sont répartis entre plusieurs processus de travail, évitant ainsi les problèmes liés au multithreading de QGIS.

Dépendance : QGIS n'est donc pas nécessaire.

## Différences à prendre en compte

- Alignement du découpage : le découpage est aligné sur la grille de pixels du raster. Avec la méthode GDAL utilisant un _cutline warp_, le contenu peut être décalé de moins d'un pixel.

- Le format de sortie est configurable via les paramètres `image_format` et `extension` :

- `GTiff` : format sans perte, compressé avec LZW et géoréférencé. Il conserve toutes les bandes du raster source ainsi que son type de données. C'est le comportement attendu de l'option `COMPRESS=LZW` dans l'appel GDAL d'origine.

- `JPEG` : format avec perte, utilisant par défaut une qualité de 75, conformément au pilote JPEG de GDAL. Il accepte uniquement les données `uint8` et conserve les trois premières bandes. Pour un raster à une seule bande, l'image est produite en niveaux de gris.

L'extension du fichier est indépendante de son contenu (par défaut, `.tif` ou `.jpg`). Elle peut donc être configurée pour correspondre aux noms attendus par le code utilisé en aval.

Le format par défaut est `"JPEG"`, car le code QGIS/GDAL d'origine produit des fichiers JPEG standard (_baseline JPEG_). L'option `COMPRESS=LZW` est propre au format GTiff et est ignorée par le pilote JPEG.

Pour comparer les paramètres de qualité, la commande suivante  peut être utilisée:

```
  identify -format '%Q' <une_image>
```

"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil, floor
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

try:  # optional: keep the recap step of the original splitter if sgis is installed
    from sgis.splitter._splitting_recap import SplittingRecap
except Exception:  # pragma: no cover
    SplittingRecap = None

logger = logging.getLogger("sgis.splitter.rasterio")

ID_ETALAB_FIELD_NAME = "ID"
ID_RNB_FIELD_NAME = "rnb_id"
MAX_CONSECUTIVE_ERRORS = 5
TMP_SUFFIX = ".tmp"


# --------------------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------------------
_STATE: dict = {}


def _init_worker(shapefile: str):
    """Load the vector layer once per worker process (reused for all its rasters)."""
    gdf = gpd.read_file(shapefile)
    if ID_ETALAB_FIELD_NAME in gdf.columns:
        id_field = ID_ETALAB_FIELD_NAME
    elif ID_RNB_FIELD_NAME in gdf.columns:
        id_field = ID_RNB_FIELD_NAME
    else:
        raise KeyError("Vector layer must have an 'ID' or 'rnb_id' attribute.")
    gdf = gdf[[id_field, "geometry"]]
    gdf.sindex  # build the spatial index now (cached on the GeoDataFrame)
    _STATE["gdf"] = gdf
    _STATE["id_field"] = id_field


class _Job(NamedTuple):
    building_id: str
    geom: object  # shapely Polygon (OMBB) in raster CRS
    r0: int
    r1: int
    c0: int
    c1: int


def _existing_ids(images_dir: Path, extension: str) -> set[str]:
    """IDs of the buildings that already have a (complete) image."""
    n = len(extension)
    with os.scandir(images_dir) as it:
        return {e.name[:-n] for e in it if e.name.endswith(extension)}


def _save_image_atomic(crop: np.ndarray, path: Path, image_format: str, quality: int,
                       transform=None, crs=None):
    """
    `crop` has shape (bands, height, width).

    Write to a temporary file, then rename. `path` therefore either does not exist or is
    complete, whatever happens to the process (crash, kill, disk full...).
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}{TMP_SUFFIX}")
    try:
        if image_format == "GTiff":  # lossless + georeferenced
            with rasterio.open(tmp, "w", driver="GTiff", height=crop.shape[1], width=crop.shape[2],
                               count=crop.shape[0], dtype=crop.dtype, crs=crs, transform=transform,
                               compress="lzw") as dst:
                dst.write(crop)
        else:
            img = crop[0] if crop.shape[0] == 1 else np.moveaxis(crop, 0, -1)
            Image.fromarray(img).save(tmp, format="JPEG", quality=quality)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def _build_jobs(sub, id_field, src, skip_ids: set[str]) -> tuple[list[_Job], int]:
    """
    OMBB + pixel window (clamped to the raster) for every building NOT already done.
    Returns (jobs, number of buildings skipped because their image already exists).
    """
    width, height = src.width, src.height
    inv = ~src.transform  # world -> pixel

    ombbs: dict[str, list] = {}
    skipped: set[str] = set()
    for bid, geom in zip(sub[id_field].astype(str), sub.geometry):
        if bid in skip_ids:
            skipped.add(bid)
            continue
        if geom is None or geom.is_empty:
            continue
        ombb = geom.minimum_rotated_rectangle
        if ombb.geom_type != "Polygon" or ombb.is_empty:
            continue
        ombbs.setdefault(bid, []).append(ombb)

    jobs = []
    for bid, parts in ombbs.items():
        # several features sharing one ID -> one image covering all of them
        # (the QGIS version used all features with that ID as the clipping mask)
        geom = parts[0] if len(parts) == 1 else unary_union(parts)
        minx, miny, maxx, maxy = geom.bounds
        # north-up raster: pixel row grows when y decreases
        cols = (inv * (minx, maxy))[0], (inv * (maxx, miny))[0]
        rows = (inv * (minx, maxy))[1], (inv * (maxx, miny))[1]
        c0, c1 = max(0, floor(min(cols))), min(width, ceil(max(cols)))
        r0, r1 = max(0, floor(min(rows))), min(height, ceil(max(rows)))
        if c1 <= c0 or r1 <= r0:  # only touches the raster edge
            continue
        jobs.append(_Job(bid, geom, r0, r1, c0, c1))
    return jobs, len(skipped)


def _group_jobs(jobs: list[_Job], band_rows: int | None) -> list[list[_Job]]:
    """Group buildings so that each group fits in a strip of ~`band_rows` rows."""
    jobs = sorted(jobs, key=lambda j: j.r0)
    if not band_rows:
        return [jobs] if jobs else []
    groups, cur, g_r0 = [], [], 0
    for job in jobs:
        if cur and job.r1 - g_r0 > band_rows:
            groups.append(cur)
            cur = []
        if not cur:
            g_r0 = job.r0
        cur.append(job)
    if cur:
        groups.append(cur)
    return groups


def _process_raster(raster_path: str, images_dir: str, repartition_dir: str,
                    band_rows: int | None, jpeg_quality: int, image_format: str, extension: str):
    t0 = perf_counter()
    raster_path = Path(raster_path)
    images_dir, repartition_dir = Path(images_dir), Path(repartition_dir)
    gdf, id_field = _STATE["gdf"], _STATE["id_field"]
    n_written = 0

    with rasterio.open(raster_path) as src:
        if image_format == "JPEG" and src.dtypes[0] != "uint8":
            raise ValueError(f"{raster_path.name}: JPEG output needs a uint8 raster, got {src.dtypes[0]} "
                             f"(use image_format='GTiff')")
        n_bands = src.count if image_format == "GTiff" else (3 if src.count >= 3 else 1)
        indexes = list(range(1, n_bands + 1))

        # 1) candidate buildings (spatial index), reprojecting only what is needed
        reproject = gdf.crs is not None and src.crs is not None and gdf.crs != src.crs
        qbounds = transform_bounds(src.crs, gdf.crs, *src.bounds) if reproject else tuple(src.bounds)
        hits = gdf.sindex.query(box(*qbounds), predicate="intersects")
        if len(hits) == 0:
            logger.info(f"No building in raster: {raster_path}.")
            return str(raster_path), 0, perf_counter() - t0
        sub = gdf.iloc[np.sort(hits)]

        # 2) drop the buildings that already have an image (resume), then OMBB + window
        done = _existing_ids(images_dir, extension)
        if reproject:
            sub = sub.to_crs(src.crs)
        jobs, n_skipped = _build_jobs(sub, id_field, src, done)
        if not jobs:
            logger.info(f"{raster_path.name}: nothing left to do ({n_skipped} buildings already done).")
            return str(raster_path), 0, perf_counter() - t0
        if n_skipped:
            logger.info(f"{raster_path.name}: resuming, {n_skipped} buildings already done, "
                        f"{len(jobs)} remaining.")

        # 3) read the raster strip by strip, crop all remaining buildings from memory.
        #    The per-raster CSV is appended to as we go, so it survives a crash.
        repartition_dir.mkdir(parents=True, exist_ok=True)
        csv_path = repartition_dir / raster_path.name.replace(".jp2", ".csv")
        new_csv = not csv_path.exists() or csv_path.stat().st_size == 0
        consecutive_errors = 0
        with open(csv_path, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            if new_csv:  # same columns as the original (no suffix any more: both are identical)
                writer.writerow(["Suffixed file name", "File name"])

            for group in _group_jobs(jobs, band_rows):
                g_r0, g_r1 = min(j.r0 for j in group), max(j.r1 for j in group)
                g_c0, g_c1 = min(j.c0 for j in group), max(j.c1 for j in group)
                data = src.read(indexes, window=Window(g_c0, g_r0, g_c1 - g_c0, g_r1 - g_r0))

                for j in group:
                    filename = f"{j.building_id}{extension}"
                    path = images_dir / filename
                    if path.exists():  # written meanwhile by another process (building in 2 rasters)
                        continue
                    try:
                        h, w = j.r1 - j.r0, j.c1 - j.c0
                        crop = data[:, j.r0 - g_r0:j.r1 - g_r0, j.c0 - g_c0:j.c1 - g_c0]
                        t = window_transform(Window(j.c0, j.r0, w, h), src.transform)
                        inside = geometry_mask([j.geom], out_shape=(h, w), transform=t, invert=True)
                        crop = np.where(inside[None, :, :], crop, 0).astype(data.dtype)
                        _save_image_atomic(crop, path, image_format, jpeg_quality, t, src.crs)
                        writer.writerow([filename, filename])
                        fcsv.flush()
                        n_written += 1
                        consecutive_errors = 0
                    except Exception as e:  # noqa: BLE001
                        consecutive_errors += 1
                        logger.error(f"{raster_path.name}: failed on {filename}: {e}")
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            raise RuntimeError(
                                f"{consecutive_errors} consecutive failures on {raster_path.name}: "
                                f"probably a systematic problem (disk? permissions?)") from e
                del data

    elapsed = perf_counter() - t0
    if n_written:
        logger.info(f"{raster_path.name}: {n_written} buildings took {elapsed:1.0f} s "
                    f"({elapsed / n_written:1.3f} s/building)")
    return str(raster_path), n_written, elapsed


# --------------------------------------------------------------------------------------
# user-facing class (same constructor as the QGIS Splitter)
# --------------------------------------------------------------------------------------
class RasterioSplitter:
    """Drop-in replacement of `Splitter` that does not need QGIS."""

    def __init__(self, vector_layer_path, raster_layers_path, output_directory_path):
        vector_layer_path = Path(vector_layer_path).expanduser().resolve()
        raster_layers_path = Path(raster_layers_path).expanduser().resolve()
        output_directory_path = Path(output_directory_path).expanduser().resolve()

        for file_ext in ("shx", "prj", "dbf"):
            if not vector_layer_path.with_suffix(f".{file_ext}").exists():
                raise FileNotFoundError(f"File '{vector_layer_path.with_suffix('.' + file_ext)}' is missing.")

        raster_files = sorted(f for f in raster_layers_path.iterdir() if f.suffix.lower() in (".jp2", ".tiff"))
        if not raster_files:
            raise FileNotFoundError(f"No raster file found in {raster_layers_path}.")

        self._vector_layer_path = vector_layer_path
        self._raster_files = raster_files
        self._raster_layers_path = raster_layers_path
        self._output_path = output_directory_path
        self._output_images_path = output_directory_path / "images"
        self._output_repartition_path = output_directory_path / "repartition_by_rasters"
        self._output_images_path.mkdir(parents=True, exist_ok=True)
        self._output_repartition_path.mkdir(parents=True, exist_ok=True)

    vector_layer_path = property(lambda self: self._vector_layer_path)
    raster_layers_path = property(lambda self: self._raster_layers_path)

    def _determine_unprocessed_rasters(self):
        """`progress.txt` = rasters that were completely processed (skipped without opening)."""
        progress_file = self._output_path / "progress.txt"
        if progress_file.exists():
            already = {line.rstrip("\n") for line in progress_file.read_text().splitlines()}
        else:
            logger.info("No existing progress file found, creating one.")
            progress_file.write_text("")
            already = set()
        remaining = [r for r in self._raster_files if str(r) not in already]
        logger.info(f"{len(self._raster_files) - len(remaining)}/{len(self._raster_files)} "
                    f"rasters were fully processed before, {len(remaining)} remaining")
        return remaining, progress_file

    def _remove_stale_temp_files(self):
        """Temporary files left by a previous run that was killed while writing an image."""
        n = 0
        with os.scandir(self._output_images_path) as it:
            for e in it:
                if e.name.endswith(TMP_SUFFIX):
                    try:
                        os.unlink(e.path)
                        n += 1
                    except OSError:
                        pass
        if n:
            logger.info(f"Removed {n} unfinished temporary image file(s) from a previous run.")

    def split(self, threads_num=None, band_rows=4096, image_format="JPEG", extension=None,
              jpeg_quality=75):
        """
        Write `<ID><extension>` for every building intersecting the rasters, skipping the buildings
        that already have an image. Safe to stop (Ctrl-C, kill, crash) and relaunch: only the
        remaining buildings are processed.

        Parameters
        ----------
        threads_num : int
            Number of worker *processes* (one raster per process at a time). Default 1.
            Each process holds the vector layer + one strip of raster in RAM.
        band_rows : int or None
            Height (rows) of the raster strips decoded at once. Lower -> less RAM.
            None -> decode each raster entirely in one go (fastest, most RAM).
        image_format : {"GTiff", "JPEG"}
            "JPEG": lossy, uint8 only, first 3 bands (default: same as the original code).
            "GTiff": lossless LZW GeoTIFF, all bands, source dtype.
        extension : str or None
            File extension of the images (e.g. ".tif", ".jpg", or whatever your downstream code
            expects). Default: ".tif" for GTiff, ".jpg" for JPEG. It does not change the content.
        jpeg_quality : int
            JPEG quality (only used for image_format="JPEG"; GDAL default is 75).
        """
        if threads_num is None:
            threads_num = 1
        if not isinstance(threads_num, int) or threads_num < 1:
            raise TypeError("`threads_num` must be a positive integer.")
        if image_format not in ("GTiff", "JPEG"):
            raise ValueError("`image_format` must be 'GTiff' or 'JPEG'.")
        if extension is None:
            extension = ".tif" if image_format == "GTiff" else ".jpg"
        if not extension.startswith("."):
            extension = "." + extension
        if extension == TMP_SUFFIX:
            raise ValueError(f"`extension` cannot be '{TMP_SUFFIX}' (reserved for temporary files).")

        self._remove_stale_temp_files()
        rasters, progress_file = self._determine_unprocessed_rasters()
        args = (str(self._output_images_path), str(self._output_repartition_path),
                band_rows, jpeg_quality, image_format, extension)
        failed = []

        def _done(raster_str, n, elapsed):
            with open(progress_file, "a") as f:  # written by the parent only -> no lock needed
                f.write(raster_str + "\n")

        if threads_num == 1 or len(rasters) <= 1:
            _init_worker(str(self._vector_layer_path))
            for r in tqdm(rasters, desc="Rasters", unit="raster", ncols=100):
                try:
                    _done(*_process_raster(str(r), *args))
                except Exception as e:  # noqa: BLE001
                    logger.error(f"{r.name} failed: {e!r}")
                    failed.append(r)
        else:
            with ProcessPoolExecutor(max_workers=min(threads_num, len(rasters)),
                                     initializer=_init_worker,
                                     initargs=(str(self._vector_layer_path),)) as ex:
                futures = {ex.submit(_process_raster, str(r), *args): r for r in rasters}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Rasters",
                                unit="raster", ncols=100):
                    r = futures[fut]
                    try:
                        _done(*fut.result())
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"{r.name} failed: {e!r}")
                        failed.append(r)

        if SplittingRecap is not None:
            SplittingRecap(self._output_path.parent).summarize()
        if failed:
            raise RuntimeError(f"{len(failed)} raster(s) failed (relaunch to retry only what is missing): "
                               f"{[r.name for r in failed]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split rasters into one JPEG per building (rasterio version).")
    ap.add_argument("vector_layer", help="path to the *.shp file")
    ap.add_argument("raster_dir", help="directory with the *.jp2 / *.tiff files")
    ap.add_argument("output_dir")
    ap.add_argument("-j", "--processes", type=int, default=1)
    ap.add_argument("--band-rows", type=int, default=4096, help="0 = decode whole rasters")
    ap.add_argument("--format", choices=["GTiff", "JPEG"], default="JPEG", help="image content (default: JPEG, like the original)")
    ap.add_argument("--extension", default=None, help="file extension, e.g. .tif, .jpg (default depends on --format)")
    ap.add_argument("--quality", type=int, default=75, help="JPEG quality (JPEG only)")
    ns = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RasterioSplitter(ns.vector_layer, ns.raster_dir, ns.output_dir).split(
        threads_num=ns.processes, band_rows=ns.band_rows or None, image_format=ns.format,
        extension=ns.extension, jpeg_quality=ns.quality)
