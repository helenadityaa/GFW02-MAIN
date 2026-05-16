import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"ERROR: library '{exc.name}' belum terinstall. "
        "Install requirements.txt terlebih dahulu."
    ) from None

from utils import (
    METADATA_DIR,
    apply_limit,
    ensure_directories,
    parse_patch_name,
    remove_file_if_exists,
    save_csv,
    save_failures,
    sanitize_filename_part,
)


CROP_COLUMNS = [
    "source_row_index",
    "original_scene_id",
    "stac_item_id",
    "timestamp",
    "lat",
    "lon",
    "category_name",
    "patch_name",
    "patch_file",
    "polarization",
    "center_x",
    "center_y",
    "patch_status",
    "processing_error",
]


def row_col_from_lat_lon(src, lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = transformer.transform(float(lon), float(lat))
    row, col = rasterio.transform.rowcol(src.transform, x, y)
    return int(row), int(col)


def read_chip(src_file, lat, lon, patch_size):
    src_file = Path(src_file)
    with rasterio.open(src_file) as src:
        if src.crs is None:
            raise RuntimeError("CRS raster tidak tersedia")
        if src.transform is None or src.transform.is_identity:
            raise RuntimeError("transform raster tidak valid atau identity")
        if src.width <= 0 or src.height <= 0 or src.count < 1:
            raise RuntimeError("ukuran raster atau band tidak valid")

        center_y, center_x = row_col_from_lat_lon(src, lat, lon)
        if center_x < 0 or center_y < 0 or center_x >= src.width or center_y >= src.height:
            raise RuntimeError("koordinat lat/lon berada di luar raster")

        half = patch_size // 2
        col_off = center_x - half
        row_off = center_y - half
        if col_off < 0 or row_off < 0:
            raise RuntimeError("patch keluar batas raster di sisi atas/kiri")
        if col_off + patch_size > src.width or row_off + patch_size > src.height:
            raise RuntimeError("patch keluar batas raster di sisi bawah/kanan")

        window = Window(col_off=col_off, row_off=row_off, width=patch_size, height=patch_size)
        data = src.read(1, window=window)
        if data.shape != (patch_size, patch_size):
            raise RuntimeError(f"ukuran patch tidak sesuai: {data.shape}")

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": patch_size,
                "width": patch_size,
                "count": 1,
                "transform": src.window_transform(window),
            }
        )
        return data, profile, center_x, center_y


def validate_written_patch(path, patch_size):
    with rasterio.open(path) as src:
        if src.count != 1:
            raise RuntimeError("patch bukan single-band")
        if src.width != patch_size or src.height != patch_size:
            raise RuntimeError("ukuran patch tertulis tidak sesuai")
        _ = src.read(1)


def write_patch(path, data, profile, patch_size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update({"count": 1})
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    validate_written_patch(path, patch_size)


def build_base_name(category_name, center_x, center_y, used_names, output_patch_dir):
    category = sanitize_filename_part(category_name, fallback="Unknown")
    raw_base = f"{category}_x{center_x}_y{center_y}"
    output_patch_dir = Path(output_patch_dir)

    while True:
        used_names[raw_base] += 1
        if used_names[raw_base] == 1:
            base_name = raw_base
        else:
            base_name = f"{raw_base}_dup{used_names[raw_base] - 1:03d}"

        vv_file = output_patch_dir / f"{base_name}_vv.tif"
        vh_file = output_patch_dir / f"{base_name}_vh.tif"
        if not vv_file.exists() and not vh_file.exists():
            return base_name


def merge_records_with_preprocess(records, preprocess):
    records = records.copy()
    preprocess = preprocess.copy()
    records["source_row_index"] = records["source_row_index"].astype(str)
    preprocess["source_row_index"] = preprocess["source_row_index"].astype(str)

    merged = preprocess.merge(
        records,
        on="source_row_index",
        how="left",
        suffixes=("", "_record"),
    )

    for column in records.columns:
        duplicate = f"{column}_record"
        if duplicate in merged.columns:
            if column not in merged.columns:
                merged[column] = merged[duplicate]
            else:
                merged[column] = merged[column].where(merged[column].notna(), merged[duplicate])
            merged = merged.drop(columns=[duplicate])

    return merged


def success_row(row, patch_name, patch_file, polarization, center_x, center_y):
    result = dict(row)
    result.update(
        {
            "patch_name": patch_name,
            "patch_file": str(patch_file),
            "polarization": polarization,
            "center_x": center_x,
            "center_y": center_y,
            "patch_status": "success",
            "processing_error": "",
        }
    )
    return result


def failed_row(row, error):
    result = dict(row)
    result.update(
        {
            "patch_name": "",
            "patch_file": "",
            "polarization": "",
            "center_x": pd.NA,
            "center_y": pd.NA,
            "patch_status": "failed",
            "processing_error": str(error),
        }
    )
    return result


def process_row(row, output_patch_dir, patch_size, used_names):
    written_files = []
    try:
        if str(row.get("preprocess_status", "")).lower() != "success":
            raise RuntimeError(row.get("processing_error") or "preprocess_status bukan success")

        lat = row["lat"]
        lon = row["lon"]
        vv_data, vv_profile, vv_x, vv_y = read_chip(
            row["preprocessed_vv_file"], lat, lon, patch_size
        )
        vh_data, vh_profile, vh_x, vh_y = read_chip(
            row["preprocessed_vh_file"], lat, lon, patch_size
        )

        base_name = build_base_name(
            row.get("category_name", "Unknown"),
            vv_x,
            vv_y,
            used_names,
            output_patch_dir,
        )
        vv_file = Path(output_patch_dir) / f"{base_name}_vv.tif"
        vh_file = Path(output_patch_dir) / f"{base_name}_vh.tif"

        write_patch(vv_file, vv_data, vv_profile, patch_size)
        written_files.append(vv_file)
        write_patch(vh_file, vh_data, vh_profile, patch_size)
        written_files.append(vh_file)

        return [
            success_row(row, parse_patch_name(vv_file), vv_file, "vv", vv_x, vv_y),
            success_row(row, parse_patch_name(vh_file), vh_file, "vh", vh_x, vh_y),
        ]
    except Exception as exc:
        for path in written_files:
            remove_file_if_exists(path)
        return [failed_row(row, exc)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_csv", default="data/selected_samples/selected_raw_records.csv")
    parser.add_argument(
        "--preprocess_manifest",
        default="data/opensarship_like/metadata/preprocess_manifest.csv",
    )
    parser.add_argument("--output_patch_dir", default="data/opensarship_like/Patch")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    if args.patch_size <= 0:
        raise ValueError("patch_size harus lebih besar dari 0")

    ensure_directories([args.output_patch_dir])
    records = pd.read_csv(args.records_csv)
    preprocess = pd.read_csv(args.preprocess_manifest)
    merged = merge_records_with_preprocess(records, preprocess)
    merged = merged[merged["preprocess_status"].astype(str).str.lower() == "success"].copy()
    merged = apply_limit(merged, args.limit)

    used_names = defaultdict(int)
    rows = []
    for row in merged.to_dict(orient="records"):
        rows.extend(process_row(row, args.output_patch_dir, args.patch_size, used_names))

    manifest = pd.DataFrame(rows)
    if manifest.empty and len(manifest.columns) == 0:
        manifest = pd.DataFrame(columns=CROP_COLUMNS)
    manifest_path = METADATA_DIR / "crop_manifest.csv"
    save_csv(manifest, manifest_path)
    failure_path = save_failures(manifest, "crop", "patch_status")

    print(f"Crop manifest: {manifest_path}")
    print(f"Failures: {failure_path}")
    if not manifest.empty:
        print(manifest["patch_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
