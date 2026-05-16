import argparse
import shutil
from pathlib import Path

import pandas as pd

try:
    import rasterio
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"ERROR: library '{exc.name}' belum terinstall. "
        "Install requirements.txt terlebih dahulu."
    ) from None

from utils import METADATA_DIR, ensure_directories, safe_path_stem, save_csv, save_failures


PREPROCESS_COLUMNS = [
    "source_row_index",
    "original_scene_id",
    "stac_item_id",
    "timestamp",
    "lat",
    "lon",
    "category_name",
    "preprocessed_vv_file",
    "preprocessed_vh_file",
    "preprocess_status",
    "vv_valid",
    "vh_valid",
    "band_count",
    "width",
    "height",
    "dtype",
    "crs",
    "processing_error",
]


def validate_raster(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file raster tidak ditemukan: {path}")

    with rasterio.open(path) as src:
        if src.crs is None:
            raise RuntimeError("CRS raster tidak tersedia")
        if src.transform is None or src.transform.is_identity:
            raise RuntimeError("transform raster tidak valid atau identity")
        if src.width <= 0 or src.height <= 0:
            raise RuntimeError("ukuran raster tidak valid")
        if src.count < 1:
            raise RuntimeError("raster tidak memiliki band")
        if abs(src.transform.a) == 0 or abs(src.transform.e) == 0:
            raise RuntimeError("resolusi transform raster tidak valid")
        return {
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
        }


def copy_and_validate(src_file, dst_file):
    dst_file = Path(dst_file)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    return validate_raster(dst_file)


def process_row(row, output_dir):
    result = dict(row)
    result.update(
        {
            "preprocessed_vv_file": "",
            "preprocessed_vh_file": "",
            "preprocess_status": "failed",
            "vv_valid": False,
            "vh_valid": False,
            "band_count": pd.NA,
            "width": pd.NA,
            "height": pd.NA,
            "dtype": pd.NA,
            "crs": pd.NA,
            "processing_error": "",
        }
    )

    try:
        if str(row.get("download_status", "")).lower() != "success":
            raise RuntimeError(row.get("processing_error") or "download_status bukan success")

        item_id = safe_path_stem(row.get("stac_item_id"), fallback="sentinel1_rtc")
        scene_dir = Path(output_dir) / item_id
        vv_dst = scene_dir / f"{item_id}_vv.tif"
        vh_dst = scene_dir / f"{item_id}_vh.tif"

        vv_info = copy_and_validate(row["downloaded_vv_file"], vv_dst)
        vh_info = copy_and_validate(row["downloaded_vh_file"], vh_dst)

        result.update(
            {
                "preprocessed_vv_file": str(vv_dst),
                "preprocessed_vh_file": str(vh_dst),
                "preprocess_status": "success",
                "vv_valid": True,
                "vh_valid": True,
                "band_count": vv_info["band_count"],
                "width": vv_info["width"],
                "height": vv_info["height"],
                "dtype": vv_info["dtype"],
                "crs": vv_info["crs"],
                "vh_width": vh_info["width"],
                "vh_height": vh_info["height"],
                "vh_band_count": vh_info["band_count"],
                "vh_dtype": vh_info["dtype"],
                "vh_crs": vh_info["crs"],
            }
        )
    except Exception as exc:
        result["processing_error"] = str(exc)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_manifest",
        default="data/opensarship_like/metadata/download_manifest.csv",
    )
    parser.add_argument("--output_dir", default="data/work/preprocessed_scenes")
    args = parser.parse_args()

    ensure_directories([args.output_dir])
    manifest = pd.read_csv(args.download_manifest)

    rows = [process_row(row, args.output_dir) for row in manifest.to_dict(orient="records")]
    output = pd.DataFrame(rows)
    if output.empty and len(output.columns) == 0:
        output = pd.DataFrame(columns=PREPROCESS_COLUMNS)

    manifest_path = METADATA_DIR / "preprocess_manifest.csv"
    save_csv(output, manifest_path)
    failure_path = save_failures(output, "preprocess", "preprocess_status")

    print(f"Preprocess manifest: {manifest_path}")
    print(f"Failures: {failure_path}")
    print(output["preprocess_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
