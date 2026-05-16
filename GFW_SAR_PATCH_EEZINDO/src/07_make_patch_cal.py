import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import rasterio
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"ERROR: library '{exc.name}' belum terinstall. "
        "Install requirements.txt terlebih dahulu."
    ) from None

from utils import (
    METADATA_DIR,
    ensure_directories,
    parse_patch_name,
    parse_patch_polarization,
    remove_file_if_exists,
    save_csv,
    save_failures,
)


PATCH_CAL_COLUMNS = [
    "patch_name",
    "patch_file",
    "patch_cal_file",
    "polarization",
    "cal_status",
    "patch_cal_status",
    "processing_error",
    "error",
]


def validate_float32_tif(path):
    with rasterio.open(path) as src:
        if src.count != 1:
            raise RuntimeError("Patch_Cal bukan single-band")
        if src.dtypes[0] != "float32":
            raise RuntimeError(f"Patch_Cal dtype bukan float32: {src.dtypes[0]}")
        _ = src.read(1)


def process_patch(patch_file, output_dir):
    patch_file = Path(patch_file)
    patch_cal_file = Path(output_dir) / patch_file.name
    row = {
        "patch_name": parse_patch_name(patch_file),
        "patch_file": str(patch_file),
        "patch_cal_file": str(patch_cal_file),
        "polarization": parse_patch_polarization(patch_file),
        "cal_status": "failed",
        "patch_cal_status": "failed",
        "processing_error": "",
        "error": "",
    }

    try:
        with rasterio.open(patch_file) as src:
            data = src.read(1).astype("float32")
            profile = src.profile.copy()

        if data.size == 0:
            raise RuntimeError("patch kosong")

        finite_mask = np.isfinite(data)
        if not finite_mask.any():
            raise RuntimeError("patch tidak memiliki nilai finite")

        # Sentinel-1 RTC sudah siap pakai. Tahap ini hanya menstabilkan nilai
        # non-finite tanpa log transform, equalization, atau smoothing agresif.
        finite_values = data[finite_mask]
        fill_value = float(np.nanmedian(finite_values))
        calibrated = np.where(finite_mask, data, fill_value).astype("float32")

        profile.update(
            {
                "driver": "GTiff",
                "count": 1,
                "dtype": "float32",
                "compress": "deflate",
            }
        )
        patch_cal_file.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(patch_cal_file, "w", **profile) as dst:
            dst.write(calibrated, 1)

        validate_float32_tif(patch_cal_file)
        row["cal_status"] = "success"
        row["patch_cal_status"] = "success"
    except Exception as exc:
        remove_file_if_exists(patch_cal_file)
        row["processing_error"] = str(exc)
        row["error"] = str(exc)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", default="data/opensarship_like/Patch")
    parser.add_argument("--output_dir", default="data/opensarship_like/Patch_Cal")
    args = parser.parse_args()

    ensure_directories([args.output_dir])
    patch_files = sorted(Path(args.patch_dir).glob("*.tif"))
    rows = [process_patch(path, args.output_dir) for path in patch_files]
    manifest = pd.DataFrame(rows)
    if manifest.empty and len(manifest.columns) == 0:
        manifest = pd.DataFrame(columns=PATCH_CAL_COLUMNS)

    manifest_path = METADATA_DIR / "patch_cal_manifest.csv"
    save_csv(manifest, manifest_path)
    failure_path = save_failures(manifest, "patch_cal", "patch_cal_status")

    print(f"Patch_Cal manifest: {manifest_path}")
    print(f"Failures: {failure_path}")
    if not manifest.empty:
        print(manifest["patch_cal_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
