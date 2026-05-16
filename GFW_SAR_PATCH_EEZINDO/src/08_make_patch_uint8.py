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


PATCH_UINT8_COLUMNS = [
    "patch_name",
    "patch_cal_file",
    "patch_uint8_file",
    "polarization",
    "uint8_status",
    "patch_uint8_status",
    "processing_error",
    "error",
]


def robust_uint8(data):
    data = data.astype("float32")
    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        raise RuntimeError("patch tidak memiliki nilai finite")

    valid = data[finite_mask]
    p2, p98 = np.percentile(valid, [2, 98])
    if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
        p2 = float(np.nanmin(valid))
        p98 = float(np.nanmax(valid))

    if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
        return np.zeros(data.shape, dtype="uint8")

    clipped = np.clip(data, p2, p98)
    scaled = (clipped - p2) / (p98 - p2)
    scaled = np.where(np.isfinite(scaled), scaled, 0)
    return np.clip(np.round(scaled * 255), 0, 255).astype("uint8")


def validate_uint8_tif(path):
    with rasterio.open(path) as src:
        if src.count != 1:
            raise RuntimeError("Patch_Uint8 bukan single-band")
        if src.dtypes[0] != "uint8":
            raise RuntimeError(f"Patch_Uint8 dtype bukan uint8: {src.dtypes[0]}")
        _ = src.read(1)


def process_patch(patch_cal_file, output_dir):
    patch_cal_file = Path(patch_cal_file)
    patch_uint8_file = Path(output_dir) / patch_cal_file.name
    row = {
        "patch_name": parse_patch_name(patch_cal_file),
        "patch_cal_file": str(patch_cal_file),
        "patch_uint8_file": str(patch_uint8_file),
        "polarization": parse_patch_polarization(patch_cal_file),
        "uint8_status": "failed",
        "patch_uint8_status": "failed",
        "processing_error": "",
        "error": "",
    }

    try:
        with rasterio.open(patch_cal_file) as src:
            data = src.read(1)
            profile = src.profile.copy()

        uint8_data = robust_uint8(data)
        profile.update(
            {
                "driver": "GTiff",
                "count": 1,
                "dtype": "uint8",
                "compress": "deflate",
                "nodata": None,
            }
        )
        patch_uint8_file.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(patch_uint8_file, "w", **profile) as dst:
            dst.write(uint8_data, 1)

        validate_uint8_tif(patch_uint8_file)
        row["uint8_status"] = "success"
        row["patch_uint8_status"] = "success"
    except Exception as exc:
        remove_file_if_exists(patch_uint8_file)
        row["processing_error"] = str(exc)
        row["error"] = str(exc)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_cal_dir", default="data/opensarship_like/Patch_Cal")
    parser.add_argument("--output_dir", default="data/opensarship_like/Patch_Uint8")
    args = parser.parse_args()

    ensure_directories([args.output_dir])
    patch_files = sorted(Path(args.patch_cal_dir).glob("*.tif"))
    rows = [process_patch(path, args.output_dir) for path in patch_files]
    manifest = pd.DataFrame(rows)
    if manifest.empty and len(manifest.columns) == 0:
        manifest = pd.DataFrame(columns=PATCH_UINT8_COLUMNS)

    manifest_path = METADATA_DIR / "patch_uint8_manifest.csv"
    save_csv(manifest, manifest_path)
    failure_path = save_failures(manifest, "patch_uint8", "patch_uint8_status")

    print(f"Patch_Uint8 manifest: {manifest_path}")
    print(f"Failures: {failure_path}")
    if not manifest.empty:
        print(manifest["patch_uint8_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
