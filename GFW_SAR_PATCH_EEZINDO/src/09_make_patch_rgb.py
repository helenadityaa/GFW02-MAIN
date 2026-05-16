import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

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


PATCH_RGB_COLUMNS = [
    "patch_name",
    "patch_uint8_file",
    "patch_rgb_file",
    "polarization",
    "rgb_status",
    "patch_rgb_status",
    "rgb_style",
    "processing_error",
    "error",
]


def to_rgb(data, style):
    data = data.astype("uint8")
    if style == "grayscale":
        return np.dstack([data, data, data])

    if style == "opensarship":
        red = data
        green = np.clip(np.sqrt(data.astype("float32") / 255.0) * 255.0, 0, 255).astype("uint8")
        blue = np.clip(255 - (data.astype("uint16") // 2), 0, 255).astype("uint8")
        return np.dstack([red, green, blue])

    raise ValueError(f"style tidak dikenal: {style}")


def validate_png_rgb(path):
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise RuntimeError(f"PNG bukan RGB: {image.mode}")


def process_patch(patch_uint8_file, output_dir, style):
    patch_uint8_file = Path(patch_uint8_file)
    patch_rgb_file = Path(output_dir) / f"{patch_uint8_file.stem}.png"
    row = {
        "patch_name": parse_patch_name(patch_uint8_file),
        "patch_uint8_file": str(patch_uint8_file),
        "patch_rgb_file": str(patch_rgb_file),
        "polarization": parse_patch_polarization(patch_uint8_file),
        "rgb_status": "failed",
        "patch_rgb_status": "failed",
        "rgb_style": style,
        "processing_error": "",
        "error": "",
    }

    try:
        with rasterio.open(patch_uint8_file) as src:
            if src.count != 1:
                raise RuntimeError("Patch_Uint8 bukan single-band")
            data = src.read(1)
        if data.dtype != np.uint8:
            data = data.astype("uint8")

        rgb = to_rgb(data, style)
        patch_rgb_file.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(patch_rgb_file)
        validate_png_rgb(patch_rgb_file)

        row["rgb_status"] = "success"
        row["patch_rgb_status"] = "success"
    except Exception as exc:
        remove_file_if_exists(patch_rgb_file)
        row["processing_error"] = str(exc)
        row["error"] = str(exc)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_uint8_dir", default="data/opensarship_like/Patch_Uint8")
    parser.add_argument("--output_dir", default="data/opensarship_like/Patch_RGB")
    parser.add_argument("--style", choices=["grayscale", "opensarship"], default="grayscale")
    args = parser.parse_args()

    ensure_directories([args.output_dir])
    patch_files = sorted(Path(args.patch_uint8_dir).glob("*.tif"))
    rows = [process_patch(path, args.output_dir, args.style) for path in patch_files]
    manifest = pd.DataFrame(rows)
    if manifest.empty and len(manifest.columns) == 0:
        manifest = pd.DataFrame(columns=PATCH_RGB_COLUMNS)

    manifest_path = METADATA_DIR / "patch_rgb_manifest.csv"
    save_csv(manifest, manifest_path)
    failure_path = save_failures(manifest, "patch_rgb", "patch_rgb_status")

    print(f"Patch_RGB manifest: {manifest_path}")
    print(f"Failures: {failure_path}")
    if not manifest.empty:
        print(manifest["patch_rgb_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
