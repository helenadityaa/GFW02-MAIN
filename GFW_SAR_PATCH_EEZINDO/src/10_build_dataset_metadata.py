import argparse
from pathlib import Path

import pandas as pd

from utils import METADATA_DIR, coerce_source_row_index, save_csv, save_json_records


PROCESS_COLUMNS = [
    "category_name",
    "original_matched_category",
    "original_scene_id",
    "stac_item_id",
    "patch_name",
    "patch_file",
    "patch_cal_file",
    "patch_uint8_file",
    "patch_rgb_file",
    "polarization",
    "center_x",
    "center_y",
    "patch_status",
    "patch_cal_status",
    "patch_uint8_status",
    "patch_rgb_status",
    "processing_error",
    "source_csv",
    "source_row_index",
]


def read_optional(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def prepare_key_columns(df, keys):
    df = df.copy()
    for key in keys:
        if key in df.columns:
            df[key] = df[key].astype(str)
    return df


def merge_manifest(base, manifest, keys, prefix):
    if manifest.empty:
        return base
    if any(key not in base.columns for key in keys):
        return base
    if any(key not in manifest.columns for key in keys):
        return base

    base = prepare_key_columns(base, keys)
    manifest = prepare_key_columns(manifest, keys)

    rename_map = {}
    selected_columns = list(keys)
    for column in manifest.columns:
        if column in keys:
            continue
        if column in base.columns:
            renamed = f"{prefix}_{column}"
            rename_map[column] = renamed
            selected_columns.append(column)
        else:
            selected_columns.append(column)

    selected = manifest[selected_columns].rename(columns=rename_map)
    return base.merge(selected, on=keys, how="left")


def merge_missing_record_columns(base, records):
    if records.empty or "source_row_index" not in base.columns or "source_row_index" not in records.columns:
        return base
    base = coerce_source_row_index(base)
    records = coerce_source_row_index(records)
    missing_columns = [
        column
        for column in records.columns
        if column not in base.columns and column != "source_row_index"
    ]
    if not missing_columns:
        return base
    return base.merge(records[["source_row_index"] + missing_columns], on="source_row_index", how="left")


def aggregate_processing_errors(df):
    error_columns = [
        column
        for column in df.columns
        if column == "processing_error"
        or column.endswith("_processing_error")
        or column.endswith("_error")
    ]
    if not error_columns:
        df["processing_error"] = ""
        return df

    def combine(row):
        values = []
        for column in error_columns:
            value = row.get(column)
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text and text.lower() != "nan" and text not in values:
                values.append(text)
        return " | ".join(values)

    combined = df.apply(combine, axis=1)
    if "processing_error" in df.columns:
        df["processing_error"] = df["processing_error"].where(
            df["processing_error"].notna() & (df["processing_error"].astype(str).str.strip() != ""),
            combined,
        )
    else:
        df["processing_error"] = combined
    return df


def build_metadata(args):
    records = read_optional(args.records_csv)
    records = coerce_source_row_index(records)
    crop = read_optional(args.crop_manifest)
    crop = coerce_source_row_index(crop)

    if not crop.empty:
        base = crop.copy()
    else:
        base = records.copy()
        if not base.empty:
            base["patch_status"] = pd.NA

    base = merge_missing_record_columns(base, records)

    download = coerce_source_row_index(read_optional(args.download_manifest))
    preprocess = coerce_source_row_index(read_optional(args.preprocess_manifest))
    patch_cal = read_optional(args.patch_cal_manifest)
    patch_uint8 = read_optional(args.patch_uint8_manifest)
    patch_rgb = read_optional(args.patch_rgb_manifest)

    base = merge_manifest(base, download, ["source_row_index"], "download")
    base = merge_manifest(base, preprocess, ["source_row_index"], "preprocess")
    base = merge_manifest(base, patch_cal, ["patch_file", "polarization"], "cal")
    base = merge_manifest(base, patch_uint8, ["patch_cal_file", "polarization"], "uint8")
    base = merge_manifest(base, patch_rgb, ["patch_uint8_file", "polarization"], "rgb")

    if "source_csv" not in base.columns:
        base["source_csv"] = str(Path(args.records_csv))

    for column in PROCESS_COLUMNS:
        if column not in base.columns:
            base[column] = pd.NA

    base = aggregate_processing_errors(base)
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_csv", default="data/selected_samples/selected_raw_records.csv")
    parser.add_argument(
        "--download_manifest",
        default="data/opensarship_like/metadata/download_manifest.csv",
    )
    parser.add_argument(
        "--preprocess_manifest",
        default="data/opensarship_like/metadata/preprocess_manifest.csv",
    )
    parser.add_argument(
        "--crop_manifest",
        default="data/opensarship_like/metadata/crop_manifest.csv",
    )
    parser.add_argument(
        "--patch_cal_manifest",
        default="data/opensarship_like/metadata/patch_cal_manifest.csv",
    )
    parser.add_argument(
        "--patch_uint8_manifest",
        default="data/opensarship_like/metadata/patch_uint8_manifest.csv",
    )
    parser.add_argument(
        "--patch_rgb_manifest",
        default="data/opensarship_like/metadata/patch_rgb_manifest.csv",
    )
    parser.add_argument("--output_dir", default=str(METADATA_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(args)
    csv_path = output_dir / "dataset_metadata.csv"
    json_path = output_dir / "dataset_metadata.json"
    save_csv(metadata, csv_path)
    save_json_records(metadata, json_path)

    print(f"Dataset metadata CSV: {csv_path}")
    print(f"Dataset metadata JSON: {json_path}")
    print(f"Rows: {len(metadata)}")


if __name__ == "__main__":
    main()
