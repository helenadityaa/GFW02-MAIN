import argparse
from pathlib import Path

import pandas as pd

from utils import (
    DEFAULT_INPUT_CSV,
    apply_limit,
    category_name_from_matched_category,
    require_columns,
    save_csv,
)


def prepare_records(input_csv, limit):
    df = pd.read_csv(input_csv)
    require_columns(df, ["lat", "lon", "timestamp"], str(input_csv))

    prepared = df.copy()
    prepared["source_row_index"] = prepared.index

    lat_valid = pd.to_numeric(prepared["lat"], errors="coerce").notna()
    lon_valid = pd.to_numeric(prepared["lon"], errors="coerce").notna()
    timestamp_valid = pd.to_datetime(prepared["timestamp"], errors="coerce", utc=True).notna()

    mask = lat_valid & lon_valid & timestamp_valid
    if "presence_score" in prepared.columns:
        mask = mask & prepared["presence_score"].notna()

    filtered = prepared.loc[mask].copy()

    if "matched_category" in filtered.columns:
        filtered["original_matched_category"] = filtered["matched_category"]
        filtered["category_name"] = filtered["matched_category"].apply(
            category_name_from_matched_category
        )
    else:
        filtered["original_matched_category"] = pd.NA
        filtered["category_name"] = "Unknown"

    if "scene_id" in filtered.columns:
        filtered["original_scene_id"] = filtered["scene_id"]
    else:
        filtered["original_scene_id"] = pd.NA

    filtered = apply_limit(filtered, limit)
    return df, filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument(
        "--output_csv",
        default="data/selected_samples/selected_raw_records.csv",
    )
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    original_df, selected_df = prepare_records(input_csv, args.limit)
    save_csv(selected_df, output_csv)

    print(f"Input rows: {len(original_df)}")
    print(f"Selected rows: {len(selected_df)}")
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
