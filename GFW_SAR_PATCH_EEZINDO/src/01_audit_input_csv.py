import argparse
from pathlib import Path

import pandas as pd

from utils import DEFAULT_INPUT_CSV, require_columns, write_text


def build_audit_text(df, input_csv):
    lines = []
    lines.append("Input CSV Audit")
    lines.append("=" * 50)
    lines.append(f"Input CSV: {Path(input_csv).resolve()}")
    lines.append(f"Jumlah baris: {len(df)}")
    lines.append(f"Jumlah kolom: {len(df.columns)}")
    lines.append("")
    lines.append("Kolom:")
    for column in df.columns:
        lines.append(f"- {column}")

    lines.append("")
    lines.append("Missing value per kolom:")
    missing = df.isna().sum().sort_values(ascending=False)
    for column, count in missing.items():
        lines.append(f"- {column}: {count}")

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        lines.append("")
        lines.append("Rentang timestamp:")
        if timestamps.notna().any():
            lines.append(f"- Min: {timestamps.min().isoformat()}")
            lines.append(f"- Max: {timestamps.max().isoformat()}")
            lines.append(f"- Invalid/NaT: {timestamps.isna().sum()}")
        else:
            lines.append("- Tidak ada timestamp valid.")

    if "scene_id" in df.columns:
        lines.append("")
        lines.append(f"Jumlah scene_id unik: {df['scene_id'].nunique(dropna=True)}")

    if "matched_category" in df.columns:
        lines.append("")
        lines.append("Jumlah matched_category:")
        counts = df["matched_category"].value_counts(dropna=False)
        for category, count in counts.items():
            label = "<NaN>" if pd.isna(category) else str(category)
            lines.append(f"- {label}: {count}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output_dir", default="outputs/metrics")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    require_columns(df, ["lat", "lon", "timestamp"], str(input_csv))

    audit_text = build_audit_text(df, input_csv)
    output_file = output_dir / "input_csv_audit.txt"
    write_text(output_file, audit_text)

    print(audit_text)
    print(f"Audit disimpan ke: {output_file}")


if __name__ == "__main__":
    main()
