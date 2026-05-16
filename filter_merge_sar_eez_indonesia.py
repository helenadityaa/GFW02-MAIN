import os
import re
import tempfile
import zipfile
from pathlib import Path

try:
    import geopandas as gpd
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"ERROR: Library Python '{exc.name}' belum terinstall. "
        "Install dependency terlebih dahulu, misalnya: "
        "python -m pip install pandas geopandas shapely"
    ) from None

try:
    from shapely.validation import make_valid
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"ERROR: Library Python '{exc.name}' belum terinstall. "
        "Install dependency terlebih dahulu, misalnya: "
        "python -m pip install pandas geopandas shapely"
    ) from None
except ImportError:
    make_valid = None


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = BASE_DIR / "CSV"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "GFW_SAR_EEZ_Indonesia_202602_202603.csv"

SAR_CSV_PATTERN = "sar_vessel_detections*.csv"
GEOMETRY_FILENAME = "geometry.geojson"

MONTH_NAMES = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}


def find_single_zip(csv_dir):
    zip_files = sorted(csv_dir.glob("*.zip"))

    if not zip_files:
        raise FileNotFoundError(f"Tidak menemukan file ZIP di folder: {csv_dir}")

    if len(zip_files) > 1:
        names = "\n".join(f"- {path.name}" for path in zip_files)
        raise RuntimeError(
            "Ditemukan lebih dari satu file ZIP. Sisakan satu file ZIP polygon EEZ "
            f"di folder CSV, atau sesuaikan script.\n{names}"
        )

    return zip_files[0]


def find_sar_csv_files(csv_dir):
    csv_files = sorted(csv_dir.glob(SAR_CSV_PATTERN))

    if not csv_files:
        raise FileNotFoundError(
            f"Tidak menemukan file CSV SAR dengan pola {SAR_CSV_PATTERN} di: {csv_dir}"
        )

    return csv_files


def safe_extract_zip(zip_path, extract_dir):
    """Extract ZIP while blocking members that try to escape extract_dir."""
    extract_root = extract_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            member_path = (extract_dir / member.filename).resolve()
            try:
                member_path.relative_to(extract_root)
            except ValueError as exc:
                raise RuntimeError(f"Path tidak aman di dalam ZIP: {member.filename}") from exc

        zip_ref.extractall(extract_dir)


def find_geometry_geojson(extract_dir):
    geometry_files = sorted(extract_dir.rglob(GEOMETRY_FILENAME))

    if not geometry_files:
        raise FileNotFoundError(
            f"Tidak menemukan {GEOMETRY_FILENAME} secara rekursif di hasil ekstraksi ZIP."
        )

    if len(geometry_files) > 1:
        print("Peringatan: ditemukan lebih dari satu geometry.geojson. File pertama dipakai:")
        for path in geometry_files:
            print(f"  - {path}")

    return geometry_files[0]


def repair_geometry(geometry):
    if geometry is None or geometry.is_empty or geometry.is_valid:
        return geometry

    if make_valid is not None:
        return make_valid(geometry)

    return geometry.buffer(0)


def read_eez_geometry(geometry_path):
    eez_gdf = gpd.read_file(geometry_path)

    if eez_gdf.empty:
        raise RuntimeError(f"File geometry kosong: {geometry_path}")

    eez_gdf = eez_gdf[eez_gdf.geometry.notna()].copy()

    if eez_gdf.empty:
        raise RuntimeError(f"Tidak ada geometry valid di file: {geometry_path}")

    if eez_gdf.crs is None:
        print("Peringatan: CRS geometry.geojson tidak tersedia. Diasumsikan EPSG:4326.")
        eez_gdf = eez_gdf.set_crs(epsg=4326)
    else:
        eez_gdf = eez_gdf.to_crs(epsg=4326)

    eez_gdf["geometry"] = eez_gdf.geometry.apply(repair_geometry)
    eez_gdf = eez_gdf[eez_gdf.geometry.notna() & ~eez_gdf.geometry.is_empty].copy()

    if eez_gdf.empty:
        raise RuntimeError(f"Tidak ada geometry EEZ yang bisa dipakai dari: {geometry_path}")

    return eez_gdf[["geometry"]]


def parse_year_month_from_filename(csv_path):
    match = re.search(r"(20\d{2})(0[1-9]|1[0-2])", csv_path.stem)

    if not match:
        raise ValueError(
            "Tidak bisa mengambil tahun dan bulan dari nama file. "
            f"Nama file harus mengandung format YYYYMM, misalnya 202602: {csv_path.name}"
        )

    year = int(match.group(1))
    month = int(match.group(2))

    return year, month, MONTH_NAMES[month]


def validate_lat_lon_columns(df, csv_path):
    required_columns = {"lat", "lon"}
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        print(f"\nERROR: File {csv_path.name} tidak memiliki kolom wajib: {missing_columns}")
        print("Daftar kolom yang tersedia:")
        for column in df.columns:
            print(f"  - {column}")
        raise ValueError(f"Kolom lat/lon tidak lengkap pada file: {csv_path}")


def spatial_filter_sar_csv(csv_path, eez_gdf):
    year, month, month_name = parse_year_month_from_filename(csv_path)

    df = pd.read_csv(csv_path)
    original_columns = list(df.columns)
    validate_lat_lon_columns(df, csv_path)

    total_rows = len(df)
    lon_numeric = pd.to_numeric(df["lon"], errors="coerce")
    lat_numeric = pd.to_numeric(df["lat"], errors="coerce")
    valid_coordinate_mask = lon_numeric.notna() & lat_numeric.notna()
    invalid_coordinate_count = total_rows - int(valid_coordinate_mask.sum())

    if invalid_coordinate_count:
        print(
            f"Peringatan: {csv_path.name} memiliki {invalid_coordinate_count} baris "
            "dengan lat/lon tidak valid. Baris tersebut dilewati untuk filter spasial."
        )

    metadata_columns = ["year", "month", "month_name", "source_file"]

    if not valid_coordinate_mask.any():
        filtered_df = pd.DataFrame(columns=original_columns + metadata_columns)
        return filtered_df, total_rows, 0

    valid_df = df.loc[valid_coordinate_mask].copy()
    point_gdf = gpd.GeoDataFrame(
        valid_df,
        geometry=gpd.points_from_xy(
            lon_numeric.loc[valid_coordinate_mask],
            lat_numeric.loc[valid_coordinate_mask],
        ),
        crs="EPSG:4326",
    )

    try:
        joined_gdf = gpd.sjoin(point_gdf, eez_gdf, how="inner", predicate="intersects")
    except TypeError:
        # Kompatibilitas untuk GeoPandas lama yang masih memakai argumen "op".
        joined_gdf = gpd.sjoin(point_gdf, eez_gdf, how="inner", op="intersects")

    if not joined_gdf.empty:
        joined_gdf = joined_gdf.loc[~joined_gdf.index.duplicated(keep="first")]

    filtered_df = joined_gdf[original_columns].copy()
    filtered_df["year"] = year
    filtered_df["month"] = month
    filtered_df["month_name"] = month_name
    filtered_df["source_file"] = csv_path.name

    return filtered_df, total_rows, len(filtered_df)


def print_final_summary(merged_df):
    print("\nRingkasan gabungan")
    print(f"- Total baris gabungan: {len(merged_df)}")

    if "scene_id" in merged_df.columns:
        print(f"- Jumlah scene_id unik: {merged_df['scene_id'].nunique(dropna=True)}")

    if "timestamp" in merged_df.columns:
        timestamp_series = pd.to_datetime(merged_df["timestamp"], errors="coerce", utc=True)

        if timestamp_series.notna().any():
            print(
                "- Rentang waktu timestamp: "
                f"{timestamp_series.min().isoformat()} sampai {timestamp_series.max().isoformat()}"
            )
        else:
            print("- Kolom timestamp tersedia, tetapi tidak ada nilai timestamp yang bisa dibaca.")

    if "matched_category" in merged_df.columns:
        print("- Jumlah matched_category:")
        matched_counts = merged_df["matched_category"].value_counts(dropna=False)

        if matched_counts.empty:
            print("  Tidak ada data matched_category.")
        else:
            for category, count in matched_counts.items():
                label = "<NaN>" if pd.isna(category) else str(category)
                print(f"  {label}: {count}")


def main():
    if not CSV_DIR.exists():
        raise FileNotFoundError(f"Folder CSV tidak ditemukan: {CSV_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = find_single_zip(CSV_DIR)
    sar_csv_files = find_sar_csv_files(CSV_DIR)

    print(f"Folder kerja: {BASE_DIR}")
    print(f"Folder CSV: {CSV_DIR}")
    print(f"ZIP polygon EEZ: {zip_path.name}")
    print("CSV SAR yang akan diproses:")
    for csv_path in sar_csv_files:
        print(f"  - {csv_path.name}")

    filtered_frames = []

    with tempfile.TemporaryDirectory(prefix="gfw_eez_indonesia_") as temp_dir:
        extract_dir = Path(temp_dir)
        safe_extract_zip(zip_path, extract_dir)

        geometry_path = find_geometry_geojson(extract_dir)
        print(f"geometry.geojson: {geometry_path}")

        eez_gdf = read_eez_geometry(geometry_path)

        print("\nRingkasan per file")
        for csv_path in sar_csv_files:
            filtered_df, total_rows, filtered_rows = spatial_filter_sar_csv(csv_path, eez_gdf)
            filtered_frames.append(filtered_df)

            print(f"- {csv_path.name}")
            print(f"  Jumlah baris awal: {total_rows}")
            print(f"  Jumlah baris masuk EEZ Indonesia: {filtered_rows}")

    if filtered_frames:
        merged_df = pd.concat(filtered_frames, ignore_index=True, sort=False)
    else:
        merged_df = pd.DataFrame()

    merged_df.to_csv(OUTPUT_FILE, index=False)

    print_final_summary(merged_df)
    print(f"\nOutput disimpan ke: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
