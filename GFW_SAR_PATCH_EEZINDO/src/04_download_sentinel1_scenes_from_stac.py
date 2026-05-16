import argparse
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

try:
    import planetary_computer
    from pystac_client import Client
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"ERROR: library '{exc.name}' belum terinstall. "
        "Install requirements.txt terlebih dahulu."
    ) from None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda value, **_: value

from utils import (
    METADATA_DIR,
    apply_limit,
    ensure_directories,
    ensure_parent,
    parse_timestamp_utc,
    save_csv,
    save_failures,
    safe_path_stem,
)


STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MANIFEST_COLUMNS = [
    "source_row_index",
    "original_scene_id",
    "stac_item_id",
    "timestamp",
    "lat",
    "lon",
    "category_name",
    "vv_asset_href",
    "vh_asset_href",
    "downloaded_vv_file",
    "downloaded_vh_file",
    "vv_file_exists",
    "vh_file_exists",
    "download_status",
    "processing_error",
]


def datetime_window(timestamp_value, hours):
    ts = parse_timestamp_utc(timestamp_value)
    if pd.isna(ts):
        raise ValueError(f"timestamp tidak valid: {timestamp_value}")
    start = ts - pd.Timedelta(hours=hours)
    end = ts + pd.Timedelta(hours=hours)
    return ts, f"{start.isoformat()}/{end.isoformat()}"


def item_datetime(item):
    dt = item.datetime
    if dt is not None:
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    raw = item.properties.get("datetime")
    return parse_timestamp_utc(raw)


def choose_best_item(items, target_timestamp):
    if not items:
        return None
    sortable = []
    for item in items:
        dt = item_datetime(item)
        if pd.isna(dt):
            delta = pd.Timedelta.max
        else:
            delta = abs(dt - target_timestamp)
        sortable.append((delta, item))
    sortable.sort(key=lambda value: value[0])
    return sortable[0][1]


def find_polarization_asset(item, polarization):
    pol = polarization.lower()
    for key, asset in item.assets.items():
        if key.lower() == pol:
            return key, asset
    for key, asset in item.assets.items():
        key_lower = key.lower()
        href_lower = str(asset.href).lower()
        if pol in key_lower or f"_{pol}" in href_lower or f"-{pol}" in href_lower:
            return key, asset
    return None, None


def clear_planetary_computer_token_cache():
    try:
        from planetary_computer.sas import TOKEN_CACHE
    except Exception:
        return
    TOKEN_CACHE.clear()


def download_file(href, output_file, retries=3, retry_wait_seconds=20):
    output_file = Path(output_file)
    ensure_parent(output_file)

    if output_file.exists() and output_file.stat().st_size > 0:
        return output_file

    tmp_file = output_file.with_suffix(output_file.suffix + ".part")
    if tmp_file.exists():
        tmp_file.unlink()

    attempts = max(1, int(retries))
    last_error = None
    for attempt in range(1, attempts + 1):
        signed_href = href() if callable(href) else href
        request = urllib.request.Request(
            signed_href,
            headers={"User-Agent": "GFW-SAR-PATCH-EEZINDO/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                with tmp_file.open("wb") as dst:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if tmp_file.exists():
                tmp_file.unlink()
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 403:
                clear_planetary_computer_token_cache()
            if attempt < attempts:
                time.sleep(max(0, float(retry_wait_seconds)))
                continue
            raise RuntimeError(f"download gagal setelah {attempts} percobaan: {exc}") from exc
        break

    if tmp_file.stat().st_size == 0:
        tmp_file.unlink()
        raise RuntimeError("download menghasilkan file kosong")

    os.replace(tmp_file, output_file)
    return output_file


def search_stac_item(client, collection, lat, lon, timestamp_value, time_window_hours):
    target_ts, datetime_query = datetime_window(timestamp_value, time_window_hours)
    search = client.search(
        collections=[collection],
        intersects={"type": "Point", "coordinates": [float(lon), float(lat)]},
        datetime=datetime_query,
        max_items=20,
    )
    items = list(search.items())
    return choose_best_item(items, target_ts)


def process_record(
    client,
    row,
    output_dir,
    collection,
    time_window_hours,
    download_retries,
    retry_wait_seconds,
):
    result = dict(row)
    result.setdefault("source_row_index", "")
    result.setdefault("original_scene_id", row.get("scene_id", ""))
    result.setdefault("category_name", row.get("category_name", "Unknown"))
    result.update(
        {
            "stac_item_id": "",
            "vv_asset_href": "",
            "vh_asset_href": "",
            "downloaded_vv_file": "",
            "downloaded_vh_file": "",
            "vv_file_exists": False,
            "vh_file_exists": False,
            "download_status": "failed",
            "processing_error": "",
        }
    )

    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
        item = search_stac_item(
            client,
            collection=collection,
            lat=lat,
            lon=lon,
            timestamp_value=row["timestamp"],
            time_window_hours=time_window_hours,
        )
        if item is None:
            raise RuntimeError("item STAC Sentinel-1 RTC tidak ditemukan")

        vv_key, vv_asset = find_polarization_asset(item, "vv")
        vh_key, vh_asset = find_polarization_asset(item, "vh")

        if vv_asset is None:
            raise RuntimeError("asset VV tidak tersedia pada item STAC")
        if vh_asset is None:
            raise RuntimeError("asset VH tidak tersedia pada item STAC")

        item_id = safe_path_stem(item.id, fallback="sentinel1_rtc")
        scene_dir = Path(output_dir) / item_id
        vv_file = scene_dir / f"{item_id}_vv.tif"
        vh_file = scene_dir / f"{item_id}_vh.tif"

        result.update(
            {
                "stac_item_id": item.id,
                "stac_collection": collection,
                "vv_asset_key": vv_key,
                "vh_asset_key": vh_key,
                "vv_asset_href": vv_asset.href,
                "vh_asset_href": vh_asset.href,
                "downloaded_vv_file": str(vv_file),
                "downloaded_vh_file": str(vh_file),
            }
        )

        download_file(
            lambda: planetary_computer.sign(vv_asset).href,
            vv_file,
            retries=download_retries,
            retry_wait_seconds=retry_wait_seconds,
        )
        result["vv_file_exists"] = vv_file.exists() and vv_file.stat().st_size > 0

        download_file(
            lambda: planetary_computer.sign(vh_asset).href,
            vh_file,
            retries=download_retries,
            retry_wait_seconds=retry_wait_seconds,
        )
        result["vh_file_exists"] = vh_file.exists() and vh_file.stat().st_size > 0

        result.update(
            {
                "download_status": "success",
                "processing_error": "",
            }
        )
    except Exception as exc:
        result["processing_error"] = str(exc)
    finally:
        for path_column, exists_column in [
            ("downloaded_vv_file", "vv_file_exists"),
            ("downloaded_vh_file", "vh_file_exists"),
        ]:
            path_value = result.get(path_column)
            if path_value:
                path = Path(path_value)
                result[exists_column] = path.exists() and path.stat().st_size > 0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_csv", default="data/selected_samples/selected_raw_records.csv")
    parser.add_argument("--output_dir", default="data/work/downloaded_scenes")
    parser.add_argument("--collection", default="sentinel-1-rtc")
    parser.add_argument("--time_window_hours", type=float, default=12)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_success", type=int, default=-1)
    parser.add_argument("--download_retries", type=int, default=3)
    parser.add_argument("--retry_wait_seconds", type=float, default=20)
    args = parser.parse_args()

    ensure_directories([args.output_dir])
    records = pd.read_csv(args.records_csv)
    records = apply_limit(records, args.limit)

    client = Client.open(STAC_API_URL)
    rows = []
    success_count = 0
    for row in tqdm(records.to_dict(orient="records"), desc="STAC download"):
        result = process_record(
            client=client,
            row=row,
            output_dir=args.output_dir,
            collection=args.collection,
            time_window_hours=args.time_window_hours,
            download_retries=args.download_retries,
            retry_wait_seconds=args.retry_wait_seconds,
        )
        rows.append(result)
        if str(result.get("download_status", "")).lower() == "success":
            success_count += 1
            if int(args.max_success) >= 0 and success_count >= int(args.max_success):
                break

    manifest = pd.DataFrame(rows)
    for column in MANIFEST_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = pd.NA

    manifest_path = METADATA_DIR / "download_manifest.csv"
    save_csv(manifest, manifest_path)
    failure_path = save_failures(manifest, "download", "download_status")

    print(f"Download manifest: {manifest_path}")
    print(f"Failures: {failure_path}")
    print(manifest["download_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
