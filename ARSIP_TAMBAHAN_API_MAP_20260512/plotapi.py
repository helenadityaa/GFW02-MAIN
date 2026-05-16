import requests
import geopandas as gpd
import pandas as pd
import os
import json
import re
from pathlib import Path
from datetime import date, timedelta

import numpy as np
from PIL import Image, ImageDraw

try:
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
except ModuleNotFoundError:
    rasterio = None
    Transformer = None
    Window = None

# --- KONFIGURASI ---
API_TOKEN = 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImtpZEtleSJ9.eyJkYXRhIjp7Im5hbWUiOiJGaXNoaW5nIFZlc3NlbCBJbmRvbmVzaWEiLCJ1c2VySWQiOjYxMTk5LCJhcHBsaWNhdGlvbk5hbWUiOiJGaXNoaW5nIFZlc3NlbCBJbmRvbmVzaWEiLCJpZCI6MTA1MTIsInR5cGUiOiJ1c2VyLWFwcGxpY2F0aW9uIn0sImlhdCI6MTc3NzI4OTUzOCwiZXhwIjoyMDkyNjQ5NTM4LCJhdWQiOiJnZnciLCJpc3MiOiJnZncifQ.bE8bCkhc9HZIqktF-xW7fTSeCho2xyNdJD7e_mZFXBDIOUVubbkFgixDd5APg0XWWbWEA2bp5340pRkP5o4g8VdDUc4aE60AbZBQvWE1A-fgM7aTI-sbD0T7TvJ2PWkay1rE7nEOGZVsb7fyl1SVRQac0987-bmwSzwtvflJH9j08HGVQ4i29cl2op6vWAy02D9TY-DKrugYw4RO4O2jWeTL0wjS_FjUAaY0GnvQDelcySLIl084GI9_sJd2ZFP56X7gbATpwhOJSB0xD0XxnwMbSJNmJWEevX_cgcJg-FWhPfNW6udfTEgFOpwlla7VlH_6utIpwiwmynAJZI1jEQHeFbbTS-Fh2xjAJhR_eseUDouAl_i3tceZRAYcv16BdoI46e5does-QwUz16r2isJbCI0ry80lnYv3Lj6JGXqFVsIY9_GgEJQH1EfJZTqNFbpopGJeT20nsn_FBZgK2owSfo5jYSvKMu0UGrzj4sKheuQ2u9TzStGKevrdEDwM'
API_TOKEN = os.getenv("GFW_API_TOKEN", API_TOKEN)

BASE_DIR = Path(__file__).resolve().parent

# Dataset ID untuk deteksi SAR publik di 4Wings API.
DATASET_ID = "public-global-sar-presence:latest"
BASE_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"

headers = {
    'Authorization': f'Bearer {API_TOKEN}',
    'Content-Type': 'application/json'
}

# Parameter Area (Contoh: Area sekitar Indonesia/Natuna)
# format: min_lon, min_lat, max_lon, max_lat
BBOX = (105, -5, 115, 5)
USE_LATEST_GFW_API_DATES = True
LATEST_API_LOOKBACK_DAYS = 30
LATEST_API_MAX_BACKTRACK_DAYS = 365
START_DATE = '2024-01-01'
END_DATE = '2024-02-05'
OUTPUT_GEOJSON_FILE = "data_sar_gfw.geojson"
OUTPUT_MAP_FILE = "map_sar_gfw.html"
LOCAL_SAR_RECORDS_FILE = (
    BASE_DIR / "output" / "GFW_SAR_EEZ_Indonesia_202602_202603.csv"
)
LOCAL_SAR_SCENE_DIR = (
    BASE_DIR
    / "GFW_SAR_PATCH_EEZINDO"
    / "data"
    / "work"
    / "downloaded_scenes"
)
SAR_PREVIEW_DIR = BASE_DIR / "sar_popup_images"
SAR_PREVIEW_SIZE = 192
OUTPUT_SAR_IMAGE_GEOJSON_FILE = "data_sar_gfw_with_images.geojson"
OUTPUT_SAR_IMAGE_MAP_FILE = "map_sar_gfw_with_images.html"
LATEST_SAR_IMAGE_LIMIT = 100

def build_gfw_params(start_date, end_date):
    return {
        'spatial-resolution': 'HIGH',
        'temporal-resolution': 'HOURLY',
        'spatial-aggregation': 'false',
        'datasets[0]': DATASET_ID,
        'date-range': f'{start_date},{end_date}',
        'format': 'JSON',
    }


def bbox_to_geojson(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox

    return {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
    }


def extract_detection_records(data):
    records = []

    for entry in data.get("entries", []):
        if isinstance(entry, dict):
            if "lat" in entry and "lon" in entry:
                records.append(entry)
                continue

            for dataset_records in entry.values():
                if isinstance(dataset_records, list):
                    records.extend(
                        record for record in dataset_records if isinstance(record, dict)
                    )
        elif isinstance(entry, list):
            records.extend(record for record in entry if isinstance(record, dict))

    return records


def iter_latest_date_ranges():
    end_day = date.today()
    searched_days = 0

    while searched_days < LATEST_API_MAX_BACKTRACK_DAYS:
        start_day = end_day - timedelta(days=LATEST_API_LOOKBACK_DAYS)
        yield start_day.isoformat(), end_day.isoformat()
        end_day = start_day
        searched_days += LATEST_API_LOOKBACK_DAYS


def request_gfw_records(start_date, end_date):
    payload = {"geojson": bbox_to_geojson(BBOX)}
    response = requests.post(
        BASE_URL,
        headers=headers,
        params=build_gfw_params(start_date, end_date),
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        print(f"Gagal! Error {response.status_code}")
        print(f"Pesan: {response.text}")
        return None

    try:
        data = response.json()
    except ValueError:
        print("Gagal membaca respons JSON dari GFW.")
        print(f"Pesan: {response.text[:1000]}")
        return None

    return extract_detection_records(data)


def get_gfw_records():
    if USE_LATEST_GFW_API_DATES:
        for start_date, end_date in iter_latest_date_ranges():
            print(f"Mencoba data GFW terbaru: {start_date} sampai {end_date}")
            records = request_gfw_records(start_date, end_date)
            if records is None:
                return [], start_date, end_date
            if records:
                return records, start_date, end_date

        print(
            "Tidak menemukan data GFW pada rentang otomatis "
            f"{LATEST_API_MAX_BACKTRACK_DAYS} hari terakhir."
        )
        return [], START_DATE, END_DATE

    records = request_gfw_records(START_DATE, END_DATE)
    return records or [], START_DATE, END_DATE


def create_leaflet_map(
    gdf,
    output_file,
    title="Layer SAR GFW",
    subtitle=None,
    layer_name="Deteksi SAR GFW",
    bbox=BBOX,
):
    minx, miny, maxx, maxy = gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    geojson_data = gdf.to_json(na="null", drop_id=True)
    bbox_data = json.dumps(bbox_to_geojson(bbox)) if bbox else "null"
    title_json = json.dumps(title)
    subtitle = subtitle or f"{START_DATE} sampai {END_DATE}"
    subtitle_json = json.dumps(subtitle)
    layer_name_json = json.dumps(layer_name)

    html = f"""<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html,
    body,
    #map {{
      height: 100%;
      margin: 0;
    }}

    body {{
      font-family: Arial, sans-serif;
    }}

    .info-panel {{
      background: rgba(255, 255, 255, 0.94);
      border-radius: 6px;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.18);
      line-height: 1.4;
      padding: 10px 12px;
    }}

    .info-panel strong {{
      display: block;
      font-size: 14px;
      margin-bottom: 4px;
    }}

    .popup-table {{
      border-collapse: collapse;
      font-size: 12px;
      min-width: 220px;
    }}

    .popup-table th {{
      color: #555;
      font-weight: 600;
      padding: 3px 8px 3px 0;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}

    .popup-table td {{
      padding: 3px 0;
      vertical-align: top;
    }}

    .sar-preview {{
      background: #111;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      display: block;
      height: auto;
      image-rendering: pixelated;
      margin-bottom: 8px;
      max-width: 280px;
      width: 280px;
    }}

    .sar-preview-caption {{
      color: #475569;
      font-size: 11px;
      margin: -2px 0 8px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const sarData = {geojson_data};
    const bboxData = {bbox_data};
    const totalDetections = {len(gdf)};
    const mapTitle = {title_json};
    const mapSubtitle = {subtitle_json};
    const sarLayerName = {layer_name_json};

    const map = L.map("map", {{
      preferCanvas: true,
      center: [{center_lat:.8f}, {center_lon:.8f}],
      zoom: 7
    }});

    const osm = L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }}).addTo(map);

    const satellite = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}",
      {{
        maxZoom: 19,
        attribution: "Tiles &copy; Esri"
      }}
    );

    function formatValue(value) {{
      if (value === null || value === undefined || value === "") {{
        return "-";
      }}
      return String(value);
    }}

    function escapeHtml(value) {{
      return formatValue(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function popupHtml(properties) {{
      const preferredKeys = [
        "scene_id",
        "date",
        "timestamp",
        "detections",
        "shipName",
        "vesselId",
        "vesselType",
        "mmsi",
        "imo",
        "callsign",
        "flag",
        "geartype",
        "matched_category",
        "length_m",
        "presence_score",
        "matching_score",
        "fishing_score",
        "entryTimestamp",
        "exitTimestamp",
        "lat",
        "lon"
      ];
      const keys = preferredKeys.filter((key) => key in properties);
      const image = properties.sar_preview
        ? `
          <img class="sar-preview" src="${{escapeHtml(properties.sar_preview)}}" alt="Preview SAR">
          <div class="sar-preview-caption">Preview SAR VV, crop sekitar koordinat kapal</div>
        `
        : "";
      const rows = keys.map((key) => (
        `<tr><th>${{escapeHtml(key)}}</th><td>${{escapeHtml(properties[key])}}</td></tr>`
      ));
      return `${{image}}<table class="popup-table">${{rows.join("")}}</table>`;
    }}

    const sarLayer = L.geoJSON(sarData, {{
      pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{
        radius: 4,
        color: "#0f766e",
        weight: 1,
        fillColor: "#14b8a6",
        fillOpacity: 0.68
      }}),
      onEachFeature: (feature, layer) => {{
        layer.bindPopup(popupHtml(feature.properties || {{}}));
      }}
    }}).addTo(map);

    const overlayLayers = {{}};
    overlayLayers[sarLayerName] = sarLayer;

    if (bboxData) {{
      const bboxLayer = L.geoJSON(bboxData, {{
        style: {{
          color: "#ef4444",
          fillColor: "#ef4444",
          fillOpacity: 0.05,
          weight: 2,
          dashArray: "6 5"
        }}
      }}).addTo(map);
      overlayLayers["Batas BBOX"] = bboxLayer;
    }}

    const bounds = sarLayer.getBounds();
    if (bounds.isValid()) {{
      map.fitBounds(bounds.pad(0.08));
    }}

    L.control.layers(
      {{
        "OpenStreetMap": osm,
        "Satellite": satellite
      }},
      overlayLayers,
      {{
        collapsed: false
      }}
    ).addTo(map);

    const info = L.control({{ position: "bottomleft" }});
    info.onAdd = function () {{
      const div = L.DomUtil.create("div", "info-panel");
      div.innerHTML = `
        <strong>${{escapeHtml(mapTitle)}}</strong>
        ${{totalDetections.toLocaleString("id-ID")}} titik deteksi<br>
        ${{escapeHtml(mapSubtitle)}}
      `;
      return div;
    }};
    info.addTo(map);
  </script>
</body>
</html>
"""

    Path(output_file).write_text(html, encoding="utf-8")


def sanitize_filename_part(value, fallback="item"):
    text = "" if value is None else str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._")
    return text or fallback


def scene_core_from_scene_id(scene_id):
    parts = str(scene_id or "").split("_")
    if len(parts) <= 1:
        return str(scene_id or "")
    return "_".join(parts[:-1])


def build_vv_raster_index(scene_dir):
    index = {}
    scene_dir = Path(scene_dir)

    if not scene_dir.exists():
        return index

    for path in scene_dir.rglob("*_vv.tif"):
        item_id = path.stem
        if item_id.endswith("_vv"):
            item_id = item_id[:-3]

        core = item_id[:-4] if item_id.endswith("_rtc") else item_id
        index[item_id] = path
        index[core] = path

    return index


def stretch_sar_to_uint8(data):
    data = np.asarray(data, dtype="float32")
    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)

    data[~np.isfinite(data)] = np.nan
    positive = data[np.isfinite(data) & (data > 0)]

    if positive.size:
        floor = max(float(np.nanpercentile(positive, 1)), 1e-8)
        work = 10 * np.log10(np.clip(data, floor, None))
    else:
        work = data

    valid = work[np.isfinite(work)]
    if valid.size == 0:
        return np.zeros(data.shape, dtype="uint8")

    low, high = np.nanpercentile(valid, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(valid))
        high = float(np.nanmax(valid))

    if high <= low:
        return np.zeros(data.shape, dtype="uint8")

    scaled = (work - low) / (high - low)
    scaled = np.clip(scaled, 0, 1) * 255
    return np.nan_to_num(scaled, nan=0).astype("uint8")


def create_sar_preview(vv_file, lat, lon, preview_file, patch_size=SAR_PREVIEW_SIZE):
    if rasterio is None or Transformer is None or Window is None:
        raise RuntimeError("rasterio, pyproj, dan pillow diperlukan untuk membuat preview SAR")

    preview_file = Path(preview_file)
    if preview_file.exists():
        return preview_file

    with rasterio.open(vv_file) as src:
        if src.crs is None:
            raise RuntimeError("CRS raster tidak tersedia")

        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(float(lon), float(lat))
        center_row, center_col = rasterio.transform.rowcol(src.transform, x, y)
        center_row = int(center_row)
        center_col = int(center_col)

        if (
            center_col < 0
            or center_row < 0
            or center_col >= src.width
            or center_row >= src.height
        ):
            raise RuntimeError("koordinat lat/lon berada di luar raster")

        width = min(patch_size, src.width)
        height = min(patch_size, src.height)
        half_w = width // 2
        half_h = height // 2
        col_off = max(0, min(center_col - half_w, src.width - width))
        row_off = max(0, min(center_row - half_h, src.height - height))
        window = Window(col_off=col_off, row_off=row_off, width=width, height=height)

        data = src.read(1, window=window, masked=True)

    image = Image.fromarray(stretch_sar_to_uint8(data), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    marker_x = int(center_col - col_off)
    marker_y = int(center_row - row_off)
    if 0 <= marker_x < image.width and 0 <= marker_y < image.height:
        marker_size = 9
        draw.line(
            [(marker_x - marker_size, marker_y), (marker_x + marker_size, marker_y)],
            fill=(255, 64, 64),
            width=1,
        )
        draw.line(
            [(marker_x, marker_y - marker_size), (marker_x, marker_y + marker_size)],
            fill=(255, 64, 64),
            width=1,
        )

    preview_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_file)
    return preview_file


def attach_sar_previews(records_df):
    vv_index = build_vv_raster_index(LOCAL_SAR_SCENE_DIR)
    rows = []

    if not vv_index:
        print(f"Tidak menemukan raster VV di: {LOCAL_SAR_SCENE_DIR}")
        return pd.DataFrame()

    for _, row in records_df.iterrows():
        scene_id = str(row.get("scene_id", ""))
        scene_core = scene_core_from_scene_id(scene_id)
        vv_file = vv_index.get(scene_core)

        if vv_file is None:
            continue

        source_row = row.get("source_row_index", len(rows))
        preview_name = (
            f"{sanitize_filename_part(source_row, 'row')}_"
            f"{sanitize_filename_part(scene_id)[:90]}_vv.png"
        )
        preview_path = SAR_PREVIEW_DIR / preview_name
        result = dict(row)

        try:
            create_sar_preview(vv_file, row["lat"], row["lon"], preview_path)
            result["sar_preview"] = preview_path.relative_to(BASE_DIR).as_posix()
            result["sar_raster"] = str(vv_file.relative_to(BASE_DIR))
            result["sar_preview_status"] = "success"
            result["sar_preview_error"] = ""
            rows.append(result)
        except Exception as exc:
            result["sar_preview"] = ""
            result["sar_raster"] = str(vv_file.relative_to(BASE_DIR))
            result["sar_preview_status"] = "failed"
            result["sar_preview_error"] = str(exc)

    return pd.DataFrame(rows)


def select_latest_records_with_local_raster(records_df):
    vv_index = build_vv_raster_index(LOCAL_SAR_SCENE_DIR)
    if not vv_index:
        print(f"Tidak menemukan raster VV di: {LOCAL_SAR_SCENE_DIR}")
        return pd.DataFrame()

    records_df = records_df.copy()
    records_df["scene_core"] = records_df["scene_id"].apply(scene_core_from_scene_id)
    records_df = records_df[records_df["scene_core"].isin(vv_index.keys())].copy()

    if records_df.empty:
        print("Tidak ada record CSV yang cocok dengan raster SAR lokal.")
        return records_df

    if "timestamp" in records_df.columns:
        records_df["_timestamp_sort"] = pd.to_datetime(
            records_df["timestamp"],
            errors="coerce",
            utc=True,
        )
        records_df = records_df.sort_values(
            ["_timestamp_sort", "scene_id"],
            ascending=[False, False],
            na_position="last",
        )
        records_df = records_df.drop(columns=["_timestamp_sort"])

    if LATEST_SAR_IMAGE_LIMIT:
        records_df = records_df.head(LATEST_SAR_IMAGE_LIMIT).copy()

    return records_df


def create_sar_image_popup_map():
    if not LOCAL_SAR_RECORDS_FILE.exists():
        print(f"File record SAR lokal tidak ditemukan: {LOCAL_SAR_RECORDS_FILE}")
        return

    records_df = pd.read_csv(LOCAL_SAR_RECORDS_FILE)
    required_columns = {"lat", "lon", "scene_id"}
    missing_columns = sorted(required_columns - set(records_df.columns))
    if missing_columns:
        print(
            "Tidak bisa membuat map popup gambar SAR. "
            f"Kolom hilang: {missing_columns}"
        )
        return

    records_df["lat"] = pd.to_numeric(records_df["lat"], errors="coerce")
    records_df["lon"] = pd.to_numeric(records_df["lon"], errors="coerce")
    records_df = records_df.dropna(subset=["lat", "lon"]).copy()

    if records_df.empty:
        print("Record SAR lokal kosong atau tidak memiliki koordinat valid.")
        return

    records_df = select_latest_records_with_local_raster(records_df)

    if records_df.empty:
        return

    print("Membuat preview gambar SAR untuk popup...")
    image_df = attach_sar_previews(records_df)

    if image_df.empty:
        print("Tidak ada preview SAR yang berhasil dibuat.")
        return

    image_gdf = gpd.GeoDataFrame(
        image_df,
        geometry=gpd.points_from_xy(image_df.lon, image_df.lat),
        crs="EPSG:4326",
    )
    Path(OUTPUT_SAR_IMAGE_GEOJSON_FILE).write_text(
        image_gdf.to_json(na="null", drop_id=True),
        encoding="utf-8",
    )
    create_leaflet_map(
        image_gdf,
        OUTPUT_SAR_IMAGE_MAP_FILE,
        title="Deteksi SAR Terbaru dengan Gambar",
        subtitle=(
            f"Preview SAR lokal terbaru dari {LOCAL_SAR_RECORDS_FILE.name}; "
            f"maksimum {LATEST_SAR_IMAGE_LIMIT} titik"
        ),
        layer_name="Deteksi + Preview SAR",
        bbox=None,
    )

    print(f"GeoJSON gambar SAR disimpan sebagai: {OUTPUT_SAR_IMAGE_GEOJSON_FILE}")
    print(f"Map dengan popup gambar SAR disimpan sebagai: {OUTPUT_SAR_IMAGE_MAP_FILE}")
    print(f"Total preview SAR: {len(image_gdf)} titik.")


def download_sar_layer():
    print("Menghubungi server GFW...")
    try:
        records, selected_start_date, selected_end_date = get_gfw_records()
    except requests.RequestException as exc:
        print(f"Gagal menghubungi GFW: {exc}")
        return

    if not records:
        print("Data kosong untuk area dan tanggal tersebut.")
        return

    # Konversi JSON ke DataFrame
    df = pd.DataFrame(records)

    if not {'lon', 'lat'}.issubset(df.columns):
        print("Respons GFW tidak memiliki kolom lat/lon.")
        print("Kolom yang tersedia:")
        for column in df.columns:
            print(f"  - {column}")
        return

    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df = df.dropna(subset=['lon', 'lat']).copy()

    if df.empty:
        print("Data diterima, tetapi tidak ada koordinat lat/lon yang valid.")
        return
    
    # Buat GeoDataFrame (titik koordinat)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )
    
    # SIMPAN SEBAGAI GEOJSON (Untuk Layer Map)
    gdf.to_file(OUTPUT_GEOJSON_FILE, driver='GeoJSON')
    create_leaflet_map(
        gdf,
        OUTPUT_MAP_FILE,
        subtitle=f"{selected_start_date} sampai {selected_end_date}",
    )
    
    print(f"Sukses! File layer peta disimpan sebagai: {OUTPUT_GEOJSON_FILE}")
    print(f"Map interaktif disimpan sebagai: {OUTPUT_MAP_FILE}")
    print(f"Rentang data GFW: {selected_start_date} sampai {selected_end_date}")
    print(f"Total deteksi: {len(gdf)} titik.")

if __name__ == "__main__":
    download_sar_layer()
    create_sar_image_popup_map()
