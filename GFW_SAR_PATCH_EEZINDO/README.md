# GFW_SAR_PATCH_EEZINDO

Project ini membuat dataset patch SAR OpenSARShip-like dari CSV Global Fishing Watch SAR Vessel Detections yang sudah difilter ke wilayah Indonesian Exclusive Economic Zone.

Input awal project adalah CSV deteksi kapal GFW, bukan citra SAR. CSV hanya dipakai sebagai metadata lokasi kapal, terutama `lat`, `lon`, `timestamp`, `scene_id`, dan kategori. Citra atau scene raster diambil dari Sentinel-1 RTC melalui Microsoft Planetary Computer STAC berdasarkan lokasi dan waktu deteksi.

Output `Patch`, `Patch_Cal`, `Patch_Uint8`, dan `Patch_RGB` hanya dibuat jika scene Sentinel-1 RTC berhasil ditemukan, asset VV dan VH tersedia, file berhasil di-download, raster valid, dan koordinat kapal masuk ke dalam raster. Jika scene tidak ditemukan, raster tidak valid, atau crop gagal, pipeline menulis status `failed` di manifest dan failure CSV. Pipeline tidak membuat patch palsu untuk menutupi kegagalan.

Project ini bukan YOLO, bukan bounding box, dan bukan object detection. Pipeline ini tidak membuat `dataset.yaml`, label YOLO, bounding box, data dummy, atau patch buatan dari CSV.

XML yang dibuat adalah XML metadata dataset buatan pipeline, bukan XML original Sentinel-1.

## Penting

- Project ini membuat dataset patch SAR OpenSARShip-like dari GFW SAR EEZ Indonesia.
- Input awal adalah CSV deteksi kapal GFW, bukan citra SAR.
- Citra/scene diambil dari Sentinel-1 RTC STAC berdasarkan `lat`, `lon`, dan `timestamp`.
- Output `Patch`, `Patch_Cal`, `Patch_Uint8`, dan `Patch_RGB` hanya dibuat jika raster Sentinel-1 berhasil ditemukan dan valid.
- Project ini bukan YOLO, bukan bounding box, dan bukan object detection.
- XML yang dibuat adalah XML metadata dataset buatan pipeline, bukan XML original Sentinel-1.

## Struktur Project

```text
GFW_SAR_PATCH_EEZINDO/
  README.md
  requirements.txt
  run_instructions.txt
  src/
    utils.py
    01_audit_input_csv.py
    02_prepare_raw_records.py
    03_build_opensarship_structure.py
    04_download_sentinel1_scenes_from_stac.py
    05_preprocess_sentinel1_scenes.py
    06_crop_ship_chips.py
    07_make_patch_cal.py
    08_make_patch_uint8.py
    09_make_patch_rgb.py
    10_build_dataset_metadata.py
    11_build_ship_xml.py
    12_build_metedata_xml.py
  data/
    selected_samples/
    work/
      downloaded_scenes/
      preprocessed_scenes/
    opensarship_like/
      Patch/
      Patch_Cal/
      Patch_Uint8/
      Patch_RGB/
      XML/
      metadata/
  outputs/
    logs/
    metrics/
    figures/
```

## Pipeline

1. `01_audit_input_csv.py` membaca CSV input, mengecek kolom wajib, missing value, rentang waktu, `scene_id`, dan `matched_category`.
2. `02_prepare_raw_records.py` memilih record awal dengan `--limit`, memvalidasi `lat/lon/timestamp`, mempertahankan semua kolom asli, dan menambah kolom kategori proses.
3. `03_build_opensarship_structure.py` membuat struktur folder dataset.
4. `04_download_sentinel1_scenes_from_stac.py` mencari item `sentinel-1-rtc` dari Microsoft Planetary Computer STAC memakai `lat/lon` dan time window timestamp, lalu download VV dan VH jika tersedia.
5. `05_preprocess_sentinel1_scenes.py` memvalidasi raster dan menyalin raster RTC valid ke folder kerja preprocessing.
6. `06_crop_ship_chips.py` mengubah `lat/lon` ke pixel raster dengan CRS dan transform raster, lalu crop patch 128x128 untuk VV dan VH.
7. `07_make_patch_cal.py` membuat `Patch_Cal` float32 dengan stabilisasi ringan tanpa double log transform, histogram equalization agresif, atau smoothing.
8. `08_make_patch_uint8.py` membuat `Patch_Uint8` single-band `uint8` memakai robust percentile clipping p2-p98.
9. `09_make_patch_rgb.py` membuat preview PNG RGB. Default: grayscale `R=G=B`.
10. `10_build_dataset_metadata.py` menggabungkan semua manifest menjadi CSV dan JSON metadata dataset.
11. `11_build_ship_xml.py` membuat `Ship.xml` sebagai metadata patch kapal.
12. `12_build_metedata_xml.py` membuat `Metedata.xml` sebagai ringkasan dataset.

## Output Utama

```text
data/opensarship_like/Patch
data/opensarship_like/Patch_Cal
data/opensarship_like/Patch_Uint8
data/opensarship_like/Patch_RGB
data/opensarship_like/XML
data/opensarship_like/metadata
```

Manifest proses disimpan di `data/opensarship_like/metadata`. Failure CSV tiap tahap disimpan di `outputs/metrics/*_failures.csv`.

## Menjalankan

Install dependency:

```powershell
Set-Location "D:\FILE ALE\GFW02\GFW_SAR_PATCH_EEZINDO"
py -m pip install -r requirements.txt
```

Lalu jalankan command satu per satu di `run_instructions.txt`.

Tahap download STAC membutuhkan koneksi internet. Mulai dari `--limit` kecil, misalnya 10, karena scene Sentinel-1 RTC dapat berukuran besar.
