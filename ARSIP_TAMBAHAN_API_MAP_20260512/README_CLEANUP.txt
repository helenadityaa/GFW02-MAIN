Folder ini berisi file tambahan visualisasi API/map yang dipindahkan dari root workspace pada 2026-05-12.

File ini dipisahkan supaya workspace utama tidak rancu dengan pipeline utama GFW_SAR_PATCH_EEZINDO.

Yang dipindahkan:
- plotapi.py
- data_sar_gfw.geojson
- data_sar_gfw_with_images.geojson
- map_sar_gfw.html
- map_sar_gfw_with_images.html
- sar_popup_images/
- __pycache__/plotapi.cpython-313.pyc.2458889146000

Yang sengaja tidak disentuh:
- CSV/
- output/
- filter_merge_sar_eez_indonesia.py
- GFW_SAR_PATCH_EEZINDO/

Pipeline download scene, preprocess, crop patch, Patch_Cal, Patch_Uint8, Patch_RGB, metadata, dan XML tetap berada di GFW_SAR_PATCH_EEZINDO.
