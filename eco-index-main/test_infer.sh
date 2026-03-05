.venv/bin/python scripts/inference.py \
    data/output/core/TTC_Verified_part2.kml/polygon_4_raw.png \
    models/final_epoch_1 \
    --output_path data/output/infer/TTC_VERIFIED.kml/polygon_0_raw.png \
    --mask_output_path debug_mask.png
