.PHONY: .venv
.venv:
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m playwright install

clean:
	rm -rf .venv

kml: .venv
	.venv/bin/python3 scripts/clean_kml.py --input data/kml/raw/core/TTC_VERIFIED.kml --output data/kml/cleaned/core/TTC_VERIFIED.kml

clean-all-kml: .venv
	.venv/bin/python3 scripts/clean_all_kml.py

screenshot: .venv
	.venv/bin/python3 scripts/screenshot_google_earth.py --kml data/kml/cleaned/extra/block_2_tf_2022.kml --output data/output/extra/block_2_tf_2022.kml

screenshot-all-kml: .venv
	.venv/bin/python3 scripts/screenshot_all_kml.py

train: .venv
	.venv/bin/python scripts/finetune_sam.py \
		--dataset_path data/output \
		--model_name facebook/sam-vit-base \
		--num_epochs 1 \
		--batch_size 2 \
		--learning_rate 1e-5 \
		--image_size 256 \
		--output_dir models/final_epoch_1 \
		--save_best_metric iou \
		--device cuda
