.venv/bin/python scripts/finetune_sam.py \
    --dataset_path data/output \
    --model_name facebook/sam-vit-base \
    --num_epochs 5 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --image_size 256 \
    --output_dir models/sam_finetune_5_epoch \
    --save_best_metric iou \
    --device cuda
