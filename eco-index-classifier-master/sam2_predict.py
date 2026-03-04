import torch, pandas as pd, cv2, os, numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

THRESH = 0.01     # same as create_dataset.py argument

def load_model(weights):
    cfg  = "segment-anything-2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    base = "segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt"
    m    = build_sam2(cfg, base, device="cuda")
    m.load_state_dict(torch.load(weights))
    return SAM2AutomaticMaskGenerator(m)

def classify(generator, img_path):
    img = cv2.imread(img_path)[:, :, ::-1]      # RGB
    masks = generator.generate(img)             # list of dicts with "segmentation"
    if len(masks) == 0:
        return "no"

    # merge all masks, compute fraction
    h, w = img.shape[:2]
    union = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        union |= m["segmentation"].astype(np.uint8)
    frac = union.sum() / (h * w)

    return "yes" if frac > THRESH else "no"

if __name__ == "__main__":
    import argparse, random
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test_csv", default="dataset/test/pairs.csv")
    args = ap.parse_args()

    gen = load_model(args.weights)
    df  = pd.read_csv(args.test_csv).sample(frac=1.0, random_state=0)

    correct = 0
    for _, row in df.iterrows():
        pred = classify(gen, os.path.join("dataset/test", row["raw_path"]))
        true = "yes" if row["masked_fraction"] > THRESH else "no"
        correct += pred == true
        print(f"{row['raw_path']} → {pred} (truth {true})")
    print(f"Accuracy {correct/len(df):.3%}")
