import os

INPUT_DATA_DIR = "plantings-validated-data-15-12-2025"
PROCESSED_DATA_DIR = "plantings-validated-processed-15-12-2025"

if __name__ == "__main__":
    examples = {}

    for file in os.listdir(INPUT_DATA_DIR):
        if not file.endswith(".png"):
            continue
        input_path = os.path.join(INPUT_DATA_DIR, file)
        is_raw = "_raw" in file
        example_name = file.split("_raw", 1)[0].split("_mask", 1)[0]
        examples.setdefault(example_name, {})["raw" if is_raw else "mask"] = input_path

    for i, (example_name, example) in enumerate(examples.items()):
        raw_input_path = example.get("raw")
        mask_input_path = example.get("mask")

        missing = False
        if not raw_input_path or not os.path.exists(raw_input_path):
            print(f"Warning: Missing raw file for example '{example_name}'. Skipping.")
            missing = True
        if not mask_input_path or not os.path.exists(mask_input_path):
            print(f"Warning: Missing mask file for example '{example_name}'. Skipping.")
            missing = True
        if missing:
            continue

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        raw_output_path = os.path.join(PROCESSED_DATA_DIR, f"polygon_{i}_raw.png")
        mask_output_path = os.path.join(PROCESSED_DATA_DIR, f"polygon_{i}_mask.png")

        with open(raw_input_path, "rb") as raw_infile, open(raw_output_path, "wb") as raw_outfile:
            raw_outfile.write(raw_infile.read())
        with open(mask_input_path, "rb") as mask_infile, open(mask_output_path, "wb") as mask_outfile:
            mask_outfile.write(mask_infile.read())
        print(f"Processed example '{example_name}' as 'polygon_{i}'.")
    print(f"Processed {len(examples)} examples into '{PROCESSED_DATA_DIR}'.")
