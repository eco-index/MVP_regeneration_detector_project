import os
import argparse
import base64
import random
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from cost_utils import MODEL_INFO, calculate_cost

load_dotenv()


# --- Helper Function to Encode Images ---
def encode_image_to_base64(image_path):
    """Encodes a single image file to a Base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


# --- Main Classification Function ---
def run_few_shot_classification(args):
    """
    Constructs and runs a few-shot classification task, then calculates the cost.
    """
    print("--- Starting Few-Shot Classification ---")

    # 1. Initialize OpenAI Client
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print(
            "Please ensure your OPENAI_API_KEY environment variable is set correctly."
        )
        return

    # 2. Load and Prepare Data
    train_csv = os.path.join(args.dataset_dir, "train", "pairs.csv")
    test_csv = os.path.join(args.dataset_dir, "test", "pairs.csv")
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(
            f"Error: 'pairs.csv' must exist in both 'train' and 'test' folders under '{args.dataset_dir}'."
        )
        return

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if len(train_df) < args.num_shots:
        print(
            f"Error: Need at least {args.num_shots} training samples, but found {len(train_df)}."
        )
        return
    if len(test_df) < args.num_test:
        print(
            f"Error: Need at least {args.num_test} test samples, but found {len(test_df)}."
        )
        return

    # 3. Randomly select examples
    print(
        f"Sampling {args.num_shots} few-shot examples from training data and {args.num_test} test images..."
    )
    few_shot_examples = train_df.sample(n=args.num_shots, random_state=args.seed)
    test_examples = test_df.sample(n=args.num_test, random_state=args.seed)

    # 4. Construct the conversation messages
    system_message = {
        "role": "system",
        "content": (
            "You are an expert satellite image analyst specializing in New Zealand ecology. "
            "Your task is to determine if an image contains evidence of new plantings of New Zealand native plants. "
            "I will provide you with several examples. For each image, you must respond with only 'yes' or 'no'.\n\n"
            "- 'yes' means the image contains organized, regular patterns of young trees or shrubs that indicate a recent, man-made planting effort.\n"
            "- 'no' means the image shows mature, established forest, empty pasture, water, or other features that are not new native plantings.\n\n"
            "Do not add any other words, explanations, or punctuation to your response."
        ),
    }
    messages = [system_message]

    print("\n--- Building Few-Shot Prompt ---")
    for _, row in few_shot_examples.iterrows():
        image_path = os.path.join(args.dataset_dir, "train", row["raw_path"])
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            continue
        label = "yes" if row["masked_fraction"] > args.threshold else "no"
        print(f"  - Adding example: {os.path.basename(row['raw_path'])} -> '{label}'")
        messages.append(
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "text",
                    #     "text": "Does this image contain new plantings of New Zealand native plants?",
                    # },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        )
        messages.append({"role": "assistant", "content": label})

    print("\n--- Sending Requests to OpenAI API ---")
    print(f"Model: {args.model}")

    total_cost = 0.0
    total_predictions = 0
    total_correct = 0

    for _, test_example in test_examples.iterrows():
        test_image_path = os.path.join(
            args.dataset_dir, "test", test_example["raw_path"]
        )
        test_base64_image = encode_image_to_base64(test_image_path)
        if not test_base64_image:
            print(f"Failed to encode test image {test_example['raw_path']}. Skipping.")
            continue

        test_messages = messages + [
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "text",
                    #     "text": "Does this image contain new plantings of New Zealand native plants?",
                    # },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{test_base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        true_label = "yes" if test_example["masked_fraction"] > args.threshold else "no"

        print(
            f"\nTesting {os.path.basename(test_example['raw_path'])} (true label '{true_label}')"
        )

        try:
            response = client.chat.completions.create(
                model=args.model, messages=test_messages, max_tokens=1, temperature=1
            )

            prediction = response.choices[0].message.content.strip().lower()
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            cached_tokens = usage.prompt_tokens_details.cached_tokens

            cost = calculate_cost(
                prompt_tokens,
                completion_tokens,
                args.model,
                cached_prompt_tokens=cached_tokens,
            )

            total_cost += cost
            total_predictions += 1

            print(f"  Model Prediction: '{prediction}'")
            cleaned_prediction = "yes" if "yes" in prediction else "no"
            if cleaned_prediction == true_label:
                total_correct += 1
                print("  ✅ Correct!")
            else:
                print(f"  ❌ Incorrect. (Cleaned prediction: '{cleaned_prediction}')")

            print(f"  Prompt Tokens: {prompt_tokens}")
            print(f"  Completion Tokens: {completion_tokens}")
            print(f"  Cached Tokens: {cached_tokens}")
            print(f"  Cost for this request: ${cost:.8f}")
        except Exception as e:
            print(f"An error occurred while calling the OpenAI API: {e}")
            continue

    print("\n--- Aggregate Cost ---")
    print(f"Total cost: ${total_cost:.8f}")
    print(f"Total predictions made: {total_predictions}")
    print(f"Total correct predictions: {total_correct}")
    print(f"Accuracy: {total_correct / total_predictions * 100:.2f}%")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a few-shot classification test and calculate cost using specific GPT-4.1 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Directory containing 'train' and 'test' subfolders with pairs.csv files.",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        required=True,
        help="The number of example images (n) to include in the prompt.",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=1,
        help="Number of test images to evaluate in one run.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help='Masked fraction threshold to define a "yes" (positive) sample.',
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_INFO.keys(),
        help="The specific GPT-4.1 series model to use for classification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling to ensure reproducibility. If not set, results will be different each run.",
    )

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 100000)
        print(
            f"Using random seed: {args.seed}. Use --seed {args.seed} to reproduce this run."
        )

    run_few_shot_classification(args)
