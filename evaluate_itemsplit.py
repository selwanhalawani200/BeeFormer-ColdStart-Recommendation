import os

# Set Torch as the backend for Keras
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import subprocess

from time import time

parser = argparse.ArgumentParser()

# General experiment settings
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--device", default=None, type=str, help="Limit device to run on")
parser.add_argument("--flag", default="none", type=str, help="Extra flag for experiment naming")

# Dataset settings
parser.add_argument("--dataset", default="-", type=str, help="Dataset to run on")

# Sentence transformer settings
parser.add_argument("--sbert", default="none", type=str, help="Sentence transformer model")
parser.add_argument("--max_seq_length", default=0, type=int, help="Maximum sequence length")
parser.add_argument("--prefix", default=None, type=str, help="Optional prefix for item descriptions")

# Image model settings
parser.add_argument("--image_model", default="none", type=str, help="Image model to test")

# Parse arguments
args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

# Limit visible GPU devices if needed
if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

import keras
import math
import numpy as np
import torch

from models import SparseKerasELSA
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import config
from _datasets.utils import *

##import images

# Automatically use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")


def main(args):

    # Create a folder to save experiment results
    folder = f"results/{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save experiment settings
    vargs = vars(args)
    vargs["cuda_or_cpu"] = DEVICE

    pd.Series(vargs).to_csv(f"{folder}/setup.csv")

    print(folder)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # Check if dataset exists in config
    if args.dataset not in config.keys():

        print("Unknown dataset. List of available datasets:\n")

        for x in config.keys():
            print(x)

        return

    print(f"Loading dataset {args.dataset}...")

    # Load dataset interactions
    dataset, params = config[args.dataset]
    dataset.load_interactions(**params)

    print("Preparing item-split evaluation...")

    # Create cold-start evaluation object
    csev = ColdStartEvaluation(dataset)

    # Print dataset statistics
    print("FINAL DATASET STATS")
    print("All interactions:", len(dataset.all_interactions))
    print("All users:", dataset.all_interactions.user_id.nunique())
    print("All items:", dataset.all_interactions.item_id.nunique())

    print("TRAIN STATS")
    print("Train interactions:", len(dataset.train_interactions))
    print("Train users:", dataset.train_interactions.user_id.nunique())
    print("Train items:", dataset.train_interactions.item_id.nunique())

    print("TEST STATS")
    print("Test interactions:", len(dataset.test_interactions))
    print("Test users:", dataset.test_interactions.user_id.nunique())
    print("Test items:", dataset.test_interactions.item_id.nunique())

    print("COLD START")
    print("Test cold-start items:", len(dataset.test_cold_start_items))
    print("Validation cold-start items:", len(dataset.val_cold_start_items))

    # Run sentence transformer embeddings
    if args.sbert != "none":

        print(f"Initializing {args.sbert} sentence transformer...")

        sbert = SentenceTransformer(
            args.sbert,
            device=DEVICE,
            trust_remote_code=True
        )

        # Set max sequence length if provided
        if args.max_seq_length > 0:
            sbert.max_seq_length = args.max_seq_length

        print("Encoding item descriptions...")

        # Add optional prefix before encoding
        if args.prefix is not None:

            print("Adding prefix to all texts")

            texts = [args.prefix + x for x in dataset.texts]

            print(texts[:10])

            embs = sbert.encode(
                texts,
                show_progress_bar=True
            )

        else:

            embs = sbert.encode(
                dataset.texts,
                show_progress_bar=True
            )

    # Run image embeddings if image model is selected
    elif args.image_model != "none":

        image_model = images.ImageModel(
            args.image_model,
            device=DEVICE
        )

        tokenized_images_dict = images.read_images_into_dict(
            dataset.all_interactions.item_id.cat.categories,
            fn=image_model.tokenize,
            path=dataset.images_dir,
            suffix=dataset.images_suffix
        )

        tokenized_test_images = images.read_images_from_dict(
            dataset.all_interactions.item_id.cat.categories,
            tokenized_images_dict
        )

        embs = image_model.encode(tokenized_test_images)

    else:
        print("Model not specified.")

    # Create recommendation model
    model = SparseKerasELSA(
        len(dataset.all_interactions.item_id.cat.categories),
        embs.shape[1],
        dataset.all_interactions.item_id.cat.categories,
        device=DEVICE,
    )

    model.to(DEVICE)

    # Load item embeddings into the model
    model.set_weights([embs])

    print("Calculating predictions...")

    # Generate recommendation predictions
    df_preds = model.predict_df(
        csev.test_src,
        candidates_df=csev.candidates_df
    )

    print("Calculating metrics...")

    # Calculate evaluation metrics
    results = csev(df_preds)

    print(results)

    # Save evaluation results
    pd.Series(results).to_csv(f"{folder}/result.csv")

    print("Results file written")

    # Save runtime information
    pd.Series(0).to_csv(f"{folder}/timer.csv")

    print("Timer file written")


if __name__ == "__main__":

    # Start the experiment
    main(args)