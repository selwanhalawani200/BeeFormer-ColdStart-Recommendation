import os

# Set Torch as the backend for Keras
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import torch
import keras

from utils import *

from callbacks import evaluateWriter


def NMSE(x, y):
    # Normalize both embeddings before calculating the loss
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)

    # Compute the mean squared difference between embeddings
    return keras.ops.mean(keras.ops.square(x - y), axis=-1)


parser = argparse.ArgumentParser()

# General training settings
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--device", default=None, type=str)
parser.add_argument("--devices", default=None, type=str)
parser.add_argument("--flag", default="amazon_beauty_finetuned", type=str)
parser.add_argument("--validation", default="false", type=str)

# Learning rate and scheduler settings
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--scheduler", default="LinearWarmup", type=str)
parser.add_argument("--init_lr", default=0.0, type=float)
parser.add_argument("--warmup_lr", default=2e-5, type=float)
parser.add_argument("--target_lr", default=1e-6, type=float)

# Training duration settings
parser.add_argument("--warmup_epochs", default=1, type=int)
parser.add_argument("--decay_epochs", default=1, type=int)
parser.add_argument("--tuning_epochs", default=1, type=int)
parser.add_argument("--epochs", default=5, type=int)

# Dataset configuration
parser.add_argument("--dataset", default="amazon-beauty-custom", type=str)

# Evaluation mode settings
parser.add_argument("--use_cold_start", default="true", type=str)
parser.add_argument("--use_time_split", default="false", type=str)

# Optional text prefix
parser.add_argument("--prefix", default=None, type=str)

# SBERT embedding model
parser.add_argument(
    "--sbert",
    default="nomic-ai/nomic-embed-text-v1.5",
    type=str,
)

# Text preprocessing and speed optimization
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--preproces_html", default="false", type=str)

# Main batch and output settings
parser.add_argument("--max_output", default=200, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--top_k", default=0, type=int)
parser.add_argument("--sbert_batch_size", default=16, type=int)

# Model output name
parser.add_argument(
    "--model_name",
    default="amazon_beauty_nomic_1000_finetuned",
    type=str,
)

# Evaluation and checkpoint settings
parser.add_argument("--evaluate", default="true", type=str)
parser.add_argument("--evaluate_epoch", default="false", type=str)
parser.add_argument("--save_every_epoch", default="false", type=str)


# Parse arguments
args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

# Limit visible GPU devices if needed
if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import pandas as pd
import sentence_transformers
import subprocess
import time
import torch

from sentence_transformers import SentenceTransformer

from callbacks import evaluateWriter
from config import config
from dataloaders import beeformerDataset
from models import NMSEbeeformer, SparseKerasELSA
from schedules import LinearWarmup
from _datasets.utils import *


# Improve matrix multiplication performance
torch.set_float32_matmul_precision("medium")

# Automatically use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")


def load_data(args):
    # Select validation or test mode
    what = "val" if args.validation == "true" else "test"

    # Load dataset from config file
    if args.dataset in config.keys():
        dataset, params = config[args.dataset]
        dataset.load_interactions(**params)

        # Choose evaluation strategy
        if args.use_time_split == "true":
            evaluator = TimeBasedEvaluation(dataset, what=what)
        elif args.use_cold_start == "true":
            evaluator = ColdStartEvaluation(dataset, what=what)
        else:
            evaluator = Evaluation(dataset, what=what)

        # Load item texts
        items_d = dataset.items_texts
        items_d["asin"] = items_d.item_id

        # Use train or full train interactions
        if args.validation == "true":
            _train_interactions = dataset.train_interactions
        else:
            _train_interactions = dataset.full_train_interactions

    else:
        print("Unknown dataset.")
        return None, None, None, None

    return dataset, evaluator, _train_interactions, items_d


def load_text_model(args, items_d, dataset, _train_interactions):
    print("Preprocessing texts.")

    # Prepare all item texts for evaluation
    if args.evaluate == "true" or args.evaluate_epoch == "true":
        am_itemids = items_d.asin.to_numpy()
        cc = np.array(dataset.all_interactions.item_id.cat.categories)

        ccdf = pd.Series(cc).to_frame()
        ccdf.columns = ["item_id"]

        amdf = pd.Series(am_itemids).to_frame().reset_index()
        amdf.columns = ["idx", "item_id"]

        am_locator = pd.merge(
            how="inner",
            left=ccdf,
            right=amdf
        ).idx.to_numpy()

        am_texts_all = items_d._text_attributes.to_numpy()[am_locator]
        am_texts_all = [str(x) for x in am_texts_all]

    else:
        am_texts_all = None

    # Prepare training item texts
    am_itemids = items_d.asin.to_numpy()
    cc = np.array(_train_interactions.item_id.cat.categories)

    ccdf = pd.Series(cc).to_frame()
    ccdf.columns = ["item_id"]

    amdf = pd.Series(am_itemids).to_frame().reset_index()
    amdf.columns = ["idx", "item_id"]

    am_locator = pd.merge(
        how="inner",
        left=ccdf,
        right=amdf
    ).idx.to_numpy()

    am_texts = items_d._text_attributes.to_numpy()[am_locator]
    am_texts = [str(x) for x in am_texts]

    # Add optional prefix to texts
    if args.prefix is not None:
        print("Adding prefix to all texts")
        am_texts = [args.prefix + x for x in am_texts]

    print("Creating SBERT model")

    # Load embedding model
    sbert = SentenceTransformer(
        args.sbert,
        device=DEVICE,
        trust_remote_code=True,
    )

    # Set max token length
    if args.max_seq_length is not None:
        sbert.max_seq_length = args.max_seq_length

    # Tokenize input texts
    am_tokenized = sbert.tokenize(list(am_texts))

    # Keep only tensor values
    am_tokenized = {
        k: v for k, v in am_tokenized.items()
        if hasattr(v, "to")
    }

    if am_texts_all is None:
        am_texts_all = am_texts

    return am_texts_all, am_tokenized, sbert


def prepare_schedule(args, steps_per_epoch):

    # Cosine decay learning rate schedule
    if args.scheduler == "CosineDecay":

        schedule = keras.optimizers.schedules.CosineDecay(
            0.0,
            steps_per_epoch * (args.decay_epochs + args.warmup_epochs),
            alpha=0.0,
            name="CosineDecay",
            warmup_target=args.warmup_lr,
            warmup_steps=steps_per_epoch * args.warmup_epochs,
        )

        epochs = (
            args.warmup_epochs
            + args.decay_epochs
            + args.tuning_epochs
        )

    # Linear warmup schedule
    elif args.scheduler == "LinearWarmup":

        schedule = LinearWarmup(
            warmup_steps=steps_per_epoch * args.warmup_epochs,
            decay_steps=steps_per_epoch * args.decay_epochs,
            starting_lr=args.init_lr,
            warmup_lr=args.warmup_lr,
            final_lr=args.target_lr,
        )

        epochs = (
            args.warmup_epochs
            + args.decay_epochs
            + args.tuning_epochs
        )

    # Use constant learning rate
    else:
        schedule = args.lr
        epochs = args.epochs
        print("Using constant learning rate")

    return schedule, epochs


def main(args):

    # Create unique folder for saving results
    timestamp = pd.Timestamp("today").strftime("%Y-%m-%d_%H-%M-%S")

    folder = os.path.join(
        "results",
        f"{timestamp}_{9 * int(1e6) + np.random.randint(999999)}",
    )

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save experiment settings
    pd.Series(vars(args)).to_csv(f"{folder}/setup.csv")
    print(f"Saving results to {folder}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Seeds set to {args.seed}")

    # Load dataset
    dataset, evaluator, _train_interactions, items_d = load_data(args)

    if dataset is None:
        return None

    # Load text embedding model
    if args.sbert is not None:

        am_texts_all, am_tokenized, sbert = load_text_model(
            args,
            items_d,
            dataset,
            _train_interactions,
        )

    else:
        print("Please specify the SBERT model.")
        return None

    # Enable multi-GPU if available
    if args.devices is not None:

        devices_to_run = eval(args.devices)

        module_sbert = torch.nn.DataParallel(
            sbert,
            device_ids=devices_to_run,
            output_device=devices_to_run[0],
        )

    else:
        module_sbert = sbert

    print("Creating interaction matrix")

    # Convert interactions into sparse matrix
    X = get_sparse_matrix_from_dataframe(_train_interactions)

    print("Creating dataloader")

    # Create Beeformer dataloader
    datal = beeformerDataset(
        X,
        am_tokenized,
        DEVICE,
        shuffle=True,
        max_output=args.max_output,
        batch_size=args.batch_size,
    )

    steps_per_epoch = len(datal)

    print(sbert)

    # Create Beeformer model
    model = NMSEbeeformer(
        tokenized_sentences=am_tokenized,
        items_idx=_train_interactions.item_id.cat.categories,
        sbert=keras.layers.TorchModuleWrapper(module_sbert),
        device=DEVICE,
        top_k=args.top_k,
        sbert_batch_size=args.sbert_batch_size,
    )

    # Prepare scheduler
    schedule, epochs = prepare_schedule(args, steps_per_epoch)

    model.to(DEVICE)

    cbs = []
    eval_cb = None

    # Add evaluation callback if needed
    if (
        args.evaluate == "true"
        or args.evaluate_epoch == "true"
        or args.save_every_epoch == "true"
    ):

        eval_cb = evaluateWriter(
            items_idx=dataset.all_interactions.item_id.cat.categories,
            sbert=sbert,
            evaluator=evaluator,
            logdir=folder,
            DEVICE=DEVICE,
            texts=am_texts_all,
            sbert_name=args.model_name,
            evaluate_epoch=args.evaluate_epoch,
            save_every_epoch=args.save_every_epoch,
        )

        cbs.append(eval_cb)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Nadam(
            learning_rate=schedule
        ),
        loss=NMSE,
        metrics=[keras.metrics.CosineSimilarity()],
    )

    print("Building the model")

    # Run one train step to initialize the model
    model.train_step(datal[0])
    model.built = True

    model.summary()

    print("Starting training loop")

    train_time = time.time()

    print(f"Training for {epochs} epochs.")

    # Start training
    f = model.fit(
        datal,
        epochs=epochs,
        callbacks=cbs,
    )

    train_time = time.time() - train_time

    # Save fine-tuned model
    sbert.save(args.model_name)

    print(f"Model saved to {args.model_name}")

    final_results = {}

    # Final evaluation
    if args.evaluate == "true":

        embs = sbert.encode(
            am_texts_all,
            show_progress_bar=True
        )

        eval_model = SparseKerasELSA(
            len(dataset.all_interactions.item_id.cat.categories),
            embs.shape[1],
            dataset.all_interactions.item_id.cat.categories,
            device=DEVICE,
        )

        eval_model.to(DEVICE)
        eval_model.set_weights([embs])

        # Cold-start evaluation
        if args.use_cold_start == "true":

            df_preds = eval_model.predict_df(
                evaluator.test_src,
                candidates_df=(
                    evaluator.cold_start_candidates_df
                    if hasattr(
                        evaluator,
                        "cold_start_candidates_df"
                    )
                    else None
                ),
                k=1000,
            )

            # Remove already interacted items
            df_preds = df_preds[
                ~df_preds.set_index(
                    ["item_id", "user_id"]
                ).index.isin(
                    evaluator.test_src.set_index(
                        ["item_id", "user_id"]
                    ).index
                )
            ]

        else:
            df_preds = eval_model.predict_df(
                evaluator.test_src
            )

        # Calculate evaluation metrics
        final_results = evaluator(df_preds)

        print(final_results)

        pd.Series(final_results).to_csv(
            f"{folder}/result.csv"
        )

        print("Results file written")

    # Save training history
    history_df = pd.DataFrame(f.history)

    history_df["epoch"] = (
        np.arange(len(history_df)) + 1
    )

    history_df.to_csv(
        f"{folder}/history.csv",
        index=False
    )

    print("History file written")

    # Save evaluation history
    try:

        pd.concat(
            [
                pd.Series(x).to_frame().T
                for x in eval_cb.results_list
            ]
        ).to_csv(f"{folder}/results-history.csv")

    except:
        print("Evaluation callback not found")

    # Save training time
    pd.Series(train_time).to_csv(
        f"{folder}/timer.csv"
    )

    print("Timer file written")

    # Save GPU information
    try:

        out = subprocess.check_output(["nvidia-smi"])

        with open(
            os.path.join(
                folder,
                f"{args.dataset}_{args.flag}.log"
            ),
            "w",
        ) as f:

            f.write(out.decode("utf-8"))

    except:
        print("nvidia-smi unavailable")

    return {
        "folder": folder,
        "train_time_seconds": train_time,
        "results": final_results,
    }


if __name__ == "__main__":

    # List of experiments
    experiments = [

        {
            "model_label": "Nomic",
            "sbert": "nomic-ai/nomic-embed-text-v1.5",
            "model_name": "amazon_beauty_nomic_1000_finetuned",
            "epochs": 5,
            "batch_size": 24,
            "lr": 1e-5,
            "max_seq_length": 256,
            "args.max_output": 150,
            "args.sbert_batch_size": 24,
        },

    ]

    summary_results = []

    # Run all experiments
    for exp in experiments:

        print("\n" + "=" * 80)
        print(f"Starting fine-tuning for: {exp['model_label']}")
        print("=" * 80)

        args.dataset = "amazon-beauty-custom"
        args.sbert = exp["sbert"]
        args.model_name = exp["model_name"]

        args.epochs = exp["epochs"]
        args.batch_size = exp["batch_size"]
        args.lr = exp["lr"]
        args.max_seq_length = exp["max_seq_length"]

        args.max_output = exp["args.max_output"]
        args.sbert_batch_size = exp["args.sbert_batch_size"]

        args.scheduler = "none"

        args.evaluate = "true"
        args.evaluate_epoch = "false"
        args.save_every_epoch = "true"

        try:

            # Run the experiment
            run_info = main(args)

            row = {
                "Model": exp["model_label"],
                "Base Model": exp["sbert"],
                "Saved Name": exp["model_name"],
                "Epochs": exp["epochs"],
                "Batch Size": exp["batch_size"],
                "Learning Rate": exp["lr"],
                "Status": "Completed",
            }

            if run_info is not None:

                row["Results Folder"] = run_info["folder"]

                row["Train Time Seconds"] = (
                    run_info["train_time_seconds"]
                )

                if isinstance(run_info["results"], dict):
                    row.update(run_info["results"])

            summary_results.append(row)

        except Exception as e:

            # Save failed experiment information
            summary_results.append(
                {
                    "Model": exp["model_label"],
                    "Base Model": exp["sbert"],
                    "Saved Name": exp["model_name"],
                    "Epochs": exp["epochs"],
                    "Batch Size": exp["batch_size"],
                    "Learning Rate": exp["lr"],
                    "Status": f"Failed: {str(e)}",
                }
            )

            print(f"Error in {exp['model_label']}: {e}")

    print("\n" + "=" * 80)
    print("FINAL FINE-TUNING SUMMARY")
    print("=" * 80)

    # Create final summary table
    summary_df = pd.DataFrame(summary_results)

    print(summary_df)

    # Save summary table
    summary_df.to_csv(
        "final_finetuning_summary.csv",
        index=False
    )

    print("\nSummary table saved as final_finetuning_summary.csv")