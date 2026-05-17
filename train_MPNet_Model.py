import os
# Set Keras backend to PyTorch for BeeFormer compatibility
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import torch
import keras
# Import utility functions and evaluation callback
from utils import *

from callbacks import evaluateWriter
# Normalized Mean Squared Error Loss Function
def NMSE(x, y):
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    return keras.ops.mean(keras.ops.square(x - y), axis=-1)

# Argument Parser Configuration
parser = argparse.ArgumentParser()

# Reproducibility / hardware
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--device", default=None, type=str)
parser.add_argument("--devices", default=None, type=str)
parser.add_argument("--flag", default="amazon_beauty_finetuned", type=str)
parser.add_argument("--validation", default="false", type=str)

# Fine-tuning optimized hyperparameters
parser.add_argument("--lr", default=2e-5, type=float)              
parser.add_argument("--scheduler", default="LinearWarmup", type=str)       
parser.add_argument("--init_lr", default=0.0, type=float)
parser.add_argument("--warmup_lr", default=2e-5, type=float)
parser.add_argument("--target_lr", default=1e-6, type=float)

# Shortened training duration
parser.add_argument("--warmup_epochs", default=1, type=int)
parser.add_argument("--decay_epochs", default=1, type=int)
parser.add_argument("--tuning_epochs", default=1, type=int)
parser.add_argument("--epochs", default=5, type=int)               

# Dataset
parser.add_argument("--dataset", default="amazon-beauty-custom", type=str)

# Task mode
parser.add_argument("--use_cold_start", default="true", type=str)
parser.add_argument("--use_time_split", default="false", type=str)

# Optional text prefix
parser.add_argument("--prefix", default=None, type=str)

# Base model
parser.add_argument(
    "--sbert",
    default="sentence-transformers/all-mpnet-base-v2",
    type=str,
)


# Speed optimization
parser.add_argument("--max_seq_length", default=128, type=int)     
parser.add_argument("--preproces_html", default="false", type=str)

# Main speed controls
parser.add_argument("--max_output", default=200, type=int)         
parser.add_argument("--batch_size", default=32, type=int)          
parser.add_argument("--top_k", default=0, type=int)
parser.add_argument("--sbert_batch_size", default=16, type=int)   

# Output
parser.add_argument(
    "--model_name",
    default="amazon_beauty_mpnet_1000_finetuned",
    type=str,
)

# Evaluation / saving
parser.add_argument("--evaluate", default="true", type=str)      
parser.add_argument("--evaluate_epoch", default="false", type=str) 
parser.add_argument("--save_every_epoch", default="false", type=str) 

# Parse arguments for notebook or script mode
args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)
# Restrict visible CUDA device if specified
if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
# Ensure backend remains PyTorch
os.environ["KERAS_BACKEND"] = "torch"
# Core libraries
import keras
import numpy as np
import pandas as pd
import sentence_transformers
import subprocess
import time
import torch

from sentence_transformers import SentenceTransformer
# Internal project modules
from callbacks import evaluateWriter
from config import config
from dataloaders import beeformerDataset
from models import NMSEbeeformer, SparseKerasELSA
from schedules import LinearWarmup
from _datasets.utils import *

# Improve GPU matrix multiplication speed
torch.set_float32_matmul_precision("medium")
# Auto-detect available hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")

# Dataset Loading Function
def load_data(args):
     # Choose validation or test split
    what = "val" if args.validation == "true" else "test"

    if args.dataset in config.keys():
        # Load dataset configuration
        dataset, params = config[args.dataset]
        dataset.load_interactions(**params)
         # Select evaluation protocol
        if args.use_time_split == "true":
            evaluator = TimeBasedEvaluation(dataset, what=what)
        elif args.use_cold_start == "true":
            evaluator = ColdStartEvaluation(dataset, what=what)
        else:
            evaluator = Evaluation(dataset, what=what)
         # Product metadata dataframe
        items_d = dataset.items_texts
        items_d["asin"] = items_d.item_id
        # Training split selection
        if args.validation == "true":
            _train_interactions = dataset.train_interactions
        else:
            _train_interactions = dataset.full_train_interactions

    else:
        # Handle unknown dataset
        print("Unknown dataset. List of available datasets:\n")
        for x in config.keys():
            print(x)
        return None, None, None, None

    return dataset, evaluator, _train_interactions, items_d

# Semantic Text Encoder Preparation
def load_text_model(args, items_d, dataset, _train_interactions):
    print("Preprocessing texts.")
      # Prepare all product texts for full evaluation
    if args.evaluate == "true" or args.evaluate_epoch == "true":
        am_itemids = items_d.asin.to_numpy()
        cc = np.array(dataset.all_interactions.item_id.cat.categories)
        
        ccdf = pd.Series(cc).to_frame()
        ccdf.columns = ["item_id"]

        amdf = pd.Series(am_itemids).to_frame().reset_index()
        amdf.columns = ["idx", "item_id"]
          # Match dataset item IDs with metadata
        am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()
        # Extract product descriptions
        am_texts_all = items_d._text_attributes.to_numpy()[am_locator]
        am_texts_all = [str(x) for x in am_texts_all]
    else:
        am_texts_all = None
     # Prepare train-only texts
    am_itemids = items_d.asin.to_numpy()
    cc = np.array(_train_interactions.item_id.cat.categories)

    ccdf = pd.Series(cc).to_frame()
    ccdf.columns = ["item_id"]

    amdf = pd.Series(am_itemids).to_frame().reset_index()
    amdf.columns = ["idx", "item_id"]
    # Align training items
    am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()
    # Training text descriptions
    am_texts = items_d._text_attributes.to_numpy()[am_locator]
    am_texts = [str(x) for x in am_texts]
    # Add optional semantic prefix
    if args.prefix is not None:
        print("adding prefix", args.prefix, "to all texts")
        am_texts = [args.prefix + x for x in am_texts]

    print("Creating sbert")
    # Load pretrained semantic encoder
    sbert = SentenceTransformer(
        args.sbert,
        device=DEVICE,
        trust_remote_code=True,
    )
      # Limit sequence length for faster training
    if args.max_seq_length is not None:
        sbert.max_seq_length = args.max_seq_length
     # Tokenize product descriptions
    am_tokenized = sbert.tokenize(list(am_texts))

    
    # beeformer dataloader expects values that have .to()
    am_tokenized = {
        k: v for k, v in am_tokenized.items()
        if hasattr(v, "to")
    }

    if am_texts_all is None:
        am_texts_all = am_texts

    return am_texts_all, am_tokenized, sbert



# Learning Rate Scheduler Preparation
def prepare_schedule(args, steps_per_epoch):
    # Cosine decay scheduler with warmup
    if args.scheduler == "CosineDecay":
        schedule = keras.optimizers.schedules.CosineDecay(
            0.0,
            steps_per_epoch * (args.decay_epochs + args.warmup_epochs),
            alpha=0.0, # Initial learning rate starts at zero
            name="CosineDecay",
            warmup_target=args.warmup_lr,
            warmup_steps=steps_per_epoch * args.warmup_epochs,
        )
        epochs = args.warmup_epochs + args.decay_epochs + args.tuning_epochs
      # Total training epochs = warmup + decay + fine-tuning
    elif args.scheduler == "LinearWarmup":
        schedule = LinearWarmup(
            warmup_steps=steps_per_epoch * args.warmup_epochs,
            decay_steps=steps_per_epoch * args.decay_epochs,
            starting_lr=args.init_lr,
            warmup_lr=args.warmup_lr,
            final_lr=args.target_lr,
        ) # Total epochs across all phases
        epochs = args.warmup_epochs + args.decay_epochs + args.tuning_epochs
     # Fallback: constant learning rate
    else:
        schedule = args.lr
        epochs = args.epochs
        print("Using constant learning rate of", schedule)

    return schedule, epochs

#  Main Fine-Tuning Pipeline
def main(args):
    # Generate unique timestamp for experiment tracking
    timestamp = pd.Timestamp("today").strftime("%Y-%m-%d_%H-%M-%S")
    # Create unique results folder
    folder = os.path.join(
        "results",
        f"{timestamp}_{9 * int(1e6) + np.random.randint(999999)}",
    )
    # Make directory if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
      # Save experiment configuration
    pd.Series(vars(args)).to_csv(f"{folder}/setup.csv")
    print(f"Saving results to {folder}")
     # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    print(f"seeds set to {args.seed}")
     # Load dataset and evaluation objects
    dataset, evaluator, _train_interactions, items_d = load_data(args)
     # Exit if dataset is invalid
    if dataset is None:
        return None
      # Load semantic text encoder
    if args.sbert is not None:
        am_texts_all, am_tokenized, sbert = load_text_model(
            args,
            items_d,
            dataset,
            _train_interactions,
        )
   
    else:
        print("Dont know what to train. Please specify the --sbert argument.")
        return None
     # Enable multi-GPU training if requested
    if args.devices is not None:
        devices_to_run = eval(args.devices)
        module_sbert = torch.nn.DataParallel(
            sbert,
            device_ids=devices_to_run,
            output_device=devices_to_run[0],
        )
    else:
        module_sbert = sbert
      # Build sparse interaction matrix
    print("Creating interaction matrix for training")
    X = get_sparse_matrix_from_dataframe(_train_interactions)
    # Create BeeFormer dataloader
    print("Creating dataloader")
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
      # Initialize BeeFormer recommendation model
    model = NMSEbeeformer(
        tokenized_sentences=am_tokenized,
        items_idx=_train_interactions.item_id.cat.categories,
        sbert=keras.layers.TorchModuleWrapper(module_sbert),
        device=DEVICE,
        top_k=args.top_k,
        sbert_batch_size=args.sbert_batch_size,
    )
    # Configure learning rate schedule
    schedule, epochs = prepare_schedule(args, steps_per_epoch)

    model.to(DEVICE)
     # Prepare callbacks
    cbs = []
    eval_cb = None
     # Add evaluation callback if enabled
    if args.evaluate == "true" or args.evaluate_epoch == "true" or args.save_every_epoch == "true":
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
     # Compile model with optimizer and loss
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=schedule),
        loss=NMSE,
        metrics=[keras.metrics.CosineSimilarity()],
    )
      # Build model structure
    print("Building the model")
    model.train_step(datal[0])
    model.built = True
    model.summary()
     # Start training
    print("Starting training loop")
     
    train_time = time.time()

    print(f"Training for {epochs} epochs.")

    f = model.fit(
        datal,
        epochs=epochs,
        callbacks=cbs,
    )
    # Record total training time
    train_time = time.time() - train_time

    sbert.save(args.model_name)
    print(f"Model saved to {args.model_name}")

    final_results = {}
   # Evaluation Phase
    if args.evaluate == "true":
         # Encode all product texts
        embs = sbert.encode(am_texts_all, show_progress_bar=True)
        # Build sparse retrieval model for recommendation evaluation
        eval_model = SparseKerasELSA(
            len(dataset.all_interactions.item_id.cat.categories),
            embs.shape[1],
            dataset.all_interactions.item_id.cat.categories,
            device=DEVICE,
        )

        eval_model.to(DEVICE)
        eval_model.set_weights([embs])

        if args.use_cold_start == "true":
            df_preds = eval_model.predict_df(
                evaluator.test_src,
                candidates_df=(
                    evaluator.cold_start_candidates_df
                    if hasattr(evaluator, "cold_start_candidates_df")
                    else None
                ),
                k=1000,
            )
             # Remove already seen interactions
            df_preds = df_preds[
                ~df_preds.set_index(["item_id", "user_id"]).index.isin(
                    evaluator.test_src.set_index(["item_id", "user_id"]).index
                )
            ]
            # Standard evaluation mode
        else:
            df_preds = eval_model.predict_df(evaluator.test_src)
          # Compute final recommendation metrics
        final_results = evaluator(df_preds)
         # Save evaluation results
        print(final_results)
        pd.Series(final_results).to_csv(f"{folder}/result.csv")
        print("results file written")
      # Save Training History
    history_df = pd.DataFrame(f.history)
    history_df["epoch"] = np.arange(len(history_df)) + 1
    history_df.to_csv(f"{folder}/history.csv", index=False)
    print("history file written")
    # Save epoch-by-epoch evaluation history
    try:
        pd.concat(
            [pd.Series(x).to_frame().T for x in eval_cb.results_list]
        ).to_csv(f"{folder}/results-history.csv")
    except:
        print("eval_cb not exist")
      # Save training duration
    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")
      # Save GPU hardware information
    try:
        out = subprocess.check_output(["nvidia-smi"])

        with open(
            os.path.join(folder, f"{args.dataset}_{args.flag}.log"),
            "w",
        ) as f:
            f.write(out.decode("utf-8"))

    except:
        print("nvidia-smi unavailable, skipping GPU log.")

    return {  # Return summary information
        "folder": folder,
        "train_time_seconds": train_time,
        "results": final_results,
    }


if __name__ == "__main__":

    experiments = [
      # Define experiments with optimized hyperparameters
  {
        "model_label": "MPNet",
        "sbert": "sentence-transformers/all-mpnet-base-v2",
        "model_name": "amazon_beauty_mpnet_1000_finetuned",
        "epochs": 5,
        "batch_size": 24,
        "lr": 1e-5,
        "max_seq_length": 256,
        "args.max_output": 150,
        "args.sbert_batch_size": 24,
    },

    ]

    summary_results = []

    for exp in experiments:
        print("\n" + "=" * 80)
        print(f"Starting fine-tuning for: {exp['model_label']}")
        print("=" * 80)
         # Assign experiment-specific parameters
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
            # Run training pipeline
            run_info = main(args)
             # Store successful experiment results
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
                row["Train Time Seconds"] = run_info["train_time_seconds"]

                if isinstance(run_info["results"], dict):
                    row.update(run_info["results"])

            summary_results.append(row)

        except Exception as e:
            # Store failed experiment information
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
    # Final Summary Export
    print("\n" + "=" * 80)
    print("FINAL FINE-TUNING SUMMARY")
    print("=" * 80)

    summary_df = pd.DataFrame(summary_results)
    print(summary_df)

    summary_df.to_csv("final_finetuning_summary.csv", index=False)
    print("\nSummary table saved as final_finetuning_summary.csv")
