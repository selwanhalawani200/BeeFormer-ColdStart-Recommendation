import os
# Set Keras backend to PyTorch for BeeFormer model compatibility
os.environ["KERAS_BACKEND"] = "torch"

import argparse
import torch
import keras
# Import utility functions and evaluation callback tools
from utils import *

from callbacks import evaluateWriter
# Normalized Mean Squared Error Loss Function
def NMSE(x, y):
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    return keras.ops.mean(keras.ops.square(x - y), axis=-1)

# Argument Parser Setup
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
    default="BAAI/bge-m3",
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
    default="amazon_beauty_bge_m3_1000_finetuned",
    type=str,
)

# Evaluation / saving
parser.add_argument("--evaluate", default="true", type=str)        
parser.add_argument("--evaluate_epoch", default="false", type=str)
parser.add_argument("--save_every_epoch", default="false", type=str) 


args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

#Device Restriction
if args.device is not None:
    print(f"Limiting devices to {args.device}")
    # Restrict visible CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
# Reconfirm backend
os.environ["KERAS_BACKEND"] = "torch"
 # Additional Core Imports
import keras
import numpy as np
import pandas as pd
import sentence_transformers
import subprocess
import time
import torch

from sentence_transformers import SentenceTransformer
#Internal project modules
from callbacks import evaluateWriter
from config import config
from dataloaders import beeformerDataset
from models import NMSEbeeformer, SparseKerasELSA
from schedules import LinearWarmup
from _datasets.utils import *

# Optimize matrix multiplication speed on supported GPUs
torch.set_float32_matmul_precision("medium")
# Automatically select GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {DEVICE}")

# Dataset Loading Function
def load_data(args):
    # Choose validation or test split
    what = "val" if args.validation == "true" else "test"
    # Check dataset exists in config
    if args.dataset in config.keys():
        # Load dataset and parameters
        dataset, params = config[args.dataset]
        dataset.load_interactions(**params)
         # Select evaluation protocol
        if args.use_time_split == "true":
            evaluator = TimeBasedEvaluation(dataset, what=what)
        elif args.use_cold_start == "true":
            evaluator = ColdStartEvaluation(dataset, what=what)
        else:
            evaluator = Evaluation(dataset, what=what)
        # Product metadata table
        items_d = dataset.items_texts
        items_d["asin"] = items_d.item_id
       # Training split selection
        if args.validation == "true":
            _train_interactions = dataset.train_interactions
        else:
            _train_interactions = dataset.full_train_interactions

    else:
        # Handle unsupported datasets
        print("Unknown dataset. List of available datasets:\n")
        for x in config.keys():
            print(x)
        return None, None, None, None

    return dataset, evaluator, _train_interactions, items_d

# Text Model Loading and Preparation
def load_text_model(args, items_d, dataset, _train_interactions):
    print("Preprocessing texts.")

    if args.evaluate == "true" or args.evaluate_epoch == "true":
        am_itemids = items_d.asin.to_numpy()
        cc = np.array(dataset.all_interactions.item_id.cat.categories)

        ccdf = pd.Series(cc).to_frame()
        ccdf.columns = ["item_id"]

        amdf = pd.Series(am_itemids).to_frame().reset_index()
        amdf.columns = ["idx", "item_id"]

        am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()

        am_texts_all = items_d._text_attributes.to_numpy()[am_locator]
        am_texts_all = [str(x) for x in am_texts_all]
    else:
        am_texts_all = None
     # Prepare Training Dataset Texts
    # Product IDs from metadata
    am_itemids = items_d.asin.to_numpy()
    cc = np.array(_train_interactions.item_id.cat.categories)
     # Convert training categories into DataFrame
    ccdf = pd.Series(cc).to_frame()
    ccdf.columns = ["item_id"]
     # Convert metadata to DataFrame
    amdf = pd.Series(am_itemids).to_frame().reset_index()
    amdf.columns = ["idx", "item_id"]
     # Match train items to metadata
    am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()
     # Extract train product descriptions
    am_texts = items_d._text_attributes.to_numpy()[am_locator]
    # Convert to string format
    am_texts = [str(x) for x in am_texts]
     # Optional Prefix Injection
    if args.prefix is not None:
        print("adding prefix", args.prefix, "to all texts")
        # Prefix can improve semantic consistency
        am_texts = [args.prefix + x for x in am_texts]
     # Load Semantic Encoder
    print("Creating sbert")
    # Load SentenceTransformer / BGE model
    sbert = SentenceTransformer(
        args.sbert,
        device=DEVICE,
        trust_remote_code=True,
    )
     # Limit sequence length for speed and memory optimization
    if args.max_seq_length is not None:
        sbert.max_seq_length = args.max_seq_length
         # Tokenization
    # Tokenize all product descriptions
    am_tokenized = sbert.tokenize(list(am_texts))

    # Important Compatibility Fix
    # Newer SentenceTransformers may return fields
    # that are not tensors, which BeeFormer cannot process.
    # Keep only fields that support .to() for device transfer.
    am_tokenized = {
        k: v for k, v in am_tokenized.items()
        if hasattr(v, "to")
    }  # Fallback Handling
     # If evaluation texts were skipped,
    # use training texts instead.
    if am_texts_all is None:
        am_texts_all = am_texts
   # Return:
    # - Full evaluation texts
    # - Tokenized training texts
    # - Loaded semantic encoder
    return am_texts_all, am_tokenized, sbert



# Learning Rate Schedule Preparation
def prepare_schedule(args, steps_per_epoch):
     # Cosine Decay Scheduler
    if args.scheduler == "CosineDecay":
        schedule = keras.optimizers.schedules.CosineDecay(
            0.0,
            steps_per_epoch * (args.decay_epochs + args.warmup_epochs),
            alpha=0.0,
            name="CosineDecay",
            warmup_target=args.warmup_lr,
            warmup_steps=steps_per_epoch * args.warmup_epochs,
        )
        # Total epochs across warmup + decay + tuning
        epochs = args.warmup_epochs + args.decay_epochs + args.tuning_epochs
    # Linear Warmup Scheduler
    elif args.scheduler == "LinearWarmup":
        schedule = LinearWarmup(
            warmup_steps=steps_per_epoch * args.warmup_epochs,
            decay_steps=steps_per_epoch * args.decay_epochs,
            starting_lr=args.init_lr,
            warmup_lr=args.warmup_lr,
            final_lr=args.target_lr,
        ) # Total training epochs
        epochs = args.warmup_epochs + args.decay_epochs + args.tuning_epochs
   # Constant Learning Rate
    else:
        schedule = args.lr
        epochs = args.epochs
        print("Using constant learning rate of", schedule)
     # Return scheduler object and total epochs
    return schedule, epochs

# Main Fine-Tuning and Evaluation Pipeline
def main(args):
     # Generate timestamp for unique experiment tracking
    timestamp = pd.Timestamp("today").strftime("%Y-%m-%d_%H-%M-%S")
     # Create unique result directory
    folder = os.path.join(
        "results",
        f"{timestamp}_{9 * int(1e6) + np.random.randint(999999)}",
    )
    # Ensure result folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Save experiment configuration for reproducibility
    pd.Series(vars(args)).to_csv(f"{folder}/setup.csv")
    print(f"Saving results to {folder}")
     # Reproducibility
    # Set all random seeds
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    print(f"seeds set to {args.seed}")
      # Data Loading
    dataset, evaluator, _train_interactions, items_d = load_data(args)
    # Exit if dataset loading fails
    if dataset is None:
        return None
     # Text Encoder Loading
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
    # Multi-GPU Support
    if args.devices is not None:
          # Parse selected GPUs
        devices_to_run = eval(args.devices)
        # Wrap model for parallel processing
        module_sbert = torch.nn.DataParallel(
            sbert,
            device_ids=devices_to_run,
            output_device=devices_to_run[0],
        )
    else:
        module_sbert = sbert
    # Interaction Matrix Construction
    print("Creating interaction matrix for training")
    # Convert user-item interactions into sparse matrix
    X = get_sparse_matrix_from_dataframe(_train_interactions)
    # Dataloader Creation
    print("Creating dataloader")
    datal = beeformerDataset(
        X,
        am_tokenized,
        DEVICE,
        shuffle=True,
        max_output=args.max_output,
        batch_size=args.batch_size,
    )
     # Number of training steps per epoch
    steps_per_epoch = len(datal)

    print(sbert)
    # BeeFormer Model Initialization
    model = NMSEbeeformer(
        tokenized_sentences=am_tokenized,
        items_idx=_train_interactions.item_id.cat.categories,
        sbert=keras.layers.TorchModuleWrapper(module_sbert),
        device=DEVICE,
        top_k=args.top_k,
        sbert_batch_size=args.sbert_batch_size,
    )
    # Learning Rate Schedule
    schedule, epochs = prepare_schedule(args, steps_per_epoch)

    model.to(DEVICE)
   # Callback Configuration
    cbs = []
    eval_cb = None
    # Add evaluation callback if needed
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
    # Model Compilation
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=schedule),
        loss=NMSE,
        metrics=[keras.metrics.CosineSimilarity()],
    )
    # Model Build
    print("Building the model")
    # Perform one train step to initialize model structure
    model.train_step(datal[0])
    model.built = True
    model.summary()
   # Training Loop
    print("Starting training loop")
    train_time = time.time()

    print(f"Training for {epochs} epochs.")

    f = model.fit(
        datal,
        epochs=epochs,
        callbacks=cbs,
    )
       # Total training duration
    train_time = time.time() - train_time
      # Save Fine-Tuned Semantic Encoder
    sbert.save(args.model_name)
    print(f"Model saved to {args.model_name}")

    final_results = {}
     # Evaluation Phase
    if args.evaluate == "true":
        embs = sbert.encode(am_texts_all, show_progress_bar=True)
         # Build sparse retrieval model
        eval_model = SparseKerasELSA(
            len(dataset.all_interactions.item_id.cat.categories),
            embs.shape[1],
            dataset.all_interactions.item_id.cat.categories,
            device=DEVICE,
        )

        eval_model.to(DEVICE)
        eval_model.set_weights([embs])
        # Cold-Start Evaluation
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
             # Remove already observed interactions
            df_preds = df_preds[
                ~df_preds.set_index(["item_id", "user_id"]).index.isin(
                    evaluator.test_src.set_index(["item_id", "user_id"]).index
                )
            ]
        else:
            # Standard recommendation evaluation
            df_preds = eval_model.predict_df(evaluator.test_src)
         # Compute recommendation metrics
        final_results = evaluator(df_preds)
        # Save final metrics
        print(final_results)
        pd.Series(final_results).to_csv(f"{folder}/result.csv")
        print("results file written")
     # Save Training History
    history_df = pd.DataFrame(f.history)
    history_df["epoch"] = np.arange(len(history_df)) + 1
    history_df.to_csv(f"{folder}/history.csv", index=False)
    print("history file written")
     # Save evaluation history if available
    try:
        pd.concat(
            [pd.Series(x).to_frame().T for x in eval_cb.results_list]
        ).to_csv(f"{folder}/results-history.csv")
    except:
        print("eval_cb not exist")
     # Save Runtime Metrics
    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")
      # Save GPU Hardware Log
    try:
        out = subprocess.check_output(["nvidia-smi"])

        with open(
            os.path.join(folder, f"{args.dataset}_{args.flag}.log"),
            "w",
        ) as f:
            f.write(out.decode("utf-8"))

    except:
        print("nvidia-smi unavailable, skipping GPU log.")
      # Return experiment summary
    return {
        "folder": folder,
        "train_time_seconds": train_time,
        "results": final_results,
    }

# Experiment Execution Block
if __name__ == "__main__":

    experiments = [
      
 {
    "model_label": "BGE-M3",
    "sbert": "BAAI/bge-m3",
    "model_name": "amazon_beauty_bge_m3_1000_finetuned",
    "epochs": 5,
    "batch_size": 24,
    "lr": 1e-5,
    "max_seq_length": 256,
    "args.max_output": 150,
    "args.sbert_batch_size": 24,
},

    ]
   # Store experiment summaries
    summary_results = []

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
            # Run main pipeline
            run_info = main(args)
            # Successful experiment record
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
           # Failed experiment record
            summary_results.append(row)

        except Exception as e:
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
     # Final Summary Generation
    print("\n" + "=" * 80)
    print("FINAL FINE-TUNING SUMMARY")
    print("=" * 80)
     # Create summary table
    summary_df = pd.DataFrame(summary_results)
    print(summary_df)
    # Save summary CSV
    summary_df.to_csv("final_finetuning_summary.csv", index=False)
    print("\nSummary table saved as final_finetuning_summary.csv")
