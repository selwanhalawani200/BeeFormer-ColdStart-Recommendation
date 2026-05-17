import os

# Set Torch as the backend for Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
import pandas as pd

from models import SparseKerasELSA
from time import time
from _datasets.utils import *


class evaluateWriter(keras.callbacks.Callback):
    def __init__(
        self,
        items_idx,
        sbert,
        texts,
        evaluator,
        logdir,
        DEVICE,
        sbert_name="sbert_temp_model",
        evaluate_epoch="false",
        save_every_epoch="false",
    ):
        super().__init__()

        # Save the objects needed for evaluation
        self.evaluator = evaluator
        self.logdir = logdir
        self.sbert = sbert
        self.texts = texts
        self.items_idx = items_idx
        self.DEVICE = DEVICE

        # Store results from each evaluated epoch
        self.results_list = []

        # Save model name and evaluation options
        self.sbert_name = sbert_name
        self.evaluate_epoch = evaluate_epoch
        self.save_every_epoch = save_every_epoch

    def on_epoch_end(self, epoch, logs=None):
        print()

        # Save the SBERT model after each epoch if this option is enabled
        if self.save_every_epoch == "true":
            print("saving sbert model")
            self.sbert.save(f"{self.sbert_name}-epoch-{epoch}")

        # Run evaluation after each epoch if this option is enabled
        if self.evaluate_epoch == "true":

            # Encode all item texts using the current SBERT model
            embs = self.sbert.encode(
                self.texts,
                show_progress_bar=True
            )

            # Build the evaluation model using the generated embeddings
            model = SparseKerasELSA(
                len(self.items_idx),
                embs.shape[1],
                self.items_idx,
                device=self.DEVICE
            )

            model.to(self.DEVICE)

            # Load the embeddings as model weights
            model.set_weights([embs])

            # Use cold-start prediction logic if the evaluator is cold-start based
            if isinstance(self.evaluator, ColdStartEvaluation):

                df_preds = model.predict_df(
                    self.evaluator.test_src,
                    candidates_df=(
                        self.evaluator.cold_start_candidates_df
                        if hasattr(self.evaluator, "cold_start_candidates_df")
                        else None
                    ),
                    k=1000,
                )

                # Remove items that already exist in the source interactions
                df_preds = df_preds[
                    ~df_preds.set_index(
                        ["item_id", "user_id"]
                    ).index.isin(
                        self.evaluator.test_src.set_index(
                            ["item_id", "user_id"]
                        ).index
                    )
                ]

            else:
                # Use the normal prediction process for non cold-start evaluation
                df_preds = model.predict_df(self.evaluator.test_src)

            # Calculate evaluation metrics
            results = self.evaluator(df_preds)

            # Save the metrics because TensorBoard does not work well here
            print(results)

            pd.Series(results).to_csv(
                f"{self.logdir}/result-epoch-{epoch}.csv"
            )

            print("results file written")

            # Keep the results inside the callback for final logging
            self.results_list.append(results)