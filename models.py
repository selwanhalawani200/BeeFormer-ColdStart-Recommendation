
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import math
import numpy as np
import pandas as pd
import torch

from dataloaders import *
from layers import *
from _datasets.utils import get_first_item


# beeformer optimized with nmse
class NMSEbeeformer(keras.models.Model):
    def __init__(
        self,
        tokenized_sentences,
        items_idx,
        sbert,
        device,
        top_k=0,
        sbert_batch_size=128,
    ):
        super().__init__()
        self.device = device
        self.sbert = LayerSBERT(
            sbert,
            device,
            tokenized_sentences,
        )

        self.items_idx = items_idx
        self.tokenized_sentences = tokenized_sentences
        self.top_k = top_k
        self.sbert_batch_size = sbert_batch_size

    def call(self, x):
        # Normalize embeddings for better stability
        return torch.nn.functional.normalize(
            self.sbert(x),
            dim=-1,
        )

    def train_step(self, data):
        # Unpack the data
        a, b = data
        x, y = a
        y = torch.hstack((x, y))
        x_out = y
        tokenized_items, slicer, negative_slicer = b
        slicer = slicer.to(self.device)

        if negative_slicer is not None:
            negative_slicer = negative_slicer.to(self.device)

        # Initialize training
        self.zero_grad()
        sbert_batch_size = self.sbert_batch_size
        len_sentences = get_first_item(
            tokenized_items
        ).shape[0]
        max_i = math.ceil(
            len_sentences / sbert_batch_size
        )

        # Forward pass #1 (without gradients)
        with torch.no_grad():
            batched_results = []

            for i in range(max_i):
                ind = i * sbert_batch_size
                ind_min = ind
                ind_max = ind + sbert_batch_size

                batch_result = self.sbert(
                    {
                        k: v[ind_min:ind_max]
                        for k, v in tokenized_items.items()
                    }
                )

                batch_result = (
                    torch.nn.functional.normalize(
                        batch_result,
                        dim=-1,
                    )
                )

                batched_results.append(
                    batch_result
                )

            A = torch.vstack(
                batched_results
            )

        # Track gradients
        A.requires_grad = True

        # Compute ELSA forward pass
        A_slicer = A[slicer]
        A_slicer = torch.nn.functional.normalize(
            A_slicer,
            dim=-1,
        )

        A_negative_slicer = A[
            negative_slicer
        ]
        A_negative_slicer = (
            torch.nn.functional.normalize(
                A_negative_slicer,
                dim=-1,
            )
        )

        # ELSA step
        xA = torch.matmul(
            x,
            A_slicer,
        )

        xAAT = torch.matmul(
            xA,
            A_negative_slicer.T,
        )

        y_pred = keras.activations.relu(
            xAAT - x_out
        )

        # Optional top-k
        if self.top_k > 0:
            val, inds = torch.topk(
                y_pred,
                self.top_k,
            )

            y = torch.gather(
                y,
                1,
                inds,
            )

            y_pred = val

        # Compute loss
        loss = self.compute_loss(
            y=y,
            y_pred=y_pred,
        )

        # Backpropagate through A
        loss.backward()

        # Forward pass #2 (with gradients)
        for i in range(max_i):
            ind = i * sbert_batch_size
            ind_min = ind
            ind_max = ind + sbert_batch_size

            temp_out = self.sbert(
                {
                    k: v[ind_min:ind_max]
                    for k, v in tokenized_items.items()
                }
            )

            temp_out = (
                torch.nn.functional.normalize(
                    temp_out,
                    dim=-1,
                )
            )

            temp_out.retain_grad()

            partial_A_grad = A.grad[
                ind_min:ind_max
            ]

            temp_out.backward(
                gradient=partial_A_grad
            )

        # Collect gradients
        trainable_weights = [
            v
            for v in self.sbert.trainable_weights
        ]

        gradients = [
            v.value.grad
            for v in trainable_weights
        ]

        # Gradient clipping
        gradients = [
            torch.clamp(
                g,
                min=-1.0,
                max=1.0,
            )
            if g is not None
            else g
            for g in gradients
        ]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(
                gradients,
                trainable_weights,
            )

        # Update metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(
                    loss
                )
            else:
                metric.update_state(
                    y,
                    y_pred,
                )

        # Return metrics
        return {
            m.name: m.result()
            for m in self.metrics
        }


# ELSA model optimized for sparse data
class SparseKerasELSA(keras.models.Model):
    def __init__(
        self,
        n_items,
        n_dims,
        items_idx,
        device,
        top_k=0,
    ):
        super().__init__()

        self.device = device
        self.ELSA = LayerELSA(
            n_items,
            n_dims,
            device=device,
        )

        self.items_idx = items_idx

        self.ELSA.build()

        self(np.zeros([1, n_items]))

        self.finetuning = False
        self.top_k = top_k

    def call(self, x):
     if isinstance(x, np.ndarray):
        x = torch.from_numpy(
            x.astype("float32")
        ).to(self.device)
     else:
        x = x.to(self.device)

     return self.ELSA(x).cpu()

    def predict_df(
        self,
        df,
        k=100,
        user_ids=None,
        candidates_df=None,
        block_reminder=True,
    ):
        # Create predictions from dataframe
        if user_ids is None:
            user_ids = np.array(
                df.user_id.cat.categories
            )

        if candidates_df is not None:
            candidates_vec = (
                get_sparse_matrix_from_dataframe(
                    candidates_df,
                    item_indices=self.items_idx,
                ).toarray()
            )

            candidates_vec = torch.from_numpy(
                candidates_vec
            )

        data = PredictDfRecSysDataset(
            df,
            self.items_idx,
            batch_size=1024,
        )

        dfs = []

        for i in tqdm(
            range(len(data)),
            total=len(data),
        ):
            x, batch_uids = data[i]

            batch = torch.from_numpy(
                self.predict_on_batch(x)
            )

            if block_reminder:
                mask = 1 - x.astype(bool)
                batch = batch * mask

            if candidates_df is not None:
                batch *= candidates_vec

            values_, indices_ = torch.topk(
                batch.to("cpu"),
                k,
            )

            temp_df = pd.DataFrame(
                {
                    "user_id": np.stack(
                        [batch_uids] * k
                    ).flatten("F"),
                    "item_id": np.array(
                        self.items_idx
                    )[indices_].flatten(),
                    "value": values_.flatten(),
                }
            )

            temp_df["user_id"] = (
                temp_df["user_id"]
                .astype(str)
                .astype("category")
            )

            temp_df["item_id"] = (
                temp_df["item_id"]
                .astype(str)
                .astype("category")
            )

            dfs.append(temp_df)

        final_df = pd.concat(dfs)

        final_df["user_id"] = (
            final_df["user_id"]
            .astype(str)
            .astype("category")
        )

        final_df["item_id"] = (
            final_df["item_id"]
            .astype(str)
            .astype("category")
        )

        return final_df

