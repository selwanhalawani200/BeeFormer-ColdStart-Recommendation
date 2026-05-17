import os

# Set Torch as the backend for Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
import math
import numpy as np
import scipy.sparse
import torch

from _datasets.utils import *


class beeformerDataset(keras.utils.PyDataset):
    """
    This dataset takes the sparse user-item interaction matrix
    and returns batches that BeeFormer can use during training.
    """

    def __init__(
        self,
        X: scipy.sparse.csr_matrix,
        tokenized_sentences,
        device,
        batch_size: int = 1024,
        shuffle=False,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10,
        max_output=None,
    ):
        # Initialize the Keras PyDataset
        super().__init__(
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size
        )

        # Store the interaction matrix, batch size, shuffle option, and tokenized item texts
        self.X, self.batch_size, self.shuffle, self.tokenized_sentences = (
            X,
            batch_size,
            shuffle,
            {k: v for k, v in tokenized_sentences.items()},
        )

        # Make sure the number of tokenized items matches the number of items in the matrix
        assert get_first_item(tokenized_sentences).shape[0] == X.shape[1]

        # Create user and item index arrays
        self.indices = np.arange(X.shape[0])
        self.items_indices = np.arange(X.shape[1])

        # Save the device used for tensors
        self.device = device

        # Set the maximum number of output items used in each batch
        if max_output is None:
            self.max_output = X.shape[1]
        else:
            self.max_output = max_output

        # Shuffle users at the beginning if shuffle is enabled
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        # Return the number of batches
        return math.ceil(self.X.shape[0] / self.batch_size)

    def __getitem__(self, n):
        # Get the row range for the current batch
        ind = n * self.batch_size
        ind_min = ind
        ind_max = ind + self.batch_size

        # Select users for this batch
        slicer = self.indices[ind_min:ind_max]
        M = self.X[slicer]

        # Find items that appear in this batch
        item_slicer = np.where(M.getnnz(0) > 0)[0]

        # Create a mask to select negative items
        mask = np.ones(self.items_indices.shape, dtype=bool)
        mask[item_slicer] = False

        # Select negative samples to keep the output size controlled
        num_negatives = max(1, self.max_output - len(item_slicer))
        item_slicer_for_negatives = np.random.choice(
            self.items_indices[mask],
            num_negatives
        )

        # Combine positive items with sampled negative items
        item_slicer_with_negatives = np.hstack([
            item_slicer,
            item_slicer_for_negatives
        ])

        # Convert positive and negative interaction matrices to COO format
        scipy_coo_x = M[:, item_slicer].tocoo()
        scipy_coo_y = M[:, item_slicer_for_negatives].tocoo()

        # Convert positive interactions to a Torch sparse tensor
        torch_coo_x = torch.sparse_coo_tensor(
            np.vstack([scipy_coo_x.row, scipy_coo_x.col]),
            scipy_coo_x.data.astype(np.float32),
            scipy_coo_x.shape,
        )

        # Convert negative interactions to a Torch sparse tensor
        torch_coo_y = torch.sparse_coo_tensor(
            np.vstack([scipy_coo_y.row, scipy_coo_y.col]),
            scipy_coo_y.data.astype(np.float32),
            scipy_coo_y.shape,
        )

        # Select the tokenized item texts for both positive and negative items
        tokenized_items = {
            k: v[item_slicer_with_negatives].to(self.device)
            for k, v in self.tokenized_sentences.items()
        }

        # Create index slicers used later by the model
        slicer = np.arange(len(item_slicer))
        slicer_neg = np.arange(len(item_slicer_with_negatives))

        # Return interaction tensors and the related tokenized item inputs
        return (
            torch_coo_x.to(self.device).to_dense(),
            torch_coo_y.to(self.device).to_dense()
        ), (
            tokenized_items,
            torch.from_numpy(slicer).long(),
            torch.from_numpy(slicer_neg).long(),
        )

    def on_epoch_end(self):
        # Shuffle users after each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)


class PredictDfRecSysDataset(keras.utils.PyDataset):
    """
    This dataset is used during prediction.
    It returns user interaction vectors and the matching user ids.
    """

    def __init__(
        self,
        df,
        item_ids,
        batch_size=128,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=10
    ):
        # Initialize the Keras PyDataset
        super().__init__(
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size
        )

        # Store user ids and item ids
        self.user_ids = np.array(df.user_id.cat.categories)
        self.df, self.batch_size, self.items_ids = df, batch_size, item_ids

        # Convert the dataframe into a sparse interaction matrix
        self.X = get_sparse_matrix_from_dataframe(
            df,
            item_indices=self.items_ids
        )

    def __len__(self):
        # Return the number of batches
        return math.ceil(self.X.shape[0] / self.batch_size)

    def __getitem__(self, n):
        # Get the row range for the current prediction batch
        ind = n * self.batch_size
        ind_min = ind
        ind_max = ind + self.batch_size

        # Select the interaction matrix part for this batch
        M = self.X[ind_min:ind_max]

        # Convert sparse matrix to dense numpy array for prediction
        R = M.toarray().astype("float32")

        # Return user vectors and their user ids
        return R, self.user_ids[ind_min:ind_max]