import numpy as np
import pandas as pd
import re
import time
import torch
import scipy
import warnings

import recpack.metrics
import scipy.sparse

from scipy.sparse import csr_matrix
from pandas.core.generic import SettingWithCopyWarning
from math import ceil, floor
from tqdm import tqdm
from bs4 import BeautifulSoup

# Ignore pandas copy warnings during preprocessing
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# Remove text inside square brackets
def striptags(data):
    try:
        p = re.compile(r"\[.*?\]")
        return p.sub("", data)

    except:
        return ""


# Remove HTML tags and keep only clean text
def preproces_html(html):
    soup = BeautifulSoup(html, "lxml")
    return soup.text


# Generate random indices for evaluation splitting
def get_random_indices(row, frac=0.2, part=0):

    a = row.indices

    # Select 20% of the interactions
    pick = ceil(len(a) * 0.2)

    if part == 0:
        return np.random.choice(a, pick)

    q = []

    # Create folds for multi-fold evaluation
    for i in range(int(1 / 0.2)):
        q.append(a[i * pick: i * pick + pick])

    return q[part]


# Split interactions into source and target matrices
def get_src_target_rand(X_val):

    X_val_src = X_val.copy()

    # Remove part of the interactions for testing
    for i in range(X_val_src.shape[0]):

        ind = get_random_indices(X_val_src[i])

        X_val_src[i, ind] = 0

    X_val_src.eliminate_zeros()

    # Remaining interactions become targets
    X_val_targets = X_val - X_val_src

    bl = torch.from_numpy(1 - X_val_src.toarray()).to("cpu")

    target = torch.from_numpy(
        X_val_targets.toarray().astype(bool)
    )

    return X_val_src, X_val_targets


# Create multiple folds for evaluation
def get_src_target_fold(X_val, fold=0):

    X = []
    XV = []

    X_val_src = X_val.copy()

    # First fold
    for i in tqdm(range(X_val_src.shape[0])):

        ind = get_random_indices(X_val_src[i], 1)

        X_val_src[i, ind] = 0

    X.append(X_val_src)
    XV.append(X_val)

    # Additional folds
    if fold != 1:

        for part in [2, 3, 4, 5]:

            X_val_src = X_val.copy()

            for i in tqdm(range(X_val_src.shape[0])):

                ind = get_random_indices(X_val_src[i], part)

                X_val_src[i, ind] = 0

            X.append(X_val_src)
            XV.append(X_val)

    # Combine all folds together
    X_val_src = scipy.sparse.vstack(X)
    X_val = scipy.sparse.vstack(XV)

    X_val_src.eliminate_zeros()

    X_val_targets = X_val - X_val_src

    return X_val_src, X_val_targets


# Convert evaluation matrices into dataframes
def get_get_src_target_rand_df(test_interactions):

    X_test = get_sparse_matrix_from_dataframe(test_interactions)

    X_test_src, X_test_target = get_src_target_rand(X_test)

    df_src = sparse_matrix_to_df(
        X_test_src,
        test_interactions.item_id.cat.categories,
        test_interactions.user_id.cat.categories
    )

    df_target = sparse_matrix_to_df(
        X_test_target,
        test_interactions.item_id.cat.categories,
        test_interactions.user_id.cat.categories
    )

    return df_src, df_target, X_test_src, X_test_target


# Same as above but using folds
def get_get_src_target_rand_df_fold(test_interactions, fold=0):

    X_test = get_sparse_matrix_from_dataframe(test_interactions)

    X_test_src, X_test_target = get_src_target_fold(X_test, fold)

    # Create temporary user ids if needed
    if X_test_src.shape[0] != len(test_interactions.user_id.cat.categories):
        uids = pd.Index(np.arange(X_test_src.shape[0]).astype(str))

    else:
        uids = test_interactions.user_id.cat.categories

    df_src = sparse_matrix_to_df(
        X_test_src,
        test_interactions.item_id.cat.categories,
        uids
    )

    df_target = sparse_matrix_to_df(
        X_test_target,
        test_interactions.item_id.cat.categories,
        uids
    )

    return df_src, df_target, X_test_src, X_test_target


# Convert sparse matrix into dataframe
def sparse_matrix_to_df(X, item_ids, user_ids, verbose=10000):

    split = np.split(X.indices, X.indptr)[1:-1]
    split2 = np.split(X.data, X.indptr)[1:-1]

    dfs = []

    # Build dataframe row by row
    for i in tqdm(range(len(split))):

        dfs.append(
            pd.DataFrame(
                {
                    "user_id": user_ids[i],
                    "item_id": item_ids[split[i]],
                    "value": split2[i]
                }
            )
        )

    ret = pd.concat(dfs)

    # Convert columns back into categorical types
    ret["user_id"] = (
        ret["user_id"]
        .astype(str)
        .astype("category")
        .cat.remove_unused_categories()
    )

    ret["item_id"] = (
        ret["item_id"]
        .astype(str)
        .astype("category")
        .cat.remove_unused_categories()
    )

    return ret


# Simple logger class
class logger:

    @staticmethod
    def info(*args):
        print(*args)

    @staticmethod
    def debug(*args):
        print(*args)


# Convert dataframe into sparse matrix
def convert_user_item_pairs_into_sparse_matrix(
    interactions: pd.DataFrame,
    sparse_type,
):

    """
    Create sparse matrix from interaction dataframe.
    """

    # Handle empty dataframe case
    if len(interactions) == 0:

        return (
            [],
            [],
            InteractionPreparator.SPARSE_MATRIXES[sparse_type](
                ([], ([], [])),
                shape=(0, 0),
                dtype=np.float64,
            ),
        )

    return (
        interactions["item_id"].cat.categories,
        interactions["user_id"].cat.categories,

        csr_matrix(
            (
                interactions["value"].values,

                (
                    interactions["item_id"].cat.codes,
                    interactions["user_id"].cat.codes,
                ),
            ),

            shape=(
                len(interactions["item_id"].cat.categories),
                len(interactions["user_id"].cat.categories),
            ),

            dtype=np.float64,
        ),
    )


# Create sparse matrix from dataframe
def get_sparse_matrix_from_dataframe(
    df,
    item_indices=None,
    user_indices=None
):

    # Use dataframe categories if indices are not provided
    if item_indices is None:
        item_indices = df.item_id.cat.categories

    if user_indices is None:
        user_indices = df.user_id.cat.categories

    df = df.copy()

    # Keep only matching items and users
    df = df[df.item_id.isin(item_indices)]
    df = df[df.user_id.isin(user_indices)]

    df["user_id"] = df.user_id.astype("category")

    # Convert ids into integer indices
    row_ind = [item_indices.get_loc(x) for x in df.item_id]
    col_ind = [user_indices.get_loc(x) for x in df.user_id]

    # Build sparse interaction matrix
    mat = csr_matrix(
        (
            df.value.values,
            (row_ind, col_ind),
        ),

        shape=(
            len(item_indices),
            len(user_indices),
        ),

        dtype=np.float64,
    )

    return mat.T.tocsr()


# Fast iterative pruning of users and items
def fast_pruning(
    interactions: pd.DataFrame,
    pruning_user: int,
    pruning_item: int,
    logger=logger,
    item_users_are_unique: bool = False,
    max_user_support: int = 0,
    max_item_support: int = 0,
    max_steps: int = 0,
) -> pd.DataFrame:

    stable = False
    step = 1

    # Convert interactions into sparse matrix
    item_map, user_map, X = convert_user_item_pairs_into_sparse_matrix(
        interactions,
        "csr"
    )

    X = X.astype(bool).T

    users_cnt_old = len(interactions["user_id"].cat.categories)
    items_cnt_old = len(interactions["item_id"].cat.categories)

    logger.info(
        "Starting reduction:"
        f" {X.getnnz()} interactions"
    )

    # Continue pruning until stable
    while not stable:

        logger.debug(
            f"Interactions at step {step}: {X.getnnz()}"
        )

        stable = True

        # Remove low-support items
        number_of_items = len(item_map)

        matching_items = np.where(X.sum(0) >= pruning_item)[1]

        X = X[:, matching_items]

        if max_item_support > 0:

            matching_items = np.where(
                X.sum(0) <= max_item_support
            )[0]

            X = X[:, matching_items]

        item_map = item_map[matching_items]

        number_of_items_with_support = len(item_map)

        logger.info(
            f"Items after pruning: {number_of_items_with_support}"
        )

        if number_of_items > number_of_items_with_support:
            stable = False

        # Remove low-support users
        number_of_users = len(user_map)

        matching_users = np.where(X.sum(1) >= pruning_user)[0]

        X = X[matching_users, :]

        if max_user_support > 0:

            matching_users = np.where(
                X.sum(1) <= max_user_support
            )[0]

            X = X[matching_users, :]

        user_map = user_map[matching_users]

        number_of_users_with_support = len(user_map)

        logger.info(
            f"Users after pruning: {number_of_users_with_support}"
        )

        if number_of_users > number_of_users_with_support:
            stable = False

        # Stop if max steps reached
        if max_steps > 0 and step >= max_steps:
            stable = True

        if stable:
            logger.info(
                f"Data stable after {step} pruning steps"
            )

        step += 1

    # Keep only remaining users and items
    interactions = interactions[
        (interactions.user_id.isin(user_map))
        &
        (interactions.item_id.isin(item_map))
    ]

    print()

    interactions["user_id"] = (
        interactions["user_id"]
        .cat.remove_unused_categories()
    )

    interactions["item_id"] = (
        interactions["item_id"]
        .cat.remove_unused_categories()
    )

    logger.info(
        f"Users: {users_cnt_old} => "
        f"{len(interactions['user_id'].cat.categories)}"
    )

    logger.info(
        f"Items: {items_cnt_old} => "
        f"{len(interactions['item_id'].cat.categories)}"
    )

    return interactions