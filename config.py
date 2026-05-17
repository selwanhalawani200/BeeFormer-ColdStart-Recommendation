import os

# Set Torch as the backend for Keras
os.environ["KERAS_BACKEND"] = "torch"

from _datasets.utils import *


# Main dataset configuration used in the experiments
config = {
    "amazon-beauty-custom": (

        # Create the dataset object
        Dataset("Amazon Beauty Custom"),

        {
            # Load the interaction dataset
            "raw_data": """pd.read_feather("_datasets/amazon_beauty/ratings.feather")""",

            # Column names used in the interaction data
            "value_name": "rating",
            "item_id_name": "item_id",
            "user_id_name": "user_id",
            "timestamp_name": "timestamp",

            # Keep only ratings greater than or equal to 4
            "min_value_to_keep": 4.0,

            # Minimum number of interactions required per user
            "user_min_support": 5,

            # Minimum number of interactions required per item
            "item_min_support": 2,

            # Convert all remaining ratings into implicit feedback
            "set_all_values_to": 1.0,

            # Number of users reserved for testing
            "num_test_users": 150,

            # Fixed random seed for reproducibility
            "random_state": 42,

            # Maximum processing steps during splitting
            "max_steps": 1000,

            # Create new dataset splits instead of loading old ones
            "load_previous_splits": False,

            # Load item description data
            "items_raw_data": """pd.read_feather("_datasets/amazon_beauty/item_text_descriptions.feather")""",

            # Item id column inside the item descriptions file
            "items_item_id_name": "item_id",

            # Use the item text field as the final item description
            "items_preprocess": """f'{row.text}'""",

            # Percentage of items used for cold-start evaluation
            "coldstart_fraction": 0.1,

            # Number of cold-start items used in testing
            "num_coldstart_items": 1000,
        },
    ),
}