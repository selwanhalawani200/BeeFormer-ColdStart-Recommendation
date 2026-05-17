import gzip
import json
import pandas as pd
from tqdm import tqdm
import os
import re
# Path to Amazon Beauty reviews dataset
REVIEWS_PATH = "All_Beauty.jsonl"
# Path to Amazon Beauty product metadata
META_PATH = "meta_All_Beauty.jsonl"
# Output directory for processed dataset files
OUTPUT_DIR = "_datasets/amazon_beauty"
# Maximum number of rows to process (None = full dataset)
MAX_ROWS = None
# Create output directory if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Text Cleaning Function
def clean_text(text):
    text = text.lower() # Convert text to lowercase
    text = re.sub(r"\s+", " ", text) # Replace multiple spaces/newlines with single spaces
    text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", "", text) # Keep only letters, numbers, and basic punctuation
    text = re.sub(r"\.{2,}", ".", text)  # Replace repeated periods
    return text.strip()
# Load and Filter Review Data
# Store processed review entries
reviews = []
 # Open review dataset file
with open(REVIEWS_PATH, "r", encoding="utf-8") as f:
     # Iterate through reviews with progress bar
    for i, line in enumerate(tqdm(f, desc="Loading reviews")):
         # Stop early if row limit is set
        if MAX_ROWS is not None and i >= MAX_ROWS:
            break
         # Parse JSON review record
        row = json.loads(line)
        # Keep only positive interactions (rating >= 4)
        if row.get("rating", 0) >= 4:
            reviews.append({
                "user_id": row["user_id"],
                "item_id": row["parent_asin"],
                "rating": row["rating"],
                "timestamp": row["timestamp"]
            })

# Convert reviews into DataFrame
reviews_df = pd.DataFrame(reviews)
# User Activity Filtering
# Count number of reviews per user
user_counts = reviews_df["user_id"].value_counts()

# Keep only users with at least 5 interactions
valid_users = user_counts[user_counts >= 5].index
# Filter dataset to active users only
reviews_df = reviews_df[reviews_df["user_id"].isin(valid_users)]
# Save Ratings Dataset
# Output path for interaction data
ratings_path = os.path.join(OUTPUT_DIR, "ratings.feather")
# Save filtered interactions
reviews_df.reset_index(drop=True).to_feather(ratings_path)

# Display summary
print("Saved ratings:", ratings_path)
print("Ratings shape:", reviews_df.shape)
# Load and Process Product Metadata
# Store cleaned product metadata
meta = []
# Keep only products present in ratings dataset
valid_items = set(reviews_df["item_id"].unique())
# Open metadata file
with open(META_PATH, "r", encoding="utf-8") as f:
    # Iterate through metadata entries
    for line in tqdm(f, desc="Loading metadata"):
        # Parse JSON metadata
        row = json.loads(line)
        
        # Product ASIN
        asin = row.get("parent_asin", None)
      # Keep only products with valid user interactions
        if asin in valid_items:
            # Extract title
            title = row.get("title", "")
             # Extract product features
            features = " ".join(row.get("features", [])) if row.get("features") else ""
            # Extract product description
            description = " ".join(row.get("description", [])) if row.get("description") else ""
             # Combine all textual fields
            full_text = f"{title}. {title}. {features}. {description}"
             # Clean combined text
            full_text = clean_text(full_text)
            # Keep only sufficiently informative descriptions
            if len(full_text.strip()) > 30:
                 # Limit text length for model efficiency
                full_text = full_text[:1500]

                meta.append({
                    "item_id": asin,
                    "text": full_text
                })
# Convert metadata to DataFrame
meta_df = pd.DataFrame(meta)
# Keep only one entry per product
meta_df = meta_df.drop_duplicates(subset=["item_id"])
# Save Product Text Descriptions
items_path = os.path.join(OUTPUT_DIR, "item_text_descriptions.feather")
# Save processed product descriptions
meta_df.reset_index(drop=True).to_feather(items_path)
# Display summary
print("Saved item texts:", items_path)
print("Items shape:", meta_df.shape)
print("Preprocessing completed successfully.")
