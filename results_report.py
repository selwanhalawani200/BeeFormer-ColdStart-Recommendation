
# results_report.py

import os
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_FILE = "final_finetuning_summary.csv"
OUTPUT_DIR = "evaluation_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load summary
df = pd.read_csv(SUMMARY_FILE)

print("\nFINAL RESULTS TABLE:\n")
print(df)

# Save clean table
df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "results_table.csv"
    ),
    index=False,
)

# Extract metrics

model_name = df["Model"].iloc[0]

metrics = {
    "Recall@20": df["recall@20"].iloc[0],
    "Recall@50": df["recall@50"].iloc[0],
    "NDCG@100": df["ndcg@100"].iloc[0],
    "Coverage@20": df["coverage@20"].iloc[0],
}

# Figure 1: Main recommendation metrics

plt.figure(figsize=(10, 6))

main_metrics = [
    "Recall@20",
    "Recall@50",
    "NDCG@100",
]

main_values = [
    metrics[m]
    for m in main_metrics
]

bars = plt.bar(
    main_metrics,
    main_values,
)

# Add value labels
for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.005,
        f"{height:.3f}",
        ha="center",
        fontsize=11,
    )

plt.title(
    f"Recommendation Performance Metrics for {model_name}",
    fontsize=14,
)

plt.ylabel("Score")
plt.ylim(
    0,
    max(main_values) + 0.08,
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "main_performance_metrics.png",
    ),
    dpi=300,
)

plt.close()


# Figure 2: Recall vs Coverage

plt.figure(figsize=(8, 6))

comparison_metrics = [
    "Recall@20",
    "Coverage@20",
]

comparison_values = [
    metrics[m]
    for m in comparison_metrics
]

bars = plt.bar(
    comparison_metrics,
    comparison_values,
)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.3f}",
        ha="center",
        fontsize=11,
    )

plt.title(
    f"Retrieval Effectiveness vs Coverage for {model_name}",
    fontsize=14,
)

plt.ylabel("Score")
plt.ylim(0, 1.1)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "retrieval_vs_coverage.png",
    ),
    dpi=300,
)

plt.close()


# summary text

summary_lines = [
    "PROJECT EVALUATION SUMMARY",
    "==========================",
    "",
    f"Model: {model_name}",
    f"Recall@20: {metrics['Recall@20']:.4f}",
    f"Recall@50: {metrics['Recall@50']:.4f}",
    f"NDCG@100: {metrics['NDCG@100']:.4f}",
    f"Coverage@20: {metrics['Coverage@20']:.4f}",
    "",
    "Generated Outputs:",
    "- results_table.csv",
    "- main_performance_metrics.png",
    "- retrieval_vs_coverage.png",
]

with open(
    os.path.join(
        OUTPUT_DIR,
        "evaluation_summary.txt",
    ),
    "w",
    encoding="utf-8",
) as f:
    f.write("\n".join(summary_lines))

print("\nAcademic-quality outputs generated successfully:")
print("- results_table.csv")
print("- main_performance_metrics.png")
print("- retrieval_vs_coverage.png")
print("- evaluation_summary.txt")
print(f"\nSaved in folder: {OUTPUT_DIR}")
