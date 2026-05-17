# Semantic Hybrid Cold-Start Recommendation Framework for Amazon Beauty

## Overview
This project presents a semantic hybrid recommendation framework designed to improve cold-start recommendation performance in sparse e-commerce environments. The system integrates transformer-based semantic understanding with collaborative filtering supervision using the BeeFormer architecture and LayerELSA behavioral modeling.

Using the Amazon Beauty dataset, the framework combines semantic product embeddings, collaborative behavioral representations, and NMSE-based optimization to improve recommendation quality for newly introduced products with limited or no historical interactions.

---

## Key Features
- Positive interaction filtering (ratings ≥ 4)
- User and item sparsity pruning
- Product metadata preprocessing
- Transformer-based semantic encoding
- Collaborative preference learning using LayerELSA
- BeeFormer hybrid recommendation framework
- Cold-start evaluation under multiple unseen product settings
- Evaluation metrics:
  - Recall@20
  - Recall@50
  - NDCG@100
  - Coverage@20

---

## Dataset

### Raw Dataset
- 701,528 interactions
- 631,986 users
- 112,565 items

### Final Processed Dataset
- 92,314 interactions
- 4,782 users
- 5,164 items

### Dataset Source
Amazon Reviews 2023 Dataset:

https://amazon-reviews-2023.github.io/

---

## Preprocessing Pipeline
1. Retain only ratings ≥ 4
2. Remove users with fewer than 5 interactions
3. Remove items with fewer than 2 interactions
4. Clean and normalize product text
5. Generate semantic product descriptions

---

## Model Architecture

### Semantic Branch
Sentence Transformers generate normalized semantic embeddings from product descriptions.

### Collaborative Branch
LayerELSA learns sparse behavioral recommendation patterns from user-item interactions.

### Hybrid Optimization
BeeFormer aligns semantic and collaborative representations using NMSE loss and Nadam optimization.

---

## Evaluated Embedding Models
- MPNet
- BGE-M3
- Nomic Embed

---

## Hyperparameters
- Epochs: 5
- Learning Rate: 1e-5
- Batch Size: 24
- Maximum Sequence Length: 256

---

## Cold-Start Evaluation
The framework reserves validation and test users while excluding unseen products from training to simulate realistic cold-start recommendation scenarios.

Evaluations were performed under multiple unseen product settings:
- 500 cold-start products
- 1000 cold-start products
- 2000 cold-start products

---

## Research Contribution
This framework contributes:
- Recommendation-aware semantic fine-tuning
- Integration of transformer semantics with collaborative supervision
- Scalable cold-start evaluation for sparse recommendation systems

---

## Practical Applications
This framework can be applied to:
- E-commerce recommendation systems
- New product recommendation
- Sparse marketplace platforms
- Personalized semantic recommendation systems

---

## Repository Resources

Large files, trained models, processed datasets, and additional project resources are available through Google Drive:

https://drive.google.com/drive/folders/15zGmBTs1n8ZZpTYXDRVJ7flIqCImKz4H?usp=drive_link
