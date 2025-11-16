# **Data Reduction Techniques for Green AI — Experimental Analysis**

This repository contains an in-depth study and experimental evaluation of several **data reduction techniques** applied to machine learning classification tasks.  
The project explores how reducing the size of training datasets, while preserving predictive performance, can significantly lower **training time**, **energy consumption**, and **CO₂ emissions**, contributing to the principles of **Green AI**.

---

## 📌 Project Overview

Modern Deep Learning models often demand large datasets and considerable computational resources. This results in:

- High training times  
- Significant energy usage  
- Increased environmental impact  

**Data Reduction** aims to mitigate these issues by removing redundant or non-informative samples while keeping the model's performance as close as possible to the original.

This project analyzes **eight different data reduction techniques**, evaluates them on two real-world datasets, and compares their performance across multiple metrics.

---

## 📂 Studied Reduction Methods

### **1. Statistic-based Methods**
| Method | Description |
|-------|-------------|
| **SRS — Stratified Random Sampling** | Random sampling per class while preserving class balance. |
| **PRD — ProtoDash Selection** | Selects representative prototypes using Maximum Mean Discrepancy (MMD). |

### **2. Geometry-based Methods**
| Method | Description |
|-------|-------------|
| **CLC — Clustering Centroids Selection** | Uses k-means centroids as reduced dataset. |
| **MMS — Max-Min Selection** | Selects points maximizing distance diversity across samples. |
| **DES — Distance-Entropy Selection** | Selects samples with highest distance-entropy relative to class prototypes. |

### **3. Ranking-based Methods**
| Method | Description |
|-------|-------------|
| **PHL — PH Landmarks Selection** | Uses persistent homology to rank points based on topological relevance. |
| **NRMD — Numerosity Reduction by Matrix Decomposition** | Ranks points using matrix factorization–based similarity scoring. |

### **4. Wrapper Methods**
| Method | Description |
|-------|-------------|
| **FES — Forgetting Events Selection** | Selects samples with the highest number of forgetting events during early training. |

### **5. NN-rule Based Methods**
| Method | Description |
|-------|-------------|
| **FCNN Rule** | Computes a consistent subset of the dataset using Fast Condensed Nearest Neighbor heuristics. |

Variations such as FCNN1, FCNN2, FCNN3, FCNN4 and α-FCNN are also analyzed.

---

## 📊 Evaluation Metrics

Reduction methods are evaluated using:

### **Classification Metrics**
- Accuracy  
- Macro Average Precision  
- Macro Average Recall  
- Macro Average F1-Score  

### **General Metrics**
- **ε-Representativeness**, measuring how well the reduced dataset preserves the structure of the original  
- **Training Time**  
- **CO₂ Emissions** (via CodeCarbon)  

---

## 🧪 Experimental Setup

### **Datasets**
1. **Collision Dataset**
   - 107,210 samples  
   - 25 numeric features  
   - Binary classification  
   - Highly unbalanced (≈69% class 1)

2. **Dry Bean Dataset**
   - 13,611 samples  
   - 16 geometric features  
   - 7 classes with varying sample sizes  

---

### **Neural Network Architecture**

A feed-forward neural network was adopted:

- 10 hidden layers with ReLU activations  
- Dropout:  
  - 0.50 for Collision  
  - 0.25 for Dry Bean  
- Optimizer: **Adam**, learning rate 0.001  
- Loss functions:  
  - **Binary Cross Entropy** for Collision  
  - **Weighted Categorical Cross Entropy** for Dry Bean  
- Training configuration:  
  - Collision: 600 epochs, batch size 1024  
  - Dry Bean: 150 epochs, batch size 32  
  - FES uses partial full-dataset pre-training  


