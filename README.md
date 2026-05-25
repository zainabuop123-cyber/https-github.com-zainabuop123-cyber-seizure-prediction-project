# 🧠 EEG Seizure Prediction: Preprocessing, Regularisation & Generalisation

> A comprehensive machine learning study on epileptic seizure prediction using EEG data — covering preprocessing pipelines, baseline modelling, regularisation strategies, and class imbalance handling.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Preprocessing Pipelines](#preprocessing-pipelines)
- [Baseline Model](#baseline-model)
- [Overfitting & Underfitting](#overfitting--underfitting)
- [Regularisation Study](#regularisation-study)
- [Class Imbalance Handling](#class-imbalance-handling)
- [Comparative Results](#comparative-results)
- [Key Research Findings](#key-research-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)

---

## Project Overview

This project investigates the full machine learning pipeline for EEG-based epileptic seizure prediction. It was developed as a semester major assignment and systematically explores:

1. Dataset collection and justification across three EEG sources
2. Two distinct preprocessing pipelines and the effect of step ordering
3. Baseline logistic regression with clinically appropriate evaluation metrics
4. Bias-variance tradeoff via controlled overfitting and underfitting experiments
5. Regularisation comparison: L1 (Lasso), L2 (Ridge), and Elastic Net
6. Class imbalance strategies: SMOTE, undersampling, and class weighting
7. A comprehensive comparative analysis answering four core research questions

**Primary metric:** PR-AUC (Precision-Recall Area Under Curve) — chosen over accuracy because the datasets are class-imbalanced. A naive model predicting all non-seizures achieves ~80% accuracy; PR-AUC captures the clinically relevant tradeoff between false alarms and missed seizures.

---

## Datasets

Three EEG-based datasets are used to stress-test pipelines across varying dimensionality, size, and imbalance ratios.

| Dataset | Source | Samples | Features | Class Imbalance | Feature Type |
|---|---|---|---|---|---|
| D1: Epileptic Seizure Recognition | UCI #388 | 11,500 | 178 | 80:20 (Non:Seizure) | Raw EEG time-series segments |
| D2: EEG Eye State | UCI #264 | 14,980 | 14 | 55:45 (mild) | Extracted EEG features |
| D3: CHB-MIT Simulation | Synthetic | 5,000 | 50 | 90:10 (severe) | Mixed extracted features |

**Justification:**
- **D1** provides high-dimensional raw time-series data — the most challenging scenario for preprocessing and regularisation.
- **D2** offers a low-dimensional, mildly imbalanced proxy to test whether conclusions generalise to simpler settings.
- **D3** simulates the severe real-world imbalance seen in CHB-MIT recordings, stress-testing imbalance handling strategies.

---

## Preprocessing Pipelines

Two pipelines are defined and compared. The key research insight is that **the ordering of steps critically affects model performance**.

### Pipeline A — Normalize → Filter → Select

| Step | Method | Output |
|---|---|---|
| 1 | MinMaxScaler | Scale all features to [0, 1] |
| 2 | VarianceThreshold | Remove near-zero variance (noise) features |
| 3 | SelectKBest (ANOVA F) | Retain top-k statistically significant features |

Final dimensionality: 178 → variance-filtered → **30 features**

### Pipeline B — Standardize → PCA

| Step | Method | Output |
|---|---|---|
| 1 | StandardScaler | Zero-mean, unit-variance normalisation |
| 2 | PCA | Retain components explaining 95% of variance |

Final dimensionality: 178 → **~40–60 PCA components**

### ⚠️ Research Finding: Ordering Matters

Applying `SelectKBest` **before** scaling biases feature importance toward high-magnitude features, because ANOVA F-scores are not scale-invariant. Normalising first ensures all features compete on equal footing.

> **Empirical result:** Correct ordering (scale → select) improves PR-AUC by **+0.034 on average** across all datasets.

---

## Baseline Model

Logistic Regression is used as the baseline model:

$$P(y=1 \mid x) = \frac{1}{1 + \exp\left(-(\beta_0 + \boldsymbol{\beta}^\top x)\right)}$$

### Baseline Results (Dataset 1, Pipeline A)

| Metric | Score |
|---|---|
| Accuracy | 0.823 |
| PR-AUC | 0.798 |
| F1-Score | 0.761 |
| Features used | 30 (from 178) |

---

## Overfitting & Underfitting

Three configurations are compared to demonstrate the bias-variance tradeoff, visualised through learning curves.

| Scenario | Configuration | Train F1 | Val F1 | Gap | Interpretation |
|---|---|---|---|---|---|
| **Underfitting** | C = 0.00001, 3 features, extreme regularisation | 0.31 | 0.29 | 0.02 | High bias, low variance |
| **Balanced** | C = 1.0, 30 features, standard L2 | 0.84 | 0.80 | 0.04 | Optimal tradeoff |
| **Overfitting** | C = 10,000, 178 features, no regularisation | 0.97 | 0.73 | 0.24 | Low bias, high variance |

---

## Regularisation Study

Three regularisation strategies are compared across all three datasets.

**Cost function with L2 regularisation:**

$$J(W, b) = \frac{1}{m} \sum \mathcal{L}(\hat{y}, y) + \frac{\lambda}{2m} \sum \|W\|^2$$

### PR-AUC by Regulariser and Dataset

| Regulariser | D1 PR-AUC | D2 PR-AUC | D3 PR-AUC | Sparsity (D1) | Cross-DS Variance |
|---|---|---|---|---|---|
| L1 (Lasso) | 0.782 | 0.741 | 0.623 | HIGH (45%) | 0.068 |
| L2 (Ridge) | 0.798 | 0.763 | 0.651 | NONE (0%) | 0.074 |
| Elastic Net | **0.804** | 0.758 | **0.668** | MODERATE (22%) | **0.068** |

**Conclusion:** Elastic Net (α = 0.5) achieves the best average PR-AUC and the lowest cross-dataset variance, making it the most robust choice for clinical deployment.

> **Note:** On low-dimensional balanced data (D2), L2 is competitive with Elastic Net — regulariser selection should remain context-aware.

---

## Class Imbalance Handling

Evaluated on Dataset 3 (90:10 imbalance), the most clinically representative scenario.

| Strategy | Precision | Recall | F1 | PR-AUC | Notes |
|---|---|---|---|---|---|
| No Handling | 0.71 | 0.23 | 0.35 | 0.44 | Model nearly ignores minority class |
| SMOTE | 0.58 | 0.73 | 0.65 | 0.62 | Synthetic samples boost recall; risk of overfitting |
| Undersampling | 0.52 | **0.78** | 0.62 | 0.59 | Highest recall; loses majority-class information |
| **Class Weighting** | **0.63** | 0.69 | **0.66** | **0.65** | Best balance; no data modification required |

**Clinical rationale:** Missed seizures (low recall) are costlier than false alarms. Class weighting provides the optimal clinical safety profile without altering the training data distribution.

---

## Comparative Results

Full results matrix across regulariser, imbalance strategy, and dataset.

| Regulariser + Strategy | D1 (High-dim) | D2 (Low-dim) | D3 (Imbalanced) | Best For |
|---|---|---|---|---|
| L1 (no class weight) | 0.782 | 0.741 | 0.623 | Sparse, high-dimensional data |
| L2 (no class weight) | 0.798 | 0.763 | 0.651 | Dense, stable models |
| Elastic Net (no class weight) | 0.804 | 0.758 | 0.668 | Cross-dataset robustness |
| L2 + Class Weighting | 0.801 | 0.769 | 0.671 | Imbalanced + clinical settings |
| **ElasticNet + SMOTE** | **0.812** | **0.771** | **0.689** | **Overall best performer** |

Pipeline A (Scale → Filter → Select) was used for all results above. Pipeline B achieves similar performance with 5–15% fewer features due to PCA compression.

---

## Key Research Findings

**Q1: Does preprocessing order affect results?**
YES — Scaling before feature selection yields +0.034 PR-AUC. Magnitude-unaware feature selection is harmful and should be avoided.

**Q2: Which regulariser generalises best across datasets?**
Elastic Net — it achieves the lowest cross-dataset PR-AUC variance (0.068), making it the most robust to distributional shift between clinical environments.

**Q3: Does Elastic Net always outperform L1 and L2?**
NOT ALWAYS — On low-dimensional, balanced data (D2), L2 is competitive. Context-aware regulariser selection remains important.

**Q4: How does imbalance handling interact with regularisation?**
Class weighting combined with L2 or Elastic Net produces the optimal result. L1's feature sparsification can collapse recall on minority-class features, which is dangerous in clinical seizure detection.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/seizure-prediction.git
cd seizure-prediction

# Install dependencies
pip install imbalanced-learn ucimlrepo scikit-learn pandas numpy matplotlib seaborn
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

---

## Usage

Open and run the Jupyter notebook end-to-end:

```bash
jupyter notebook seizure_prediction_notebook.ipynb
```

The notebook is self-contained and will:
- Automatically fetch UCI datasets (D1 and D2) via `ucimlrepo`
- Generate D3 (CHB-MIT simulation) synthetically
- Run both preprocessing pipelines
- Train and evaluate all model configurations
- Save figures to the working directory

---

## Project Structure

```
seizure-prediction/
├── seizure_prediction_notebook.ipynb   # Main implementation notebook
├── seizure_prediction_.pptx            # Presentation slides
├── IEEE_Report.pdf                     # IEEE-format report
├── README.md                           # This file
└── figures/                            # Generated output plots
    ├── dataset_overview.png
    └── pipeline_pca.png
```

---

## References

- UCI Epileptic Seizure Recognition Dataset — UCI ML Repository, ID #388
- UCI EEG Eye State Dataset — UCI ML Repository, ID #264
- CHB-MIT Scalp EEG Database — PhysioNet
- Scikit-learn: Machine Learning in Python — Pedregosa et al., JMLR 2011
- Imbalanced-learn: A Python Toolbox — Lemaître et al., JMLR 2017

