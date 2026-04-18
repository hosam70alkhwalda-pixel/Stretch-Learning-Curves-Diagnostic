# Learning Curve Analysis – Logistic Regression (Telecom Churn)

##  Overview
This project analyzes the performance of Logistic Regression models using learning curves on a telecom churn dataset. The evaluation metrics used are **ROC-AUC and PR-AUC**, which are more appropriate than accuracy for imbalanced classification problems.

Two models were compared:
- Logistic Regression with default regularization (C=1)
- Logistic Regression with stronger regularization (C=0.01)

---

##  Results

### ROC-AUC

| Model | Train ROC-AUC | Validation ROC-AUC |
|------|-------------|-------------------|
| LR (C=1) | 0.700 | 0.661 |
| LR (C=0.01) | 0.673 | 0.641 |

### PR-AUC

| Model | Train PR-AUC | Validation PR-AUC |
|------|-------------|-------------------|
| LR (C=1) | 0.301 | 0.282 |
| LR (C=0.01) | 0.293 | 0.277 |

---

##  Data Distribution

The dataset is imbalanced:

- Class 0 (non-churn): 83.7%  
- Class 1 (churn): 16.3%

This imbalance significantly affects evaluation, making PR-AUC a more reliable metric than ROC-AUC.

---

##  Analysis

### 1. Bias vs Variance

The model shows **low to moderate overfitting**, but overall performance is stable.

This is evident from the gap between training and validation scores:

- ROC-AUC (C=1): gap ≈ 0.039  
- PR-AUC (C=1): gap ≈ 0.019  

- ROC-AUC (C=0.01): gap ≈ 0.032  
- PR-AUC (C=0.01): gap ≈ 0.016  

The small gap indicates that the model generalizes reasonably well, and there is no severe overfitting.

---

### 2. Model Performance Insight

Although ROC-AUC values (~0.66) suggest moderate discrimination ability, PR-AUC (~0.28) reveals that the model struggles to correctly identify churn cases.

This indicates:
- The model performs significantly better than random (baseline PR-AUC ≈ 0.16)
- However, its ability to correctly detect the minority class is still limited

---

### 3. Would More Data Help?

Yes, more data would likely improve performance.

Since the model is trained on an imbalanced dataset, additional data (especially more churn cases) would help improve PR-AUC and generalization.

---

### 4. Would Increasing Model Complexity Help?

Not significantly at this stage.

The model is not strongly overfitting, but PR-AUC suggests limited predictive power. Increasing complexity alone may not solve the issue unless feature quality is improved.

---

### 5. Recommended Next Steps

To improve performance:

- Improve feature engineering (interaction features, ratio-based features)
- Try more powerful models (Random Forest, XGBoost)
- Tune decision threshold instead of using default 0.5
- Consider resampling techniques (SMOTE or undersampling)
- Focus optimization on **PR-AUC instead of ROC-AUC**

---

##  Key Takeaways

- Dataset is clearly imbalanced (83.7% vs 16.3%)
- ROC-AUC alone is misleading for this problem
- PR-AUC provides a more realistic evaluation
- Logistic Regression is stable but has limited predictive power
- The main limitation is **data signal quality, not model variance**