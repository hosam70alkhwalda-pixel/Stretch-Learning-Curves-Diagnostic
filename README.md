# Learning Curve Analysis – Logistic Regression (Telecom Churn)

##  Overview
This project analyzes the performance of Logistic Regression models using learning curves on a telecom churn dataset. The evaluation metric used is **ROC-AUC**, which is more appropriate than accuracy for imbalanced classification problems.

Two models were compared:
- Logistic Regression with default regularization (C=1)
- Logistic Regression with stronger regularization (C=0.01)

---

##  Results

| Model | Train ROC-AUC | Validation ROC-AUC |
|------|-------------|-------------------|
| LR (C=1) | 0.700 | 0.661 |
| LR (C=0.01) | 0.673 | 0.641 |

---

##  Analysis

### 1. Bias vs Variance

The model shows **mild overfitting (slight high variance)** rather than underfitting.

This is evident from the gap between training and validation scores:
- For C=1: gap ≈ 0.039  
- For C=0.01: gap ≈ 0.032  

The training performance is consistently higher than validation performance, indicating that the model is learning patterns from the training data that do not fully generalize to unseen data. However, the gap is relatively small, so the overfitting is moderate.

---

### 2. Would More Data Help?

Yes, collecting more data would likely improve validation performance.

The presence of a gap between training and validation scores suggests that the model could benefit from additional data. More data would help reduce overfitting and improve generalization.

---

### 3. Would Increasing Model Complexity Help?

No, increasing model complexity is not recommended at this stage.

Since the model already shows signs of overfitting, making it more complex (e.g., adding polynomial features or using more flexible models) could increase variance and worsen performance.

---

### 4. Recommended Next Steps

To improve the model:

- Tune regularization strength (e.g., try values between 0.01 and 1 such as 0.1)
- Collect more data if possible
- Perform feature selection or improve feature quality

---


---

##  Key Takeaways

- ROC-AUC is a better metric than accuracy for imbalanced datasets
- The default model (C=1) performs better overall
- Stronger regularization reduces overfitting but may lower performance
- The model is variance-limited, not bias-limited