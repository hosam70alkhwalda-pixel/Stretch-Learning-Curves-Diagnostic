import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# =========================
# FEATURES
# =========================
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "senior_citizen",
    "has_partner",
    "has_dependents"
]

CATEGORICAL_FEATURES = [
    "gender",
    "contract_type",
    "internet_service",
    "payment_method"
]

# =========================
# TRAIN SIZES
# =========================
TRAIN_SIZES = np.linspace(0.1, 1.0, 9)

# =========================
# PALETTE (COLORS)
# =========================
PALETTE = {
    "LR - default regularization (C=1)": ("#1F77B4", "#AEC7E8"),   # blue
    "LR - strong regularization (C=0.01)": ("#FF7F0E", "#FFBB78") # orange
}

# =========================
# PREPROCESSOR
# =========================
def build_preprocessor():
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor

# =========================
# MODELS
# =========================
def define_models():
    models = {
        "LR - default regularization (C=1)": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(
                C=1,
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ))
        ]),

        "LR - strong regularization (C=0.01)": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(
                C=0.01,
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ))
        ]),
    }

    return models

# =========================
# MAIN
# =========================
def main():

    # load dataset safely
    file_path = os.path.join(os.path.dirname(__file__), "telecom_churn.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "telecom_churn.csv not found in the same folder as this script."
        )

    df = pd.read_csv(file_path)

    # clean numeric column
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df = df.dropna()

    # split features/target
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churned"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = define_models()

    plt.figure(figsize=(12, 7))

    # =========================
    # LEARNING CURVES
    # =========================
    for name, model in models.items():

        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=5,
            train_sizes=TRAIN_SIZES,
            scoring="accuracy",
            n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)

        train_color, val_color = PALETTE[name]

        # train curve
        plt.plot(
            train_sizes,
            train_mean,
            linestyle="--",
            color=train_color,
            label=f"{name} (train)"
        )

        # validation curve
        plt.plot(
            train_sizes,
            val_mean,
            color=val_color,
            label=f"{name} (val)"
        )

        print(f"\n{name}")
        print(f"Final Train Score: {train_mean[-1]:.3f}")
        print(f"Final Validation Score: {val_mean[-1]:.3f}")
        print("-" * 40)

    # =========================
    # PLOT STYLE
    # =========================
    plt.title("Learning Curves - Logistic Regression Comparison")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.savefig("learning_curves.png")
    plt.show()


if __name__ == "__main__":
    main()