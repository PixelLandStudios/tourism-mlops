import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, login
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def train():
    # Authenticate with Hugging Face via environment variable or cached token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # 1. Load train and test data directly from Hugging Face data space
    print("Loading train and test data from Hugging Face dataset space...")
    dataset = load_dataset("malawn/tourism-dataset")
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    target_col = "ProdTaken"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numerical features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # 2. Define preprocessing and Random Forest pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    rf = RandomForestClassifier(random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    # 3. Define hyperparameter tuning grid
    param_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5],
    }

    # 4. Experimentation tracking with MLflow
    mlflow.set_experiment("tourism_package_prediction")

    with mlflow.start_run(run_name="RandomForest_Tuned"):
        print("Tuning model with GridSearchCV...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"\nBest Tuned Parameters: {best_params}")

        # Log tuned parameters to MLflow
        mlflow.log_params(best_params)

        # 5. Evaluate model performance on test set
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        print("\n--- Model Performance Evaluation on Test Set ---")
        for metric_name, score in metrics.items():
            print(f"{metric_name.capitalize()}: {score:.4f}")
            mlflow.log_metric(metric_name, score)

        # 6. Save model pipeline locally
        model_path = "tourism_project/model_building/best_model.joblib"
        joblib.dump(best_pipeline, model_path)
        print(f"\nModel successfully saved locally at: {model_path}")

        # 7. Register the best model in Hugging Face Model Hub
        model_repo_id = "malawn/tourism-package-model"
        api = HfApi()
        api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_model.joblib",
            repo_id=model_repo_id,
            repo_type="model"
        )
        print(f"Best model successfully registered at: https://huggingface.co/{model_repo_id}")

if __name__ == "__main__":
    train()
