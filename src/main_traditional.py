import argparse
import os

import joblib  # type: ignore
import polars as pl
from dotenv import load_dotenv

from src.data.preprocessor import DataPreprocessor
from src.models.traditional import (
    LightGBMModel,
    RandomForestModel,
    SVMModel,
    XGBoostModel,
)
from src.utils.data_manager import DataSyncManager
from src.utils.logger import ProjectLogger

# Load environment variables from .env if present
load_dotenv()


def get_model(model_name: str):
    """Factory to instantiate models based on argument string."""
    models = {
        "XGBoost": XGBoostModel,
        "RandomForest": RandomForestModel,
        "LightGBM": LightGBMModel,
        "SVM": SVMModel,
    }
    if model_name not in models:
        raise ValueError(
            f"Model {model_name} not supported. Choose from {list(models.keys())}"
        )
    return models[model_name]()  # type: ignore


def main() -> None:
    """
    Main orchestration function to run the traditional models pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run baseline traditional ML experiments."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HI-Small",
        help="Dataset prefix (e.g., 'HI-Small', 'HI-Large', 'LI-Medium'). Defaults to 'HI-Small'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="XGBoost",
        choices=["XGBoost", "RandomForest", "LightGBM", "SVM"],
        help="Machine Learning model to train. Defaults to 'XGBoost'.",
    )
    args = parser.parse_args()

    # 1. Initialize Logger
    ProjectLogger.initialize()
    logger = ProjectLogger.get_logger("MainTraditional")
    logger.info(
        f"Starting Traditional ML Pipeline with Model: {args.model}, Dataset: {args.dataset}"
    )

    # 2. Download from Kaggle
    dataset_handle = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
    sync_manager = DataSyncManager()

    logger.info("Ensuring dataset is downloaded...")
    dataset_path = sync_manager.download_kaggle_dataset(dataset_handle)

    # 3. Preprocess Dataset
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(dataset_path, dataset_prefix=args.dataset)
    df = preprocessor.clean_data(df)

    # Automatically find string columns to encode
    categorical_cols = df.select(pl.col(pl.Utf8)).columns
    if "Is Laundering" in categorical_cols:
        categorical_cols.remove("Is Laundering")

    logger.info(f"Categorical features to encode: {categorical_cols}")
    df = preprocessor.encode_features(df, categorical_cols)

    # 4. Train-Validate-Test Split (60/20/20)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, target_col="Is Laundering"
    )
    logger.info(
        f"Splits: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}"
    )

    # 5. Train Model
    model = get_model(args.model)
    model.train(X_train, y_train)

    # 6. Evaluate
    logger.info("Evaluating Validation Set:")
    model.evaluate(X_val, y_val)

    logger.info("Evaluating Test Set:")
    model.evaluate(X_test, y_test)

    # 7. Save and Upload Model
    model_filename = f"{args.model.lower()}_{args.dataset.lower()}_model.pkl"
    joblib.dump(model.model, model_filename)
    logger.info(f"Model saved locally at {model_filename}")

    # Use environment variable for the target repository on HF
    hf_repo_id = os.getenv("HF_MODEL_REPO_ID")
    if hf_repo_id:
        logger.info(f"Uploading model to {hf_repo_id}...")
        sync_manager.upload_artifact(
            local_path=model_filename,
            repo_id=hf_repo_id,
            remote_filename=model_filename,
        )
    else:
        logger.warning(
            "HF_MODEL_REPO_ID not set. Skipping model upload to Hugging Face Hub."
        )


if __name__ == "__main__":
    main()
