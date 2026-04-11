import os

import joblib  # type: ignore
import polars as pl

from src.data.preprocessor import DataPreprocessor
from src.models.traditional import XGBoostModel
from src.utils.data_manager import DataSyncManager
from src.utils.logger import ProjectLogger


def main() -> None:
    """
    Main orchestration function to run the traditional models pipeline.
    """
    # 1. Initialize Logger
    ProjectLogger.initialize()
    logger = ProjectLogger.get_logger("MainTraditional")
    logger.info("Starting Traditional ML Pipeline")

    # 2. Download from Kaggle
    dataset_handle = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
    sync_manager = DataSyncManager()

    logger.info("Ensuring dataset is downloaded...")
    dataset_path = sync_manager.download_kaggle_dataset(dataset_handle)

    # 3. Preprocess Dataset
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(dataset_path)
    df = preprocessor.clean_data(df)

    # Automatically find string columns to encode
    categorical_cols = df.select(pl.col(pl.Utf8)).columns
    if "Is Laundering" in categorical_cols:
        categorical_cols.remove("Is Laundering")

    logger.info(f"Categorical features to encode: {categorical_cols}")
    df = preprocessor.encode_features(df, categorical_cols)

    # 4. Train-Test Split based on 'Is Laundering'
    # 'Is Laundering' is expected to be numeric (0/1), but we ensure it works
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df, target_col="Is Laundering"
    )
    logger.info(f"Training shapes: X={X_train.shape}, y={y_train.shape}")

    # 5. Train Model
    model = XGBoostModel()
    model.train(X_train, y_train)

    # 6. Evaluate
    model.evaluate(X_test, y_test)

    # 7. Save and Upload Model
    model_path = "xgboost_model.pkl"
    joblib.dump(model.model, model_path)
    logger.info(f"Model saved locally at {model_path}")

    # Use environment variable for the target repository on HF
    hf_repo_id = os.getenv("HF_MODEL_REPO_ID")
    if hf_repo_id:
        logger.info(f"Uploading model to {hf_repo_id}...")
        sync_manager.upload_artifact(
            local_path=model_path,
            repo_id=hf_repo_id,
            remote_filename="xgboost_baseline.pkl",
        )
    else:
        logger.warning(
            "HF_MODEL_REPO_ID not set. Skipping model upload to Hugging Face Hub."
        )


if __name__ == "__main__":
    main()
