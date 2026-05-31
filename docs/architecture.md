# Project Architecture

## Directory Structure

```
TFG-Fraud-Detection-GNN/
├── .devcontainer/                        # Docker / DevContainer configuration
│   ├── Dockerfile
│   └── devcontainer.json
├── configs/                              # Experiment configuration files (TOML)
│   ├── traditional_hi_small.toml
│   └── traditional_hi_large.toml
├── src/                                  # Source code
│   ├── cli.py                            # Unified CLI entry point
│   ├── data/
│   │   ├── preprocessor.py               # Data loading, cleaning, encoding, splitting
│   │   └── graph_builder.py              # Generates PyG graphs from CSVs
│   ├── explainability/                   # [FUTURE] XAI module
│   │   ├── base.py                       # IExplainer interface
│   │   ├── shap_explainer.py             # SHAP for traditional models
│   │   ├── lime_explainer.py             # LIME for traditional models
│   │   └── gnn_explainer.py              # GNNExplainer for graph models
│   ├── models/
│   │   ├── base.py                       # IClassificationModel abstract interface
│   │   ├── classification_metrics.py     # Shared metrics evaluation
│   │   ├── traditional/                  # Traditional ML model implementations
│   │   │   ├── xgboost_model.py
│   │   │   ├── random_forest_model.py
│   │   │   ├── lightgbm_model.py
│   │   │   └── svm_model.py
│   │   └── gnn/                          # Graph Neural Network models
│   │       ├── evaluator.py              # Evaluates GNN predictions (optimizing PR-AUC)
│   │       ├── layers.py                 # MEGA-PNA layers, Multi-edge aggregation
│   │       ├── loss.py                   # Focal Loss handling for class imbalance
│   │       └── model.py                  # GNNFraudDetector wrapper model
│   ├── pipelines/                        # Orchestration layer
│   │   ├── traditional_pipeline.py       # Preprocess once → train N traditional models
│   │   ├── gnn_pipeline.py               # Graph construction → train MEGA-PNA model
│   │   ├── results_exporter.py           # Exports MLflow results
│   │   └── experiment_config.py          # Handles TOML config loading
│   ├── tracking/
│   │   └── experiment_tracker.py         # MLflow wrapper + HF upload
│   └── utils/
│       ├── data_manager.py               # Kaggle / HuggingFace download & upload
│       ├── gpu_availability.py           # Device checks
│       ├── paths.py                      # Project paths
│       └── logger.py                     # Centralized logging to file + console
├── tests/                                # Mirrors src/ structure
│   ├── data/
│   ├── models/
│   ├── pipelines/
│   └── utils/
├── Makefile                              # Dev shortcuts (lint, test, run)
└── pyproject.toml                        # Python packaging and dependencies
```

## Module Responsibilities

### `src/cli.py`
Unified entry point. Uses `argparse` subcommands (`traditional`, `gnn`, `explain`).
Supports `--models all` to run every model in a single invocation.

### `src/data/preprocessor.py` & `src/data/graph_builder.py`
- `preprocessor.py`: Loads CSV datasets via Polars, handles null cleaning, label-encodes categorical features, and splits into train/validate/test.
- `graph_builder.py`: Constructs PyTorch Geometric (PyG) `Data` objects containing edge attributes and features specifically designed for Anti-Money Laundering.

### `src/models/base.py` & `src/models/classification_metrics.py`
Defines `IClassificationModel` — the abstract interface all traditional models implement.
Key contract: `evaluate()` returns `dict[str, float]` for structured metric tracking.

### `src/models/traditional/`
One file per algorithm (XGBoost, RandomForest, LightGBM, SVM).
Each implements `IClassificationModel`. The sub-package `__init__.py` exposes a
`MODEL_REGISTRY` dictionary for factory-style instantiation.

### `src/models/gnn/`
Contains the state-of-the-art **MEGA-PNA** architecture tailored for AML graph structures.
Features an aggregation in two stages (multi-edge and neighborhood), bidirectional message passing,
and focal loss heavily adapted to the 0.1% fraud class imbalance.

### `src/pipelines/traditional_pipeline.py` & `src/pipelines/gnn_pipeline.py`
- `traditional_pipeline.py`: Loads config → downloads dataset → preprocesses **once** → iterates over requested models → trains, evaluates, logs.
- `gnn_pipeline.py`: Loads config → builds global graph representation → instantiates `GNNFraudDetector` → trains across epochs storing best state via Validation PR-AUC.

### `src/tracking/experiment_tracker.py`
Thin wrapper around MLflow. Creates experiments, logs runs with metrics and model
artifacts. At pipeline completion, uploads the MLflow store to Hugging Face Hub.

### `src/utils/`
- **`logger.py`**: Singleton `ProjectLogger` writing to `logs.txt` + console.
  Redirects `stdout`/`stderr` to the logger for full capture.
- **`data_manager.py`**: Downloads datasets from Kaggle, uploads artifacts to HF Hub.

## Data Flow

```
┌───────────┐      ┌──────────────┐      ┌───────────────────────────┐
│  Config   │────▶ │   Pipeline   │────▶ │ Preprocessor / GraphBldr  │
│  (.toml)  │      │ (Trad or GNN)│      │   (clean → encode / build)│
└───────────┘      │              │      └────────┬──────────────────┘
                   │              │               │ preprocessed splits / graph
                   │              │      ┌────────▼──────────────────┐
                   │              │────▶ │ Model / GNNFraudDetector  │
                   │              │      │   train → evaluate        │
                   │              │      └────────┬──────────────────┘
                   │              │               │ metrics + artifact
                   │              │      ┌────────▼──────────────────┐
                   │              │────▶ │  ExperimentTracker        │
                   │              │      │  MLflow log + HF ↑        │
                   └──────────────┘      └───────────────────────────┘
```

## Execution Examples

```bash
# Install the project as an editable package
pip install -e ".[dev]"

# Run ALL traditional models on HI-Small
python -m src.cli traditional --config configs/traditional_hi_small.toml --models all

# Run only XGBoost and LightGBM on HI-Large
python -m src.cli traditional --config configs/traditional_hi_large.toml --models XGBoost LightGBM

# Run GNN model on HI-Small
python -m src.cli gnn --config configs/gnn_hi_small.toml

# [FUTURE] Run explainability
python -m src.cli explain --pipeline traditional --config configs/traditional_hi_small.toml
```
