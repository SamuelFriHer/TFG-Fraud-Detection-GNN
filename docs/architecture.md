# Project Architecture

## Directory Structure

```
TFG-Fraud-Detection-GNN/
├── .devcontainer/                        # Docker / DevContainer configuration
│   ├── Dockerfile
│   └── devcontainer.json
├── .env                                  # Environment variables (HF_TOKEN, etc.)
├── configs/                              # Experiment configuration files (TOML)
│   ├── traditional_hi_small.toml
│   └── traditional_hi_large.toml
├── docs/                                 # Project documentation
│   ├── architecture.md                   # This file
│   ├── Proyecto de TFG.pdf
│   ├── Seminario Seguimiento 1.md
│   └── [Borrador] Memoria TFG.pdf
├── outputs/                              # Runtime artifacts (gitignored)
│   ├── models/                           # Serialized trained models
│   └── mlflow/                           # MLflow experiment tracking store
├── src/                                  # Source code
│   ├── cli.py                            # Unified CLI entry point
│   ├── data/
│   │   └── preprocessor.py               # Data loading, cleaning, encoding, splitting
│   ├── explainability/                   # [FUTURE] XAI module
│   │   ├── base.py                       # IExplainer interface
│   │   ├── shap_explainer.py             # SHAP for traditional models
│   │   ├── lime_explainer.py             # LIME for traditional models
│   │   └── gnn_explainer.py              # GNNExplainer for graph models
│   ├── models/
│   │   ├── base.py                       # IClassificationModel abstract interface
│   │   ├── traditional/                  # Traditional ML model implementations
│   │   │   ├── xgboost_model.py
│   │   │   ├── random_forest_model.py
│   │   │   ├── lightgbm_model.py
│   │   │   └── svm_model.py
│   │   └── gnn/                          # [FUTURE] Graph Neural Network models
│   │       ├── gcn_model.py
│   │       └── gat_model.py
│   ├── pipelines/                        # Orchestration layer
│   │   ├── traditional_pipeline.py       # Preprocess once → train N models
│   │   └── gnn_pipeline.py               # [FUTURE] Graph construction → GNN training
│   ├── tracking/
│   │   └── experiment_tracker.py         # MLflow wrapper + HF upload
│   └── utils/
│       ├── data_manager.py               # Kaggle / HuggingFace download & upload
│       └── logger.py                     # Centralized logging to file + console
├── tests/                                # Mirrors src/ structure
│   ├── data/
│   ├── models/
│   └── utils/
│       ├── test_data_manager.py
│       └── test_logger.py
├── Makefile                              # Dev shortcuts (lint, test, run)
└── pyproject.toml                        # Python packaging and dependencies
```

## Module Responsibilities

### `src/cli.py`
Unified entry point. Uses `argparse` subcommands (`traditional`, `gnn`, `explain`).
Supports `--models all` to run every model in a single invocation.

### `src/data/preprocessor.py`
Loads CSV datasets via Polars, handles null cleaning, label-encodes categorical features,
and splits into train (60%) / validate (20%) / test (20%).

### `src/models/base.py`
Defines `IClassificationModel` — the abstract interface all models implement.
Key contract: `evaluate()` returns `dict[str, float]` for structured metric tracking.

### `src/models/traditional/`
One file per algorithm (XGBoost, RandomForest, LightGBM, SVM).
Each implements `IClassificationModel`. The sub-package `__init__.py` exposes a
`MODEL_REGISTRY` dictionary for factory-style instantiation.

### `src/pipelines/traditional_pipeline.py`
Core orchestration: loads TOML config → downloads dataset → preprocesses **once** →
iterates over requested models → trains, evaluates, logs metrics, saves artifacts.

### `src/tracking/experiment_tracker.py`
Thin wrapper around MLflow. Creates experiments, logs runs with metrics and model
artifacts. At pipeline completion, uploads the MLflow store to Hugging Face Hub.

### `src/utils/`
- **`logger.py`**: Singleton `ProjectLogger` writing to `logs.txt` + console.
  Redirects `stdout`/`stderr` to the logger for full capture.
- **`data_manager.py`**: Downloads datasets from Kaggle, uploads artifacts to HF Hub.

## Data Flow

```
┌───────────┐      ┌──────────────┐      ┌─────────────────────┐
│  Config   │────▶│   Pipeline   │────▶│   Preprocessor      │
│  (.toml)  │      │              │      │   (load → clean →   │
└───────────┘      │              │      │   encode → split)   │
                   │              │      └────────┬────────────┘
                   │              │               │ preprocessed splits
                   │              │      ┌────────▼────────────┐
                   │              │────▶│   Model (N times)   │
                   │              │      │   train → evaluate  │
                   │              │      └────────┬────────────┘
                   │              │               │ metrics + artifact
                   │              │      ┌────────▼────────────┐
                   │              │────▶│  ExperimentTracker  │
                   │              │      │  MLflow log + HF ↑  │
                   └──────────────┘      └─────────────────────┘
```

## Execution Examples

```bash
# Install the project as an editable package
pip install -e ".[dev]"

# Run ALL traditional models on HI-Small
python -m src.cli traditional --config configs/traditional_hi_small.toml --models all

# Run only XGBoost and LightGBM on HI-Large
python -m src.cli traditional --config configs/traditional_hi_large.toml --models XGBoost LightGBM

# [FUTURE] Run GNN models
python -m src.cli gnn --config configs/gnn_hi_large.toml --models GCN GAT

# [FUTURE] Run explainability
python -m src.cli explain --pipeline traditional --config configs/traditional_hi_small.toml
```
