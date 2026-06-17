# TFG: Fraud Detection (Traditional ML vs GNNs)

This repository contains the codebase for the Bachelor's Thesis (_Trabajo de Fin de Grado_ - TFG) on **Anti-Money Laundering (AML) Fraud Detection**. 
It offers a comprehensive comparison framework between traditional Machine Learning models and Graph Neural Networks (GNNs) using the **IBM Transactions for Anti-Money Laundering (AML)** dataset.

## Key Features

- **Traditional ML Pipeline**: Out-of-the-box support for **XGBoost**, **LightGBM**, **Random Forest**, and **SVM**. Includes automatic GPU acceleration via **RAPIDS cuML** for SVM and Random Forest when a compatible GPU is present, with seamless fallback to **scikit-learn** on CPU.
- **GNN Pipeline**: Native implementation of a custom **MEGA-PNA** (Multi-Edge & Neighborhood Aggregation using Principal Neighborhood Aggregation) encoder with an Edge Classifier to classify transaction edges, built on **PyTorch Geometric**.
- **Graph Databases & GDS Integration**: Integrates with **Neo4j** and **Neo4j Graph Data Science (GDS)** to automatically project graphs and extract topological features, including PageRank, Weakly Connected Components (WCC), and Fast Random Projection (FastRP) embeddings. Supports automatic fallback to basic node/edge statistics if Neo4j is offline or unavailable.
- **Explainability (XAI)**: Native support for model interpretability. Uses **SHAP** (TreeExplainer and KernelExplainer) for traditional ML models, and **GNNExplainer** for explaining edge-level GNN predictions.
- **MLOps & Experiment Tracking**: Seamless logging of parameters, evaluation metrics, and model artifacts to a local **MLflow** database (SQLite backend).
- **Hugging Face Hub Integration**: Functions to sync MLflow experiment archives directly with a Hugging Face model repository.

---

## Project Structure

```text
TFG-Fraud-Detection-GNN/
├── .devcontainer/      # Docker Dev Container setup (includes python environment and Neo4j)
├── configs/            # TOML configuration files for small/large traditional and GNN runs
├── src/                # Core source code
│   ├── config/         # TOML configuration loading and schema validation
│   ├── data/           # Data preprocessing, graph construction, and Neo4j loaders/extractors
│   ├── explainability/ # SHAP and GNNExplainer wrappers and visualizers
│   ├── models/         # Model architectures (Traditional wrappers & custom MEGA-PNA GNN)
│   ├── pipelines/      # Orchestrators (Traditional, GNN, Grid Search, and Results Exporter)
│   ├── tracking/       # MLflow execution tracker and HF Hub uploading
│   ├── utils/          # Utilities (GPU checking, paths, data downloading, and logging)
│   └── cli.py          # Unified CLI entry point (`fraud-detect`)
├── tests/              # Unit tests covering all components
├── Makefile            # Commands for installation, tests, linting, and running pipelines
└── pyproject.toml      # Dependency specification and project metadata
```

---

## Requirements

- **Python 3.11+**
- **Docker & Docker Compose** (highly recommended to run Neo4j with Graph Data Science & APOC plugins)
- **CUDA Toolkit** (optional, but highly recommended for PyTorch Geometric GNN training and RAPIDS cuML acceleration)

---

## Environment Configuration

Create a `.env` file in the root directory to store environment credentials:

```ini
HF_TOKEN="your_hugging_face_token_here"
HF_MODEL_REPO_ID="your_hf_username/your_repo_id"
NEO4J_URI="bolt://localhost:7687"  # Use "bolt://neo4j:7687" when running inside the Dev Container
NEO4J_PASSWORD="your_neo4j_password"
```

---

## Installation & Environment Setup

The repository is fully configured for development using VS Code **Dev Containers**. This is the recommended setup as it provisions Python 3.11 with CUDA-enabled PyTorch, PyTorch Geometric, and a running Neo4j instance with GDS automatically.

### Option A: Using Dev Containers (Recommended)

1. Ensure Docker and the VS Code "Dev Containers" extension are installed.
2. Open the project folder in VS Code.
3. Click on the pop-up "Reopen in Container" or run the command `Dev Containers: Reopen in Container` from the command palette.
4. The container will build, start Neo4j, and install python dependencies in editable mode.

### Option B: Local Setup

1. **Start Neo4j**: Ensure you have a running Neo4j 5.12+ instance with the `apoc` and `graph-data-science` plugins enabled.
2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   make install
   ```

---

## Usage

You can launch pipelines either through the CLI (`fraud-detect`) or using the provided `Makefile` shortcuts.

### 1. Using the CLI

```bash
# Traditional ML pipeline (XGBoost, RandomForest, LightGBM, SVM)
fraud-detect traditional --config configs/traditional_hi_small.toml --models all

# GNN pipeline (MEGA-PNA GNN with Neo4j topological features)
fraud-detect gnn --config configs/gnn_hi_small.toml

# GNN Hyperparameter Grid Search
fraud-detect gnn-grid --config configs/gnn_hi_small.toml

# Download MLflow results archive from HF Hub and export to CSV
fraud-detect fetch-results --experiment traditional_HI-Small
```

### 2. Using the Makefile

The `Makefile` defines target shortcuts for small and large configurations:

```bash
make run-traditional        # Run traditional ML pipeline (Small)
make run-traditional-large  # Run traditional ML pipeline (Large)
make run-gnn                # Run GNN pipeline (Small)
make run-gnn-large          # Run GNN pipeline (Large)
make run-gnn-grid           # Run GNN Grid Search (Small)
make run-gnn-grid-large     # Run GNN Grid Search (Large)
```

> [!NOTE]
> When building the graph, the GNN pipeline will automatically attempt to connect to the Neo4j database configured in `.env`. If it cannot connect, it will log a warning and fallback to using standard node features, allowing GNN training to proceed without GDS properties.

---

## Testing & Code Quality

You can verify the codebase by running tests and style checkers inside your environment:

```bash
# Run the test suite via pytest
make test

# Lint code using Ruff
make lint

# Clean cache directories and logs
make clean
```

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** License. See the [LICENSE](LICENSE) file for more details.
