"""Unified CLI entry point for all fraud detection pipelines."""

import argparse
import sys

from dotenv import load_dotenv

from src.models.traditional import ALL_MODEL_NAMES
from src.pipelines.gnn_pipeline import GNNPipeline
from src.pipelines.results_exporter import ResultsExporter
from src.pipelines.traditional_pipeline import TraditionalPipeline
from src.utils.logger import ProjectLogger

load_dotenv()


def _build_traditional_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore
    """Registers the 'traditional' subcommand with its arguments."""
    parser = subparsers.add_parser(
        "traditional",
        help="Run traditional ML models (XGBoost, RandomForest, LightGBM, SVM).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the TOML experiment config file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to train. Use 'all' for every registered model.",
    )


def _build_gnn_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore
    """Registers the 'gnn' subcommand with its arguments."""
    parser = subparsers.add_parser(
        "gnn",
        help="Run Graph Neural Network models (GraphSAGE, etc.).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the TOML experiment config file.",
    )


def _build_fetch_results_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore
    """Registers the 'fetch-results' subcommand with its arguments."""
    parser = subparsers.add_parser(
        "fetch-results",
        help="Download MLflow results from HF Hub and export them to CSV.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="MLflow experiment name to fetch (e.g. 'traditional_HI-Small').",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="HF model repo ID. Defaults to HF_MODEL_REPO_ID env variable.",
    )


def _resolve_model_names(raw_names: list[str]) -> list[str]:
    """Expands 'all' into the full model registry list."""
    if "all" in raw_names:
        return ALL_MODEL_NAMES
    for name in raw_names:
        if name not in ALL_MODEL_NAMES:
            supported = ", ".join(ALL_MODEL_NAMES)
            raise ValueError(f"Unknown model '{name}'. Supported: {supported}")
    return raw_names


def _run_traditional(args: argparse.Namespace) -> None:
    """Handles execution of the traditional ML pipeline."""
    model_names = _resolve_model_names(args.models)
    logger = ProjectLogger.get_logger("CLI")
    logger.info(
        "Launching traditional pipeline — config: %s, models: %s",
        args.config,
        model_names,
    )
    pipeline = TraditionalPipeline(config_path=args.config)
    pipeline.run(requested_models=model_names)


def _run_gnn(args: argparse.Namespace) -> None:
    """Handles execution of the GNN pipeline."""
    logger = ProjectLogger.get_logger("CLI")
    logger.info("Launching GNN pipeline — config: %s", args.config)
    pipeline = GNNPipeline(config_path=args.config)
    pipeline.run()


def _run_fetch_results(args: argparse.Namespace) -> None:
    """Downloads the MLflow archive from HF Hub and exports runs to CSV."""
    logger = ProjectLogger.get_logger("CLI")
    logger.info("Fetching results for experiment: %s", args.experiment)
    exporter = ResultsExporter()
    csv_path = exporter.fetch_and_export(
        experiment_name=args.experiment,
        hf_repo_id=args.repo,
    )
    logger.info("Done. CSV available at: %s", csv_path)


def main() -> None:
    """Parses CLI arguments and dispatches to the appropriate pipeline."""
    ProjectLogger.initialize()

    parser = argparse.ArgumentParser(
        prog="fraud-detect",
        description="Fraud Detection: Traditional ML vs GNN comparison toolkit.",
    )
    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    _build_traditional_parser(subparsers)
    _build_gnn_parser(subparsers)
    _build_fetch_results_parser(subparsers)

    args = parser.parse_args()

    dispatch = {
        "traditional": _run_traditional,
        "gnn": _run_gnn,
        "fetch-results": _run_fetch_results,
    }

    handler = dispatch.get(args.pipeline)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
