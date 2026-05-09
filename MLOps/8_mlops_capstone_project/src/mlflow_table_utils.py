"""Shared MLflow table logging helpers for capstone steps."""

from __future__ import annotations

import json
import logging
import os

import pandas as pd
import mlflow


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def markdown_table(df: pd.DataFrame) -> str:
    """Build a simple markdown table without optional dependencies."""
    if df.empty:
        return "(empty table)"
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.fillna("").astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def log_table_with_markdown(df: pd.DataFrame, json_artifact: str, md_artifact: str) -> None:
    """Log a table to MLflow.

    By default we always log JSON (for MLflow table UI), and markdown is optional
    via environment variable `MLOPS_LOG_TABLE_MARKDOWN=1`.
    """
    if df.empty:
        return

    mlflow.log_table(df, artifact_file=json_artifact)

    if _env_bool("MLOPS_LOG_TABLE_MARKDOWN", False):
        mlflow.log_text(markdown_table(df), md_artifact)


def log_decision_json(decision: dict, artifact_file: str) -> None:
    """Log decision dictionaries in a consistent JSON format."""
    mlflow.log_text(json.dumps(decision, indent=2), artifact_file)


def log_step_header(logger: logging.Logger, title: str) -> None:
    """Log a consistent step banner across all pipeline modules."""
    logger.info("=" * 60)
    logger.info(title)


def bool_str(value: bool) -> str:
    """Convert bool-like values to lowercase MLflow-friendly strings."""
    return str(bool(value)).lower()


def set_bool_tag(name: str, value: bool) -> None:
    """Set a boolean MLflow tag with normalized lowercase string value."""
    mlflow.set_tag(name, bool_str(value))


def log_bool_param(name: str, value: bool) -> None:
    """Log a boolean MLflow param with normalized lowercase string value."""
    mlflow.log_param(name, bool_str(value))


def log_bool_param_and_tag(name: str, value: bool) -> None:
    """Log a boolean value to both MLflow params and tags consistently."""
    log_bool_param(name, value)
    set_bool_tag(name, value)
