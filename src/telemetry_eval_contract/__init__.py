"""Offline-only SPEC-EVAL contract.

Production runtime packages must not import this package.
"""

from .models import EVAL_CONTRACT_VERSION, EVALUATION_TABLES, EvaluationCapability
from .validate import validate_eval_bundle

__all__ = [
    "EVAL_CONTRACT_VERSION",
    "EVALUATION_TABLES",
    "EvaluationCapability",
    "validate_eval_bundle",
]
