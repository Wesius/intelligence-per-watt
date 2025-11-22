from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .mcq import BaseMCQHandler


@EvaluationRegistry.register("mmlu-pro")
class MMLUProHandler(BaseMCQHandler):
    """Evaluation for MMLU-Pro multiple-choice tasks."""

    evaluation_method = "mmlu-pro"

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        # For MMLU-Pro, the reference answer is already a letter (A/B/C/...).
        return self._evaluate_mcq(
            problem=problem,
            reference_letter=reference,
            model_answer=model_answer,
            metadata=metadata,
        )
