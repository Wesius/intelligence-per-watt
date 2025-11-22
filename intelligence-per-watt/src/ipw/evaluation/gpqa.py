from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .mcq import BaseMCQHandler


@EvaluationRegistry.register("supergpqa")
class SuperGPQAHandler(BaseMCQHandler):
    """Evaluation for SuperGPQA tasks."""

    evaluation_method = "supergpqa"

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        # SuperGPQA stores the correct letter as answer_letter in the original
        # dataset; in the mixed IPW dataset, reference is already that letter.
        return self._evaluate_mcq(
            problem=problem,
            reference_letter=reference,
            model_answer=model_answer,
            metadata=metadata,
        )


@EvaluationRegistry.register("gpqa")
class GPQAHandler(BaseMCQHandler):
    """Evaluation for GPQA tasks."""

    evaluation_method = "gpqa"

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        # GPQA normalizes all correct answers to letter "A" in the base dataset.
        # The IPW "answer" field should therefore be the gold letter already.
        return self._evaluate_mcq(
            problem=problem,
            reference_letter=reference,
            model_answer=model_answer,
            metadata=metadata,
        )
