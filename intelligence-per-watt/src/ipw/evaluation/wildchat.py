from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler

LOGGER = logging.getLogger(__name__)


@EvaluationRegistry.register("wildchat")
@EvaluationRegistry.register("chat")  # Alias used in IPW dataset
class WildChatHandler(EvaluationHandler):
    """LLM-based semantic equivalence check for free-form chat responses."""

    evaluation_method = "wildchat"

    SYSTEM_PROMPT = """You are an impartial judge evaluating the quality of two AI-assistant replies to the same user prompt.

Step 1 – Generate your own answer
Write the response *you* would give to the user. Keep it separate from later analysis.

Step 2 – Decide the query type
Classify the user prompt as either
• **Subjective / open-ended** (creative writing, opinion, advice, brainstorming)
• **Objective / technical** (code, math, logical derivations with a single correct outcome)
If uncertain, default to "Subjective".

Step 3 – Score each assistant with the correct rubric

| Query type | Criteria |
|------------|----------|
| Subjective / open-ended | 1. Correctness / factual soundness 2. Helpfulness 3. Relevance 4. Conciseness 5. Creativity & novelty |
| Objective / technical   | 1. Correctness only |

When using the multi-criteria rubric, note strengths and weaknesses for **each** dimension.
When using the single-criterion rubric, focus exclusively on factual / functional accuracy and ignore style or flair.

Step 4 – Compare & justify
Explain which assistant is better and why, correcting any mistakes you find. Highlight missing but important details. **Be concise.**

Step 5 – Verdict
1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, Assistant A is equal: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Choose exactly one token from: `[[A>>B]]`, `[[A>B]]`, `[[A=B]]`, `[[B>A]]`, `[[B>>A]]`.

---

### Output format (strict)
Return **only** a JSON object that matches the provided schema:

<Your Response To The User Prompt>

```json
{
"query_type": "<query type>",
"explanation": "<multi-criteria explanation> | <single-criteria explanation> (if query_type is \"Objective / technical\")",
"verdict": "<one verdict token from: [[A>>B]], [[A>B]], [[A=B]], [[B>A]], [[B>>A]]>"
}
```"""

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        if not reference.strip():
            raise RuntimeError("WildChatHandler requires a non-empty reference answer.")

        # Perform two comparisons: (model_answer vs reference) and (reference vs model_answer).
        # This accounts for possible asymmetry in the evaluation. Self-comparison is not a concern,
        # as this function is always called with distinct candidate and reference answers.
        verdict1, response1 = self._get_judge_verdict(problem, model_answer, reference)
        verdict2, response2 = self._get_judge_verdict(problem, reference, model_answer)

        if verdict1 is None or verdict2 is None:
            return None, {
                "reason": "missing_verdicts",
                "verdict1": verdict1,
                "verdict2": verdict2,
            }

        result1 = self._verdict_to_bool(verdict1, generated_is_a=True)
        result2 = self._verdict_to_bool(verdict2, generated_is_a=False)

        if result1 is None or result2 is None:
            meta = {
                "generated_as_a": {"verdict": verdict1, "response": response1},
                "generated_as_b": {"verdict": verdict2, "response": response2},
            }
            return None, meta
        else:
            final_result = result1 or result2
            meta = {
                "generated_as_a": {"verdict": verdict1, "response": response1},
                "generated_as_b": {"verdict": verdict2, "response": response2},
            }
            return final_result, meta
    def _get_judge_verdict(
        self, problem: str, response_a: str, response_b: str
    ) -> Tuple[Optional[str], Optional[str]]:
        if not hasattr(self._client, "chat"):
            raise RuntimeError(
                "WildChatHandler requires a client with a .chat() helper (e.g. OpenAIClient)."
            )

        prompt = (
            f"<|User Prompt|>\n{problem}\n\n"
            f"<|The Start of Assistant A's Answer|>\n{response_a}\n"
            f"<|The End of Assistant A's Answer|>\n\n"
            f"<|The Start of Assistant B's Answer|>\n{response_b}\n"
            f"<|The End of Assistant B's Answer|>"
        )

        raw = self._client.chat(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.0,
            max_output_tokens=1024,
        )
        content = raw.strip()

        verdict_match = re.search(r"\[\[([AB][><=]{1,2}[AB])\]\]", content)
        if verdict_match:
            return verdict_match.group(1), content

        return None, content

    def _verdict_to_bool(self, verdict: Optional[str], generated_is_a: bool) -> Optional[bool]:
        if not verdict:
            return None

        verdict_map_a = {
            "A>>B": True,
            "A>B": True,
            "A=B": True,
            "B>A": False,
            "B>>A": False,
        }

        verdict_map_b = {
            "A>>B": False,
            "A>B": False,
            "A=B": True,
            "B>A": True,
            "B>>A": True,
        }

        verdict_map = verdict_map_a if generated_is_a else verdict_map_b
        return verdict_map.get(verdict)
