from __future__ import annotations

import re
from typing import Optional


def normalize_math_answer(text: Optional[str]) -> Optional[str]:
    """Best-effort normalization for simple mathematical expressions."""

    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()
    if not s:
        return None

    # Remove LaTeX math delimiters
    s = s.replace("$", "")

    # Extract content inside the last \boxed{...} if present
    boxed_matches = list(re.finditer(r"\\boxed\{([^}]*)\}", s))
    if boxed_matches:
        s = boxed_matches[-1].group(1)

    # Remove thin spaces / !, etc.
    s = re.sub(r"\\[!,]", "", s)

    # Replace \frac{a}{b} with a/b for simple integer fractions
    s = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", s)

    # Replace \sqrt{n} with √n for small integers
    s = re.sub(r"\\sqrt\{(\d+)\}", r"√\1", s)

    # Remove thousands separators
    s = s.replace(",", "")

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s or None
