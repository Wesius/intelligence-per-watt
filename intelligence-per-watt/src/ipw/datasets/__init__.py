"""Dataset implementations bundled with Intelligence Per Watt.

Datasets register themselves with ``ipw.core.DatasetRegistry``.
"""

from .base import DatasetProvider


def ensure_registered() -> None:
    """Import built-in dataset providers to populate the registry."""
    from . import (  # noqa: F401
        ipw,
        ipw_pro,
        mmlu_pro,
        supergpqa,
    )


__all__ = ["DatasetProvider", "ensure_registered"]
