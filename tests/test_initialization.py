"""Initial environment and project structure test suite for PACE."""

import importlib
import pytest


@pytest.mark.parametrize(
    "subpackage",
    [
        "src.data",
        "src.features",
        "src.models",
        "src.simulation",
        "src.evaluation",
        "src.reporting",
        "src.utils",
    ],
)
def test_subpackages_importable(subpackage: str) -> None:
    """Validate that core source subpackages are defined and importable."""
    module = importlib.import_module(subpackage)
    assert module is not None
