"""Packaging smoke test for runtime imports."""

import importlib


def test_runtime_module_imports() -> None:
    """Ensure key runtime modules import without optional extras."""
    importlib.import_module("echopress.core.rmcpe")
    importlib.import_module("echopress.cli")
