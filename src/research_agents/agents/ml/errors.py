from __future__ import annotations


class LibraryImportError(Exception):
    """Raised when ml.library cannot be imported."""


class ModelFitError(Exception):
    """Raised when model fitting fails."""
