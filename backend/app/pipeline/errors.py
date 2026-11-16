"""User-facing pipeline errors. These are terminal: retrying the same file will not help."""
from __future__ import annotations


class PipelineError(Exception):
    def __init__(self, code: str, message: str, suggestion: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion
