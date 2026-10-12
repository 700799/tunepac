"""Exception hierarchy for Tunepac.

Every error raised deliberately by Tunepac derives from
:class:`TunepacError`, so the CLI can catch one type, print a readable
message, and exit non-zero without a traceback. Unexpected exceptions are
allowed to propagate so bugs stay visible.
"""

from __future__ import annotations


class TunepacError(Exception):
    """Base class for all errors Tunepac raises on purpose."""


class ConfigError(TunepacError):
    """The YAML config file is missing, unparsable, or fails schema validation."""


class DatasetError(TunepacError):
    """The dataset cannot be read or does not match the declared format."""


class ChatTemplateError(DatasetError):
    """The dataset format and the tokenizer/chat-template combination are incompatible."""


class ExportError(TunepacError):
    """A model/adapter export could not be completed."""


class OOMRiskError(TunepacError):
    """Training aborted because the GPU ran out of memory (or is about to)."""

    def __init__(self, message: str, *, hints: list[str] | None = None) -> None:
        self.hints = hints or []
        if self.hints:
            bullet_list = "\n".join(f"  - {hint}" for hint in self.hints)
            message = f"{message}\nSuggestions:\n{bullet_list}"
        super().__init__(message)
