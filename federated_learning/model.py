"""Backward-compatible model import.

Prefer importing TBML_DetectionModel from tbml.models in new code.
"""

from tbml.models import TBML_DetectionModel

__all__ = ["TBML_DetectionModel"]