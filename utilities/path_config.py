"""Backward-compatible path helpers.

Prefer importing path helpers from tbml.config in new code.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'federated_learning')))
from tbml.config.paths import BANKS_ROOT, PROJECT_ROOT, get_bank_file, get_bank_path

__all__ = ["PROJECT_ROOT", "BANKS_ROOT", "get_bank_path", "get_bank_file"]