"""
QA framework for running different scenarios.
"""

from dpprivqa.qa.scenarios import (
    run_local_only,
    run_local_with_cot,
    run_remote_only,
    run_local_with_selective_cot
)
from dpprivqa.qa.dpprivqa import run_dpprivqa

__all__ = ['run_local_only', 'run_local_with_cot', 'run_remote_only', 'run_local_with_selective_cot', 'run_dpprivqa']

