"""Runner surfaces for research benchmarks."""

from .path1_cli import cli_main as path1_cli_main
from .mini_moe import run_mini_moe_variant
from .mini_moe_autoresearch import run_mini_moe_autoresearch
from .mini_moe_policy_search import run_mini_moe_policy_search
from .path1 import run_path1_variant
