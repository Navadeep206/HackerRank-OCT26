"""Suggested terminal entry point for HackerRank Orchestrate claim pipeline."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from code.run_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
