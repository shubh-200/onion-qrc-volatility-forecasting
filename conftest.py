"""Root conftest.py — adds the repo root to sys.path so 'gic' is importable."""
import sys
from pathlib import Path

# The repo root is one level above this conftest (gic_new/).
# We need c:\Users\Shubham\Desktop\gic_new\ on sys.path so that
# `import gic` resolves to c:\Users\Shubham\Desktop\gic_new\gic\.
_REPO_PARENT = Path(__file__).resolve().parent.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))
