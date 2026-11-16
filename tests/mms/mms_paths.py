"""Repository locations and shared test support for directly executable MMS tools."""
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_SUPPORT = REPOSITORY_ROOT/'tests'
if str(TEST_SUPPORT) not in sys.path:
    sys.path.insert(0, str(TEST_SUPPORT))
