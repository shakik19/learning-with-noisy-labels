import sys
from pathlib import Path

# Make tests/fake_cifar.py importable from the tests.
sys.path.insert(0, str(Path(__file__).parent))
