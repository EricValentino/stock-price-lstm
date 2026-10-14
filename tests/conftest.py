import sys
from pathlib import Path

# The scripts in src/ import each other as top-level modules (python src/train.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
