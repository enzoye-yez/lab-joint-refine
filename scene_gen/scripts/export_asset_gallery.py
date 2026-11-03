"""Export the existing asset images without running modeling scripts or calling the model."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from scene_generation.gallery import export

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(export(PROJECT, args.run, args.package, refresh=args.refresh))
