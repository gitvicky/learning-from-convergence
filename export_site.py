"""Copy only reproducible article assets to an existing website checkout."""
import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("website", type=Path)
args = parser.parse_args()
website = args.website.resolve()
if not (website / "_quarto.yml").is_file():
    parser.error("website must be an existing Quarto project")
destination = website / "Blog" / "assets" / "learning_from_convergence"
destination.mkdir(parents=True, exist_ok=True)
results = Path(__file__).parent / "results"
for name in ("noisy_expectation.svg", "noisy_expectation.png", "budget_allocation.svg",
             "budget_allocation.png", "allocation_table.md"):
    shutil.copy2(results / name, destination / name)
print(f"Copied five generated assets to {destination}")
