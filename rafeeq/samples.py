"""Helper for reading the agreed sample JSON files (contracts/samples/*.json).

Use it while another member's part is not ready yet, and in your own tests:

    from rafeeq.samples import load_sample
    chunks = load_sample("search_material")["chunks"]
"""
import json
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "contracts" / "samples"


def load_sample(name: str) -> dict:
    """Return the sample called `name` (file name without .json) as a Python dict."""
    path = SAMPLES_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))
