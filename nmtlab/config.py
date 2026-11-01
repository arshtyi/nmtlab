import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = tomllib.loads((ROOT / "config" / "experiment.toml").read_text())
CACHE = ROOT / ".cache"
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "manifest.json"
EXAMPLES = (
    "The students are learning how neural networks translate English into Chinese.",
    "Although the train was delayed by heavy rain, we arrived in time for the meeting.",
    "The scientist who studied memory explained why people sometimes forget familiar names.",
)
