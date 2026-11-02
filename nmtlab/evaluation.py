import csv
import importlib.metadata
import json
import math
import os
import platform
import statistics
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sacrebleu.metrics import BLEU, CHRF

from nmtlab.config import CONFIG, MANIFEST, RESULTS, ROOT
from nmtlab.data import LENGTH_GROUPS, load_samples, sha256_file

if TYPE_CHECKING:
    from nmtlab.engine import Translator

DECODINGS = tuple(("greedy" if beams == 1 else "beam", beams) for beams in sorted(CONFIG["decoding"]["beams"]))
METRIC_FIELDS = ("decoding", "num_beams", "n", "bleu", "chrf", "mean_ms", "median_ms", "p95_ms", "capped")


def _metrics() -> tuple[BLEU, CHRF]:
    return BLEU(**CONFIG["metrics"]["bleu"]), CHRF(**CONFIG["metrics"]["chrf"])


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def aggregate_metrics(samples: list[dict], decoding: str, bleu: BLEU, chrf: CHRF) -> dict:
    """Score each corpus jointly; aggregate the median latency of each sample."""
    if decoding not in {name for name, _ in DECODINGS}:
        raise ValueError(f"Unknown decoding mode: {decoding}")
    record = {"decoding": decoding, "num_beams": dict(DECODINGS)[decoding], "n": len(samples)}
    if not samples:
        return {**record, **dict.fromkeys(METRIC_FIELDS[3:])}
    predictions = [sample[decoding]["text"] for sample in samples]
    references = [[sample["reference"] for sample in samples]]
    times = []
    for sample in samples:
        repeats = sample[decoding]["elapsed_ms"]
        if not repeats or any(not math.isfinite(elapsed) or elapsed < 0 for elapsed in repeats):
            raise ValueError("Latency measurements must be finite and nonnegative.")
        times.append(statistics.median(repeats))
    return {
        **record,
        "bleu": bleu.corpus_score(predictions, references).score,
        "chrf": chrf.corpus_score(predictions, references).score,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": _percentile(times, 0.95),
        "capped": sum(bool(sample[decoding]["capped"]) for sample in samples),
    }


def select_cases(samples: list[dict]) -> list[dict]:
    """Select metric-defined examples, with sample id breaking all score ties."""
    if not samples:
        return []
    scored = [
        {
            **sample,
            "average_chrf": (sample["greedy"]["chrf"] + sample["beam"]["chrf"]) / 2,
            "delta_chrf": sample["beam"]["chrf"] - sample["greedy"]["chrf"],
        }
        for sample in samples
    ]
    highest = min(scored, key=lambda sample: (-sample["average_chrf"], sample["id"]))
    lowest = min(scored, key=lambda sample: (sample["average_chrf"], sample["id"]))
    cases = [
        {**highest, "category": "highest_average_chrf"},
        {**lowest, "category": "lowest_average_chrf"},
    ]
    positive = [sample for sample in scored if sample["delta_chrf"] > 0]
    negative = [sample for sample in scored if sample["delta_chrf"] < 0]
    if positive:
        best = min(positive, key=lambda sample: (-sample["delta_chrf"], sample["id"]))
        cases.append({**best, "category": "largest_positive_delta"})
    if negative:
        worst = min(negative, key=lambda sample: (sample["delta_chrf"], sample["id"]))
        cases.append({**worst, "category": "largest_negative_delta"})
    return cases


def _measure_sample(translator: "Translator", sample: dict, position: int, chrf: CHRF) -> dict:
    row = dict(sample)
    results = {name: [] for name, _ in DECODINGS}
    for repeat in range(CONFIG["measurement"]["repeats"]):
        order = DECODINGS if (position + repeat) % 2 == 0 else tuple(reversed(DECODINGS))
        for name, beams in order:
            result = translator.translate(sample["source"], beams=beams)
            if result.source != sample["source"].strip() or result.beams != beams:
                raise ValueError(f"Translation does not match sample {sample['id']} and decoding {name}.")
            if not math.isfinite(result.elapsed_ms) or result.elapsed_ms < 0:
                raise ValueError(f"Invalid latency for sample {sample['id']} ({name}).")
            previous = results[name]
            if previous:
                first = previous[0]
                if (result.text, result.source_ids, result.output_ids, result.capped) != (
                    first.text,
                    first.source_ids,
                    first.output_ids,
                    first.capped,
                ):
                    raise ValueError(f"Nondeterministic output for sample {sample['id']} ({name}).")
            previous.append(result)
    for name, _ in DECODINGS:
        measured = results[name]
        first = measured[0]
        times = [result.elapsed_ms for result in measured]
        row[name] = {
            "num_beams": first.beams,
            "text": first.text,
            "elapsed_ms": times,
            "median_ms": statistics.median(times),
            "chrf": chrf.sentence_score(first.text, [sample["reference"]]).score,
            "capped": first.capped,
        }
    return row


def _metadata(translator: "Translator", samples: list[dict], manifest_hash: str, lock_hash: str) -> dict:
    packages = ("nmtlab", "torch", "transformers", "sentencepiece", "sacrebleu", "huggingface-hub", "numpy", "gradio")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "config": CONFIG,
        "generation_configs": {str(beams): generation.to_dict() for beams, generation in translator.generations.items()},
        "manifest_sha256": manifest_hash,
        "uv_lock_sha256": lock_hash,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in [ROOT / "config" / "experiment.toml", *sorted((ROOT / "nmtlab").glob("*.py"))]
        },
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "runtime": {
            "device": translator.device,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "dtype": str(translator.model.dtype),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "sample_ids": [sample["id"] for sample in samples],
        "timing": {
            "scope": CONFIG["measurement"]["scope"],
            "unit": "milliseconds",
            "repeats": CONFIG["measurement"]["repeats"],
            "warmup_per_mode": CONFIG["measurement"]["warmup"],
            "order": "Greedy first when (zero-based sample position + repetition) is even; beam first otherwise.",
            "aggregation": "Mean, median, and p95 across per-sample median latencies; all three raw repetitions retained.",
            "p95": "Linear interpolation at sorted index (n - 1) * 0.95.",
        },
        "length_group_definition": "Whitespace-separated source words: <=15, 16-30, >30; language prefix excluded.",
        "capped_definition": "Generated length reaches max_new_tokens, including a possible forced EOS; conservative flag.",
        "case_selection": "Highest/lowest mean sentence chrF, positive/negative beam-minus-greedy chrF extremes; ties by id.",
    }


def _save(output: Path, summary: dict, samples: list[dict], cases: list[dict]) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".evaluation-") as temporary:
        staging = Path(temporary)
        for name, value in (("summary.json", summary), ("cases.json", cases)):
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=4, allow_nan=False) + "\n", encoding="utf-8")
        with (staging / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
        with (staging / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("length_group", *METRIC_FIELDS))
            writer.writeheader()
            for record in summary["overall"]:
                writer.writerow({"length_group": "all", **record})
            writer.writerows(summary["by_length"])
        output.mkdir(parents=True, exist_ok=True)
        # Publish only complete files after successful inference and serialization, with summary last.
        for name in ("samples.jsonl", "cases.json", "metrics.csv", "summary.json"):
            os.replace(staging / name, output / name)


def evaluate(translator: "Translator", output: Path = RESULTS, progress: Callable[[int, int], None] | None = None) -> dict:
    """Run the paired frozen experiment and publish all artifacts after it succeeds."""
    beams = CONFIG["decoding"]["beams"]
    if len(beams) != 2 or any(type(value) is not int for value in beams) or min(beams) != 1 or max(beams) <= 1:
        raise ValueError("The frozen experiment requires exactly one greedy mode and one beam mode with num_beams > 1.")
    if CONFIG["measurement"]["repeats"] != 3:
        raise ValueError("The frozen experiment requires three repetitions.")
    manifest_hash = sha256_file(MANIFEST)
    lock_hash = sha256_file(ROOT / "uv.lock")
    samples = load_samples()
    if len(samples) != 200:
        raise ValueError("The frozen evaluation requires exactly 200 aligned samples.")
    bleu, chrf = _metrics()
    translator.warmup()
    rows = []
    if progress:
        progress(0, len(samples))
    for position, sample in enumerate(samples):
        rows.append(_measure_sample(translator, sample, position, chrf))
        if progress:
            progress(position + 1, len(samples))
    overall = [aggregate_metrics(rows, name, bleu, chrf) for name, _ in DECODINGS]
    by_length = [
        {"length_group": group, **aggregate_metrics([row for row in rows if row["length_group"] == group], name, bleu, chrf)}
        for group in LENGTH_GROUPS
        for name, _ in DECODINGS
    ]
    if sha256_file(MANIFEST) != manifest_hash or sha256_file(ROOT / "uv.lock") != lock_hash:
        raise ValueError("The manifest or dependency lock changed during evaluation; results were not saved.")
    summary = {
        "schema_version": 1,
        "metadata": _metadata(translator, samples, manifest_hash, lock_hash),
        "signatures": {"bleu": str(bleu.get_signature()), "chrf": str(chrf.get_signature())},
        "overall": overall,
        "by_length": by_length,
    }
    _save(output, summary, rows, select_cases(rows))
    return summary
