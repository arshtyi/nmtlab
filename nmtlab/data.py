import hashlib
import json
import os
import random
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

from nmtlab.config import CACHE, CONFIG, MANIFEST

MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "spiece.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
)
LENGTH_GROUPS = ("<=15", "16-30", ">30")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def length_group(source_words: int) -> str:
    if source_words < 0:
        raise ValueError("Source word count cannot be negative.")
    lower, upper = CONFIG["measurement"]["length_boundaries"]
    if source_words <= lower:
        return LENGTH_GROUPS[0]
    if source_words <= upper:
        return LENGTH_GROUPS[1]
    return LENGTH_GROUPS[2]


def _manifest() -> dict:
    manifest = json.loads(MANIFEST.read_bytes())
    data = CONFIG["data"]
    if (manifest["dataset"], manifest["archive_sha256"], manifest["split"]) != (data["name"], data["sha256"], data["split"]):
        raise ValueError("Dataset configuration does not match the frozen manifest.")
    source = manifest["files"][_member_name(data["source"])]
    target = manifest["files"][_member_name(data["target"])]
    if source["count"] != target["count"]:
        raise ValueError("Source and reference manifest counts differ.")
    expected_ids = sorted(random.Random(data["seed"]).sample(range(1, source["count"] + 1), data["size"]))
    if manifest["sample_ids"] != expected_ids:
        raise ValueError("Sample selection does not match the frozen manifest.")
    return manifest


def _member_name(language: str) -> str:
    split = CONFIG["data"]["split"]
    return f"./flores200_dataset/{split}/{language}.{split}"


def _validated_lines(content: bytes, entry: dict, label: str) -> list[str]:
    if hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA256 mismatch for {label}; run prepare to restore the dataset.")
    lines = content.decode("utf-8").splitlines()
    if len(lines) != entry["count"] or any(not line.strip() for line in lines):
        raise ValueError(f"Invalid line count or blank sentence in {label}.")
    return lines


def _select_samples(sources: list[str], references: list[str], ids: list[int]) -> list[dict]:
    if len(sources) != len(references):
        raise ValueError("Source and reference files are not aligned.")
    if not ids or ids != sorted(set(ids)) or any(type(index) is not int or not 1 <= index <= len(sources) for index in ids):
        raise ValueError("Sample ids must be unique, sorted, one-based line numbers within the dataset.")
    samples = []
    for index in ids:
        source, reference = sources[index - 1], references[index - 1]
        if not source.strip() or not reference.strip():
            raise ValueError(f"Empty source or reference for sample {index}.")
        words = len(source.split())
        samples.append({"id": index, "source": source, "reference": reference, "source_words": words, "length_group": length_group(words)})
    return samples


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare() -> None:
    """Fetch the pinned model and verify the original FLORES archive before caching two files."""
    manifest = _manifest()
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / "flores200_dataset.tar.gz"
    if not archive.exists():
        descriptor, temporary = tempfile.mkstemp(dir=CACHE, prefix=".flores-download-")
        try:
            with os.fdopen(descriptor, "wb") as destination, urllib.request.urlopen(CONFIG["data"]["url"], timeout=120) as source:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            if sha256_file(Path(temporary)) != CONFIG["data"]["sha256"]:
                raise ValueError("Downloaded FLORES archive SHA256 does not match config/experiment.toml.")
            os.replace(temporary, archive)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if sha256_file(archive) != CONFIG["data"]["sha256"]:
        raise ValueError("Cached FLORES archive SHA256 mismatch; remove the invalid archive and run prepare again.")
    selected = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for language in (CONFIG["data"]["source"], CONFIG["data"]["target"]):
            name = _member_name(language)
            member = bundle.getmember(name)
            if not member.isfile():
                raise ValueError(f"Expected a regular archive member: {name}")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read archive member: {name}")
            with stream:
                content = stream.read()
            _validated_lines(content, manifest["files"][name], name)
            selected[Path(name).name] = content
    snapshot_download(
        repo_id=CONFIG["model"]["id"],
        revision=CONFIG["model"]["revision"],
        allow_patterns=list(MODEL_FILES),
    )
    for filename, content in selected.items():
        _atomic_bytes(CACHE / "flores200" / filename, content)


def load_samples() -> list[dict]:
    """Load only the frozen sample ids, after checking both complete cached split files."""
    manifest = _manifest()
    lines = []
    for language in (CONFIG["data"]["source"], CONFIG["data"]["target"]):
        name = _member_name(language)
        path = CACHE / "flores200" / Path(name).name
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached dataset file {path}; run prepare first.")
        lines.append(_validated_lines(path.read_bytes(), manifest["files"][name], name))
    return _select_samples(lines[0], lines[1], manifest["sample_ids"])
