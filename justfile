default: list

list:
    @just --list

prepare:
    uv run --locked python -m nmtlab prepare

run:
    HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False uv run --locked python -m nmtlab app

evaluate:
    HF_HUB_OFFLINE=1 uv run --locked python -m nmtlab evaluate

check:
    uv run --locked ruff check .
    uv run --locked ruff format --check .
    uv run --locked ty check
    uv run --locked python -m pytest
