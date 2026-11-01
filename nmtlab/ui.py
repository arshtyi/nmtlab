import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import TypedDict

import gradio as gr

from nmtlab.config import CONFIG, EXAMPLES, RESULTS
from nmtlab.engine import Translation, Translator
from nmtlab.evaluation import evaluate
from nmtlab.plots import attention_figure


class ModelEvent(TypedDict):
    concurrency_id: str
    concurrency_limit: int


LAUNCH_OPTIONS = {
    "theme": gr.themes.Base(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        font=("Lato", "Noto Sans", "Noto Sans CJK SC", "sans-serif"),
        font_mono=("monospace",),
    ),
    "css": """
        .gradio-container { width: 100% !important; max-width: 1180px !important; margin: auto; }
        #project-heading { margin-bottom: 0.4rem; }
        .snapshot-note { color: var(--body-text-color-subdued); }
    """,
    "footer_links": [],
}

DECODING = [("Greedy" if beams == 1 else f"Beam search · {beams} beams", beams) for beams in CONFIG["decoding"]["beams"]]
METRIC_HEADERS = ["Group", "Decoding", "Samples", "BLEU", "chrF", "Mean ms", "Median ms", "P95 ms", "Capped"]
CASE_LABELS = {
    "highest_average_chrf": "Highest average chrF",
    "lowest_average_chrf": "Lowest average chrF",
    "largest_positive_delta": "Largest beam gain in chrF",
    "largest_negative_delta": "Largest beam loss in chrF",
}
RESULT_FILES = ("summary.json", "samples.jsonl", "cases.json", "metrics.csv")
EVALUATION_PROGRESS = gr.Progress()


def _timing(result: Translation) -> str:
    label = "Greedy" if result.beams == 1 else f"Beam search ({result.beams} beams)"
    capped = " · Output reached the token cap; it may be incomplete." if result.capped else ""
    return f"{label} · {result.elapsed_ms:,.1f} ms{capped}"


def _rounded(value):
    return None if value is None else round(value, 2)


def _saved_results(directory: Path, formal: bool = True) -> tuple:
    label = "Formal evaluation" if formal else "Separate UI evaluation"
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return (
            "No saved evaluation. Run the fixed experiment below.",
            [],
            "",
            None,
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"**Saved evaluation could not be read:** {error}", [], "", None
    saved_config = summary["metadata"]["config"]
    saved_beams = {"greedy" if beams == 1 else "beam": beams for beams in saved_config["decoding"]["beams"]}
    metrics = [
        [
            record.get("length_group", "Overall"),
            "Greedy" if record["decoding"] == "greedy" else f"Beam ({record.get('num_beams', saved_beams['beam'])})",
            record["n"],
            *[_rounded(record[key]) for key in ("bleu", "chrf", "mean_ms", "median_ms", "p95_ms")],
            record["capped"],
        ]
        for record in [*summary["overall"], *summary["by_length"]]
    ]
    case_rows = []
    for case in cases:
        title = f"{CASE_LABELS[case['category']]} · {case['id']}"
        entries = (
            ("English", case["source"]),
            ("Reference", case["reference"]),
            ("Greedy", case["greedy"]["text"]),
            (f"Beam ({case['beam'].get('num_beams', saved_beams['beam'])})", case["beam"]["text"]),
        )
        body = "".join(f"<p><strong>{label}</strong><br>{escape(text)}</p>" for label, text in entries)
        scores = f"chrF: {case['greedy']['chrf']:.2f} / {case['beam']['chrf']:.2f}"
        case_rows.append(f"<details><summary>{escape(title)} · {scores}</summary>{body}</details>")
    downloads = [str(directory / name) for name in RESULT_FILES if (directory / name).is_file()]
    status = f"**{label}** · `{directory.relative_to(RESULTS.parent)}`\n\n"
    status += f"{saved_config['model']['id'].rsplit('/', 1)[-1]} · FLORES-200 devtest · 200 fixed English-Chinese pairs."
    return status, metrics, "".join(case_rows), downloads


def create_app(translator: Translator) -> gr.Blocks:
    saved_status, saved_metrics, saved_cases, saved_downloads = _saved_results(RESULTS)
    layers = len(translator.model.decoder.block)
    heads = translator.model.config.num_heads
    beam_count = next(beams for beams in CONFIG["decoding"]["beams"] if beams > 1)
    model_event: ModelEvent = {"concurrency_id": "model", "concurrency_limit": 1}

    def translate(text, beams):
        try:
            result = translator.translate(text, beams=int(beams))
        except ValueError as error:
            raise gr.Error(str(error)) from error
        return result.text, _timing(result)

    def compare(text):
        try:
            greedy = translator.translate(text, beams=1)
            beam = translator.translate(text, beams=beam_count)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        return greedy.text, _timing(greedy), beam.text, _timing(beam)

    def capture_attention(text, beams, layer, head, show_special):
        try:
            result = translator.translate(text, beams=int(beams))
            attention = translator.attention(result)
            figure = attention_figure(attention, layer=int(layer), head=int(head), show_special=show_special)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        snapshot = {"result": result, "attention": attention}
        return snapshot, result.source, result.text, _timing(result), figure

    def redraw_attention(snapshot, layer, head, show_special):
        if snapshot is None:
            return None
        try:
            return attention_figure(snapshot["attention"], layer=int(layer), head=int(head), show_special=show_special)
        except ValueError as error:
            raise gr.Error(str(error)) from error

    def rerun_evaluation(progress=EVALUATION_PROGRESS):
        directory = RESULTS / f"ui-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        progress(0, desc="Warming up and evaluating the fixed manifest")
        try:
            evaluate(
                translator,
                output=directory,
                progress=lambda completed, total: progress((completed, total), desc=f"Evaluating sample {completed}/{total}"),
            )
        except (FileNotFoundError, ValueError) as error:
            raise gr.Error(str(error)) from error
        return _saved_results(directory, formal=False)

    with gr.Blocks(title="NMT Lab · English to Chinese", analytics_enabled=False) as app:
        gr.Markdown("# NMT Lab", elem_id="project-heading")
        gr.Markdown(
            f"English to Chinese · {CONFIG['model']['id'].rsplit('/', 1)[-1]} · {translator.device.upper()} · "
            f"{CONFIG['model']['dtype']} · {CONFIG['runtime']['threads']} CPU threads"
        )
        with gr.Tab("Translate"):
            source = gr.Textbox(
                label="English input",
                placeholder="Enter an English sentence or short paragraph.",
                info="Maximum 512 model tokens, including language and end tokens; longer input is rejected.",
                lines=4,
            )
            gr.Examples(examples=[[example] for example in EXAMPLES], inputs=source, label="Try an example")
            with gr.Row():
                decoding = gr.Dropdown(DECODING, value=CONFIG["decoding"]["default_beams"], label="Decoding", filterable=False)
                translate_button = gr.Button("Translate", variant="primary")
            translated = gr.Textbox(label="Chinese translation", lines=4, interactive=False)
            translation_timing = gr.Markdown()
            gr.Markdown("Timing: tokenize + transfer + generate + decode; excludes loading, warmup, and attention replay.")
            translate_button.click(translate, [source, decoding], [translated, translation_timing], **model_event)

        with gr.Tab("Compare"):
            comparison_source = gr.Textbox(label="English input", value=EXAMPLES[0], lines=4)
            compare_button = gr.Button("Compare greedy and beam", variant="primary")
            with gr.Row():
                with gr.Column():
                    greedy_text = gr.Textbox(label="Greedy translation", lines=5, interactive=False)
                    greedy_timing = gr.Markdown()
                with gr.Column():
                    beam_text = gr.Textbox(label=f"Beam search translation · {beam_count} beams", lines=5, interactive=False)
                    beam_timing = gr.Markdown()
            gr.Markdown("Single runs, greedy first; use Evaluation for repeated latency measurements.")
            compare_button.click(
                compare,
                comparison_source,
                [greedy_text, greedy_timing, beam_text, beam_timing],
                **model_event,
            )

        with gr.Tab("Attention"):
            attention_state = gr.State(value=None)
            attention_source = gr.Textbox(label="English input for a new snapshot", value=EXAMPLES[0], lines=3)
            with gr.Row():
                attention_decoding = gr.Dropdown(
                    DECODING, value=CONFIG["decoding"]["default_beams"], label="Decoding for a new snapshot", filterable=False
                )
                attention_button = gr.Button("Translate and inspect attention", variant="primary")
            gr.Markdown("Plot stays tied to the captured input.", elem_classes="snapshot-note")
            with gr.Row():
                snapshot_source = gr.Textbox(label="Snapshot · English source", lines=3, interactive=False)
                snapshot_translation = gr.Textbox(label="Snapshot · Chinese translation", lines=3, interactive=False)
            snapshot_timing = gr.Markdown("No attention snapshot yet.")
            with gr.Row():
                layer = gr.Dropdown(list(range(1, layers + 1)), value=layers, label="Decoder layer", filterable=False)
                head = gr.Dropdown(
                    [("Mean of all heads", 0), *[(f"Head {index}", index) for index in range(1, heads + 1)]],
                    value=0,
                    label="Attention head",
                    filterable=False,
                )
                show_special = gr.Checkbox(value=False, label="Show special tokens")
            attention_plot = gr.Plot(label="Decoder cross-attention", show_label=False, format="png")
            gr.Markdown(
                "Cross-attention replay on generated tokens; rows: target, columns: source. Raw subwords are preserved; "
                "hidden special tokens are not renormalized. Attention is not alignment or a causal explanation."
            )
            attention_button.click(
                capture_attention,
                [attention_source, attention_decoding, layer, head, show_special],
                [attention_state, snapshot_source, snapshot_translation, snapshot_timing, attention_plot],
                **model_event,
            )
            for control in (layer, head, show_special):
                control.change(
                    redraw_attention,
                    [attention_state, layer, head, show_special],
                    attention_plot,
                    **model_event,
                )

        with gr.Tab("Evaluation"):
            evaluation_status = gr.Markdown(saved_status)
            gr.Markdown(
                "Two warmups per mode, three timed repeats per sentence, alternating mode order; latency summaries use sentence medians. "
                "Length groups count English whitespace-separated words: <=15, 16-30, >30."
            )
            metrics = gr.Dataframe(
                value=saved_metrics,
                headers=METRIC_HEADERS,
                datatype=["str", "str", "number", "number", "number", "number", "number", "number", "number"],
                label="Overall and length-group results",
                interactive=False,
                wrap=True,
            )
            gr.Markdown("BLEU / chrF: higher is better. Latency: milliseconds. Capped: outputs reaching the 256-token limit.")
            cases = gr.HTML(value=saved_cases, label="Metric-selected translation cases")
            gr.Markdown(
                "Cases rank average sentence chrF and signed beam-minus-greedy changes, not semantic correctness. "
                "Gain or loss cases are omitted when no matching sign exists."
            )
            downloads = gr.File(value=saved_downloads, file_count="multiple", label="Download evaluation artifacts", interactive=False)
            gr.Markdown(
                "Reruns occupy the model for several minutes and save to `results/ui-<timestamp>`. "
                "Tables switch to that run; restarting restores formal results."
            )
            rerun = gr.Button("Run full evaluation in a separate folder", variant="secondary")
            rerun.click(rerun_evaluation, outputs=[evaluation_status, metrics, cases, downloads], **model_event)

    return app
