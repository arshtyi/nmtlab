# nmtlab

Local English-to-Chinese neural machine translation and visual analysis for a single-user course project on macOS with Apple Silicon.

Built with PyTorch, Transformers, Gradio, and pretrained [google/madlad400-3b-mt](https://huggingface.co/google/madlad400-3b-mt). No training or fine-tuning is performed. Project work covers inference integration, controlled decoding comparisons, attention visualization, and reproducible evaluation—not new translation algorithms.

## Features

- Translate: English input, Chinese output, independent demo examples, and inference latency.
- Compare: side-by-side greedy and beam-search translations with individual timings.
- Attention: real decoder cross-attention with layer, head, and special-token controls.
- Evaluation: fixed-sample BLEU and chrF, source-length groups, metric-selected cases, and downloadable JSON / JSONL / CSV results.

## Usage

```sh
uv sync --locked
just prepare
just run
```

Open the local URL printed in the terminal.

Inference defaults to beam search with width 8 on MPS, using bfloat16 weights and four compute threads, as configured in `config/experiment.toml`.

Inputs exceeding 512 model tokens, including language-control tokens and EOS, are rejected rather than silently truncated. Outputs reaching the 256-token cap are flagged as potentially incomplete.

## Evaluation

```sh
just evaluate
```

The protocol is defined in [`config/experiment.toml`](config/experiment.toml), with sample IDs and dataset checksums in [`data/manifest.json`](data/manifest.json):

- Model revision `fa184c675da0b5c9e1c8694fccd4e12e2d422094`; source prefix `<2zh>` selects Chinese; bfloat16 and eager attention.
- FLORES-200 `eng_Latn` → `zho_Hans`, 200 devtest pairs sampled without replacement using seed 42. IDs are one-based line numbers, fixed before inspecting translation results. Demo and warmup sentences are separate.
- Greedy versus beam=8; batch size 1, no sampling, length penalty 1, `early_stopping=false`, and at most 256 new tokens.
- Corpus BLEU with `tokenize=zh`, mixed case, exponential smoothing, and effective order disabled. chrF uses character order 6, word order 0, β=2, mixed case, no whitespace, and no epsilon smoothing. Metric signatures are saved.
- Source-length groups use whitespace-separated English words: ≤15, 16–30, and >30. Each group is scored as a corpus, not by averaging sentence scores.
- Two warmups per mode, then three measurements per sentence and mode with alternating execution order. Latency summaries aggregate per-sentence medians.

Timing includes tokenization, device transfer, generation, and text decoding. It excludes loading, warmup, attention replay, plotting, and UI rendering. MPS is synchronized at both timing boundaries. The Compare tab shows single runs, greedy first; use evaluation results for repeated measurements.

Outputs are saved under `results/`:

| File            | Contents                                                                                                              |
| --------------- | --------------------------------------------------------------------------------------------------------------------- |
| `summary.json`  | Overall and grouped metrics, signatures, resolved generation settings, versions, hashes, and timing protocol          |
| `samples.jsonl` | Every sample ID, source, reference, both translations, three raw timings per mode, sentence chrF, and cap flags       |
| `cases.json`    | Highest/lowest average sentence chrF and largest positive/negative beam-minus-greedy differences; ties resolved by ID |
| `metrics.csv`   | Overall and grouped metrics in tabular form                                                                           |

`just evaluate` replaces the formal result files on success. The UI's evaluation button instead writes to a separate `results/ui-<timestamp>/` directory. Reloading the page restores the formal results.

## Attention interpretation

Attention is recomputed on the exact generated sequence after translation timing ends. Decoder inputs use the sequence without its final token; row labels use the sequence without the decoder-start token. This keeps each row associated with the token being predicted, including for the selected beam-search output.

Raw SentencePiece subwords are retained. Hiding control tokens does not renormalize the remaining weights; the mean-head view averages heads within one layer. The plot remains tied to its captured input until a new snapshot is generated.

Cross-attention is an internal model signal, not strict word alignment or a causal explanation.
