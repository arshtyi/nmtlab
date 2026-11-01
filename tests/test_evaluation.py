import hashlib
import statistics

import pytest
from sacrebleu.metrics import BLEU, CHRF

from nmtlab.data import _select_samples, _validated_lines, length_group
from nmtlab.evaluation import aggregate_metrics, select_cases


@pytest.mark.parametrize(
    ("words", "expected"),
    [(0, "<=15"), (1, "<=15"), (15, "<=15"), (16, "16-30"), (30, "16-30"), (31, ">30"), (100, ">30")],
)
def test_source_length_boundaries(words, expected):
    assert length_group(words) == expected


def test_negative_source_length_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        length_group(-1)


def test_manifest_ids_select_aligned_one_based_rows():
    sources = ["first source", " ".join(["word"] * 16), "last source"]
    references = ["第一句", "第二句", "第三句"]
    rows = _select_samples(sources, references, [2, 3])
    assert [row["id"] for row in rows] == [2, 3]
    assert [row["source"] for row in rows] == sources[1:]
    assert [row["reference"] for row in rows] == references[1:]
    assert [row["source_words"] for row in rows] == [16, 2]
    assert [row["length_group"] for row in rows] == ["16-30", "<=15"]


@pytest.mark.parametrize("ids", [[], [0], [-1], [4], [1, 1], [2, 1], [True], [1.5]])
def test_invalid_sample_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="Sample ids"):
        _select_samples(["a", "b", "c"], ["甲", "乙", "丙"], ids)


def test_misaligned_corpora_are_rejected():
    with pytest.raises(ValueError, match="not aligned"):
        _select_samples(["a", "b"], ["甲"], [1])


@pytest.mark.parametrize(("source", "reference"), [("  ", "甲"), ("a", "")])
def test_empty_selected_sentence_is_rejected(source, reference):
    with pytest.raises(ValueError, match="Empty source or reference"):
        _select_samples([source], [reference], [1])


def test_cached_bytes_require_original_hash_and_line_count():
    content = b"first\nsecond\n"
    entry = {"sha256": hashlib.sha256(content).hexdigest(), "count": 2}
    assert _validated_lines(content, entry, "source") == ["first", "second"]
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _validated_lines(content + b"third\n", entry, "source")
    with pytest.raises(ValueError, match="line count"):
        _validated_lines(content, {**entry, "count": 3}, "source")


def test_blank_cached_sentence_is_rejected_even_with_matching_hash():
    content = b"first\n \n"
    entry = {"sha256": hashlib.sha256(content).hexdigest(), "count": 2}
    with pytest.raises(ValueError, match="blank sentence"):
        _validated_lines(content, entry, "source")


def _row(index, prediction, reference, greedy_chrf=0.0, beam_chrf=0.0):
    return {
        "id": index,
        "source": f"Source {index}",
        "reference": reference,
        "source_words": 2,
        "length_group": "<=15",
        "greedy": {
            "text": prediction,
            "elapsed_ms": [10.0, 20.0, 30.0],
            "median_ms": 20.0,
            "chrf": greedy_chrf,
            "capped": False,
        },
        "beam": {
            "text": prediction,
            "elapsed_ms": [20.0, 30.0, 40.0],
            "median_ms": 30.0,
            "chrf": beam_chrf,
            "capped": False,
        },
    }


def test_group_metrics_are_corpus_scores_not_sentence_averages():
    rows = [
        _row(1, "今天下雨我们留在家里认真读书学习翻译模型。", "今天下雨我们留在家里认真读书学习翻译模型。"),
        _row(2, "猫", "狗"),
    ]
    bleu, chrf = BLEU(tokenize="zh", effective_order=False), CHRF()
    result = aggregate_metrics(rows, "greedy", bleu, chrf)
    predictions = [row["greedy"]["text"] for row in rows]
    references = [row["reference"] for row in rows]
    assert result["bleu"] == pytest.approx(bleu.corpus_score(predictions, [references]).score)
    assert result["chrf"] == pytest.approx(chrf.corpus_score(predictions, [references]).score)
    sentence_bleu = statistics.mean(
        bleu.sentence_score(prediction, [reference]).score for prediction, reference in zip(predictions, references, strict=True)
    )
    sentence_chrf = statistics.mean(
        chrf.sentence_score(prediction, [reference]).score for prediction, reference in zip(predictions, references, strict=True)
    )
    assert result["bleu"] != pytest.approx(sentence_bleu)
    assert result["chrf"] != pytest.approx(sentence_chrf)


def test_empty_groups_are_explicitly_null():
    result = aggregate_metrics([], "beam", BLEU(tokenize="zh"), CHRF())
    assert result["n"] == 0
    assert result["decoding"] == "beam"
    assert result["num_beams"] > 1
    assert all(value is None for key, value in result.items() if key not in {"n", "decoding", "num_beams"})


def test_latency_statistics_use_sample_medians_and_count_capped_samples():
    rows = [_row(1, "甲", "甲"), _row(2, "乙", "乙")]
    rows[0]["greedy"]["elapsed_ms"] = [1.0, 2.0, 100.0]
    rows[1]["greedy"]["elapsed_ms"] = [3.0, 4.0, 5.0]
    rows[1]["greedy"]["capped"] = True
    result = aggregate_metrics(rows, "greedy", BLEU(tokenize="zh"), CHRF())
    assert result["n"] == 2
    assert result["mean_ms"] == pytest.approx(3.0)
    assert result["median_ms"] == pytest.approx(3.0)
    assert result["p95_ms"] == pytest.approx(3.9)
    assert result["capped"] == 1


@pytest.mark.parametrize("times", [[], [-1.0], [float("nan")], [float("inf")]])
def test_invalid_latency_is_rejected(times):
    row = _row(1, "甲", "甲")
    row["greedy"]["elapsed_ms"] = times
    with pytest.raises(ValueError, match="Latency"):
        aggregate_metrics([row], "greedy", BLEU(tokenize="zh"), CHRF())


def test_unknown_decoding_is_rejected():
    with pytest.raises(ValueError, match="Unknown decoding"):
        aggregate_metrics([], "sampling", BLEU(tokenize="zh"), CHRF())


def test_case_selection_uses_signed_extremes_and_id_ties():
    rows = [
        _row(4, "甲", "甲", 70, 90),
        _row(3, "乙", "乙", 20, 0),
        _row(2, "丙", "丙", 70, 90),
        _row(1, "丁", "丁", 20, 0),
    ]
    cases = {case["category"]: case for case in select_cases(rows)}
    assert cases["highest_average_chrf"]["id"] == 2
    assert cases["lowest_average_chrf"]["id"] == 1
    assert cases["largest_positive_delta"]["id"] == 2
    assert cases["largest_positive_delta"]["delta_chrf"] == 20
    assert cases["largest_negative_delta"]["id"] == 1
    assert cases["largest_negative_delta"]["delta_chrf"] == -20


def test_case_selection_does_not_invent_positive_or_negative_outcomes():
    rows = [_row(1, "甲", "甲", 50, 50)]
    assert {case["category"] for case in select_cases(rows)} == {"highest_average_chrf", "lowest_average_chrf"}
    assert select_cases([]) == []
