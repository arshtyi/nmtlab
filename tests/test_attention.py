import pytest
import torch

from nmtlab.engine import Attention
from nmtlab.plots import attention_figure


def test_hidden_control_tokens_preserve_mass_and_subword_positions():
    attention = Attention(
        source_tokens=["<2zh>", "▁neural", "net", "</s>"],
        target_tokens=["<s>", "▁神", "经"],
        source_special=[True, False, False, True],
        target_special=[True, False, False],
        weights=torch.tensor([[[[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.5, 0.2], [0.1, 0.1, 0.1, 0.7]]]]),
    )
    figure = attention_figure(attention, layer=1)
    axis = figure.axes[0]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["▁neural", "net"]
    assert [tick.get_text() for tick in axis.get_yticklabels()] == ["▁神", "经"]
    values = axis.images[0].get_array()
    assert values is not None
    assert values[0].tolist() == pytest.approx([0.1, 0.5])
    assert values[1].tolist() == pytest.approx([0.1, 0.1])


def test_head_average_and_explicit_head_are_distinct():
    attention = Attention(
        source_tokens=["a", "b"],
        target_tokens=["x"],
        source_special=[False, False],
        target_special=[False],
        weights=torch.tensor([[[[0.2, 0.8]], [[0.6, 0.4]]]]),
    )
    averaged = attention_figure(attention, layer=1, head=0).axes[0].images[0].get_array()
    selected = attention_figure(attention, layer=1, head=2).axes[0].images[0].get_array()
    assert averaged is not None and selected is not None
    assert averaged.tolist()[0] == pytest.approx([0.4, 0.6])
    assert selected.tolist()[0] == pytest.approx([0.6, 0.4])
    with pytest.raises(ValueError, match="out of range"):
        attention_figure(attention, layer=2)


def test_empty_lexical_output_requires_explicit_control_view():
    attention = Attention(["</s>"], ["</s>"], [True], [True], torch.ones(1, 1, 1, 1))
    with pytest.raises(ValueError, match="No lexical tokens"):
        attention_figure(attention, layer=1)
    shown = attention_figure(attention, layer=1, show_special=True)
    values = shown.axes[0].images[0].get_array()
    assert values is not None
    assert values[0, 0] == 1


def test_default_view_uses_final_decoder_layer():
    attention = Attention(
        source_tokens=["a", "b"],
        target_tokens=["x"],
        source_special=[False, False],
        target_special=[False],
        weights=torch.tensor([[[[0.9, 0.1]]], [[[0.2, 0.8]]]]),
    )
    values = attention_figure(attention).axes[0].images[0].get_array()
    assert values is not None
    assert values[0].tolist() == pytest.approx([0.2, 0.8])
