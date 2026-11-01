import matplotlib
import torch
from matplotlib.figure import Figure

from nmtlab.engine import Attention

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": ["Lato", "Noto Sans CJK SC"], "axes.unicode_minus": False, "svg.fonttype": "path"})


def attention_figure(attention: Attention, layer: int | None = None, head: int = 0, show_special: bool = False) -> Figure:
    layers, heads, _, _ = attention.weights.shape
    layer = layers if layer is None else layer
    if not 1 <= layer <= layers or not 0 <= head <= heads:
        raise ValueError("Layer or head is out of range.")
    weights = attention.weights[layer - 1]
    matrix = weights.mean(dim=0) if head == 0 else weights[head - 1]
    source_indices = [i for i, special in enumerate(attention.source_special) if show_special or not special]
    target_indices = [i for i, special in enumerate(attention.target_special) if show_special or not special]
    if not source_indices or not target_indices:
        raise ValueError("No lexical tokens to display. Include control tokens to inspect this output.")
    matrix = matrix[torch.tensor(target_indices)][:, torch.tensor(source_indices)]
    figure = Figure(figsize=(max(8, len(source_indices) * 0.34), max(4, len(target_indices) * 0.30)), dpi=160, layout="constrained")
    axis = figure.subplots()
    image = axis.imshow(matrix.numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axis.set_xticks(range(len(source_indices)), [attention.source_tokens[i] for i in source_indices], rotation=60, ha="right", fontsize=9)
    axis.set_yticks(range(len(target_indices)), [attention.target_tokens[i] for i in target_indices], fontsize=9)
    axis.set_xlabel("Source subwords")
    axis.set_ylabel("Predicted target subwords")
    head_label = "mean of heads" if head == 0 else f"head {head}"
    axis.set_title(f"Cross-attention · layer {layer} · {head_label}", loc="left", fontsize=12, pad=12)
    figure.colorbar(image, ax=axis, label="Attention weight", fraction=0.025, pad=0.03)
    return figure
