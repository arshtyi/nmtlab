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
    source_indices = [i for i, special in enumerate(attention.source_special) if show_special or not (special or attention.source_tokens[i] == "▁")]
    target_indices = [i for i, special in enumerate(attention.target_special) if show_special or not (special or attention.target_tokens[i] == "▁")]
    if not source_indices or not target_indices:
        raise ValueError("No lexical tokens to display. Include control and boundary tokens to inspect this output.")
    hidden_indices = [i for i in range(matrix.shape[1]) if i not in source_indices]
    rows = matrix[torch.tensor(target_indices)]
    visible = rows[:, torch.tensor(source_indices)]
    figure = Figure(figsize=(max(9, len(source_indices) * 0.34 + 2), max(4, len(target_indices) * 0.30)), dpi=160, layout="constrained")
    if hidden_indices:
        axis, mass_axis = figure.subplots(1, 2, gridspec_kw={"width_ratios": [max(6, len(source_indices) * 0.34), 1.2]})
        hidden_mass = rows[:, torch.tensor(hidden_indices)].sum(dim=-1)
        mass_axis.barh(range(len(target_indices)), hidden_mass.numpy(), height=0.8, color="#64748b")
        mass_axis.set_xlim(0, 1)
        mass_axis.set_ylim(len(target_indices) - 0.5, -0.5)
        mass_axis.set_xticks([0, 0.5, 1])
        mass_axis.set_yticks([])
        mass_axis.set_title("Hidden mass", fontsize=10, pad=12)
        mass_axis.set_xlabel("Weight")
        mass_axis.spines[["top", "right", "left"]].set_visible(False)
    else:
        axis = figure.subplots()
    # Scale colors for this view only; retain the original probability values.
    maximum = float(visible.max()) or 1.0
    image = axis.imshow(visible.numpy(), cmap="Blues", vmin=0, vmax=maximum, aspect="auto", interpolation="nearest")
    source_labels = ["<boundary>" if attention.source_tokens[i] == "▁" else attention.source_tokens[i] for i in source_indices]
    target_labels = ["<boundary>" if attention.target_tokens[i] == "▁" else attention.target_tokens[i] for i in target_indices]
    axis.set_xticks(range(len(source_indices)), source_labels, rotation=60, ha="right", fontsize=9)
    axis.set_yticks(range(len(target_indices)), target_labels, fontsize=9)
    axis.set_xlabel("Source subwords")
    axis.set_ylabel("Predicted target subwords")
    head_label = "mean of heads" if head == 0 else f"head {head}"
    axis.set_title(f"Cross-attention · layer {layer} · {head_label}", loc="left", fontsize=12, pad=12)
    figure.colorbar(image, ax=axis, label="Original weight (view-scaled colors)", fraction=0.035, pad=0.03)
    return figure
