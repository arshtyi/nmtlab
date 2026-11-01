import json
from dataclasses import asdict

from nmtlab.config import EXAMPLES, RESULTS
from nmtlab.engine import Translator
from nmtlab.plots import attention_figure


def main() -> None:
    translator = Translator()
    translator.warmup()
    result = translator.translate(EXAMPLES[0])
    attention = translator.attention(result)
    output = RESULTS / "figures"
    output.mkdir(parents=True, exist_ok=True)
    layer = attention.weights.shape[0]
    figure = attention_figure(attention, layer=layer, head=0)
    figure.savefig(output / "attention.svg")
    record = {
        "translation": asdict(result),
        "source_tokens": attention.source_tokens,
        "target_tokens": attention.target_tokens,
        "source_special": attention.source_special,
        "target_special": attention.target_special,
        "weights": attention.weights.tolist(),
        "view": {"layer": layer, "head": "mean", "show_special": False, "renormalized": False},
    }
    (output / "attention.json").write_text(json.dumps(record, ensure_ascii=False, indent=4) + "\n")
    print(f"Attention figure and underlying weights saved to {output}.")


if __name__ == "__main__":
    main()
