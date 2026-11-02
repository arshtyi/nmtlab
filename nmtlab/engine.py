from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import cast

import torch
from transformers import GenerationConfig, T5ForConditionalGeneration, T5Tokenizer

from nmtlab.config import CONFIG, EXAMPLES


@dataclass(frozen=True)
class Translation:
    source: str
    text: str
    elapsed_ms: float
    source_ids: list[int]
    output_ids: list[int]
    beams: int
    capped: bool


@dataclass(frozen=True)
class Attention:
    source_tokens: list[str]
    target_tokens: list[str]
    source_special: list[bool]
    target_special: list[bool]
    weights: torch.Tensor


class Translator:
    def __init__(self, device: str = CONFIG["runtime"]["device"]) -> None:
        if device not in {"cpu", "mps"}:
            raise ValueError("Device must be cpu or mps.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available on this machine.")
        self.device = device
        torch.set_num_threads(CONFIG["runtime"]["threads"])
        torch.manual_seed(CONFIG["measurement"]["seed"])
        model_config = CONFIG["model"]
        self.tokenizer = T5Tokenizer.from_pretrained(model_config["id"], revision=model_config["revision"], local_files_only=True)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
            local_files_only=True,
            attn_implementation=model_config["attention"],
            dtype=getattr(torch, model_config["dtype"]),
        )
        cast(Callable[..., T5ForConditionalGeneration], self.model.to)(device=device)
        self.model.eval()
        self.model.generation_config.max_length = None
        self.max_source_tokens = model_config["max_source_tokens"]
        self.target_prefix = model_config["target_prefix"]
        self.generations = {
            beams: GenerationConfig.from_pretrained(
                model_config["id"],
                revision=model_config["revision"],
                local_files_only=True,
                max_length=None,
                max_new_tokens=CONFIG["decoding"]["max_new_tokens"],
                do_sample=CONFIG["decoding"]["do_sample"],
                num_beams=beams,
                length_penalty=CONFIG["decoding"]["length_penalty"],
                early_stopping=CONFIG["decoding"]["early_stopping"],
                use_cache=True,
            )
            for beams in CONFIG["decoding"]["beams"]
        }

    def _synchronize(self) -> None:
        if self.device == "mps":
            torch.mps.synchronize()

    def warmup(self) -> None:
        for beams in self.generations:
            for _ in range(CONFIG["measurement"]["warmup"]):
                self.translate(EXAMPLES[0], beams)

    @torch.inference_mode()
    def translate(self, text: str, beams: int = CONFIG["decoding"]["default_beams"]) -> Translation:
        source = text.strip()
        if not source:
            raise ValueError("Enter an English sentence.")
        if beams not in self.generations:
            raise ValueError("Choose one of the configured decoding modes.")
        self._synchronize()
        start = perf_counter()
        encoded = self.tokenizer(f"{self.target_prefix} {source}", return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] > self.max_source_tokens:
            raise ValueError(f"Input exceeds {self.max_source_tokens} model tokens, including language and end tokens. Shorten the input.")
        inputs = encoded.to(self.device)
        generated = cast(Callable[..., torch.Tensor], self.model.generate)(**inputs, generation_config=self.generations[beams])
        output_ids = generated[0].cpu().tolist()
        translated = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        self._synchronize()
        elapsed_ms = (perf_counter() - start) * 1000
        return Translation(
            source=source,
            text=translated,
            elapsed_ms=elapsed_ms,
            source_ids=inputs["input_ids"][0].cpu().tolist(),
            output_ids=output_ids,
            beams=beams,
            capped=len(output_ids) - 1 >= CONFIG["decoding"]["max_new_tokens"],
        )

    @torch.inference_mode()
    def attention(self, result: Translation) -> Attention:
        source = torch.tensor([result.source_ids], device=self.device)
        decoder = torch.tensor([result.output_ids[:-1]], device=self.device)
        encoded = self.model.encoder(input_ids=source, attention_mask=torch.ones_like(source), return_dict=True)
        output = self.model.decoder(
            input_ids=decoder,
            attention_mask=torch.ones_like(decoder),
            encoder_hidden_states=encoded.last_hidden_state,
            encoder_attention_mask=torch.ones_like(source),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        if output.cross_attentions is None:
            raise RuntimeError("The model did not return cross-attention weights.")
        weights = torch.stack([layer[0].cpu().float() for layer in output.cross_attentions])
        source_tokens = self.tokenizer.convert_ids_to_tokens(result.source_ids)
        target_ids = result.output_ids[1:]
        target_tokens = self.tokenizer.convert_ids_to_tokens(target_ids)
        special_ids = set(self.tokenizer.all_special_ids) - {self.tokenizer.unk_token_id}
        special_ids.add(self.tokenizer.convert_tokens_to_ids(self.target_prefix))
        return Attention(
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            source_special=[token in special_ids for token in result.source_ids],
            target_special=[token in special_ids for token in target_ids],
            weights=weights,
        )
