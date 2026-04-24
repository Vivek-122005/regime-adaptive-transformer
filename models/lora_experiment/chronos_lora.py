from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from transformers import T5EncoderModel


class ChronosLoRARanker(nn.Module):
    """
    Chronos-T5 encoder with LoRA adapters and a scalar ranking head.

    Inputs are daily feature sequences with shape (batch, seq_len, input_dim).
    Output is one scalar alpha prediction per sample.
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        input_dim: int = 10,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()

        base_encoder = T5EncoderModel.from_pretrained(model_name)
        for param in base_encoder.parameters():
            param.requires_grad = False

        peft_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q", "k", "v", "o"],
        )
        self.encoder = get_peft_model(base_encoder, peft_cfg)

        hidden_dim = int(self.encoder.config.d_model)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.ranking_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape (batch, seq_len, features), got {tuple(x.shape)}")

        inputs_embeds = self.input_projection(x)
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], inputs_embeds.shape[1]),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.ranking_head(pooled).squeeze(-1)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_adapter(self, out_path: str | Path) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder_lora_state_dict": get_peft_model_state_dict(self.encoder),
            "input_projection_state_dict": self.input_projection.state_dict(),
            "ranking_head_state_dict": self.ranking_head.state_dict(),
            "meta": {
                "model_name": "amazon/chronos-t5-small",
                "notes": "Backbone frozen; LoRA adapters + projection/head are trainable.",
            },
        }
        torch.save(payload, path)
