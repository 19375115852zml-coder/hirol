from typing import Optional

import torch
import torch.nn as nn


class Dinov3Encoder(nn.Module):
    """HuggingFace DINOv3 wrapper that matches the ResNet encoder interface."""

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        pooling: str = "pooler",
        freeze: bool = False,
        output_dim: Optional[int] = None,
        trust_remote_code: bool = False,
        interpolate_pos_encoding: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "Dinov3Encoder requires transformers. Install it in the training "
                "environment before using get_dino_v3()."
            ) from exc

        if pooling not in {"pooler", "cls", "mean_patch"}:
            raise ValueError(
                f"Unsupported DINOv3 pooling={pooling!r}. "
                "Expected 'pooler', 'cls', or 'mean_patch'."
            )

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.pooling = pooling
        self.interpolate_pos_encoding = interpolate_pos_encoding

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError(
                f"Cannot infer hidden_size from DINOv3 config for {model_name!r}."
            )

        self.proj = nn.Identity()
        if output_dim is not None and int(output_dim) != int(hidden_size):
            self.proj = nn.Linear(int(hidden_size), int(output_dim))

        if freeze:
            self.model.eval()
            self.model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.model.parameters()):
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kwargs = {"pixel_values": x}
        if self.interpolate_pos_encoding:
            kwargs["interpolate_pos_encoding"] = True
        outputs = self.model(**kwargs)

        if self.pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
            features = outputs.pooler_output
        elif self.pooling == "cls":
            features = outputs.last_hidden_state[:, 0]
        else:
            features = outputs.last_hidden_state[:, 1:].mean(dim=1)

        return self.proj(features)
