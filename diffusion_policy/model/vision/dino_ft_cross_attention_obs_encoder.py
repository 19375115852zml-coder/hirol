from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def _to_2tuple(value: Optional[Union[int, Sequence[int]]]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    value = tuple(value)
    if len(value) != 2:
        raise ValueError(f"Expected a 2D size, got {value}.")
    return int(value[0]), int(value[1])


class DinoFtCrossAttentionObsEncoder(ModuleAttrMixin):
    """Encode RGB with DINOv3, FT windows with LSTM, then fuse with cross attention.

    Input tensors follow the policy convention after time flattening:
      - RGB keys: (B, C, H, W)
      - ft_data: (B, K, D_ft)
      - ft_mask: (B, K), where True/1 means valid
      - low_dim keys: (B, D), projected after cross attention and not used as KV tokens
    """

    def __init__(
        self,
        shape_meta: dict,
        image_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", #
        image_freeze: bool = True,
        image_resize_shape: Optional[Union[int, Sequence[int]]] = (224, 224),
        image_input_range: str = "minus_one_one",
        image_imagenet_norm: bool = True,
        trust_remote_code: bool = False,
        interpolate_pos_encoding: bool = True,
        ft_key: str = "ft_data",
        ft_mask_key: str = "ft_mask",
        ft_input_dim: Optional[int] = None,
        ft_hidden_dim: int = 128,
        ft_num_layers: int = 1,
        ft_bidirectional: bool = False,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_queries: int = 1,
        dropout: float = 0.0,
        lowdim_embed_dim: int = 128,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "DinoFtCrossAttentionObsEncoder requires transformers for DINOv3."
            ) from exc

        self.shape_meta = shape_meta
        self.ft_key = ft_key
        self.ft_mask_key = ft_mask_key
        self.image_resize_shape = _to_2tuple(image_resize_shape)
        if image_input_range not in {"zero_one", "minus_one_one"}:
            raise ValueError(
                f"Unsupported image_input_range={image_input_range!r}. "
                "Expected 'zero_one' or 'minus_one_one'."
            )
        self.image_input_range = image_input_range
        self.image_imagenet_norm = bool(image_imagenet_norm)
        self.interpolate_pos_encoding = bool(interpolate_pos_encoding)
        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)
        self.lowdim_embed_dim = int(lowdim_embed_dim)

        obs_shape_meta = shape_meta["obs"]
        self.rgb_keys = sorted(
            key for key, attr in obs_shape_meta.items() if attr.get("type") == "rgb"
        )
        self.low_dim_keys = sorted(
            key for key, attr in obs_shape_meta.items() if attr.get("type") == "low_dim"
        )
        ft_keys = [key for key, attr in obs_shape_meta.items() if attr.get("type") == "ft"]
        if len(ft_keys) != 1:
            raise ValueError(f"Expected exactly one type='ft' obs key, got {ft_keys}.")
        if self.ft_key not in ft_keys:
            raise ValueError(f"ft_key={self.ft_key!r} is not declared as type='ft'.")
        if len(self.rgb_keys) == 0:
            raise ValueError("At least one RGB obs key is required for DINO image encoding.")
        if self.num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {self.num_queries}.")

        self.key_shape_map = {
            key: tuple(attr["shape"])
            for key, attr in obs_shape_meta.items()
        }
        ft_shape = self.key_shape_map[self.ft_key]
        if len(ft_shape) != 2:
            raise ValueError(f"FT obs {self.ft_key!r} must have shape [K, D], got {ft_shape}.")
        inferred_ft_dim = int(ft_shape[-1])
        if ft_input_dim is None:
            ft_input_dim = inferred_ft_dim
        if int(ft_input_dim) != inferred_ft_dim:
            raise ValueError(
                f"ft_input_dim={ft_input_dim} does not match shape_meta dim {inferred_ft_dim}."
            )

        self.image_model = AutoModel.from_pretrained(
            image_model_name,
            trust_remote_code=trust_remote_code,
        )
        image_hidden_dim = getattr(self.image_model.config, "hidden_size", None)
        if image_hidden_dim is None:
            raise AttributeError(
                f"Cannot infer hidden_size from DINO config for {image_model_name!r}."
            )
        if image_freeze:
            self.image_model.eval()
            self.image_model.requires_grad_(False)

        ft_num_directions = 2 if ft_bidirectional else 1
        self.ft_lstm = nn.LSTM(
            input_size=int(ft_input_dim),
            hidden_size=int(ft_hidden_dim),
            num_layers=int(ft_num_layers),
            batch_first=True,
            bidirectional=bool(ft_bidirectional),
            dropout=float(dropout) if int(ft_num_layers) > 1 else 0.0,
        )

        self.image_proj = nn.Linear(int(image_hidden_dim), self.embed_dim)
        self.camera_embedding = nn.Embedding(len(self.rgb_keys), self.embed_dim)
        self.ft_proj = nn.Linear(int(ft_hidden_dim) * ft_num_directions, self.embed_dim)
        self.query = nn.Parameter(torch.empty(self.num_queries, self.embed_dim))
        nn.init.normal_(self.query, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)
        self.out_norm = nn.LayerNorm(self.embed_dim)

        lowdim_dim = 0
        for key in self.low_dim_keys:
            key_dim = 1
            for size in self.key_shape_map[key]:
                key_dim *= int(size)
            lowdim_dim += key_dim
        self.lowdim_dim = lowdim_dim
        if lowdim_dim > 0 and self.lowdim_embed_dim > 0:
            self.lowdim_proj = nn.Sequential(
                nn.Linear(lowdim_dim, self.lowdim_embed_dim),
                nn.LayerNorm(self.lowdim_embed_dim),
                nn.SiLU(),
            )
            self._output_dim = self.embed_dim * self.num_queries + self.lowdim_embed_dim
        else:
            self.lowdim_proj = None
            self._output_dim = self.embed_dim * self.num_queries

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.image_model.parameters()):
            self.image_model.eval()
        return self

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.image_input_range == "minus_one_one":
            image = (image + 1.0) * 0.5
        image = image.clamp(0.0, 1.0)
        if self.image_resize_shape is not None:
            image = F.interpolate(
                image,
                size=self.image_resize_shape,
                mode="bilinear",
                align_corners=False,
            )
        if self.image_imagenet_norm:
            image = (image - self.image_mean.to(dtype=image.dtype)) / self.image_std.to(dtype=image.dtype)
        return image

    def _encode_image_tokens(self, obs_dict) -> torch.Tensor:
        image_tokens = []
        for cam_idx, key in enumerate(self.rgb_keys):
            image = obs_dict[key]
            expected_shape = self.key_shape_map[key]
            if tuple(image.shape[1:]) != expected_shape:
                raise AssertionError(
                    f"RGB obs {key!r} shape mismatch, expected {expected_shape}, got {tuple(image.shape[1:])}."
                )
            image = self._preprocess_image(image)
            kwargs = {"pixel_values": image}
            if self.interpolate_pos_encoding:
                kwargs["interpolate_pos_encoding"] = True
            try:
                outputs = self.image_model(**kwargs)
            except TypeError:
                kwargs.pop("interpolate_pos_encoding", None)
                outputs = self.image_model(**kwargs)

            tokens = outputs.last_hidden_state
            if tokens.shape[1] > 1:
                tokens = tokens[:, 1:, :]
            tokens = self.image_proj(tokens)
            cam_embed = self.camera_embedding.weight[cam_idx].view(1, 1, -1)
            image_tokens.append(tokens + cam_embed)

        return torch.cat(image_tokens, dim=1)

    def _encode_ft_tokens(self, obs_dict) -> tuple[torch.Tensor, torch.Tensor]:
        ft = obs_dict[self.ft_key]
        if tuple(ft.shape[1:]) != self.key_shape_map[self.ft_key]:
            raise AssertionError(
                f"FT obs {self.ft_key!r} shape mismatch, expected {self.key_shape_map[self.ft_key]}, "
                f"got {tuple(ft.shape[1:])}."
            )
        ft = ft.to(dtype=self.dtype)
        ft_tokens, _ = self.ft_lstm(ft)
        ft_tokens = self.ft_proj(ft_tokens)

        if self.ft_mask_key in obs_dict:
            ft_valid = obs_dict[self.ft_mask_key].to(device=ft_tokens.device)
            ft_valid = ft_valid > 0.5 if ft_valid.dtype != torch.bool else ft_valid
        else:
            ft_valid = torch.ones(ft.shape[:2], dtype=torch.bool, device=ft_tokens.device)
        return ft_tokens, ft_valid

    def _encode_lowdim_feature(self, obs_dict) -> Optional[torch.Tensor]:
        if self.lowdim_proj is None:
            return None
        lowdim_values = []
        for key in self.low_dim_keys:
            value = obs_dict[key]
            if tuple(value.shape[1:]) != self.key_shape_map[key]:
                raise AssertionError(
                    f"Lowdim obs {key!r} shape mismatch, expected {self.key_shape_map[key]}, "
                    f"got {tuple(value.shape[1:])}."
                )
            lowdim_values.append(value.reshape(value.shape[0], -1))
        lowdim = torch.cat(lowdim_values, dim=-1).to(dtype=self.dtype)
        return self.lowdim_proj(lowdim)

    def forward(self, obs_dict):
        image_tokens = self._encode_image_tokens(obs_dict)
        ft_tokens, ft_valid = self._encode_ft_tokens(obs_dict)

        tokens = torch.cat([image_tokens, ft_tokens], dim=1)
        image_valid = torch.ones(
            image_tokens.shape[:2],
            dtype=torch.bool,
            device=image_tokens.device,
        )
        key_valid = torch.cat([image_valid, ft_valid], dim=1)
        key_padding_mask = ~key_valid
        all_masked = key_padding_mask.all(dim=1)
        if torch.any(all_masked):
            key_padding_mask[all_masked, 0] = False

        batch_size = tokens.shape[0]
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attn(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = self.out_norm(self.attn_norm(attn_out + query))
        fused = fused.reshape(batch_size, -1)

        lowdim_feature = self._encode_lowdim_feature(obs_dict)
        if lowdim_feature is not None:
            fused = torch.cat([fused, lowdim_feature], dim=-1)
        return fused

    @torch.no_grad()
    def output_shape(self):
        return (self._output_dim,)
