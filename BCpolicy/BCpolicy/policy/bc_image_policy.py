from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class BCImagePolicy(BaseImagePolicy):
    """Behavior cloning ablation policy for DP image-style batches.

    This policy intentionally consumes the same batch contract as the diffusion
    image policies:

    - batch["obs"][key]: [B, To, ...]
    - batch["action"]: [B, horizon, action_dim]

    It removes diffusion/noise scheduling from the experiment and directly
    regresses the normalized action trajectory from encoded observations.
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        hidden_dims: Sequence[int] = (1024, 1024, 512),
        activation: str = "gelu",
        dropout: float = 0.0,
        loss_on_action_steps_only: bool = False,
    ):
        super().__init__()

        action_shape = tuple(shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise ValueError(f"BCImagePolicy expects 1-D action shape, got {action_shape}")

        self.shape_meta = shape_meta
        self.obs_encoder = obs_encoder
        self.horizon = int(horizon)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.action_dim = int(action_shape[0])
        self.obs_feature_dim = int(obs_encoder.output_shape()[0])
        self.loss_on_action_steps_only = bool(loss_on_action_steps_only)

        mlp_layers = []
        input_dim = self.obs_feature_dim * self.n_obs_steps
        cur_dim = input_dim
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            mlp_layers.append(nn.Linear(cur_dim, hidden_dim))
            mlp_layers.append(_make_activation(activation))
            if dropout > 0:
                mlp_layers.append(nn.Dropout(float(dropout)))
            cur_dim = hidden_dim
        mlp_layers.append(nn.Linear(cur_dim, self.horizon * self.action_dim))

        # Train workspace expects image policies to expose .model for stats.
        self.model = nn.Sequential(*mlp_layers)
        self.normalizer = LinearNormalizer()

    def _encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]

        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        obs_features = self.obs_encoder(this_nobs)
        return obs_features.reshape(batch_size, -1)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs_features = self._encode_obs(obs_dict)
        naction_pred = self.model(obs_features)
        return naction_pred.reshape(-1, self.horizon, self.action_dim)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        naction_pred = self.forward(obs_dict)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        naction_pred = self.forward(batch["obs"])
        naction = self.normalizer["action"].normalize(batch["action"])

        if self.loss_on_action_steps_only:
            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            naction_pred = naction_pred[:, start:end]
            naction = naction[:, start:end]

        return F.mse_loss(naction_pred, naction)
