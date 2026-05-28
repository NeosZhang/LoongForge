"""
XVLA Policy Model

Core XVLA implementation combining Florence2 VLM encoder with temporal action transformer.
Implements flow matching for action generation with optional soft prompts and domain conditioning.
"""
#!/usr/bin/env python

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig, ListConfig
from torch import Tensor, nn

from loongforge.embodied.model.registry import register_model

from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig
from .soft_transformer import SoftPromptedTransformer
from .util.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE
from .util.import_utils import _transformers_available
from .util.types import FeatureType, PolicyFeature

logger = logging.getLogger(__name__)


# Florence2 config and modeling depend on transformers
if TYPE_CHECKING or _transformers_available:
    from .configuration_florence2 import Florence2Config
    from .modeling_florence2 import Florence2ForConditionalGeneration
else:
    Florence2Config = None
    Florence2ForConditionalGeneration = None


class XVLAModel(nn.Module):
    """
    XVLA backbone that stitches Florence-2 embeddings with the temporal/action transformer head.
    """

    def __init__(
        self,
        config: XVLAConfig,
        florence_config: Florence2Config,
        proprio_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size: int = config.chunk_size
        self.use_proprio: bool = config.use_proprio

        # Build action space with auto-detection for "auto" mode
        if config.action_mode.lower() == "auto":
            # Auto-detect real action dim from config.action_feature
            real_dim = (
                config.action_feature.shape[-1]
                if config.action_feature is not None
                else config.max_action_dim
            )
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=real_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())

        self.dim_action = self.action_space.dim_action
        self.dim_proprio = proprio_dim

        self.vlm = Florence2ForConditionalGeneration(florence_config)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        # Apply freezing based on config
        self._apply_freezing()

        # Apply dtype casting based on config
        self._apply_dtype()

    def _get_target_dtype(self) -> torch.dtype:
        """Get the target dtype based on config."""
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _apply_dtype(self) -> None:
        """
        Apply dtype casting to model components based on config.
        """
        target_dtype = self._get_target_dtype()
        self.to(dtype=target_dtype)

    def _apply_freezing(self) -> None:
        """
        Freeze VLM vision and language encoders based on config options.
        Keep only policy transformer and soft prompts trainable.
        """
        # Freeze vision encoder
        if self.config.freeze_vision_encoder and hasattr(self.vlm, "vision_tower"):
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = False

        # Freeze language encoder
        if self.config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            # Freeze encoder
            if hasattr(lm, "model") and hasattr(lm.model, "encoder"):
                for param in lm.model.encoder.parameters():
                    param.requires_grad = False
            # Freeze shared embeddings
            if hasattr(lm, "model") and hasattr(lm.model, "shared"):
                for param in lm.model.shared.parameters():
                    param.requires_grad = False

        # Freeze or unfreeze policy transformer
        if not self.config.train_policy_transformer:
            for name, param in self.transformer.named_parameters():
                if "soft_prompts" not in name:
                    param.requires_grad = False

        # Freeze or unfreeze soft prompts
        if not self.config.train_soft_prompts and hasattr(self.transformer, "soft_prompt_hub"):
            for param in self.transformer.soft_prompt_hub.parameters():
                param.requires_grad = False

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Encode text and multi-view images via Florence2 encoder.
        """
        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")
        valid_images = flat_images[flat_mask]

        valid_feats = self.vlm._encode_image(valid_images)

        tokens_per_view, hidden_dim = valid_feats.shape[1:]
        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]

        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the XVLA model.
        """
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device, dtype=target_dtype)
            + torch.arange(batch_size, device=input_ids.device, dtype=target_dtype) / batch_size
        ) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)

        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )

        losses = self.action_space.compute_loss(pred_action, action)
        return losses

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Generate action sequences using flow matching denoising."""
        self.eval()

        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        action_dim = self.dim_action
        x1 = torch.randn(batch_size, self.chunk_size, action_dim, device=proprio.device, dtype=target_dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((batch_size,), i / steps, device=proprio.device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )

        action = self.action_space.postprocess(action)
        return action


def resize_with_pad(img: torch.Tensor, height: int, width: int, pad_value: float = 0.0) -> torch.Tensor:
    """Resize image tensor to target size with zero-padding to maintain aspect ratio."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad or truncate the last dimension of a tensor to new_dim."""
    if vector.shape[-1] == new_dim:
        return vector
    if new_dim == 0:
        shape = list(vector.shape)
        shape[-1] = 0
        return vector.new_zeros(*shape)
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = vector.new_zeros(*shape)
    length = min(current_dim, new_dim)
    new_vector[..., :length] = vector[..., :length]
    return new_vector


def pad_tensor_along_dim(tensor: Tensor, target_len: int, dim: int = 1) -> Tensor:
    """Pad or truncate a tensor along the specified dimension to target_len."""
    current_len = tensor.size(dim)
    if current_len == target_len:
        return tensor
    if current_len > target_len:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, target_len)
        return tensor[tuple(slices)]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - current_len
    pad_tensor = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, pad_tensor], dim=dim)


def _dictconfig_to_dict(obj):
    if isinstance(obj, DictConfig):
        return {key: _dictconfig_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, ListConfig):
        return [_dictconfig_to_dict(item) for item in obj]
    return obj


def _pil_to_tensor(images_pil: list, image_size: int, device: torch.device) -> torch.Tensor:
    from torchvision.transforms.functional import resize, to_tensor

    tensors = []
    for img in images_pil:
        if not isinstance(img, torch.Tensor):
            t = to_tensor(img)
        else:
            t = img.float()
            if t.max() > 1.0:
                t = t / 255.0
        if t.shape[-2] != image_size or t.shape[-1] != image_size:
            t = resize(t, [image_size, image_size])
        tensors.append(t)
    return torch.stack(tensors).to(device)


def _imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def _get_config_value(config_or_path, key: str, default=None):
    if hasattr(config_or_path, "get"):
        return config_or_path.get(key, default)
    return getattr(config_or_path, key, default)


def _has_key(container, key: str) -> bool:
    return hasattr(container, "__contains__") and key in container


@register_model("xvla")
class XVLAPolicy(nn.Module):
    """XVLA policy model combining vision-language backbone with action head."""

    def __init__(self, config):
        """Initialize XVLAPolicy from a config dict or XVLAConfig object."""
        super().__init__()

        backbone_cfg = _dictconfig_to_dict(
            config.get("backbone", {}) if hasattr(config, "get") else getattr(config, "backbone", {})
        )
        action_cfg = _dictconfig_to_dict(
            config.get("action_model", {}) if hasattr(config, "get") else getattr(config, "action_model", {})
        )

        self.image_size = backbone_cfg.get("image_size", 224)
        self.num_images = backbone_cfg.get("num_images", backbone_cfg.get("num_image_views", 2))
        self.image_mask = backbone_cfg.get("image_mask", [True] * self.num_images)
        self.max_token_len = backbone_cfg.get("max_token_len", backbone_cfg.get("tokenizer_max_length", 64))
        backbone_cfg["dtype"] = backbone_cfg.get("dtype", "float32")

        self.action_dim = action_cfg.get("action_dim", 7)
        self.action_horizon = action_cfg.get("action_horizon", backbone_cfg.get("n_action_steps", 32))
        self.chunk_size = action_cfg.get("chunk_size", backbone_cfg.get("chunk_size", 32))
        self.max_state_dim = action_cfg.get("max_state_dim", backbone_cfg.get("max_state_dim", 32))
        action_dtype_str = action_cfg.get("dtype", backbone_cfg.get("dtype", "bfloat16"))
        backbone_cfg["tokenizer_name"] = os.environ.get("TOKENIZER_PATH", "") or backbone_cfg.get(
            "tokenizer_name", "facebook/bart-large"
        )
        backbone_cfg["chunk_size"] = self.chunk_size
        backbone_cfg["n_action_steps"] = self.action_horizon
        backbone_cfg["max_state_dim"] = self.max_state_dim
        backbone_cfg["max_action_dim"] = action_cfg.get(
            "max_action_dim", backbone_cfg.get("max_action_dim", self.action_dim)
        )
        backbone_cfg["action_mode"] = action_cfg.get("action_mode", backbone_cfg.get("action_mode", "ee6d"))

        resize_cfg = backbone_cfg.get("resize_imgs_with_padding") or (self.image_size, self.image_size)
        resize_shape = tuple(resize_cfg)
        input_features = {
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *resize_shape)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *resize_shape)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,)),
        }
        if self.num_images > 2:
            for idx in range(2, self.num_images):
                suffix = idx + 1
                input_features[f"observation.images.image{suffix}"] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, *resize_shape),
                )
        output_features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
        }

        self.config = XVLAConfig(**self._filter_xvla_config_kwargs(backbone_cfg))
        self.config.florence_config = _dictconfig_to_dict(backbone_cfg.get("florence_config", {}))
        self.config.input_features = input_features
        self.config.output_features = output_features
        self.config.validate_features()

        self.model = XVLAModel(
            config=self.config,
            florence_config=self.config.get_florence_config(),
            proprio_dim=self.max_state_dim if self.config.use_proprio else 0,
        )
        self._tokenizer = None
        self.action_dtype = torch.bfloat16 if action_dtype_str == "bfloat16" else torch.float32

    @staticmethod
    def _filter_xvla_config_kwargs(backbone_cfg) -> Dict[str, Any]:
        """Filter backbone config keys to only those accepted by XVLAConfig."""
        valid_fields = {f.name for f in dataclasses.fields(XVLAConfig)}
        return {k: v for k, v in backbone_cfg.items() if k in valid_fields}

    @property
    def tokenizer(self):
        """Lazily initialize and return the tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        return self._tokenizer

    @property
    def backbone(self) -> nn.Module:
        """Return the Florence2 VLM backbone."""
        return self.model.vlm

    @property
    def action_head(self) -> nn.Module:
        """Return the temporal action transformer head."""
        return self.model.transformer

    def gradient_checkpointing_enable(self):
        """Disable fused attention in transformer blocks to enable gradient checkpointing."""
        if hasattr(self.model, "transformer"):
            for block in self.model.transformer.blocks:
                block.attn.fused_attn = False

    def gradient_checkpointing_disable(self):
        """Re-enable fused attention in transformer blocks after gradient checkpointing."""
        if hasattr(self.model, "transformer"):
            for block in self.model.transformer.blocks:
                block.attn.fused_attn = True

    def _prepare_state(self, batch: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if not self.config.use_proprio:
            return torch.zeros(batch_size, 0, device=device)
        if hasattr(batch, "state") and batch.state is not None:
            state = batch.state
        elif _has_key(batch, OBS_STATE):
            state = batch[OBS_STATE]
        else:
            return torch.zeros(batch_size, self.max_state_dim, device=device)
        if state.ndim > 2:
            state = state[:, -1, :]
        return pad_vector(state.to(device=device), self.max_state_dim)

    def _prepare_images(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(batch, "images_list") and batch.images_list is not None:
            images = [img[:, -1] if img.ndim == 5 else img for img in batch.images_list]
            masks = batch.img_masks
            stacked_imgs = torch.stack(images, dim=1)
            stacked_masks = torch.stack(masks, dim=1)
        else:
            present_img_keys = [key for key in self.config.image_features if _has_key(batch, key)]
            if len(present_img_keys) == 0:
                raise ValueError(
                    "All image features are missing from the batch. "
                    f"Batch keys: {list(batch.keys())}, expected at least one of {list(self.config.image_features)}."
                )
            images = []
            masks = []
            for key in present_img_keys:
                img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
                images.append(img)
                masks.append(torch.ones(img.size(0), dtype=torch.bool, device=img.device))
            stacked_imgs = torch.stack(images, dim=1)
            stacked_masks = torch.stack(masks, dim=1)

        if self.config.resize_imgs_with_padding is not None:
            batch_size, num_views = stacked_imgs.shape[:2]
            channels = stacked_imgs.shape[2]
            target_h, target_w = tuple[int, int](self.config.resize_imgs_with_padding)
            stacked_imgs = resize_with_pad(
                stacked_imgs.flatten(0, 1),
                target_h,
                target_w,
            ).view(batch_size, num_views, channels, target_h, target_w)

        total_views = self.config.num_image_views or stacked_imgs.size(1)
        total_views = max(total_views, stacked_imgs.size(1))
        num_pad = total_views - stacked_imgs.size(1)
        if num_pad > 0:
            pad_shape = (stacked_imgs.size(0), num_pad, *stacked_imgs.shape[2:])
            pad_imgs = stacked_imgs.new_zeros(pad_shape)
            pad_masks = stacked_masks.new_zeros((stacked_masks.size(0), num_pad))
            stacked_imgs = torch.cat([stacked_imgs, pad_imgs], dim=1)
            stacked_masks = torch.cat([stacked_masks, pad_masks], dim=1)

        return stacked_imgs, stacked_masks

    def _get_domain_id(self, batch, batch_size: int, device: torch.device) -> torch.Tensor:
        candidate = None
        if hasattr(batch, "domain_id") and batch.domain_id is not None:
            candidate = batch.domain_id
        elif self.config.domain_feature_key and _has_key(batch, self.config.domain_feature_key):
            candidate = batch[self.config.domain_feature_key]
        elif _has_key(batch, "domain_id"):
            candidate = batch["domain_id"]

        if candidate is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if not isinstance(candidate, torch.Tensor):
            candidate = torch.as_tensor(candidate, device=device)
        else:
            candidate = candidate.to(device=device)

        if candidate.ndim == 0:
            candidate = candidate.expand(batch_size)
        if candidate.ndim > 1:
            candidate = candidate.view(candidate.shape[0], -1)[:, 0]
        if candidate.shape[0] != batch_size:
            candidate = candidate.expand(batch_size)
        return candidate.to(dtype=torch.long)

    def _prepare_action_targets(self, batch) -> torch.Tensor:
        if hasattr(batch, "actions") and batch.actions is not None:
            actions = batch.actions
        elif _has_key(batch, ACTION):
            actions = batch[ACTION]
        else:
            raise ValueError("Batch is missing action targets required for training.")
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = pad_tensor_along_dim(actions, self.config.chunk_size, dim=1)
        if actions.shape[-1] != self.model.dim_action:
            actions = pad_vector(actions, self.model.dim_action)
        return actions

    def _build_model_inputs(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = batch.input_ids if hasattr(batch, "input_ids") else batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        images, image_mask = self._prepare_images(batch)
        device = images.device
        return {
            "input_ids": input_ids.to(device=device),
            "image_input": images,
            "image_mask": image_mask.to(device=device),
            "domain_id": self._get_domain_id(batch, batch_size, device),
            "proprio": self._prepare_state(batch, batch_size, device),
        }

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Compute flow matching loss for a training batch."""
        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch).to(device=inputs["image_input"].device)
        loss_map = self.model(action=targets, **inputs)
        total_loss = sum(loss_map.values())
        result = {"action_loss": total_loss}
        result.update({k: v.detach().item() for k, v in loss_map.items()})
        result["total_loss"] = total_loss.detach().item()
        return result

    @torch.no_grad()
    def predict_action_chunk(self, batch, steps: int | None = None, **kwargs) -> torch.Tensor:
        """Predict a full action chunk for the given batch."""
        self.eval()
        inputs = self._build_model_inputs(batch)
        return self.model.generate_actions(
            **inputs,
            steps=steps if steps is not None else self.config.num_denoising_steps,
        )

    @torch.no_grad()
    def select_action(self, batch, **kwargs) -> torch.Tensor:
        """Select a single action (first step of predicted chunk)."""
        return self.predict_action_chunk(batch, **kwargs)[:, 0]

    @torch.no_grad()
    def predict_action(self, images, instructions, state=None, steps=10, **kwargs) -> Dict[str, np.ndarray]:
        """Run end-to-end inference from raw images and instructions to normalized actions."""
        device = next(self.parameters()).device
        self.eval()

        if isinstance(images[0], (list, tuple)):
            batch_size = len(images)
        else:
            batch_size = 1
            images = [images]
            instructions = [instructions]

        images_list = []
        img_masks = []
        for view_idx in range(self.num_images):
            view_images = []
            for batch_idx in range(batch_size):
                img_list = images[batch_idx] if isinstance(images[batch_idx], list) else [images[batch_idx]]
                if view_idx < len(img_list):
                    view_images.append(img_list[view_idx])
                else:
                    view_images.append(torch.zeros(3, self.image_size, self.image_size))
            images_list.append(_pil_to_tensor(view_images, self.image_size, device))
            mask_val = self.image_mask[view_idx] if view_idx < len(self.image_mask) else False
            img_masks.append(torch.full((batch_size,), mask_val, dtype=torch.bool, device=device))

        images_stacked = torch.stack(images_list, dim=1)
        img_masks_stacked = torch.stack(img_masks, dim=1)
        images_normalized = _imagenet_normalize(images_stacked)

        prompts = [f"Task: {instr.strip()};\nAction: " for instr in instructions]
        tok_out = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="pt",
        )
        tokens = tok_out["input_ids"].to(device)

        if state is not None:
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            state = pad_vector(state, self.max_state_dim).to(dtype=self.action_dtype)
        else:
            state = torch.zeros(batch_size, self.max_state_dim, device=device, dtype=self.action_dtype)

        domain_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        pred_actions = self.model.generate_actions(
            input_ids=tokens,
            image_input=images_normalized.to(self.action_dtype),
            image_mask=img_masks_stacked,
            domain_id=domain_id,
            proprio=state,
            steps=steps,
        )
        return {"normalized_actions": pred_actions.cpu().numpy()}

    def load_pretrained(self, pretrained_path: str, strict: bool = True, device=None):
        """Load model weights from a safetensors checkpoint file or directory."""
        path = Path(pretrained_path)
        safetensors_file = path / "model.safetensors" if path.is_dir() else path

        try:
            from safetensors.torch import load_file

            load_kwargs = {"device": str(device)} if device is not None else {}
            state_dict = load_file(str(safetensors_file), **load_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Could not load safetensors from {safetensors_file}: {exc}") from exc

        encoder_key = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shared_key = "model.vlm.language_model.model.shared.weight"
        if encoder_key in state_dict:
            state_dict[shared_key] = state_dict[encoder_key]
        self.load_state_dict(state_dict, strict=strict)
        logging.info("Loaded XVLA checkpoint")
        return self

    @classmethod
    def from_pretrained(cls, config_or_path) -> "XVLAPolicy":
        """Construct an XVLAPolicy from a config or pretrained path and load weights."""
        if isinstance(config_or_path, XVLAConfig):
            cfg = {"backbone": dataclasses.asdict(config_or_path), "action_model": {}}
            pretrained_path = config_or_path.pretrained_path
        else:
            cfg = config_or_path
            pretrained_path = _get_config_value(config_or_path, "pretrained_path", None)

        model = cls(cfg)
        if pretrained_path:
            model.load_pretrained(pretrained_path, strict=False)
        return model
