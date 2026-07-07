# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge policy adapters for the standalone eval server."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.protocol import PROTOCOL_VERSION
from loongforge.embodied.eval.servers.predict_action_interface import PredictActionModel, call_predict_action


@dataclass
class PredictActionModelSpec:
    """Model instance and metadata needed by the generic eval policy."""

    model: PredictActionModel
    metadata: Dict[str, Any]


class GenericPredictActionPolicy:
    """Generic eval RPC policy for models exposing predict_action."""

    def __init__(
        self,
        model: PredictActionModel,
        metadata: Dict[str, Any],
        dataset_statistics_path: str = "",
        action_dim: int = 7,
        request_id_prefix: str = "predict-action",
    ) -> None:
        """Initialize the generic predict_action eval policy."""
        self._model = model
        self._dataset_stats = self._load_dataset_stats(dataset_statistics_path)
        self._action_stats = self._extract_action_stats(self._dataset_stats)
        self._chunk_cache: Dict[str, tuple[int, np.ndarray]] = {}
        self._request_id_prefix = request_id_prefix
        self._metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "action_dim": int(action_dim),
            "action_unnormalization": self._action_unnormalization_mode(),
            "supports_preempt": False,
            **metadata,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return policy metadata for the eval server."""
        return dict(self._metadata)

    def reset(self, episode_id: str) -> Dict[str, Any]:
        """Reset action cache for an episode."""
        self._chunk_cache.pop(episode_id, None)
        return {"episode_id": episode_id}

    def predict_action(
        self,
        images: Dict[str, np.ndarray],
        instruction: str,
        episode_id: str = "default",
        episode_step: int = 0,
        state: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Handle an eval RPC predict_action request with chunk caching."""
        disable_action_cache = bool(kwargs.pop("disable_action_cache", False))
        return_action_chunk = bool(kwargs.pop("return_action_chunk", False))
        cache_entry = None if disable_action_cache else self._chunk_cache.get(episode_id)
        if cache_entry is not None:
            chunk_index, chunk = cache_entry
            chunk_index += 1
            if chunk_index < chunk.shape[0]:
                self._chunk_cache[episode_id] = (chunk_index, chunk)
                return {
                    "actions": chunk if return_action_chunk else chunk[chunk_index : chunk_index + 1],
                    "inference_latency_ms": None,
                    "request_id": self._request_id(episode_id, episode_step),
                }

        image_input = self._build_image_input(images)
        start_time = time.perf_counter()
        chunk = call_predict_action(
            self._model,
            images=[image_input],
            instructions=[instruction],
            state=state,
            dataset_stats=self._dataset_stats,
            action_dim=int(self._metadata["action_dim"]),
        )
        inference_latency_ms = (time.perf_counter() - start_time) * 1000.0
        chunk = self._unnormalize_actions(chunk)
        if not disable_action_cache:
            self._chunk_cache[episode_id] = (0, chunk)
        return {
            "actions": chunk if return_action_chunk else chunk[0:1],
            "inference_latency_ms": inference_latency_ms,
            "request_id": self._request_id(episode_id, episode_step),
        }

    def _request_id(self, episode_id: str, episode_step: int) -> str:
        """Build a stable request id for eval responses."""
        return f"{self._request_id_prefix}-{episode_id}-{episode_step}"

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Apply eval-side action unnormalization when dataset statistics are present."""
        if not self._action_stats:
            return actions
        q01 = np.asarray(self._action_stats["q01"], dtype=np.float32)
        q99 = np.asarray(self._action_stats["q99"], dtype=np.float32)
        dim = min(actions.shape[-1], q01.shape[-1])
        out = actions.copy()
        out[..., :dim] = (out[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim]) + q01[:dim]
        return out

    def _action_unnormalization_mode(self) -> str:
        """Describe whether eval-side action unnormalization is active."""
        if self._action_stats and "q01" in self._action_stats and "q99" in self._action_stats:
            return "q99"
        return "none"

    @staticmethod
    def _extract_action_stats(dataset_stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract action statistics used by eval-side unnormalization."""
        if not dataset_stats:
            return None
        stats = dataset_stats.get("action")
        if not isinstance(stats, dict):
            return None
        if "q01" not in stats or "q99" not in stats:
            raise ValueError("action unnormalization requires action.q01 and action.q99")
        return stats

    @staticmethod
    def _build_image_input(images: Dict[str, np.ndarray]) -> list[np.ndarray]:
        """Convert canonical eval image views into the common predict_action image input."""
        primary = images.get("primary")
        if primary is None:
            primary = images.get("head")
        if primary is None:
            raise ValueError("images.primary or images.head is required for predict_action inference")
        image_input = [np.asarray(primary)]

        wrist = images.get("wrist")
        if wrist is None:
            wrist = images.get("right")
        if wrist is None:
            wrist = images.get("left")
        if wrist is not None:
            image_input.append(np.asarray(wrist))
        return image_input

    @staticmethod
    def _load_dataset_stats(path: str) -> Optional[Dict[str, Any]]:
        """Load dataset statistics passed to model predict_action calls."""
        if not path:
            return None
        stats_path = Path(path).expanduser()
        if not stats_path.exists():
            raise FileNotFoundError(f"dataset statistics file not found: {path}")
        with stats_path.open("r", encoding="utf-8") as f:
            return json.load(f)


class PI05ModelFactory:
    """Build a PI05 model instance that implements the common predict_action interface."""

    @classmethod
    def build(
        cls,
        ckpt_path: str,
        loongforge_root: str = "",
        device: str = "cuda",
        use_bf16: bool = True,
        dataset_statistics_path: str = "",
        tokenizer_path: str = "",
        action_dim: int = 7,
        state_dim: int = 7,
        action_horizon: int = 50,
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        random_init: bool = False,
    ) -> PredictActionModelSpec:
        """Create PI05 and return it with metadata for the generic eval policy."""
        import torch
        from transformers import PaliGemmaConfig
        from transformers.models.auto import CONFIG_MAPPING

        CONFIG_MAPPING.register("paligemma", PaliGemmaConfig, exist_ok=True)
        cls._patch_transformers_no_init_weights()
        from loongforge.embodied.model.pi05.model_configuration_pi05 import Pi05ModelConfig
        from loongforge.embodied.model.pi05.modeling_pi05 import PI05Policy

        pretrained_path = str(Path(ckpt_path).expanduser()) if ckpt_path else ""
        config = Pi05ModelConfig(
            action_dim=int(action_dim),
            state_dim=int(state_dim),
            action_horizon=int(action_horizon),
            max_action_dim=int(max_action_dim),
            max_state_dim=int(max_state_dim),
            compile_model=bool(compile_model),
            compile_mode=compile_mode,
        )
        resolved_device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        model = PI05Policy.from_pretrained(config)
        model._tokenizer_path = tokenizer_path or os.environ.get("TOKENIZER_PATH", "")
        if not random_init:
            cls._load_checkpoint(model.model, pretrained_path, resolved_device)
        model = model.to(resolved_device)
        model.eval()
        if use_bf16 and resolved_device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)

        metadata = {
            "framework": "loongforge",
            "model_type": "pi05",
            "ckpt_path": pretrained_path if not random_init else "random_init://pi05",
            "random_init": bool(random_init),
            "loongforge_root": loongforge_root,
            "action_horizon": int(action_horizon),
            "dataset_statistics_path": dataset_statistics_path,
            "tokenizer_path": model._tokenizer_path,
        }
        return PredictActionModelSpec(model=model, metadata=metadata)

    @staticmethod
    def _patch_transformers_no_init_weights() -> None:
        """Provide transformers.modeling_utils.no_init_weights for eager model imports."""
        import contextlib

        import transformers.modeling_utils as modeling_utils

        if hasattr(modeling_utils, "no_init_weights"):
            return

        @contextlib.contextmanager
        def no_init_weights() -> Any:
            yield

        modeling_utils.no_init_weights = no_init_weights

    @staticmethod
    def _load_checkpoint(model: Any, ckpt_path: str, device: Any) -> None:
        """Load a PI05 checkpoint while normalizing historical key prefixes."""
        from safetensors.torch import load_file

        path = Path(ckpt_path)
        safetensors_file = path / "model.safetensors" if path.is_dir() else path
        original_state_dict = load_file(str(safetensors_file), device=str(device))
        prefixes = (
            "model.architecture.pi05_model.",
            "architecture.pi05_model.",
            "model.pi05.",
            "pi05.",
            "model.",
        )
        fixed = {}
        for key, value in original_state_dict.items():
            new_key = key
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    break
            new_key = new_key.replace("action_time_mlp_in.", "time_mlp_in.").replace(
                "action_time_mlp_out.", "time_mlp_out."
            )
            if new_key.startswith("state_proj."):
                continue
            if new_key == "paligemma_with_expert.paligemma.lm_head.weight":
                fixed["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = value.clone()
            fixed[new_key] = value

        missing, unexpected = model.load_state_dict(fixed, strict=False)
        allowed_missing = [key for key in missing if "state_proj" not in key]
        if allowed_missing:
            raise RuntimeError(f"Missing checkpoint keys after prefix normalization: {allowed_missing[:8]}")
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys after prefix normalization: {unexpected[:8]}")


class LoongForgePI05Policy(GenericPredictActionPolicy):
    """Backward-compatible PI05 policy built from the generic predict_action policy."""

    def __init__(
        self,
        ckpt_path: str,
        loongforge_root: str = "",
        device: str = "cuda",
        use_bf16: bool = True,
        dataset_statistics_path: str = "",
        tokenizer_path: str = "",
        action_dim: int = 7,
        state_dim: int = 7,
        action_horizon: int = 50,
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        random_init: bool = False,
    ) -> None:
        """Build PI05 through its factory and attach it to the generic eval policy."""
        spec = PI05ModelFactory.build(
            ckpt_path=ckpt_path,
            loongforge_root=loongforge_root,
            device=device,
            use_bf16=use_bf16,
            dataset_statistics_path=dataset_statistics_path,
            tokenizer_path=tokenizer_path,
            action_dim=action_dim,
            state_dim=state_dim,
            action_horizon=action_horizon,
            max_action_dim=max_action_dim,
            max_state_dim=max_state_dim,
            compile_model=compile_model,
            compile_mode=compile_mode,
            random_init=random_init,
        )
        super().__init__(
            model=spec.model,
            metadata=spec.metadata,
            dataset_statistics_path=dataset_statistics_path,
            action_dim=action_dim,
            request_id_prefix="loongforge-pi05",
        )
