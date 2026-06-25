# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge policy adapters for the standalone eval server."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

from loongforge.embodied.eval.protocol import PROTOCOL_VERSION


class LoongForgePI05Policy:
    """Adapter from the eval RPC protocol to LoongForge PI05Policy."""

    def __init__(
        self,
        ckpt_path: str,
        loongforge_root: str = "",
        device: str = "cuda",
        use_bf16: bool = True,
        dataset_statistics_path: str = "",
        tokenizer_name: str = "google/paligemma-3b-pt-224",
        action_dim: int = 7,
        state_dim: int = 7,
        action_horizon: int = 50,
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        random_init: bool = False,
    ) -> None:
        """Run __init__."""
        import torch
        from transformers import PaliGemmaConfig
        from transformers.models.auto import CONFIG_MAPPING

        CONFIG_MAPPING.register("paligemma", PaliGemmaConfig, exist_ok=True)
        from loongforge.embodied.model.pi05.modeling_pi05 import PI05Policy

        pretrained_path = str(Path(ckpt_path).expanduser())
        config = {
            "tokenizer_name": tokenizer_name,
            "action_dim": int(action_dim),
            "state_dim": int(state_dim),
            "action_horizon": int(action_horizon),
            "max_action_dim": int(max_action_dim),
            "max_state_dim": int(max_state_dim),
            "compile_model": bool(compile_model),
            "compile_mode": compile_mode,
        }
        self._device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self._policy = PI05Policy.from_pretrained(config)
        if not random_init:
            self._load_checkpoint(self._policy.model, pretrained_path, self._device)
        self._policy = self._policy.to(self._device)
        self._policy.eval()
        if use_bf16 and self._device.type == "cuda":
            self._policy = self._policy.to(dtype=torch.bfloat16)
        self._dataset_stats = self._load_dataset_stats(dataset_statistics_path)
        self._action_stats = self._extract_action_stats(self._dataset_stats)
        self._chunk_cache: Dict[str, tuple[int, np.ndarray]] = {}
        self._metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "framework": "loongforge",
            "model_type": "pi05",
            "ckpt_path": pretrained_path if not random_init else "random_init://pi05",
            "random_init": bool(random_init),
            "loongforge_root": loongforge_root,
            "action_dim": int(action_dim),
            "action_horizon": int(action_horizon),
            "action_unnormalization": (
                "q99" if self._action_stats and "q01" in self._action_stats and "q99" in self._action_stats else "none"
            ),
            "supports_preempt": False,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        """Run metadata."""
        return dict(self._metadata)

    def reset(self, episode_id: str) -> Dict[str, Any]:
        """Run reset."""
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
        """Run predict_action."""
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
                    "request_id": f"loongforge-pi05-{episode_id}-{episode_step}",
                }

        image_input = self._build_image_input(images)
        start_time = time.perf_counter()
        result = self._predict_action_chunk(image_input, instruction)
        inference_latency_ms = (time.perf_counter() - start_time) * 1000.0
        chunk = np.asarray(result, dtype=np.float32).reshape(-1, result.shape[-1])
        chunk = self._unnormalize_actions(chunk)
        if not disable_action_cache:
            self._chunk_cache[episode_id] = (0, chunk)
        return {
            "actions": chunk if return_action_chunk else chunk[0:1],
            "inference_latency_ms": inference_latency_ms,
            "request_id": f"loongforge-pi05-{episode_id}-{episode_step}",
        }

    def _predict_action_chunk(self, image_input: list[np.ndarray], instruction: str) -> Any:
        """Run _predict_action_chunk."""
        import torch
        from loongforge.embodied.data.transforms.image_transform import ImageTransform
        from loongforge.embodied.model.pi05.modeling_pi05 import tokenize_prompts

        image_size = int(getattr(getattr(self._policy.model, "config", None), "image_resolution", (224, 224))[0])
        image_transform = ImageTransform(apply_to=[], image_size=image_size)
        images_list = []
        img_masks = []
        model_dtype = next(self._policy.parameters()).dtype
        for view_index in range(2):
            if view_index < len(image_input):
                view = image_transform.process_batch([image_input[view_index]]).to(
                    device=self._device, dtype=model_dtype
                )
                mask = torch.ones(1, dtype=torch.bool, device=self._device)
            else:
                view = torch.zeros(
                    1,
                    3,
                    image_size,
                    image_size,
                    dtype=model_dtype,
                    device=self._device,
                )
                mask = torch.zeros(1, dtype=torch.bool, device=self._device)
            images_list.append(view)
            img_masks.append(mask)

        prompt = f"Task: {instruction.strip()};\nAction: "
        tokenized = tokenize_prompts([prompt], self._policy.tokenizer, max_length=200)
        batch = SimpleNamespace(
            images_list=images_list,
            img_masks=img_masks,
            input_ids=tokenized["input_ids"].to(self._device),
            attention_mask=tokenized["attention_mask"].bool().to(self._device),
        )
        return self._policy.predict_action_chunk(batch).detach().cpu().numpy()

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Run _unnormalize_actions."""
        if not self._action_stats:
            return actions
        q01 = np.asarray(self._action_stats["q01"], dtype=np.float32)
        q99 = np.asarray(self._action_stats["q99"], dtype=np.float32)
        dim = min(actions.shape[-1], q01.shape[-1])
        out = actions.copy()
        out[..., :dim] = (out[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim]) + q01[:dim]
        return out

    @staticmethod
    def _extract_action_stats(dataset_stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Run _extract_action_stats."""
        if not dataset_stats:
            return None
        stats = dataset_stats.get("action")
        if not isinstance(stats, dict):
            return None
        if "q01" not in stats or "q99" not in stats:
            raise ValueError("LoongForge pi05 action unnormalization requires action.q01 and action.q99")
        return stats

    @staticmethod
    def _load_checkpoint(model: Any, ckpt_path: str, device: Any) -> None:
        """Run _load_checkpoint."""
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

    @staticmethod
    def _build_image_input(images: Dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run _build_image_input."""
        primary = images.get("primary")
        if primary is None:
            primary = images.get("head")
        if primary is None:
            raise ValueError("images.primary or images.head is required for LoongForge pi05 inference")
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
        """Run _load_dataset_stats."""
        if not path:
            return None
        stats_path = Path(path).expanduser()
        if not stats_path.exists():
            raise FileNotFoundError(f"dataset statistics file not found: {path}")
        with stats_path.open("r", encoding="utf-8") as f:
            return json.load(f)
