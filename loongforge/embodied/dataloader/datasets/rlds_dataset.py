"""
RLDS Dataset Adapter - Load RLDS/TFDS format datasets

Supports Open X-Embodiment standard format (via HuggingFace datasets or tensorflow_datasets):
  - Bridge V2
  - RT-1 (Fractal)
  - Kuka
  - Language Table
  - And 70+ datasets in OXE

RLDS data format:
  Each episode is a dict:
    {
        "steps": [{
            "observation": {
                "image": (H, W, 3) uint8,
                "state": (D,) float32,
            },
            "action": (7,) float32,
            "language_instruction": str,
            "is_terminal": bool,
        }, ...]
    }

Usage (from config):
    datasets:
      vla_data:
        dataloader_module: rlds_datasets
        dataset_name: bridge_v2       # HuggingFace dataset name
        split: train
        per_device_batch_size: 4
        action_horizon: 7
        image_size: 224
        streaming: true               # Recommended streaming mode for large datasets

Usage (programmatic):
    from dataloader.datasets.rlds_dataset import RLDSVLADataset

    dataset = RLDSVLADataset(
        dataset_name="bridge_v2",
        split="train[:1000]",
        action_horizon=7,
    )
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

from dataloader import collate_fn
from dataloader.transforms import ActionTransform, ComposedTransform, ImageTransform

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# OXE dataset name mapping (HuggingFace Hub paths)
# ═══════════════════════════════════════════════════════════════

OXE_DATASET_REGISTRY = {
    "bridge_v2": "rail-berkeley/bridge_v2",
    "rt1": "google/fractal20220817_data",
    "kuka": "google/kuka",
    "language_table": "google/language_table",
    "taco_play": "taco-robot/taco_play",
    "jaco_play": "jaco-robot/jaco_play",
    "berkeley_cable_routing": "rail-berkeley/cable_routing",
    "berkeley_autolab_ur5": "rail-berkeley/autolab_ur5",
    "roboturk": "roboturk/roboturk",
    "columbia_cairlab_pusht_real": "columbia/cairlab_pusht_real",
}


class RLDSVLADataset(IterableDataset):
    """
    RLDS format VLA dataset (via HuggingFace datasets streaming).

    Features:
      - Large dataset friendly (streaming mode, no full download needed)
      - Automatic episode boundary handling
      - Supports action chunking
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        action_horizon: int = 7,
        image_size: int = 224,
        image_key: str = "image",
        state_key: Optional[str] = "state",
        action_key: str = "action",
        language_key: str = "language_instruction",
        streaming: bool = True,
        normalization_mode: str = "min_max",
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name or local path
            split: Dataset split
            action_horizon: Action chunk length
            image_size: Image size
            image_key: Image key in observation
            state_key: State key in observation (None to skip loading)
            action_key: Action key
            language_key: Language instruction key
            streaming: Whether to use streaming mode
            normalization_mode: Normalization mode
        """
        self.dataset_name = dataset_name
        self.split = split
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.image_key = image_key
        self.state_key = state_key
        self.action_key = action_key
        self.language_key = language_key
        self.streaming = streaming
        self.normalization_mode = normalization_mode

        # Resolve HuggingFace path
        self.hf_path = OXE_DATASET_REGISTRY.get(dataset_name, dataset_name)

        # Image transform (no action transform here since we don't have stats upfront)
        self.image_transform = ImageTransform(
            apply_to=["image"],
            size=(image_size, image_size),
            crop_scale=0.95,
            color_jitter=True,
        )

        logger.info(
            f"RLDSVLADataset: {dataset_name} ({self.hf_path}), "
            f"split={split}, streaming={streaming}"
        )

    def __iter__(self):
        """Iterate episodes, expand into (obs, action_chunk) samples."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library required. Install with: pip install datasets"
            )

        # Load dataset
        ds = load_dataset(
            self.hf_path,
            split=self.split,
            streaming=self.streaming,
            trust_remote_code=True,
        )

        for episode in ds:
            steps = episode.get("steps", [episode])
            if not isinstance(steps, list):
                steps = list(steps)

            # Collect all actions for chunking
            episode_actions = []
            for step in steps:
                action = step.get(self.action_key, step.get("action", np.zeros(7)))
                if isinstance(action, (list, np.ndarray)):
                    episode_actions.append(np.array(action, dtype=np.float32))
                else:
                    episode_actions.append(np.zeros(7, dtype=np.float32))

            # Yield each step with action chunk
            for t, step in enumerate(steps):
                # Image
                obs = step.get("observation", step)
                img_data = obs.get(self.image_key, None)
                if img_data is not None:
                    if isinstance(img_data, np.ndarray):
                        img = Image.fromarray(img_data.astype(np.uint8)).convert("RGB")
                    elif isinstance(img_data, Image.Image):
                        img = img_data.convert("RGB")
                    else:
                        img = Image.new("RGB", (self.image_size, self.image_size))
                else:
                    img = Image.new("RGB", (self.image_size, self.image_size))

                # Action chunk
                action_chunk = episode_actions[t: t + self.action_horizon]
                if len(action_chunk) < self.action_horizon:
                    pad_len = self.action_horizon - len(action_chunk)
                    padding = [action_chunk[-1]] * pad_len
                    action_chunk = action_chunk + padding
                action_chunk = np.stack(action_chunk, axis=0)

                # State
                state = None
                if self.state_key and self.state_key in obs:
                    state_data = obs[self.state_key]
                    if isinstance(state_data, (list, np.ndarray)):
                        state = np.array(state_data, dtype=np.float32).reshape(1, -1)

                # Language
                lang = step.get(self.language_key, "perform the task")
                if isinstance(lang, bytes):
                    lang = lang.decode()

                sample = {
                    "image": [img],
                    "lang": lang,
                    "action": action_chunk,
                    "state": state,
                }

                # Apply image transform
                sample = self.image_transform(sample)
                yield sample


def build_rlds_dataloader(cfg, vla_cfg, batch_size: int, num_workers: int) -> DataLoader:
    """Build RLDS dataloader."""
    dataset_name = vla_cfg.get("dataset_name", "bridge_v2")
    split = vla_cfg.get("split", "train")
    action_horizon = vla_cfg.get("action_horizon", 7)
    image_size = vla_cfg.get("image_size", 224)
    streaming = vla_cfg.get("streaming", True)
    normalization_mode = vla_cfg.get("normalization_mode", "min_max")

    dataset = RLDSVLADataset(
        dataset_name=dataset_name,
        split=split,
        action_horizon=action_horizon,
        image_size=image_size,
        streaming=streaming,
        normalization_mode=normalization_mode,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
