"""Video dataset for activity-biometrics.

All three benchmarks (NTU RGB-AB, Charades-AB, MEVID) are consumed
through one annotation schema, produced by the ``tools/prepare_*.py`` scripts::

    {
      "name": "ntu_rgb_ab",
      "num_actions": 94,
      "action_names": ["drink water", ...],
      "multilabel": false,
      "splits": {
        "train":   [<sample>, ...],
        "query":   [<sample>, ...],
        "gallery": [<sample>, ...]
      }
    }

where each ``<sample>`` is::

    {
      "id":         "S001C001P001R001A001",   # unique tracklet id
      "frames_dir": "ntu_rgb_ab/frames/S001C001P001R001A001",  # relative to data_root
      "num_frames": 103,
      "video":      null,          # alternative to frames_dir: a video file
      "pid":        12,            # identity label
      "action":     7,             # int, or list[int] when multilabel
      "view":       2,             # camera / viewpoint id
      "keyframe":   51             # frame the VLM prompts were generated from
    }

Identity labels are remapped to a contiguous ``[0, num_train_identities)``
range for the train split (the classifier needs that); query/gallery keep the
original ids so retrieval can match across splits.
"""

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from disenq.data.prompts import PROMPT_KEYS, PromptBank
from disenq.data.transforms import ClipTransform
from disenq.utils.logger import get_logger


def load_annotation(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def primary_action(action) -> int:
    """Single action id for a sample; ``-1`` when unlabelled.

    Multi-label samples (Charades-AB) collapse to their first label, which is
    only used for the cross-activity gallery filter, never for the loss.
    """
    if action is None:
        return -1
    if isinstance(action, (list, tuple)):
        return int(action[0]) if len(action) else -1
    return int(action)


def sample_frame_indices(
    num_frames: int,
    num_sampled: int = 8,
    stride: int = 4,
    mode: str = "train",
    sampling: str = "stride",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Pick ``num_sampled`` frame indices from a clip of ``num_frames`` frames.

    ``sampling="stride"`` implements the paper's recipe -- "8 frames which are
    randomly selected with a stride of 4 from each original video" -- as a
    contiguous strided window whose offset is random during training and
    centred during evaluation.

    ``sampling="linspace"`` spreads the frames over the whole clip instead
    (random within each of ``num_sampled`` equal segments when training), which
    is the better choice for long untrimmed videos such as Charades.

    Indices are clamped into range, so short clips repeat their last frames.
    """
    num_frames = max(int(num_frames), 1)
    rng = rng or np.random.default_rng()

    if sampling == "linspace":
        bounds = np.linspace(0, num_frames, num_sampled + 1)
        if mode == "train":
            lows, highs = bounds[:-1], bounds[1:]
            indices = lows + rng.random(num_sampled) * np.maximum(highs - lows, 1e-6)
            indices = np.floor(indices)
        else:
            indices = np.rint((bounds[:-1] + bounds[1:]) / 2.0 - 0.5)
        return np.clip(indices, 0, num_frames - 1).astype(np.int64)

    if sampling != "stride":
        raise ValueError(f"unknown sampling '{sampling}'; use 'stride' or 'linspace'")

    if num_sampled > 1 and num_frames < (num_sampled - 1) * stride + 1:
        # Clip too short for the requested stride. Shrink the stride to the
        # largest value that still fits rather than clamping, which would
        # otherwise return many copies of the final frame.
        stride = max((num_frames - 1) // (num_sampled - 1), 1)

    span = (num_sampled - 1) * stride + 1
    slack = num_frames - span
    if slack > 0:
        start = int(rng.integers(0, slack + 1)) if mode == "train" else slack // 2
    else:
        start = 0
    indices = start + np.arange(num_sampled) * stride
    return np.clip(indices, 0, num_frames - 1).astype(np.int64)


class VideoReIDDataset(Dataset):
    """One split of an activity-biometrics benchmark."""

    def __init__(
        self,
        samples: List[Dict],
        data_root: str,
        transform: ClipTransform,
        num_sampled_frames: int = 8,
        frame_stride: int = 4,
        sampling: str = "stride",
        mode: str = "train",
        frame_template: str = "frame{:06d}.jpg",
        frame_index_base: int = 1,
        num_actions: int = 1,
        multilabel: bool = False,
        prompt_bank: Optional[PromptBank] = None,
        pid_map: Optional[Dict[int, int]] = None,
        cached_vit_features_root: Optional[str] = None,
        load_prompts: bool = True,
    ):
        self.samples = samples
        self.data_root = data_root
        self.transform = transform
        self.num_sampled_frames = num_sampled_frames
        self.frame_stride = frame_stride
        self.sampling = sampling
        self.mode = mode
        self.frame_template = frame_template
        self.frame_index_base = frame_index_base
        self.num_actions = num_actions
        self.multilabel = multilabel
        self.prompt_bank = prompt_bank or PromptBank()
        self.pid_map = pid_map
        self.cached_vit_features_root = cached_vit_features_root
        self.load_prompts = load_prompts
        self._video_readers = {}

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------
    @property
    def pids(self) -> np.ndarray:
        return np.array([s["pid"] for s in self.samples], dtype=np.int64)

    @property
    def views(self) -> np.ndarray:
        return np.array([s.get("view", 0) for s in self.samples], dtype=np.int64)

    @property
    def actions(self) -> List:
        return [s.get("action", -1) for s in self.samples]

    def mapped_pid(self, raw_pid: int) -> int:
        if self.pid_map is None:
            return int(raw_pid)
        return int(self.pid_map.get(int(raw_pid), -1))

    def _action_target(self, action):
        if self.multilabel:
            target = torch.zeros(self.num_actions, dtype=torch.float32)
            if action is None:
                return target
            if isinstance(action, (list, tuple, set)):
                for a in action:
                    if 0 <= int(a) < self.num_actions:
                        target[int(a)] = 1.0
            elif 0 <= int(action) < self.num_actions:
                target[int(action)] = 1.0
            return target
        if action is None:
            return torch.tensor(-1, dtype=torch.long)
        if isinstance(action, (list, tuple)):
            action = action[0] if action else -1
        return torch.tensor(int(action), dtype=torch.long)

    # ------------------------------------------------------------------
    # frame / feature loading
    # ------------------------------------------------------------------
    def _resolve(self, relative: str) -> str:
        if os.path.isabs(relative):
            return relative
        return os.path.join(self.data_root, relative)

    def _load_cached_features(self, sample: Dict, indices: np.ndarray) -> Optional[torch.Tensor]:
        """Fetch pre-extracted ViT tokens for the requested original frame ids.

        ``tools/extract_vit_features.py`` may cache only a uniform subset of a
        clip's frames, so the payload records which original frame each cached
        row came from. Requested ids are mapped to the nearest cached frame.
        A bare tensor (no index record) is treated as a dense, frame-aligned
        cache.
        """
        if not self.cached_vit_features_root:
            return None
        path = os.path.join(self.cached_vit_features_root, f"{sample['id']}.pt")
        if not os.path.isabs(path):
            path = self._resolve(path)
        if not os.path.exists(path):
            return None

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            features = payload["features"]
            cached_ids = np.asarray(payload.get("frame_indices"))
        else:
            features = payload
            cached_ids = None

        if cached_ids is None or cached_ids.size != features.shape[0]:
            rows = np.clip(indices, 0, features.shape[0] - 1)
        else:
            # Nearest cached frame for each requested original frame id.
            positions = np.searchsorted(cached_ids, indices)
            positions = np.clip(positions, 1, len(cached_ids) - 1)
            left, right = cached_ids[positions - 1], cached_ids[positions]
            take_left = np.abs(indices - left) <= np.abs(right - indices)
            rows = np.where(take_left, positions - 1, positions)

        return features[torch.from_numpy(np.ascontiguousarray(rows))].float()

    def _frame_path(self, sample: Dict, frames_dir: str, index: int) -> str:
        """Resolve one frame of a sample to a file path.

        Three layouts are supported:

        * ``frame_names`` -- an explicit per-frame file list, used by tracklet
          datasets such as MEVID whose frames are named arbitrarily;
        * ``frame_offset`` -- the clip is a segment of a longer video whose
          frames all live in one directory, as for the action segments cut out
          of an untrimmed recording;
        * otherwise ``frame_template`` formatted with the frame number.
        """
        names = sample.get("frame_names")
        if names:
            name = names[min(int(index), len(names) - 1)]
            return name if os.path.isabs(name) else os.path.join(frames_dir, os.path.basename(name))

        offset = int(sample.get("frame_offset", 0))
        return os.path.join(
            frames_dir, self.frame_template.format(int(index) + offset + self.frame_index_base)
        )

    def _load_frames_from_dir(self, sample: Dict, indices: np.ndarray) -> torch.Tensor:
        from PIL import Image

        frames_dir = self._resolve(sample["frames_dir"])
        images = []
        for i in indices:
            path = self._frame_path(sample, frames_dir, int(i))
            with Image.open(path) as img:
                images.append(torch.from_numpy(np.array(img.convert("RGB"))))
        clip = torch.stack(images).permute(0, 3, 1, 2)  # [T, 3, H, W], uint8
        return clip.float() / 255.0

    def _load_frames_from_video(self, sample: Dict, indices: np.ndarray) -> torch.Tensor:
        import decord

        path = self._resolve(sample["video"])
        reader = decord.VideoReader(path, num_threads=1)
        offset = int(sample.get("frame_offset", 0))
        idx = np.clip(indices + offset, 0, len(reader) - 1)
        batch = reader.get_batch(idx.tolist()).asnumpy()  # [T, H, W, 3]
        clip = torch.from_numpy(batch).permute(0, 3, 1, 2)
        return clip.float() / 255.0

    def _num_frames(self, sample: Dict) -> int:
        if sample.get("num_frames"):
            return int(sample["num_frames"])
        if sample.get("frame_names"):
            return len(sample["frame_names"])
        if sample.get("frames_dir"):
            frames_dir = self._resolve(sample["frames_dir"])
            count = len([f for f in os.listdir(frames_dir) if not f.startswith(".")])
            sample["num_frames"] = count
            return count
        if sample.get("video"):
            import decord

            count = len(decord.VideoReader(self._resolve(sample["video"]), num_threads=1))
            sample["num_frames"] = count
            return count
        raise ValueError(f"sample {sample.get('id')} has neither frames_dir nor video")

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        total_frames = self._num_frames(sample)
        indices = sample_frame_indices(
            total_frames,
            num_sampled=self.num_sampled_frames,
            stride=self.frame_stride,
            mode=self.mode,
            sampling=self.sampling,
        )

        item = {
            "index": index,
            "id": sample["id"],
            "pid": torch.tensor(self.mapped_pid(sample["pid"]), dtype=torch.long),
            "raw_pid": torch.tensor(int(sample["pid"]), dtype=torch.long),
            "view": torch.tensor(int(sample.get("view", 0)), dtype=torch.long),
            "action": self._action_target(sample.get("action")),
            "action_id": torch.tensor(primary_action(sample.get("action")), dtype=torch.long),
            "frame_mask": torch.ones(self.num_sampled_frames, dtype=torch.float32),
        }

        cached = self._load_cached_features(sample, indices)
        if cached is not None:
            item["vit_features"] = cached
        else:
            if sample.get("frames_dir"):
                clip = self._load_frames_from_dir(sample, indices)
            elif sample.get("video"):
                clip = self._load_frames_from_video(sample, indices)
            else:
                raise ValueError(f"sample {sample['id']} has neither frames_dir nor video")
            item["frames"] = self.transform(clip)

        if self.load_prompts:
            item["prompts"] = self.prompt_bank.get(sample["id"], sample["pid"])

        return item


def collate_video_batch(batch: Sequence[Dict]) -> Dict:
    """Collate that keeps the three prompt categories as lists of strings."""
    out: Dict = {}
    tensor_keys = [k for k, v in batch[0].items() if isinstance(v, torch.Tensor)]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch])

    out["id"] = [item["id"] for item in batch]
    out["index"] = torch.tensor([item["index"] for item in batch], dtype=torch.long)

    if "prompts" in batch[0]:
        out["prompts"] = {
            key: [item["prompts"].get(key, "") for item in batch] for key in PROMPT_KEYS
        }
    return out


def build_pid_map(samples: Sequence[Dict]) -> Dict[int, int]:
    """Map the raw identity labels of a split onto ``[0, num_identities)``."""
    unique = sorted({int(s["pid"]) for s in samples})
    return {pid: i for i, pid in enumerate(unique)}


def describe_split(name: str, samples: Sequence[Dict]) -> str:
    pids = {int(s["pid"]) for s in samples}
    views = {int(s.get("view", 0)) for s in samples}
    return (
        f"{name:>8}: {len(samples):6d} clips | {len(pids):4d} identities | "
        f"{len(views):3d} views"
    )


def log_dataset_summary(annotation: Dict) -> None:
    logger = get_logger()
    logger.info(
        f"dataset '{annotation.get('name', '?')}' "
        f"({annotation.get('num_actions')} actions, "
        f"multilabel={annotation.get('multilabel', False)})"
    )
    for split, samples in annotation["splits"].items():
        logger.info("  " + describe_split(split, samples))
