"""FSA paired datasets for the PyTorch DnCNN TCNS benchmark."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


LEVEL_DIRS = {
    "tx01": "input_tx01",
    "tx11": "input_tx11",
    "tx75": "input_tx75",
    "txAll": "gt_txAll",
}


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def reflect_pad_to(image: np.ndarray, size: int) -> np.ndarray:
    """Reflect-pad a 2-D image so both spatial axes are at least size."""
    height, width = image.shape
    pad_h = max(0, size - height)
    pad_w = max(0, size - width)
    if not (pad_h or pad_w):
        return image
    top = pad_h // 2
    left = pad_w // 2
    return np.pad(
        image,
        ((top, pad_h - top), (left, pad_w - left)),
        mode="reflect",
    )


def crop_origin(
    height: int,
    width: int,
    size: int,
    mode: str,
    rng: random.Random | None,
) -> tuple[int, int]:
    max_row = height - size
    max_col = width - size
    if max_row < 0 or max_col < 0:
        raise ValueError(f"Cannot crop {size} from shape {(height, width)}")
    if mode == "random":
        if rng is None:
            raise ValueError("random crop requires an RNG")
        return rng.randint(0, max_row), rng.randint(0, max_col)
    if mode == "center":
        return max_row // 2, max_col // 2
    raise ValueError(f"Unknown crop mode {mode!r}")


def center_crop_array(image: np.ndarray, size: int) -> np.ndarray:
    padded = reflect_pad_to(image, size)
    row, col = crop_origin(padded.shape[0], padded.shape[1], size, "center", None)
    return np.ascontiguousarray(padded[row : row + size, col : col + size])


def center_crop_path(path: Path, size: int) -> np.ndarray:
    return center_crop_array(load_gray(path), size).astype(np.float32) / 255.0


def acquisition_group(filename: str) -> str:
    """Match the acquisition grouping used by TCNS conditional training."""
    parts = Path(filename).stem.split("_")
    date_index = next(
        (
            index
            for index, part in enumerate(parts)
            if len(part) == 8 and part.isdigit() and part.startswith("20")
        ),
        len(parts) - 1,
    )
    source = parts[0]
    if source == "murine":
        return "_".join(parts[: date_index + 1])
    time_value = parts[date_index + 1] if date_index + 1 < len(parts) else "unknown"
    return f"{'_'.join(parts[: date_index + 1])}_{time_value[:2]}h"


def split_files_by_acquisition(
    files: list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if validation_fraction <= 0:
        return list(files), [], []
    groups: dict[str, list[str]] = defaultdict(list)
    for filename in files:
        groups[acquisition_group(filename)].append(filename)
    ordered_groups = sorted(
        groups,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
    )
    target_count = max(1, round(len(files) * validation_fraction))
    selected_groups: list[str] = []
    selected_count = 0
    for group in ordered_groups:
        if selected_groups and selected_count >= target_count:
            break
        selected_groups.append(group)
        selected_count += len(groups[group])
    validation_set = {
        filename
        for group in selected_groups
        for filename in groups[group]
    }
    train_files = [filename for filename in files if filename not in validation_set]
    validation_files = [filename for filename in files if filename in validation_set]
    if not train_files:
        raise ValueError("Validation split consumed all training files")
    return train_files, validation_files, selected_groups


def list_target_files(data_dir: str | Path, split: str, target_level: str) -> list[str]:
    target_dir = Path(data_dir) / split / LEVEL_DIRS[target_level]
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")
    files = sorted(path.name for path in target_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files found in {target_dir}")
    return files


def validate_dataset_layout(
    data_dir: str | Path,
    split: str,
    files: list[str],
    input_levels: list[str],
    target_level: str,
) -> None:
    root = Path(data_dir) / split
    directories = [LEVEL_DIRS[level] for level in input_levels]
    directories.append(LEVEL_DIRS[target_level])
    missing: list[str] = []
    mismatches: list[tuple[str, dict[str, tuple[int, int]]]] = []
    for filename in files:
        shapes = {}
        for directory in directories:
            path = root / directory / filename
            if not path.is_file():
                missing.append(str(path))
                continue
            with Image.open(path) as image:
                shapes[directory] = image.size
        if len(set(shapes.values())) > 1:
            mismatches.append((filename, shapes))
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing paired images:\n{preview}")
    if mismatches:
        preview = "\n".join(
            f"{filename}: {shapes}" for filename, shapes in mismatches[:20]
        )
        raise ValueError(f"Paired image size mismatch:\n{preview}")


def validate_train_test_isolation(data_dir: str | Path, train_files: list[str]) -> None:
    test_dir = Path(data_dir) / "test" / LEVEL_DIRS["tx75"]
    if not test_dir.is_dir():
        return
    train_groups = {acquisition_group(filename) for filename in train_files}
    test_groups = {acquisition_group(path.name) for path in test_dir.glob("*.png")}
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise ValueError(
            "Acquisition leakage between train and test splits: "
            + ", ".join(overlap)
        )


class FSADnCNNDataset(Dataset):
    """Balanced tx01/tx11 sparse-transmit inputs paired with tx75 targets."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        files: list[str],
        input_levels: list[str],
        target_level: str,
        crop_size: int,
        repeat: int = 1,
        mode: str = "center",
        augment: bool = False,
        seed: int = 0,
        max_black_fraction: float = 0.5,
        crop_tries: int = 10,
    ) -> None:
        self.root = Path(data_dir) / split
        self.files = list(files)
        self.input_levels = list(input_levels)
        self.target_level = target_level
        self.crop_size = int(crop_size)
        self.repeat = max(1, int(repeat))
        self.mode = mode
        self.augment = bool(augment)
        self.seed = int(seed)
        self.max_black_fraction = float(max_black_fraction)
        self.crop_tries = int(crop_tries)
        self.epoch = 0
        if not self.files:
            raise ValueError("Dataset file list is empty")
        if target_level not in LEVEL_DIRS:
            raise ValueError(f"Unknown target_level {target_level!r}")
        for level in self.input_levels:
            if level not in {"tx01", "tx11"}:
                raise ValueError(f"DnCNN inputs must be tx01/tx11, got {level!r}")
        self.target_dir = self.root / LEVEL_DIRS[target_level]

    def __len__(self) -> int:
        return len(self.files) * self.repeat * len(self.input_levels)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_origin(
        self,
        target: np.ndarray,
        rng: random.Random,
    ) -> tuple[int, int]:
        if self.mode != "random":
            return crop_origin(
                target.shape[0],
                target.shape[1],
                self.crop_size,
                "center",
                None,
            )
        best_origin = None
        best_black = float("inf")
        for _ in range(max(1, self.crop_tries)):
            row, col = crop_origin(
                target.shape[0],
                target.shape[1],
                self.crop_size,
                "random",
                rng,
            )
            patch = target[row : row + self.crop_size, col : col + self.crop_size]
            black_fraction = float((patch <= 8).mean())
            if black_fraction <= self.max_black_fraction:
                return row, col
            if black_fraction < best_black:
                best_origin = (row, col)
                best_black = black_fraction
        if best_origin is None:
            raise RuntimeError("Failed to sample a crop origin")
        return best_origin

    def __getitem__(self, index: int) -> dict[str, object]:
        level_index = index % len(self.input_levels)
        pair_index = index // len(self.input_levels)
        file_index = pair_index % len(self.files)
        filename = self.files[file_index]
        level = self.input_levels[level_index]

        target = reflect_pad_to(
            load_gray(self.target_dir / filename), self.crop_size
        )
        observation = reflect_pad_to(
            load_gray(self.root / LEVEL_DIRS[level] / filename), self.crop_size
        )
        if observation.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {filename}: {observation.shape} vs {target.shape}"
            )
        rng = random.Random(
            (
                self.seed * 1_000_003
                + self.epoch * 10_000_019
                + pair_index
            )
            & 0xFFFFFFFF
        )
        row, col = self._sample_origin(target, rng)
        crop = (slice(row, row + self.crop_size), slice(col, col + self.crop_size))
        target = target[crop]
        observation = observation[crop]
        if self.augment and rng.random() < 0.5:
            target = target[:, ::-1]
            observation = observation[:, ::-1]

        target_tensor = torch.from_numpy(
            np.ascontiguousarray(target).astype(np.float32) / 255.0
        ).unsqueeze(0)
        observation_tensor = torch.from_numpy(
            np.ascontiguousarray(observation).astype(np.float32) / 255.0
        ).unsqueeze(0)
        return {
            "input": observation_tensor,
            "target": target_tensor,
            "level": level,
            "name": filename,
        }
