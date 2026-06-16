#!/usr/bin/env python3
"""Evaluate a PyTorch DnCNN checkpoint on FSA tx01/tx11 -> tx75 restoration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
JY_ROOT = ROOT.parent
TCNS_ROOT = JY_ROOT / "TCNS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TCNS_ROOT))
sys.path.insert(0, str(TCNS_ROOT / "evaluation"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from evaluation.tx75_metrics import paired_image_metrics
from fsa_dncnn.dataset import (
    LEVEL_DIRS,
    center_crop_path,
    list_target_files,
    validate_dataset_layout,
)
from fsa_dncnn.model import DnCNN


DEFAULT_DATA_DIR = TCNS_ROOT / "data" / "mydata" / "fsa_dataset_fullres"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--input_levels", nargs="+", default=["tx01", "tx11"])
    parser.add_argument("--target_level", default="tx75", choices=["tx75", "txAll"])
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results" / "fsa_dncnn_tx75" / "model_best.pt")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "fsa_dncnn_tx75" / "eval")
    parser.add_argument("--depth", type=int, default=17)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--grid_cases_per_level", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def composite(metrics: dict[str, float]) -> float:
    return float(
        0.25 * np.clip(metrics["psnr"] / 30.0, 0.0, 1.0)
        + 0.25 * np.clip(metrics["ssim"], 0.0, 1.0)
        + 0.30 * np.clip(metrics["gcnr"], 0.0, 1.0)
        + 0.20 * np.clip(metrics["cnr"] / 3.0, 0.0, 1.0)
    )


def metric_row(
    method: str,
    level: str,
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    metrics = paired_image_metrics(prediction, target)
    return {
        "method": method,
        "level": level,
        "name": name,
        **metrics,
        "composite": composite(metrics),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {"count": float(len(rows))}
    if not rows:
        return result
    for key in ("psnr", "ssim", "mae", "rmse", "gcnr", "cnr", "composite"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def save_grid(
    path: Path,
    cases: list[tuple[str, str]],
    images: dict[tuple[str, str, str], np.ndarray],
) -> None:
    if not cases:
        return
    columns = ["input", "dncnn", "tx75"]
    figure, axes = plt.subplots(
        len(cases),
        len(columns),
        figsize=(3.0 * len(columns), 2.8 * len(cases)),
        squeeze=False,
    )
    for row_index, (level, name) in enumerate(cases):
        for col_index, method in enumerate(columns):
            axes[row_index, col_index].imshow(
                images[(method, level, name)],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row_index, col_index].axis("off")
            if row_index == 0:
                axes[row_index, col_index].set_title(method)
            if col_index == 0:
                axes[row_index, col_index].set_ylabel(f"{level}\n{name[:18]}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def load_checkpoint(path: Path, args: argparse.Namespace, device: torch.device) -> DnCNN:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    depth = int(config.get("depth", args.depth))
    features = int(config.get("features", args.features))
    model = DnCNN(depth=depth, features=features).to(device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def infer_batch(
    model: DnCNN,
    batch: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(np.stack(batch)[:, None]).float().to(device)
    output = model.restore(tensor)
    return output[:, 0].cpu().numpy()


def main() -> None:
    args = parse_args()
    if set(args.input_levels) != {"tx01", "tx11"}:
        raise ValueError("--input_levels must be exactly tx01 tx11 for this benchmark")
    filenames = list_target_files(args.data_dir, args.split, args.target_level)
    if args.max_cases:
        filenames = filenames[: args.max_cases]
    validate_dataset_layout(
        args.data_dir,
        args.split,
        filenames,
        args.input_levels,
        args.target_level,
    )
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    preview_images: dict[tuple[str, str, str], np.ndarray] = {}
    grid_cases: list[tuple[str, str]] = []

    root = args.data_dir / args.split
    for level in args.input_levels:
        inputs = [
            center_crop_path(root / LEVEL_DIRS[level] / name, args.crop_size)
            for name in filenames
        ]
        targets = [
            center_crop_path(root / LEVEL_DIRS[args.target_level] / name, args.crop_size)
            for name in filenames
        ]
        outputs: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(filenames), args.batch_size),
            desc=f"Evaluating {level}",
        ):
            outputs.extend(
                infer_batch(
                    model,
                    inputs[start : start + args.batch_size],
                    device,
                )
            )
        for index, name in enumerate(filenames):
            prediction = np.asarray(outputs[index], dtype=np.float32)
            observation = np.asarray(inputs[index], dtype=np.float32)
            target = np.asarray(targets[index], dtype=np.float32)
            rows.append(metric_row("input", level, name, observation, target))
            rows.append(metric_row("dncnn", level, name, prediction, target))
            save_png(args.output_dir / "denoised" / level / name, prediction)
            if index < args.grid_cases_per_level:
                grid_cases.append((level, name))
                preview_images[("input", level, name)] = observation
                preview_images[("dncnn", level, name)] = prediction
                preview_images[("tx75", level, name)] = target

    with (args.output_dir / "per_case_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    levels = list(args.input_levels)
    summary = {}
    for method in ("input", "dncnn"):
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            "overall": aggregate(method_rows),
            "by_level": {
                level: aggregate(
                    [row for row in method_rows if row["level"] == level]
                )
                for level in levels
            },
        }
    gains = {
        "overall": {
            "psnr_gain": summary["dncnn"]["overall"]["psnr_mean"]
            - summary["input"]["overall"]["psnr_mean"],
            "ssim_gain": summary["dncnn"]["overall"]["ssim_mean"]
            - summary["input"]["overall"]["ssim_mean"],
            "composite_gain": summary["dncnn"]["overall"]["composite_mean"]
            - summary["input"]["overall"]["composite_mean"],
        },
        "by_level": {
            level: {
                "psnr_gain": summary["dncnn"]["by_level"][level]["psnr_mean"]
                - summary["input"]["by_level"][level]["psnr_mean"],
                "ssim_gain": summary["dncnn"]["by_level"][level]["ssim_mean"]
                - summary["input"]["by_level"][level]["ssim_mean"],
                "composite_gain": summary["dncnn"]["by_level"][level]["composite_mean"]
                - summary["input"]["by_level"][level]["composite_mean"],
            }
            for level in levels
        },
    }
    report = {
        "data_dir": str(args.data_dir),
        "split": args.split,
        "target_directory": LEVEL_DIRS[args.target_level],
        "checkpoint": str(args.checkpoint),
        "crop_size": args.crop_size,
        "case_count_per_level": len(filenames),
        "summary": summary,
        "gains": gains,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    save_grid(args.output_dir / "comparison_grid.png", grid_cases, preview_images)
    print(json.dumps(gains, indent=2))


if __name__ == "__main__":
    main()
