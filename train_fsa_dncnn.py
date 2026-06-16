#!/usr/bin/env python3
"""Train a PyTorch DnCNN baseline for tx01/tx11 -> tx75 restoration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
JY_ROOT = ROOT.parent
TCNS_ROOT = JY_ROOT / "TCNS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TCNS_ROOT))
sys.path.insert(0, str(TCNS_ROOT / "evaluation"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.tx75_metrics import paired_image_metrics
from fsa_dncnn.dataset import (
    FSADnCNNDataset,
    LEVEL_DIRS,
    list_target_files,
    split_files_by_acquisition,
    validate_dataset_layout,
    validate_train_test_isolation,
)
from fsa_dncnn.model import DnCNN


DEFAULT_DATA_DIR = TCNS_ROOT / "data" / "mydata" / "fsa_dataset_fullres"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--input_levels", nargs="+", default=["tx01", "tx11"])
    parser.add_argument("--target_level", default="tx75", choices=["tx75", "txAll"])
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step_epoch", type=int, default=50)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=17)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--loss", choices=["mse", "l1"], default="mse")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_black_fraction", type=float, default=0.5)
    parser.add_argument("--crop_tries", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results" / "fsa_dncnn_tx75")
    parser.add_argument("--verify_data_only", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_names(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input": torch.stack([item["input"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "level": [str(item["level"]) for item in batch],
        "name": [str(item["name"]) for item in batch],
    }


def composite(metrics: dict[str, float]) -> float:
    return float(
        0.25 * np.clip(metrics["psnr"] / 30.0, 0.0, 1.0)
        + 0.25 * np.clip(metrics["ssim"], 0.0, 1.0)
        + 0.30 * np.clip(metrics["gcnr"], 0.0, 1.0)
        + 0.20 * np.clip(metrics["cnr"] / 3.0, 0.0, 1.0)
    )


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {"count": float(len(rows))}
    if not rows:
        return result
    for key in ("psnr", "ssim", "mae", "rmse", "gcnr", "cnr", "composite"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


@torch.no_grad()
def evaluate(
    model: DnCNN,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        outputs = model.restore(inputs)
        for prediction, target in zip(outputs.cpu().numpy(), targets.cpu().numpy()):
            metrics = paired_image_metrics(prediction[0], target[0])
            metrics["composite"] = composite(metrics)
            rows.append(metrics)
    return aggregate(rows)


def make_checkpoint(
    model: DnCNN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_psnr: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
        "config": config,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if set(args.input_levels) != {"tx01", "tx11"}:
        raise ValueError("--input_levels must be exactly tx01 tx11 for this benchmark")
    files = list_target_files(args.data_dir, args.split, args.target_level)
    validate_dataset_layout(
        args.data_dir,
        args.split,
        files,
        args.input_levels,
        args.target_level,
    )
    train_files, val_files, val_groups = split_files_by_acquisition(
        files,
        args.val_fraction,
        args.seed,
    )
    validate_train_test_isolation(args.data_dir, files)
    test_files = []
    test_root = args.data_dir / "test" / LEVEL_DIRS[args.target_level]
    if test_root.is_dir():
        test_files = sorted(path.name for path in test_root.glob("*.png"))
        validate_dataset_layout(
            args.data_dir,
            "test",
            test_files,
            args.input_levels,
            args.target_level,
        )

    summary = {
        "data_dir": str(args.data_dir),
        "target_directory": LEVEL_DIRS[args.target_level],
        "train_split_files": len(files),
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "test_files": len(test_files),
        "validation_groups": val_groups,
        "input_levels": args.input_levels,
    }
    if args.verify_data_only:
        print(json.dumps(summary, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(summary)
    config["data_dir"] = str(args.data_dir)
    config["output_dir"] = str(args.output_dir)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))

    train_dataset = FSADnCNNDataset(
        args.data_dir,
        args.split,
        train_files,
        args.input_levels,
        args.target_level,
        args.crop_size,
        repeat=args.repeat,
        mode="random",
        augment=True,
        seed=args.seed,
        max_black_fraction=args.max_black_fraction,
        crop_tries=args.crop_tries,
    )
    val_dataset = FSADnCNNDataset(
        args.data_dir,
        args.split,
        val_files,
        args.input_levels,
        args.target_level,
        args.crop_size,
        repeat=1,
        mode="center",
        augment=False,
        seed=args.seed,
        max_black_fraction=1.0,
        crop_tries=1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_names,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_names,
    )

    device = torch.device(args.device)
    model = DnCNN(depth=args.depth, features=args.features).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, args.lr_step_epoch),
        gamma=args.lr_gamma,
    )

    log_path = args.output_dir / "train_log.csv"
    best_psnr = float("-inf")
    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "val_psnr",
                "val_ssim",
                "val_gcnr",
                "val_cnr",
                "val_composite",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_dataset.set_epoch(epoch)
            total_loss = 0.0
            total_count = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            for batch_index, batch in enumerate(progress):
                if args.max_train_batches and batch_index >= args.max_train_batches:
                    break
                inputs = batch["input"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                residual = model(inputs)
                outputs = torch.clamp(inputs - residual, 0.0, 1.0)
                if args.loss == "mse":
                    loss = F.mse_loss(outputs, targets)
                else:
                    loss = F.l1_loss(outputs, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_size = int(inputs.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_count += batch_size
                progress.set_postfix(loss=f"{total_loss / max(1, total_count):.6f}")
            scheduler.step()

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                max_batches=args.max_val_batches,
            )
            train_loss = total_loss / max(1, total_count)
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_psnr": val_metrics.get("psnr_mean", 0.0),
                "val_ssim": val_metrics.get("ssim_mean", 0.0),
                "val_gcnr": val_metrics.get("gcnr_mean", 0.0),
                "val_cnr": val_metrics.get("cnr_mean", 0.0),
                "val_composite": val_metrics.get("composite_mean", 0.0),
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, indent=2))

            checkpoint = make_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_psnr,
                config,
            )
            torch.save(checkpoint, args.output_dir / "model_latest.pt")
            if float(row["val_psnr"]) > best_psnr:
                best_psnr = float(row["val_psnr"])
                checkpoint["best_psnr"] = best_psnr
                torch.save(checkpoint, args.output_dir / "model_best.pt")

    final_checkpoint = make_checkpoint(
        model,
        optimizer,
        scheduler,
        args.epochs,
        best_psnr,
        config,
    )
    torch.save(final_checkpoint, args.output_dir / "model_final.pt")
    print(f"Best validation PSNR: {best_psnr:.4f} dB")


if __name__ == "__main__":
    main()
