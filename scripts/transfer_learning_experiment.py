#!/usr/bin/env python
"""CDS521 transfer learning experiment.

Reproducible 4-setting comparison on a 5-class CIFAR-10 subset:
1) scratch
2) feature_extraction
3) partial_finetune
4) full_finetune
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchvision import datasets, models, transforms


AUG_CLASSES = [0, 1, 2, 3, 4]
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "dog"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RemappedSubset(Dataset):
    """Subset dataset wrapper that remaps original CIFAR labels to local 0..N-1 labels."""

    def __init__(self, base_dataset: Dataset, indices: List[int], class_to_local: Dict[int, int], transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.class_to_local = class_to_local
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, target = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_local[int(target)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices_by_class(
    targets: List[int],
    classes: List[int],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_per_class: int | None = None,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    rng = np.random.RandomState(seed)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    class_indices: Dict[int, List[int]] = {c: [] for c in classes}

    for idx, t in enumerate(targets):
        if t in class_to_idx:
            class_indices[t].append(idx)

    splits = {"train": [], "val": [], "test": []}
    for c, idxs in class_indices.items():
        idxs = np.array(idxs)
        if len(idxs) == 0:
            continue
        rng.shuffle(idxs)
        if max_per_class is not None:
            idxs = idxs[:max_per_class]
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_end = n_train
        val_end = n_train + n_val

        splits["train"].extend(idxs[:train_end].tolist())
        splits["val"].extend(idxs[train_end:val_end].tolist())
        splits["test"].extend(idxs[val_end:].tolist())

    return splits


def build_dataloaders(data_dir: Path, seed: int, batch_size: int, max_per_class: int | None) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, int]]:
    data_dir.mkdir(parents=True, exist_ok=True)

    train_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    cifar_train = datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    target_to_local = {c: i for i, c in enumerate(AUG_CLASSES)}

    splits = split_indices_by_class(cifar_train.targets, AUG_CLASSES, seed, max_per_class=max_per_class)

    train_set = RemappedSubset(cifar_train, splits["train"], target_to_local, transform=train_tfms)
    val_set = RemappedSubset(cifar_train, splits["val"], target_to_local, transform=eval_tfms)
    test_set = RemappedSubset(cifar_train, splits["test"], target_to_local, transform=eval_tfms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, target_to_local


def get_model(regime: str, num_classes: int, seed: int) -> Tuple[nn.Module, List[nn.Parameter], float]:
    if regime == "scratch":
        model = models.resnet18(weights=None)
        lr = 1e-3
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        lr = 1e-3 if regime == "feature_extraction" else (1e-4 if regime == "partial_finetune" else 5e-5)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if regime == "scratch":
        for p in model.parameters():
            p.requires_grad = True

    elif regime == "feature_extraction":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    elif regime == "partial_finetune":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True

    elif regime == "full_finetune":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown regime: {regime}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    torch.manual_seed(seed)
    return model, trainable, lr


def count_params(model: nn.Module) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = float(trainable) / float(total) if total > 0 else 0.0
    return total, trainable, ratio


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_pred, all_label, all_prob, total_loss = [], [], [], 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item())

            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)

            all_pred.extend(pred.cpu().tolist())
            all_label.extend(y.cpu().tolist())
            all_prob.extend(probs.cpu().numpy())

    if len(all_label) == 0:
        return {"loss": 0.0, "acc": 0.0, "f1": 0.0, "probs": np.array([]), "y_true": np.array([]), "y_pred": np.array([])}

    y_true = np.array(all_label)
    y_pred = np.array(all_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    avg_loss = total_loss / max(len(loader), 1)
    return {"loss": avg_loss, "acc": acc, "f1": f1, "probs": np.stack(all_prob), "y_true": y_true, "y_pred": y_pred}


def train_regime(regime: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, lr: float, device: torch.device, seed: int, patience: int):
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    best_state = None
    best_score = -1.0
    patience_left = patience

    records = []
    for ep in range(1, epochs + 1):
        start = time.time()
        model.train()
        tr_loss = 0.0
        tr_pred, tr_true = [], []

        for x, y in tqdm(train_loader, desc=f"{regime} epoch {ep}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            tr_pred.extend(pred.detach().cpu().tolist())
            tr_true.extend(y.cpu().tolist())

        tr_acc = accuracy_score(tr_true, tr_pred) if tr_true else 0.0
        tr_f = f1_score(tr_true, tr_pred, average="macro", zero_division=0) if tr_true else 0.0

        val_metrics = evaluate(model, val_loader, device)

        epoch_time = time.time() - start
        record = {
            "epoch": ep,
            "train_loss": tr_loss / max(len(train_loader), 1),
            "train_acc": float(tr_acc),
            "train_f1": float(tr_f),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
            "val_f1": float(val_metrics["f1"]),
            "epoch_time_sec": float(epoch_time),
        }
        records.append(record)

        # Early stop by validation accuracy
        if val_metrics["acc"] > best_score:
            best_score = val_metrics["acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return records


def denorm(x: torch.Tensor) -> np.ndarray:
    x = x.cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    x = (x * std + mean)
    x = np.clip(x, 0, 1)
    return x


def build_prediction_figure(model: nn.Module, loader: DataLoader, device: torch.device, out_path: Path, num_samples: int = 6):
    model.eval()
    samples = []

    # collect original input before normalization for visualization
    raw_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    # load from original dataset for plotting only
    cifar_train = datasets.CIFAR10(root="./data", train=True, download=True)
    class_to_idx = {c: i for i, c in enumerate(AUG_CLASSES)}
    selected = [i for i, t in enumerate(cifar_train.targets) if t in class_to_idx]
    rng = np.random.RandomState(2026)
    rng.shuffle(selected)
    selected = selected[:max(num_samples * 2, num_samples)]

    for idx in selected:
        img, target = cifar_train[idx]
        x = raw_transform(img)
        x_norm = transforms.functional.normalize(x.clone(), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x_norm)
            prob = torch.softmax(logits, dim=1)[0]
            p, pred = torch.max(prob, dim=0)

        pred_name = CLASS_NAMES[pred.item()]
        true_name = CLASS_NAMES[class_to_idx[target]]
        samples.append((x.numpy(), true_name, pred_name, float(p.item())))

        if len(samples) >= num_samples:
            break

    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (x, tname, pname, p) in zip(axes, samples):
        ax.imshow(np.transpose(x, (1, 2, 0)))
        ax.axis("off")
        color = "green" if tname == pname else "red"
        ax.set_title(f"T:{tname}\nP:{pname} ({p:.2f})", fontsize=10, color=color)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_curves(history_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for regime in history_df["regime"].unique():
        sub = history_df[history_df["regime"] == regime]
        axes[0].plot(sub["epoch"], sub["val_acc"], label=regime)
        axes[1].plot(sub["epoch"], sub["val_loss"], label=regime)

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CrossEntropy")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_conf_matrix(metrics: Dict[str, float], regime: str, out_path: Path) -> None:
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Pred")
    ax.set_ylabel("Truth")
    ax.set_title(f"Confusion Matrix - {regime}")
    plt.colorbar(im, ax=ax)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Optional upper bound of images per selected CIFAR-10 class; use smaller value for faster CPU runs.",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./outputs")
    args = parser.parse_args()

    set_seed(args.seed)

    base_path = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = base_path / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = run_dir / "metrics"
    fig_dir = run_dir / "figures"
    metric_dir.mkdir(parents=True)
    fig_dir.mkdir(parents=True)

    train_loader, val_loader, test_loader, _ = build_dataloaders(
        data_dir, args.seed, args.batch_size, args.samples_per_class
    )

    all_records = []
    summary = []
    all_histories = []

    regimes = ["scratch", "feature_extraction", "partial_finetune", "full_finetune"]
    for regime in regimes:
        print(f"\nRunning regime: {regime}")
        set_seed(args.seed)

        model, _, lr = get_model(regime, num_classes=len(CLASS_NAMES), seed=args.seed)

        total_params, trainable_params, ratio = count_params(model)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        start = time.time()
        history = train_regime(
            regime=regime,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=lr,
            device=device,
            seed=args.seed,
            patience=args.patience,
        )
        elapsed = time.time() - start

        # Save per-epoch history
        regime_hist = pd.DataFrame(history)
        regime_hist["regime"] = regime
        regime_hist.to_csv(metric_dir / f"{regime}_history.csv", index=False)
        all_histories.append(regime_hist)

        test_metrics = evaluate(model, test_loader, device)

        reg_summary = {
            "regime": regime,
            "seed": args.seed,
            "best_val_acc": float(max(r["val_acc"] for r in history) if history else 0.0),
            "best_val_f1": float(max(r["val_f1"] for r in history) if history else 0.0),
            "test_acc": float(test_metrics["acc"]),
            "test_f1": float(test_metrics["f1"]),
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "trainable_param_ratio": float(ratio),
            "trainable_param_pct": float(ratio * 100.0),
            "total_train_time_sec": float(elapsed),
            "epochs_ran": int(len(history)),
            "final_lr": float(lr),
        }
        summary.append(reg_summary)

        # Save confusion matrix and one prediction plot per regime
        build_conf_matrix(test_metrics, regime, fig_dir / f"{regime}_confusion_matrix.png")
        build_prediction_figure(model, test_loader, device, fig_dir / f"{regime}_predictions.png", num_samples=6)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(metric_dir / "summary.csv", index=False)

    history_all = pd.concat(all_histories, ignore_index=True)
    history_all.to_csv(metric_dir / "all_histories.csv", index=False)
    plot_curves(history_all, fig_dir / "training_curves.png")

    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "data_classes": CLASS_NAMES,
                "generated_at": datetime.now().isoformat(),
                "regimes": regimes,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nExperiment finished. Results saved to", run_dir)
    print(summary_df)


if __name__ == "__main__":
    main()
