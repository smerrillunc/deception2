#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
    set_seed,
)

from localization_next_token import (
    assign_group_splits,
    binary_classification_metrics,
    build_prefix_dataset,
)


DEFAULT_ENVIRONMENTS = ("AdvisorAudit", "BS", "Gridworld")
DEFAULT_EMPTY_REASONING_TEXT = "(no prior reasoning yet)"


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-").lower()


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_confusion_artifacts(
    confusion: np.ndarray,
    *,
    output_prefix: Path,
    title: str,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(confusion, dtype=int)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_not_deceptive", "actual_deceptive"],
        columns=["pred_not_deceptive", "pred_deceptive"],
    )
    cm_df.to_csv(output_prefix.with_suffix(".csv"), index=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["truthful", "deceptive"])
    ax.set_yticks([0, 1], labels=["truthful", "deceptive"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    max_count = int(cm.max()) if cm.size else 0
    threshold = max_count / 2.0 if max_count else 0.0
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            value = int(cm[row_idx, col_idx])
            text_color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=text_color)

    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def value_slug(value: Any) -> str:
    return str(value).replace(".", "p")


def max_records_slug(max_records: Optional[int]) -> str:
    return "all" if max_records is None else str(int(max_records))


def split_config_slug(args: argparse.Namespace) -> str:
    return (
        f"max_{max_records_slug(args.max_records)}"
        f"__seed_{args.seed}"
        f"__tr_{value_slug(args.train_frac)}"
        f"__va_{value_slug(args.val_frac)}"
        f"__te_{value_slug(args.test_frac)}"
    )


def environment_cache_dir(
    *,
    dataset_cache_root: Path,
    model_slug: str,
    threshold_slug: str,
    environment: str,
) -> Path:
    return dataset_cache_root / model_slug / f"threshold_{threshold_slug}" / environment


def cached_base_split_path(
    *,
    cache_dir: Path,
    args: argparse.Namespace,
) -> Path:
    return cache_dir / f"base_split__{split_config_slug(args)}.parquet"


def strict_binary_labels(values: Any, threshold: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) > float(threshold)).astype(int)


def ensure_tokenizer(model_name_or_path: str, *, trust_remote_code: bool) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"
    return tokenizer


def discover_localization_source(
    dataset_root: Path,
    environment: str,
    model_dirname: str,
    *,
    prefer_jsonl: bool,
) -> Path:
    base_dir = dataset_root / environment / model_dirname
    jsonl_path = base_dir / "localization.jsonl"
    dir_path = base_dir / "localization"
    if prefer_jsonl and jsonl_path.exists():
        return jsonl_path
    if dir_path.exists():
        return dir_path
    if jsonl_path.exists():
        return jsonl_path
    raise FileNotFoundError(f"Could not find localization source under {base_dir}")


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def build_quant_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if not args.load_in_4bit:
        return None
    if not torch.cuda.is_available():
        return None
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def get_runtime_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return get_runtime_device()


def build_model_load_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    quant_config = build_quant_config(args)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return model_kwargs


def configure_model_for_tokenizer(model, tokenizer, *, use_cache: bool) -> None:
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = use_cache


def get_hidden_size(model) -> int:
    config = getattr(model, "config", None)
    for attr_name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attr_name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer hidden size from model config.")


def freeze_module_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    try:
        r2 = float(r2_score(y_true_arr, y_pred_arr))
    except Exception:
        r2 = np.nan
    if np.std(y_true_arr) < 1e-12 or np.std(y_pred_arr) < 1e-12:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
    }


def task_specific_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    task_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    targets = targets.float()
    if task_type == "classification":
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        predictions = torch.sigmoid(logits)
        return loss, predictions

    predictions = torch.sigmoid(logits)
    loss = F.mse_loss(predictions, targets)
    return loss, predictions


def validation_metrics_for_task(
    *,
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    if task_type == "classification":
        metrics = binary_classification_metrics(y_true, y_pred, cutoff=0.5)
        return {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "roc_auc": float(metrics["roc_auc"]) if not pd.isna(metrics["roc_auc"]) else np.nan,
        }

    reg = regression_metrics(y_true, y_pred)
    threshold_metrics = binary_classification_metrics(
        strict_binary_labels(y_true, threshold),
        y_pred,
        cutoff=threshold,
    )
    return {
        "rmse": float(reg["rmse"]),
        "mae": float(reg["mae"]),
        "r2": float(reg["r2"]) if not pd.isna(reg["r2"]) else np.nan,
        "pearson": float(reg["pearson"]) if not pd.isna(reg["pearson"]) else np.nan,
        "threshold_accuracy": float(threshold_metrics["accuracy"]),
        "threshold_precision": float(threshold_metrics["precision"]),
        "threshold_recall": float(threshold_metrics["recall"]),
        "threshold_f1": float(threshold_metrics["f1"]),
        "threshold_roc_auc": float(threshold_metrics["roc_auc"])
        if not pd.isna(threshold_metrics["roc_auc"])
        else np.nan,
    }


def selection_score(task_type: str, metrics: Dict[str, Any]) -> tuple[float, str]:
    if task_type == "classification":
        roc_auc = float(metrics.get("roc_auc", np.nan))
        if not pd.isna(roc_auc):
            return roc_auc, "roc_auc"
        return float(metrics.get("accuracy", -np.inf)), "accuracy"

    rmse = float(metrics.get("rmse", np.nan))
    if not pd.isna(rmse):
        return -rmse, "neg_rmse"
    mae = float(metrics.get("mae", np.nan))
    return -mae, "neg_mae"


def build_model_input_text(args: argparse.Namespace, df: pd.DataFrame) -> pd.Series:
    prefix_text = df["prefix_text"].fillna("").astype(str)
    target_sentence_text = df["target_sentence_text"].fillna("").astype(str)

    if args.input_view == "reasoning_only":
        input_text = prefix_text
    elif args.input_view == "prefix_plus_target_sentence":
        input_text = prefix_text + target_sentence_text
    else:
        raise ValueError(f"Unsupported input_view={args.input_view!r}")

    input_text = input_text.astype(str).str.strip()
    empty_mask = input_text == ""
    if empty_mask.any():
        input_text = input_text.copy()
        input_text.loc[empty_mask] = args.empty_reasoning_text
    return input_text


def prepare_reasoning_dataframe(args: argparse.Namespace, base_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy()
    df["input_text"] = build_model_input_text(args, df)
    df["label_binary"] = strict_binary_labels(df["deception_rate"], args.threshold)
    df["label_name"] = np.where(df["label_binary"] == 1, "deceptive", "not_deceptive")
    if args.task_type == "classification":
        df["target_value"] = df["label_binary"].astype(float)
    else:
        df["target_value"] = df["deception_rate"].astype(float)
    return df


class ReasoningTokenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, *, max_length: int):
        self.df = df.reset_index(drop=True).copy()
        self.examples: list[Dict[str, Any]] = []
        self.num_truncated = 0

        for row_idx, row in enumerate(self.df.itertuples(index=False)):
            encoding = tokenizer(
                str(row.input_text),
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            input_ids = list(encoding["input_ids"])
            attention_mask = list(encoding["attention_mask"])
            if not input_ids:
                continue
            if len(input_ids) >= max_length:
                self.num_truncated += 1
            self.examples.append(
                {
                    "row_index": row_idx,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "targets": float(row.target_value),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


class ReasoningTokenCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_targets = []
        batch_row_index = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_targets.append(float(feature["targets"]))
            batch_row_index.append(int(feature["row_index"]))

        return {
            "row_index": torch.tensor(batch_row_index, dtype=torch.long),
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "targets": torch.tensor(batch_targets, dtype=torch.float32),
        }


class FeatureTensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features.to(torch.float32).cpu()
        self.targets = targets.to(torch.float32).cpu()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "targets": self.targets[idx],
        }


class ScalarMLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Sequential(nn.Linear(input_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        first_param = next(self.parameters(), None)
        if first_param is not None and (
            features.dtype != first_param.dtype or features.device != first_param.device
        ):
            features = features.to(device=first_param.device, dtype=first_param.dtype)
        return self.net(features).squeeze(-1)


class HiddenStateHeadModel(nn.Module):
    def __init__(self, backbone, head: ScalarMLPHead, *, task_type: str):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.task_type = task_type

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        valid_lengths = attention_mask.long().sum(dim=1).clamp(min=1)
        last_indices = valid_lengths - 1
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, last_indices]

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(features)
        predictions = torch.sigmoid(logits.float())
        loss = None
        if targets is not None:
            loss, predictions = task_specific_loss(logits, targets, task_type=self.task_type)
        return {
            "loss": loss,
            "logits": logits.float(),
            "predictions": predictions,
        }


def make_token_loader(
    df: pd.DataFrame,
    tokenizer,
    *,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> tuple[ReasoningTokenDataset, DataLoader]:
    dataset = ReasoningTokenDataset(df, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ReasoningTokenCollator(tokenizer.pad_token_id),
    )
    return dataset, loader


def move_tensor_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def weighted_mean(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        return np.nan
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return np.nan
    return float(sum(value * weight for value, weight in pairs) / total_weight)


def evaluate_joint_model_on_loader(
    model: HiddenStateHeadModel,
    loader: DataLoader,
    *,
    device: torch.device,
    task_type: str,
    threshold: float,
    progress_desc: str,
) -> Dict[str, Any]:
    model.eval()
    losses: list[tuple[float, int]] = []
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=progress_desc, leave=False):
            model_batch = move_tensor_batch_to_device(batch, device)
            outputs = model(
                input_ids=model_batch["input_ids"],
                attention_mask=model_batch["attention_mask"],
                targets=model_batch["targets"],
            )
            batch_size = int(model_batch["targets"].shape[0])
            if outputs["loss"] is not None:
                losses.append((float(outputs["loss"].detach().cpu().item()), batch_size))
            all_targets.append(model_batch["targets"].detach().cpu().numpy())
            all_predictions.append(outputs["predictions"].detach().cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.asarray([], dtype=float)
    y_pred = np.concatenate(all_predictions) if all_predictions else np.asarray([], dtype=float)
    metrics = validation_metrics_for_task(
        task_type=task_type,
        y_true=y_true,
        y_pred=y_pred,
        threshold=threshold,
    )
    metrics["loss"] = weighted_mean(losses)
    return metrics


def evaluate_head_on_loader(
    head: ScalarMLPHead,
    loader: DataLoader,
    *,
    device: torch.device,
    task_type: str,
    threshold: float,
    progress_desc: str,
) -> Dict[str, Any]:
    head.eval()
    losses: list[tuple[float, int]] = []
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=progress_desc, leave=False):
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            logits = head(features)
            loss, predictions = task_specific_loss(logits, targets, task_type=task_type)
            batch_size = int(targets.shape[0])
            losses.append((float(loss.detach().cpu().item()), batch_size))
            all_targets.append(targets.detach().cpu().numpy())
            all_predictions.append(predictions.detach().cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.asarray([], dtype=float)
    y_pred = np.concatenate(all_predictions) if all_predictions else np.asarray([], dtype=float)
    metrics = validation_metrics_for_task(
        task_type=task_type,
        y_true=y_true,
        y_pred=y_pred,
        threshold=threshold,
    )
    metrics["loss"] = weighted_mean(losses)
    return metrics


def count_optimizer_steps(train_loader: DataLoader, gradient_accumulation_steps: int) -> int:
    return max(1, math.ceil(len(train_loader) / max(1, gradient_accumulation_steps)))


def resolve_total_training_steps(args: argparse.Namespace, train_loader: DataLoader) -> int:
    if args.max_steps is not None and args.max_steps > 0:
        return int(args.max_steps)
    return max(1, math.ceil(args.num_train_epochs * count_optimizer_steps(train_loader, args.gradient_accumulation_steps)))


def save_head_state(head: ScalarMLPHead, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), path)


def load_head_state(
    *,
    path: Path,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> ScalarMLPHead:
    head = ScalarMLPHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    state_dict = torch.load(path, map_location="cpu")
    head.load_state_dict(state_dict)
    head.to(device)
    head.eval()
    return head


def extract_features_from_loader(
    backbone,
    loader: DataLoader,
    *,
    device: torch.device,
    progress_desc: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    feature_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=progress_desc, leave=False):
            model_batch = move_tensor_batch_to_device(batch, device)
            outputs = backbone(
                input_ids=model_batch["input_ids"],
                attention_mask=model_batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]
            valid_lengths = model_batch["attention_mask"].long().sum(dim=1).clamp(min=1)
            last_indices = valid_lengths - 1
            batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            features = hidden_states[batch_indices, last_indices].detach().to(torch.float32).cpu()
            feature_chunks.append(features)
            target_chunks.append(batch["targets"].detach().cpu().to(torch.float32))

    features = torch.cat(feature_chunks, dim=0) if feature_chunks else torch.empty((0, 0), dtype=torch.float32)
    targets = torch.cat(target_chunks, dim=0) if target_chunks else torch.empty((0,), dtype=torch.float32)
    return features, targets


def predict_scores_from_head(
    head: ScalarMLPHead,
    features: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    head.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), batch_size):
            stop = min(int(features.shape[0]), start + batch_size)
            batch_features = features[start:stop].to(device)
            logits = head(batch_features)
            outputs.append(torch.sigmoid(logits.float()).detach().cpu())
    if not outputs:
        return np.asarray([], dtype=float)
    return torch.cat(outputs, dim=0).numpy()


def write_training_history(history_rows: list[Dict[str, Any]], path: Path) -> None:
    if history_rows:
        pd.DataFrame(history_rows).to_csv(path, index=False)


def load_trainable_backbone(args: argparse.Namespace, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **build_model_load_kwargs(args),
    )
    configure_model_for_tokenizer(model, tokenizer, use_cache=False)
    if args.load_in_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    if not args.load_in_4bit and torch.cuda.is_available():
        model.to(get_runtime_device())

    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    return model


def load_frozen_backbone(args: argparse.Namespace, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **build_model_load_kwargs(args),
    )
    configure_model_for_tokenizer(model, tokenizer, use_cache=False)
    freeze_module_parameters(model)
    if not args.load_in_4bit and torch.cuda.is_available():
        model.to(get_runtime_device())
    model.eval()
    return model


def load_saved_lora_pipeline(args: argparse.Namespace, tokenizer, *, run_dir: Path) -> HiddenStateHeadModel:
    artifact_meta = json.loads((run_dir / "model_artifact.json").read_text(encoding="utf-8"))
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **build_model_load_kwargs(args),
    )
    configure_model_for_tokenizer(backbone, tokenizer, use_cache=False)
    backbone = PeftModel.from_pretrained(
        backbone,
        str(run_dir / "adapter"),
        is_trainable=False,
    )
    if not args.load_in_4bit and torch.cuda.is_available():
        backbone.to(get_runtime_device())
    backbone.eval()
    device = get_module_device(backbone)
    head = load_head_state(
        path=run_dir / "mlp_head.pt",
        input_dim=int(artifact_meta["input_dim"]),
        hidden_dim=int(artifact_meta["mlp_hidden_dim"]),
        dropout=float(artifact_meta["mlp_dropout"]),
        device=device,
    )
    model = HiddenStateHeadModel(backbone, head, task_type=str(artifact_meta["task_type"]))
    model.eval()
    return model


def load_saved_frozen_pipeline(
    args: argparse.Namespace,
    tokenizer,
    *,
    run_dir: Path,
) -> tuple[Any, ScalarMLPHead]:
    artifact_meta = json.loads((run_dir / "model_artifact.json").read_text(encoding="utf-8"))
    backbone = load_frozen_backbone(args, tokenizer)
    device = get_runtime_device()
    head = load_head_state(
        path=run_dir / "mlp_head.pt",
        input_dim=int(artifact_meta["input_dim"]),
        hidden_dim=int(artifact_meta["mlp_hidden_dim"]),
        dropout=float(artifact_meta["mlp_dropout"]),
        device=device,
    )
    return backbone, head


def score_dataframe_with_joint_model(
    df: pd.DataFrame,
    tokenizer,
    model: HiddenStateHeadModel,
    *,
    batch_size: int,
    max_length: int,
    progress_desc: str,
) -> pd.DataFrame:
    dataset, loader = make_token_loader(
        df,
        tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
    )
    if len(dataset) != len(df):
        raise RuntimeError("Token dataset row count mismatch during evaluation.")

    device = get_module_device(model.head)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=progress_desc, leave=False):
            model_batch = move_tensor_batch_to_device(batch, device)
            outputs = model(
                input_ids=model_batch["input_ids"],
                attention_mask=model_batch["attention_mask"],
                targets=None,
            )
            predictions.append(outputs["predictions"].detach().cpu().numpy())

    pred_scores = np.concatenate(predictions) if predictions else np.asarray([], dtype=float)
    out = df.reset_index(drop=True).copy()
    out["pred_score"] = pred_scores
    return out


def score_dataframe_with_frozen_backbone(
    df: pd.DataFrame,
    tokenizer,
    backbone,
    head: ScalarMLPHead,
    *,
    max_length: int,
    feature_batch_size: int,
    score_batch_size: int,
    progress_desc: str,
) -> pd.DataFrame:
    dataset, loader = make_token_loader(
        df,
        tokenizer,
        max_length=max_length,
        batch_size=feature_batch_size,
        shuffle=False,
    )
    if len(dataset) != len(df):
        raise RuntimeError("Token dataset row count mismatch during frozen feature scoring.")

    backbone_device = get_module_device(backbone)
    features, _ = extract_features_from_loader(
        backbone,
        loader,
        device=backbone_device,
        progress_desc=f"{progress_desc}:extract",
    )
    head_device = get_module_device(head)
    pred_scores = predict_scores_from_head(
        head,
        features,
        batch_size=score_batch_size,
        device=head_device,
    )
    out = df.reset_index(drop=True).copy()
    out["pred_score"] = pred_scores
    return out


def model_artifacts_exist(args: argparse.Namespace, run_dir: Path) -> bool:
    base_exists = (run_dir / "mlp_head.pt").exists() and (run_dir / "model_artifact.json").exists()
    if not base_exists:
        return False
    if args.training_mode == "lora":
        return (run_dir / "adapter").exists()
    return True


def load_summary_rows(
    summary_path: Path,
    *,
    expected_train_dataset: Optional[str] = None,
    expected_eval_datasets: Optional[list[str]] = None,
) -> Optional[list[Dict[str, Any]]]:
    summary_df = pd.read_csv(summary_path)
    required_columns = {"train_dataset", "eval_dataset", "eval_kind", "split"}
    if not required_columns.issubset(summary_df.columns):
        missing = sorted(required_columns - set(summary_df.columns))
        print(
            f"Existing summary at {summary_path} is missing columns {missing}; recomputing evaluation outputs.",
            flush=True,
        )
        return None
    if expected_train_dataset is not None:
        found_train_datasets = set(summary_df["train_dataset"].dropna().astype(str).unique().tolist())
        if found_train_datasets != {expected_train_dataset}:
            print(
                f"Existing summary at {summary_path} has train datasets {sorted(found_train_datasets)}; "
                f"expected {[expected_train_dataset]}. Recomputing evaluation outputs.",
                flush=True,
            )
            return None
    if expected_eval_datasets is not None:
        found_eval_datasets = set(summary_df["eval_dataset"].dropna().astype(str).unique().tolist())
        if found_eval_datasets != set(expected_eval_datasets):
            print(
                f"Existing summary at {summary_path} has eval datasets {sorted(found_eval_datasets)}; "
                f"expected {sorted(expected_eval_datasets)}. Recomputing evaluation outputs.",
                flush=True,
            )
            return None
    return summary_df.to_dict(orient="records")


def split_metrics_rows(
    predictions_df: pd.DataFrame,
    *,
    run_name: str,
    train_dataset: str,
    task_type: str,
    training_mode: str,
    threshold: float,
) -> list[Dict[str, Any]]:
    rows = []
    group_cols = ["eval_dataset", "eval_kind", "split"]
    for (eval_dataset, eval_kind, split_name), split_df in predictions_df.groupby(group_cols, sort=False):
        y_true_rate = split_df["deception_rate"].to_numpy(dtype=float)
        y_pred = split_df["pred_score"].to_numpy(dtype=float)
        y_true_binary = strict_binary_labels(y_true_rate, threshold)

        row: Dict[str, Any] = {
            "run_name": run_name,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "eval_kind": eval_kind,
            "task_type": task_type,
            "training_mode": training_mode,
            "threshold": threshold,
            "split": split_name,
            "n_rows": int(len(split_df)),
            "n_examples": int(split_df["example_id"].nunique()),
        }

        if task_type == "classification":
            metrics = binary_classification_metrics(y_true_binary, y_pred, cutoff=0.5)
            cm = np.asarray(metrics["confusion_matrix"], dtype=int)
            tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
            row.update(
                {
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "roc_auc": float(metrics["roc_auc"]) if not pd.isna(metrics["roc_auc"]) else np.nan,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )
        else:
            reg = regression_metrics(y_true_rate, y_pred)
            metrics = binary_classification_metrics(y_true_binary, y_pred, cutoff=threshold)
            cm = np.asarray(metrics["confusion_matrix"], dtype=int)
            tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
            row.update(
                {
                    "rmse": float(reg["rmse"]),
                    "mae": float(reg["mae"]),
                    "r2": float(reg["r2"]) if not pd.isna(reg["r2"]) else np.nan,
                    "pearson": float(reg["pearson"]) if not pd.isna(reg["pearson"]) else np.nan,
                    "threshold_accuracy": float(metrics["accuracy"]),
                    "threshold_precision": float(metrics["precision"]),
                    "threshold_recall": float(metrics["recall"]),
                    "threshold_f1": float(metrics["f1"]),
                    "threshold_roc_auc": float(metrics["roc_auc"])
                    if not pd.isna(metrics["roc_auc"])
                    else np.nan,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )
        rows.append(row)
    return rows


def save_evaluation_outputs(
    *,
    predictions_df: pd.DataFrame,
    run_dir: Path,
    run_name: str,
    train_dataset: str,
    task_type: str,
    training_mode: str,
    threshold: float,
) -> list[Dict[str, Any]]:
    if predictions_df.empty:
        return []

    predictions_df.to_parquet(run_dir / "predictions_test_across_datasets.parquet", index=False)
    summary_rows = split_metrics_rows(
        predictions_df,
        run_name=run_name,
        train_dataset=train_dataset,
        task_type=task_type,
        training_mode=training_mode,
        threshold=threshold,
    )
    pd.DataFrame(summary_rows).to_csv(run_dir / "metrics_summary.csv", index=False)

    group_cols = ["eval_dataset", "eval_kind", "split"]
    for (eval_dataset, eval_kind, split_name), split_df in predictions_df.groupby(group_cols, sort=False):
        y_true_rate = split_df["deception_rate"].to_numpy(dtype=float)
        y_pred = split_df["pred_score"].to_numpy(dtype=float)
        y_true_binary = strict_binary_labels(y_true_rate, threshold)

        payload: Dict[str, Any] = {
            "run_name": run_name,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "eval_kind": eval_kind,
            "task_type": task_type,
            "training_mode": training_mode,
            "threshold": threshold,
            "split": split_name,
            "n_rows": int(len(split_df)),
            "n_examples": int(split_df["example_id"].nunique()),
        }

        if task_type == "classification":
            metrics = binary_classification_metrics(y_true_binary, y_pred, cutoff=0.5)
            payload.update(
                {
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "roc_auc": None if pd.isna(metrics["roc_auc"]) else float(metrics["roc_auc"]),
                    "confusion_matrix": np.asarray(metrics["confusion_matrix"], dtype=int).tolist(),
                }
            )
            confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
            title = f"train={train_dataset} | eval={eval_dataset} | {training_mode} | classification | {split_name}"
        else:
            reg = regression_metrics(y_true_rate, y_pred)
            metrics = binary_classification_metrics(y_true_binary, y_pred, cutoff=threshold)
            payload.update(
                {
                    "rmse": float(reg["rmse"]),
                    "mae": float(reg["mae"]),
                    "r2": None if pd.isna(reg["r2"]) else float(reg["r2"]),
                    "pearson": None if pd.isna(reg["pearson"]) else float(reg["pearson"]),
                    "threshold_accuracy": float(metrics["accuracy"]),
                    "threshold_precision": float(metrics["precision"]),
                    "threshold_recall": float(metrics["recall"]),
                    "threshold_f1": float(metrics["f1"]),
                    "threshold_roc_auc": None if pd.isna(metrics["roc_auc"]) else float(metrics["roc_auc"]),
                    "confusion_matrix": np.asarray(metrics["confusion_matrix"], dtype=int).tolist(),
                }
            )
            confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
            title = (
                f"train={train_dataset} | eval={eval_dataset} | {training_mode} | "
                f"regression thresholded@{threshold} | {split_name}"
            )

        eval_slug = slugify(eval_dataset)
        save_json(payload, run_dir / f"metrics_{split_name}__{eval_slug}.json")
        save_confusion_artifacts(
            confusion,
            output_prefix=run_dir / f"confusion_matrix_{split_name}__{eval_slug}",
            title=title,
        )

    return summary_rows


def load_or_build_base_split_df(
    args: argparse.Namespace,
    *,
    environment: str,
    localization_source: Path,
    model_slug: str,
    threshold_slug: str,
) -> pd.DataFrame:
    cache_dir = environment_cache_dir(
        dataset_cache_root=args.dataset_cache_root,
        model_slug=model_slug,
        threshold_slug=threshold_slug,
        environment=environment,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cached_base_split_path(cache_dir=cache_dir, args=args)

    if args.cache_datasets and cache_path.exists() and not args.rebuild_cache:
        print(f"Loading cached base split dataset: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Building base split dataset for {environment}")
    base_df = build_prefix_dataset(
        localization_source,
        dataset_name=environment,
        low_threshold=args.threshold,
        high_threshold=args.threshold,
        drop_ambiguous=False,
        max_records=args.max_records,
    )
    base_df = assign_group_splits(
        base_df,
        group_col="example_id",
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    if args.cache_datasets:
        base_df.to_parquet(cache_path, index=False)
        save_json(
            {
                "environment": environment,
                "localization_source": str(localization_source),
                "threshold": args.threshold,
                "seed": args.seed,
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "test_frac": args.test_frac,
                "max_records": args.max_records,
                "drop_ambiguous": False,
                "n_rows": int(len(base_df)),
                "n_examples": int(base_df["example_id"].nunique()) if not base_df.empty else 0,
            },
            cache_path.with_suffix(".json"),
        )
        print(f"Saved base split dataset cache: {cache_path}")

    return base_df


def load_environment_base_bundle(
    args: argparse.Namespace,
    *,
    environment: str,
    model_slug: str,
    threshold_slug: str,
) -> tuple[Path, pd.DataFrame]:
    localization_source = discover_localization_source(
        args.dataset_root,
        environment,
        args.model_dirname,
        prefer_jsonl=args.prefer_jsonl,
    )
    print(f"Localization source for {environment}: {localization_source}")

    base_df = load_or_build_base_split_df(
        args,
        environment=environment,
        localization_source=localization_source,
        model_slug=model_slug,
        threshold_slug=threshold_slug,
    )
    return localization_source, base_df


def build_run_spec(
    args: argparse.Namespace,
    *,
    environment: str,
    output_root: Path,
) -> Dict[str, Any]:
    model_slug = slugify(Path(args.model_name_or_path).name or args.model_name_or_path)
    threshold_slug = value_slug(args.threshold)
    input_view_slug = slugify(args.input_view)
    run_name = f"{environment}__{args.training_mode}__{args.task_type}__{input_view_slug}__thr_{threshold_slug}"
    run_dir = (
        output_root
        / model_slug
        / f"task_{args.task_type}"
        / f"mode_{args.training_mode}"
        / f"input_{input_view_slug}"
        / f"threshold_{threshold_slug}"
        / environment
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "model_slug": model_slug,
        "threshold_slug": threshold_slug,
        "run_name": run_name,
        "run_dir": run_dir,
        "train_dataset": environment,
        "task_type": args.task_type,
        "training_mode": args.training_mode,
        "eval_environments": list(args.eval_environments),
    }


def build_eval_base_dfs(
    args: argparse.Namespace,
    *,
    train_dataset: str,
    model_slug: str,
    threshold_slug: str,
    eval_environments: list[str],
) -> "OrderedDict[str, pd.DataFrame]":
    eval_base_dfs: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    ordered_eval_environments = [train_dataset] + [env for env in eval_environments if env != train_dataset]
    for eval_environment in ordered_eval_environments:
        print(f"Preparing evaluation dataset bundle for {eval_environment}", flush=True)
        _, eval_base_df = load_environment_base_bundle(
            args,
            environment=eval_environment,
            model_slug=model_slug,
            threshold_slug=threshold_slug,
        )
        eval_base_dfs[eval_environment] = prepare_reasoning_dataframe(args, eval_base_df)
    return eval_base_dfs


def save_model_artifact_metadata(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    input_dim: int,
    best_val_metrics: Dict[str, Any],
    selected_by: str,
) -> None:
    save_json(
        {
            "model_name_or_path": args.model_name_or_path,
            "task_type": args.task_type,
            "training_mode": args.training_mode,
            "input_view": args.input_view,
            "threshold": args.threshold,
            "input_dim": int(input_dim),
            "mlp_hidden_dim": int(args.mlp_hidden_dim),
            "mlp_dropout": float(args.mlp_dropout),
            "max_seq_length": int(args.max_seq_length),
            "selected_by": selected_by,
            "best_val_metrics": best_val_metrics,
        },
        run_dir / "model_artifact.json",
    )


def train_lora_pipeline(
    args: argparse.Namespace,
    *,
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_dir: Path,
) -> Dict[str, Any]:
    train_dataset, train_loader = make_token_loader(
        train_df,
        tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
    )
    val_dataset, val_loader = make_token_loader(
        val_df,
        tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.scoring_batch_size,
        shuffle=False,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset empty after tokenization.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset empty after tokenization.")

    backbone = None
    model = None
    optimizer = None
    scheduler = None
    history_rows: list[Dict[str, Any]] = []
    best_metrics: Dict[str, Any] | None = None
    best_tmp_dir = run_dir / "_best_tmp"

    try:
        backbone = load_trainable_backbone(args, tokenizer)
        device = get_module_device(backbone)
        head = ScalarMLPHead(get_hidden_size(backbone), args.mlp_hidden_dim, args.mlp_dropout).to(device)
        model = HiddenStateHeadModel(backbone, head, task_type=args.task_type)

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        total_steps = resolve_total_training_steps(args, train_loader)
        warmup_steps = int(args.warmup_ratio * total_steps)
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        completed_steps = 0
        epoch_idx = 0
        best_score: Optional[float] = None
        selected_by = ""
        while completed_steps < total_steps:
            epoch_idx += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss_pairs: list[tuple[float, int]] = []

            progress = tqdm(train_loader, desc=f"train:lora:epoch{epoch_idx}", leave=False)
            for batch_idx, batch in enumerate(progress, start=1):
                model_batch = move_tensor_batch_to_device(batch, device)
                outputs = model(
                    input_ids=model_batch["input_ids"],
                    attention_mask=model_batch["attention_mask"],
                    targets=model_batch["targets"],
                )
                loss = outputs["loss"]
                if loss is None:
                    raise RuntimeError("Expected training loss for LoRA pipeline.")

                batch_size = int(model_batch["targets"].shape[0])
                train_loss_pairs.append((float(loss.detach().cpu().item()), batch_size))
                (loss / max(1, args.gradient_accumulation_steps)).backward()

                should_step = (
                    batch_idx % max(1, args.gradient_accumulation_steps) == 0
                    or batch_idx == len(train_loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    completed_steps += 1
                    progress.set_postfix(step=completed_steps, loss=f"{loss.detach().cpu().item():.4f}")
                    if completed_steps >= total_steps:
                        break

            val_metrics = evaluate_joint_model_on_loader(
                model,
                val_loader,
                device=device,
                task_type=args.task_type,
                threshold=args.threshold,
                progress_desc=f"val:lora:epoch{epoch_idx}",
            )
            val_score, selected_by = selection_score(args.task_type, val_metrics)
            train_loss = weighted_mean(train_loss_pairs)
            history_row = {
                "epoch": epoch_idx,
                "optimizer_steps": completed_steps,
                "train_loss": train_loss,
                **val_metrics,
                "selection_metric": selected_by,
                "selection_score": val_score,
            }
            history_rows.append(history_row)

            if best_score is None or val_score > best_score:
                best_score = float(val_score)
                best_metrics = dict(val_metrics)
                if best_tmp_dir.exists():
                    shutil.rmtree(best_tmp_dir)
                (best_tmp_dir / "adapter").mkdir(parents=True, exist_ok=True)
                model.backbone.save_pretrained(str(best_tmp_dir / "adapter"))
                save_head_state(model.head, best_tmp_dir / "mlp_head.pt")

        if best_metrics is None or not best_tmp_dir.exists():
            raise RuntimeError("LoRA training did not produce a best checkpoint.")

        final_adapter_dir = run_dir / "adapter"
        if final_adapter_dir.exists():
            shutil.rmtree(final_adapter_dir)
        shutil.copytree(best_tmp_dir / "adapter", final_adapter_dir)
        shutil.copy2(best_tmp_dir / "mlp_head.pt", run_dir / "mlp_head.pt")
        tokenizer.save_pretrained(str(run_dir / "tokenizer"))
        write_training_history(history_rows, run_dir / "training_history.csv")
        save_json(best_metrics, run_dir / "best_val_metrics.json")
        save_model_artifact_metadata(
            args=args,
            run_dir=run_dir,
            input_dim=get_hidden_size(backbone),
            best_val_metrics=best_metrics,
            selected_by=selected_by,
        )
        return {
            "tokenized_train_rows": int(len(train_dataset)),
            "tokenized_val_rows": int(len(val_dataset)),
            "train_truncated_rows": int(train_dataset.num_truncated),
            "val_truncated_rows": int(val_dataset.num_truncated),
            "best_val_metrics": best_metrics,
        }
    finally:
        if best_tmp_dir.exists():
            shutil.rmtree(best_tmp_dir)
        if model is not None:
            del model
        if backbone is not None:
            del backbone
        if optimizer is not None:
            del optimizer
        if scheduler is not None:
            del scheduler
        release_cuda_memory()


def train_frozen_mlp_pipeline(
    args: argparse.Namespace,
    *,
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_dir: Path,
) -> Dict[str, Any]:
    train_token_dataset, train_token_loader = make_token_loader(
        train_df,
        tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.feature_batch_size or args.scoring_batch_size,
        shuffle=False,
    )
    val_token_dataset, val_token_loader = make_token_loader(
        val_df,
        tokenizer,
        max_length=args.max_seq_length,
        batch_size=args.feature_batch_size or args.scoring_batch_size,
        shuffle=False,
    )
    if len(train_token_dataset) == 0:
        raise RuntimeError("Training dataset empty after tokenization.")
    if len(val_token_dataset) == 0:
        raise RuntimeError("Validation dataset empty after tokenization.")

    backbone = None
    head = None
    optimizer = None
    scheduler = None
    history_rows: list[Dict[str, Any]] = []
    best_metrics: Dict[str, Any] | None = None

    try:
        backbone = load_frozen_backbone(args, tokenizer)
        backbone_device = get_module_device(backbone)
        train_features, train_targets = extract_features_from_loader(
            backbone,
            train_token_loader,
            device=backbone_device,
            progress_desc="extract:train",
        )
        val_features, val_targets = extract_features_from_loader(
            backbone,
            val_token_loader,
            device=backbone_device,
            progress_desc="extract:val",
        )
        torch.save(
            {
                "train_shape": list(train_features.shape),
                "val_shape": list(val_features.shape),
            },
            run_dir / "feature_shapes.pt",
        )

        head_device = get_runtime_device()
        input_dim = int(train_features.shape[1])
        head = ScalarMLPHead(input_dim, args.mlp_hidden_dim, args.mlp_dropout).to(head_device)

        train_feature_loader = DataLoader(
            FeatureTensorDataset(train_features, train_targets),
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
        )
        val_feature_loader = DataLoader(
            FeatureTensorDataset(val_features, val_targets),
            batch_size=args.scoring_batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        total_steps = resolve_total_training_steps(args, train_feature_loader)
        warmup_steps = int(args.warmup_ratio * total_steps)
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        completed_steps = 0
        epoch_idx = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_score: Optional[float] = None
        selected_by = ""

        while completed_steps < total_steps:
            epoch_idx += 1
            head.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss_pairs: list[tuple[float, int]] = []

            progress = tqdm(train_feature_loader, desc=f"train:frozen_mlp:epoch{epoch_idx}", leave=False)
            for batch_idx, batch in enumerate(progress, start=1):
                features = batch["features"].to(head_device)
                targets = batch["targets"].to(head_device)
                logits = head(features)
                loss, _ = task_specific_loss(logits, targets, task_type=args.task_type)
                batch_size = int(targets.shape[0])
                train_loss_pairs.append((float(loss.detach().cpu().item()), batch_size))
                (loss / max(1, args.gradient_accumulation_steps)).backward()

                should_step = (
                    batch_idx % max(1, args.gradient_accumulation_steps) == 0
                    or batch_idx == len(train_feature_loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    completed_steps += 1
                    progress.set_postfix(step=completed_steps, loss=f"{loss.detach().cpu().item():.4f}")
                    if completed_steps >= total_steps:
                        break

            val_metrics = evaluate_head_on_loader(
                head,
                val_feature_loader,
                device=head_device,
                task_type=args.task_type,
                threshold=args.threshold,
                progress_desc=f"val:frozen_mlp:epoch{epoch_idx}",
            )
            val_score, selected_by = selection_score(args.task_type, val_metrics)
            train_loss = weighted_mean(train_loss_pairs)
            history_rows.append(
                {
                    "epoch": epoch_idx,
                    "optimizer_steps": completed_steps,
                    "train_loss": train_loss,
                    **val_metrics,
                    "selection_metric": selected_by,
                    "selection_score": val_score,
                }
            )

            if best_score is None or val_score > best_score:
                best_score = float(val_score)
                best_metrics = dict(val_metrics)
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        if best_metrics is None or best_state is None:
            raise RuntimeError("Frozen MLP training did not produce a best checkpoint.")

        head.load_state_dict(best_state)
        save_head_state(head, run_dir / "mlp_head.pt")
        tokenizer.save_pretrained(str(run_dir / "tokenizer"))
        write_training_history(history_rows, run_dir / "training_history.csv")
        save_json(best_metrics, run_dir / "best_val_metrics.json")
        save_model_artifact_metadata(
            args=args,
            run_dir=run_dir,
            input_dim=input_dim,
            best_val_metrics=best_metrics,
            selected_by=selected_by,
        )
        return {
            "tokenized_train_rows": int(len(train_token_dataset)),
            "tokenized_val_rows": int(len(val_token_dataset)),
            "train_truncated_rows": int(train_token_dataset.num_truncated),
            "val_truncated_rows": int(val_token_dataset.num_truncated),
            "best_val_metrics": best_metrics,
        }
    finally:
        if head is not None:
            del head
        if backbone is not None:
            del backbone
        if optimizer is not None:
            del optimizer
        if scheduler is not None:
            del scheduler
        release_cuda_memory()


def run_single_training(
    args: argparse.Namespace,
    *,
    environment: str,
    tokenizer,
    output_root: Path,
) -> Dict[str, Any]:
    run_spec = build_run_spec(
        args,
        environment=environment,
        output_root=output_root,
    )
    model_slug = str(run_spec["model_slug"])
    threshold_slug = str(run_spec["threshold_slug"])
    run_name = str(run_spec["run_name"])
    run_dir = Path(run_spec["run_dir"])
    eval_environments = list(run_spec["eval_environments"])

    print(f"\n=== {run_name} ===")
    print(f"Train dataset: {environment}")
    print(f"Eval datasets: {eval_environments}")
    print(f"Task type: {args.task_type}")
    print(f"Training mode: {args.training_mode}")
    print(f"Input view: {args.input_view}")

    localization_source, base_df = load_environment_base_bundle(
        args,
        environment=environment,
        model_slug=model_slug,
        threshold_slug=threshold_slug,
    )
    model_df = prepare_reasoning_dataframe(args, base_df)
    model_df.to_parquet(run_dir / "model_input_dataset.parquet", index=False)

    split_counts_df = (
        model_df.groupby(["split", "label_name"], as_index=False)
        .agg(n_rows=("example_id", "size"), n_examples=("example_id", "nunique"))
    )
    split_counts_df.to_csv(run_dir / "split_counts.csv", index=False)

    train_df = model_df[model_df["split"] == "train"].reset_index(drop=True)
    val_df = model_df[model_df["split"] == "val"].reset_index(drop=True)
    test_df = model_df[model_df["split"] == "test"].reset_index(drop=True)

    if train_df.empty:
        raise RuntimeError(f"No training rows for {run_name}")
    if val_df.empty or test_df.empty:
        raise RuntimeError(f"Expected non-empty val and test splits for {run_name}")

    base_run_config = {
        "run_name": run_name,
        "train_dataset": environment,
        "eval_datasets": eval_environments,
        "task_type": args.task_type,
        "training_mode": args.training_mode,
        "input_view": args.input_view,
        "threshold": args.threshold,
        "localization_source": str(localization_source),
        "split_counts": split_counts_df.to_dict(orient="records"),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "load_in_4bit": bool(args.load_in_4bit),
        "force_retrain": bool(args.force_retrain),
        "force_rescore": bool(args.force_rescore),
        "prepare_only": bool(args.prepare_only),
    }

    if args.prepare_only:
        save_json(base_run_config, run_dir / "run_config.json")
        print(f"Prepared datasets only for {run_name}; skipping training.")
        run_spec["prepared_only_row"] = {
            "run_name": run_name,
            "train_dataset": environment,
            "eval_dataset": environment,
            "eval_kind": "prepared_only",
            "task_type": args.task_type,
            "training_mode": args.training_mode,
            "threshold": args.threshold,
            "split": "prepared_only",
            "n_rows": int(len(model_df)),
            "n_examples": int(model_df["example_id"].nunique()),
        }
        return run_spec

    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for training runs.")

    if model_artifacts_exist(args, run_dir) and not args.force_retrain:
        print(f"Reusing saved model artifacts for {run_name}: {run_dir}", flush=True)
        return run_spec

    if args.training_mode == "lora":
        training_info = train_lora_pipeline(
            args,
            tokenizer=tokenizer,
            train_df=train_df,
            val_df=val_df,
            run_dir=run_dir,
        )
    else:
        training_info = train_frozen_mlp_pipeline(
            args,
            tokenizer=tokenizer,
            train_df=train_df,
            val_df=val_df,
            run_dir=run_dir,
        )

    save_json(
        {
            **base_run_config,
            **training_info,
        },
        run_dir / "run_config.json",
    )
    return run_spec


def run_single_evaluation(
    args: argparse.Namespace,
    *,
    run_spec: Dict[str, Any],
    tokenizer,
) -> list[Dict[str, Any]]:
    train_dataset = str(run_spec["train_dataset"])
    task_type = str(run_spec["task_type"])
    training_mode = str(run_spec["training_mode"])
    run_name = str(run_spec["run_name"])
    run_dir = Path(run_spec["run_dir"])
    model_slug = str(run_spec["model_slug"])
    threshold_slug = str(run_spec["threshold_slug"])
    eval_environments = list(run_spec["eval_environments"])
    predictions_path = run_dir / "predictions_test_across_datasets.parquet"
    metrics_summary_path = run_dir / "metrics_summary.csv"

    print(f"\n=== Evaluation: {run_name} ===")
    print(f"Train dataset: {train_dataset}")
    print(f"Eval datasets: {eval_environments}")

    if not model_artifacts_exist(args, run_dir):
        raise FileNotFoundError(f"Expected saved model artifacts before evaluation: {run_dir}")

    if predictions_path.exists() and metrics_summary_path.exists() and not args.force_rescore:
        print(f"Found existing evaluation outputs for {run_name}; checking compatibility.", flush=True)
        existing_summary_rows = load_summary_rows(
            metrics_summary_path,
            expected_train_dataset=train_dataset,
            expected_eval_datasets=eval_environments,
        )
        if existing_summary_rows is not None:
            print(f"Reusing existing evaluation outputs for {run_name}.", flush=True)
            return existing_summary_rows

    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for evaluation runs.")

    eval_base_dfs = build_eval_base_dfs(
        args,
        train_dataset=train_dataset,
        model_slug=model_slug,
        threshold_slug=threshold_slug,
        eval_environments=eval_environments,
    )

    eval_frames = []
    if training_mode == "lora":
        inference_model = load_saved_lora_pipeline(args, tokenizer, run_dir=run_dir)
        try:
            for eval_dataset, df in eval_base_dfs.items():
                split_df = df[df["split"] == "test"].reset_index(drop=True)
                if split_df.empty:
                    raise RuntimeError(f"Expected non-empty test split for eval dataset {eval_dataset}")
                print(
                    f"Scoring test split for eval dataset {eval_dataset} with {len(split_df):,} rows "
                    f"(batch_size={args.scoring_batch_size}, max_seq_length={args.max_seq_length})",
                    flush=True,
                )
                scored_df = score_dataframe_with_joint_model(
                    split_df,
                    tokenizer,
                    inference_model,
                    batch_size=args.scoring_batch_size,
                    max_length=args.max_seq_length,
                    progress_desc=f"score:{eval_dataset}",
                )
                scored_df["train_dataset"] = train_dataset
                scored_df["eval_dataset"] = eval_dataset
                scored_df["eval_kind"] = "id_test" if eval_dataset == train_dataset else "ood_test"
                eval_frames.append(scored_df)
        finally:
            del inference_model
            release_cuda_memory()
    else:
        backbone, head = load_saved_frozen_pipeline(args, tokenizer, run_dir=run_dir)
        try:
            for eval_dataset, df in eval_base_dfs.items():
                split_df = df[df["split"] == "test"].reset_index(drop=True)
                if split_df.empty:
                    raise RuntimeError(f"Expected non-empty test split for eval dataset {eval_dataset}")
                print(
                    f"Scoring test split for eval dataset {eval_dataset} with {len(split_df):,} rows "
                    f"(feature_batch_size={args.feature_batch_size or args.scoring_batch_size}, "
                    f"score_batch_size={args.scoring_batch_size}, max_seq_length={args.max_seq_length})",
                    flush=True,
                )
                scored_df = score_dataframe_with_frozen_backbone(
                    split_df,
                    tokenizer,
                    backbone,
                    head,
                    max_length=args.max_seq_length,
                    feature_batch_size=args.feature_batch_size or args.scoring_batch_size,
                    score_batch_size=args.scoring_batch_size,
                    progress_desc=f"score:{eval_dataset}",
                )
                scored_df["train_dataset"] = train_dataset
                scored_df["eval_dataset"] = eval_dataset
                scored_df["eval_kind"] = "id_test" if eval_dataset == train_dataset else "ood_test"
                eval_frames.append(scored_df)
        finally:
            del backbone
            del head
            release_cuda_memory()

    predictions_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    return save_evaluation_outputs(
        predictions_df=predictions_df,
        run_dir=run_dir,
        run_name=run_name,
        train_dataset=train_dataset,
        task_type=task_type,
        training_mode=training_mode,
        threshold=args.threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a localization model from sentence-prefix views with either LoRA adapters or a "
            "frozen LLM feature extractor + MLP head, and evaluate the trained model on ID/OOD test sets."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/playpen-ssd/smerrill/deception2/Dataset"),
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        default=["BS"],
        choices=list(DEFAULT_ENVIRONMENTS),
        help="Training dataset for this invocation. Exactly one environment is supported so separate GPUs can run separate datasets.",
    )
    parser.add_argument(
        "--eval-environments",
        nargs="+",
        default=None,
        choices=list(DEFAULT_ENVIRONMENTS),
        help="Datasets whose test splits should be evaluated for the trained model. Defaults to all datasets, with the train dataset always included for ID metrics.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Whether to predict deception_rate > threshold or the raw deception_rate.",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default="lora",
        choices=["lora", "frozen_mlp"],
        help="Train LoRA adapters plus an MLP head, or freeze the LLM and train only the MLP head on last-hidden-state features.",
    )
    parser.add_argument(
        "--input-view",
        type=str,
        default="prefix_plus_target_sentence",
        choices=["reasoning_only", "prefix_plus_target_sentence"],
        help=(
            "Which localized text span to feed the model. It should always be prefix_plus_target.  (Datasets are constructed so the"
            "target sentence means deception rate after that sentence.  `reasoning_only` uses prefix_text only; "
            "`prefix_plus_target_sentence` uses prefix_text + target_sentence_text."
        ),
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--model-dirname",
        type=str,
        default="DeepSeek-R1-Distill-Qwen-7B",
        help="Subdirectory under each dataset environment that contains localization outputs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Classification threshold and the threshold used for thresholded regression diagnostics.",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/playpen-ssd/smerrill/deception2/next_token_training_runs"),
    )
    parser.add_argument(
        "--dataset-cache-root",
        type=Path,
        default=Path("/playpen-ssd/smerrill/deception2/next_token_dataset_cache"),
    )
    parser.add_argument(
        "--cache-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache thresholded/split datasets to parquet for reuse across reruns.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore any existing dataset cache files and rebuild them from localization inputs.",
    )
    parser.add_argument(
        "--prefer-jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="If set to a positive integer, train for this many optimizer steps and ignore epoch-based stopping.",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scoring-batch-size", type=int, default=8)
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=None,
        help="Batch size used when extracting frozen hidden-state features. Defaults to scoring-batch-size.",
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--empty-reasoning-text", type=str, default=DEFAULT_EMPTY_REASONING_TEXT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-chat-template-for-instructional",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore any saved model artifact for a run and retrain it from scratch.",
    )
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Recompute evaluation outputs even when saved predictions and metrics already exist.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build datasets and splits for the run, save them, and exit before training.",
    )
    args = parser.parse_args()
    if len(args.environments) != 1:
        parser.error(
            "--environments must contain exactly one dataset per invocation. "
            "Launch separate runs for BS, Gridworld, and AdvisorAudit on separate GPUs."
        )
    train_environment = args.environments[0]
    if args.eval_environments is None:
        args.eval_environments = list(DEFAULT_ENVIRONMENTS)
    else:
        args.eval_environments = list(args.eval_environments)
    if train_environment not in args.eval_environments:
        args.eval_environments = [train_environment] + [
            env for env in args.eval_environments if env != train_environment
        ]
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.cache_datasets:
        args.dataset_cache_root.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if not args.prepare_only:
        tokenizer = ensure_tokenizer(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )

    train_environment = args.environments[0]
    run_spec = run_single_training(
        args,
        environment=train_environment,
        tokenizer=tokenizer,
        output_root=args.output_root,
    )

    all_rows: list[Dict[str, Any]] = []
    if args.prepare_only:
        prepared_only_row = run_spec.get("prepared_only_row")
        if prepared_only_row is not None:
            all_rows.append(prepared_only_row)
    else:
        print("\n=== Evaluation Phase ===", flush=True)
        run_rows = run_single_evaluation(
            args,
            run_spec=run_spec,
            tokenizer=tokenizer,
        )
        all_rows.extend(run_rows)

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        summary_path = args.output_root / (
            f"summary__{slugify(args.training_mode)}__{slugify(args.task_type)}"
            f"__input_{slugify(args.input_view)}"
            f"__train_{slugify(train_environment)}"
            f"__threshold_{str(args.threshold).replace('.', 'p')}.csv"
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary table to: {summary_path}")


if __name__ == "__main__":
    main()
