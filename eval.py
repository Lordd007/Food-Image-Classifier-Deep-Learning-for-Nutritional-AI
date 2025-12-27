#!/usr/bin/env python3
"""
Quick eval for Food-101 fine-tuning (TensorFlow/Keras).
Writes metrics.json (val_top1, val_top5, etc.) so CI/Colab can compare runs.

Usage examples:
  python eval.py --quick --out metrics.json
  python eval.py --epochs 2 --train-split "train[:8000]" --val-split "validation[:2000]" --out metrics.json
"""

import argparse
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np

# TF logs quieter
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import tensorflow_datasets as tfds


@dataclass
class Metrics:
    val_top1: float
    val_top5: float
    val_loss: float
    train_top1_last: float
    train_loss_last: float
    seconds: float
    git_sha: str
    tf_version: str
    device: str
    notes: str


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    # Best effort determinism (may reduce speed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def get_git_sha() -> str:
    # Works if git is available; otherwise "unknown"
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def build_model(
    num_classes: int,
    img_size: int,
    base_name: str,
    dropout: float,
    label_smoothing: float,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")

    # Mild, safe augmentation for quick benchmark (you can expand later)
    x = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augment",
    )(inputs)

    # Choose backbone + correct preprocess
    base_name = base_name.lower()
    if base_name == "efficientnetb0":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        Base = tf.keras.applications.EfficientNetB0
    elif base_name == "mobilenetv2":
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        Base = tf.keras.applications.MobileNetV2
    else:
        raise ValueError(f"Unsupported base model: {base_name}")

    x = tf.keras.layers.Lambda(preprocess, name="preprocess")(x)

    base = Base(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False  # stage A

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{Base.__name__}_food101")

    # Loss + metrics
    loss = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )

    return model


def make_datasets(
    img_size: int,
    batch_size: int,
    train_split: str,
    val_split: str,
    seed: int,
    data_dir: str | None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    ds_train = tfds.load(
        "food101",
        split=train_split,
        as_supervised=True,
        shuffle_files=True,
        data_dir=data_dir,
    )
    ds_val = tfds.load(
        "food101",
        split=val_split,
        as_supervised=True,
        shuffle_files=False,
        data_dir=data_dir,
    )

    # Get class count from builder info (stable)
    builder = tfds.builder("food101", data_dir=data_dir)
    builder.download_and_prepare()
    num_classes = builder.info.features["label"].num_classes

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(image, label):
        image = tf.image.resize(image, (img_size, img_size), method="bilinear")
        image = tf.cast(image, tf.float32)
        return image, label

    ds_train = (
        ds_train
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .shuffle(2048, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    ds_val = (
        ds_val
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return ds_train, ds_val, num_classes


def fine_tune(
    model: tf.keras.Model,
    base_unfreeze_last_n: int,
    lr: float,
) -> None:
    # Find base model inside
    # (the base is the only large Application model in the graph)
    base_candidates = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    base = None
    for cand in base_candidates:
        # heuristic: backbone model has many layers
        if len(cand.layers) > 50:
            base = cand
            break
    if base is None:
        return

    base.trainable = True
    if base_unfreeze_last_n is not None and base_unfreeze_last_n > 0:
        for layer in base.layers[:-base_unfreeze_last_n]:
            layer.trainable = False

    # Re-compile with low LR for fine-tuning
    loss = model.loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=model.metrics,
    )


def detect_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"GPU({gpus[0].name})"
    return "CPU"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="metrics.json")
    p.add_argument("--seed", type=int, default=42)

    # Data
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--data-dir", default=None)

    # Splits (quick benchmark defaults)
    p.add_argument("--train-split", default="train[:8000]")
    p.add_argument("--val-split", default="validation[:2000]")

    # Model/training
    p.add_argument("--base", default="efficientnetb0", choices=["efficientnetb0", "mobilenetv2"])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--head-epochs", type=int, default=2)
    p.add_argument("--ft-epochs", type=int, default=4)
    p.add_argument("--unfreeze-last-n", type=int, default=40)
    p.add_argument("--ft-lr", type=float, default=3e-5)

    p.add_argument("--quick", action="store_true", help="Use defaults intended for ~5-10 min GPU run.")
    args = p.parse_args()

    # Quick mode just enforces consistent defaults if user passes partial args
    if args.quick:
        args.train_split = "train[:8000]"
        args.val_split = "validation[:2000]"
        args.epochs = 6
        args.head_epochs = 2
        args.ft_epochs = 4
        args.unfreeze_last_n = 40
        args.ft_lr = 3e-5

    set_determinism(args.seed)

    # Enable mixed precision on GPU to speed up (safe for these models)
    if tf.config.list_physical_devices("GPU"):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    t0 = time.time()

    ds_train, ds_val, num_classes = make_datasets(
        img_size=args.img_size,
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    model = build_model(
        num_classes=num_classes,
        img_size=args.img_size,
        base_name=args.base,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_top1", patience=2, restore_best_weights=True),
    ]

    # Stage A: train head
    hist_a = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.head_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Stage B: fine-tune top layers
    fine_tune(model, base_unfreeze_last_n=args.unfreeze_last_n, lr=args.ft_lr)

    hist_b = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.head_epochs + args.ft_epochs,
        initial_epoch=args.head_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Final eval
    results = model.evaluate(ds_val, verbose=0)
    # results order: loss, top1, top5
    val_loss, val_top1, val_top5 = [float(x) for x in results]

    # last train metrics best-effort
    train_top1_last = float(hist_b.history.get("top1", hist_a.history.get("top1", [0.0]))[-1])
    train_loss_last = float(hist_b.history.get("loss", hist_a.history.get("loss", [0.0]))[-1])

    secs = time.time() - t0

    m = Metrics(
        val_top1=val_top1,
        val_top5=val_top5,
        val_loss=val_loss,
        train_top1_last=train_top1_last,
        train_loss_last=train_loss_last,
        seconds=secs,
        git_sha=get_git_sha(),
        tf_version=tf.__version__,
        device=detect_device(),
        notes=f"{args.base} img={args.img_size} split={args.train_split}/{args.val_split} seed={args.seed}",
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(asdict(m), f, indent=2)

    # Also print a GitHub-friendly summary
    print("\n=== QUICK EVAL SUMMARY ===")
    print(f"val_top1: {m.val_top1:.4f}")
    print(f"val_top5: {m.val_top5:.4f}")
    print(f"val_loss: {m.val_loss:.4f}")
    print(f"seconds:  {m.seconds:.1f}")
    print(f"device:   {m.device}")
    print(f"git_sha:  {m.git_sha}")
    print(f"wrote:    {args.out}")


if __name__ == "__main__":
    main()
