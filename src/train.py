"""Train the Task 3 CNN on MNIST, with and without data augmentation, and log the results.

Run from the repository root:

    python src/train.py

The script loads data/mnist.npz (downloading MNIST through Keras on first use), holds out a
stratified 10% of the training images for validation and trains the same CNN twice from the
same starting weights: once on the original images and once with random rotation, shift and
zoom. Each model keeps the weights of the epoch with the lowest validation loss. Both models
are then evaluated on the 10,000 test images, and the script writes:

    results/train.log                  run log (also printed to the console)
    results/history_baseline.csv       loss and accuracy per epoch, without augmentation
    results/history_augmented.csv      loss and accuracy per epoch, with augmentation
    results/metrics.json               accuracy, loss and macro F1 of both models on each split
    results/model_comparison.csv       one row per model
    results/classification_report.csv  precision, recall and F1 per digit for the final model
    models/cnn_baseline.keras          trained model, no augmentation
    models/cnn_augmented.keras         trained model, with augmentation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TensorFlow start-up messages

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [str(digit) for digit in range(10)]
VARIANTS = {"baseline": "CNN without augmentation", "augmented": "CNN with augmentation"}

log = logging.getLogger("train")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the MNIST CNN with and without data augmentation.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "mnist.npz", help="path to mnist.npz")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results", help="where to write logs and metrics")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models", help="where to save the trained models")
    parser.add_argument("--epochs", type=int, default=20, help="training epochs for each model")
    parser.add_argument("--batch-size", type=int, default=128, help="images per training batch")
    parser.add_argument("--val-size", type=float, default=0.1, help="share of training images held out for validation")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N training and test images (quick check)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split, the weights and the augmentation")
    return parser.parse_args(argv)


def setup_logging(log_file: Path) -> None:
    """Send log records to the console and to results/train.log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def relative(path: Path) -> str:
    """Show paths relative to the repository root in the log."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_mnist(path: Path):
    """Read the local copy of MNIST; download it through Keras and cache it on first use."""
    if not path.exists():
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    with np.load(path) as data:
        return (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"])


def to_model_input(images: np.ndarray) -> np.ndarray:
    """(n, 28, 28) uint8 -> (n, 28, 28, 1) float32. Pixel values stay 0-255; the model rescales them."""
    return images[..., None].astype("float32")


def split_data(x_pool, y_pool, x_test, y_test, val_size: float, seed: int) -> dict:
    """Hold out a stratified validation set; the official test set is kept for the final evaluation."""
    x_train, x_val, y_train, y_val = train_test_split(
        x_pool, y_pool, test_size=val_size, stratify=y_pool, random_state=seed
    )
    return {
        "train": (to_model_input(x_train), y_train),
        "validation": (to_model_input(x_val), y_val),
        "test": (to_model_input(x_test), y_test),
    }


def build_augmentation(seed: int) -> keras.Sequential:
    """Random rotation, shift and zoom for training batches.

    No flips: a mirrored digit is no longer the same symbol.
    """
    fill = {"fill_mode": "constant", "fill_value": 0.0}  # pixels moved in from outside the image are black
    return keras.Sequential([
        layers.RandomRotation(10 / 360, seed=seed, **fill),     # up to ±10 degrees
        layers.RandomTranslation(0.1, 0.1, seed=seed, **fill),  # up to ±10% of the image size (about 3 pixels)
        layers.RandomZoom(0.1, seed=seed, **fill),              # up to ±10%
    ], name="augmentation")


def build_cnn(name: str, augmentation: keras.Sequential | None = None) -> keras.Model:
    """Three convolutional layers, two max-pooling layers and two fully connected layers."""
    inputs = keras.Input(shape=(28, 28, 1), name="image")
    x = inputs if augmentation is None else augmentation(inputs)    # active during training only
    x = layers.Rescaling(1 / 255, name="rescale")(x)                 # pixel values 0-255 -> 0-1
    x = layers.Conv2D(32, 3, activation="relu", name="conv1")(x)     # -> 26 x 26 x 32
    x = layers.MaxPooling2D(2, name="pool1")(x)                      # -> 13 x 13 x 32
    x = layers.Conv2D(64, 3, activation="relu", name="conv2")(x)     # -> 11 x 11 x 64
    x = layers.MaxPooling2D(2, name="pool2")(x)                      # -> 5 x 5 x 64
    x = layers.Conv2D(128, 3, activation="relu", name="conv3")(x)    # -> 3 x 3 x 128
    x = layers.Flatten(name="flatten")(x)                            # -> 1,152 values
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(10, activation="softmax", name="output")(x)
    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class EpochLogger(keras.callbacks.Callback):
    """Write one log line per epoch."""

    def __init__(self, label: str):
        super().__init__()
        self.label = label

    def on_epoch_begin(self, epoch, logs=None):
        self.started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log.info(f"{self.label} | epoch {epoch + 1:>2}/{self.params['epochs']} | "
                 f"train loss {logs['loss']:.4f}, acc {logs['accuracy']:.4f} | "
                 f"val loss {logs['val_loss']:.4f}, acc {logs['val_accuracy']:.4f} | "
                 f"{time.perf_counter() - self.started:.0f} s")


def train_variant(key: str, splits: dict, epochs: int, batch_size: int, models_dir: Path, seed: int):
    """Train one model, then reload the weights of its epoch with the lowest validation loss."""
    keras.utils.set_random_seed(seed)  # both models start from the same weights and batch order
    augmentation = build_augmentation(seed) if key == "augmented" else None
    model = build_cnn(f"cnn_{key}", augmentation)
    (x_train, y_train), (x_val, y_val) = splits["train"], splits["validation"]
    path = Path(models_dir) / f"cnn_{key}.keras"
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Training {VARIANTS[key]}: {model.count_params():,} parameters, "
             f"{epochs} epochs, batch size {batch_size}")
    started = time.perf_counter()
    fit = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,
        shuffle=True, verbose=0,
        callbacks=[keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
                   EpochLogger(VARIANTS[key])],
    )
    seconds = time.perf_counter() - started
    history = pd.DataFrame(fit.history)[["loss", "accuracy", "val_loss", "val_accuracy"]]
    history.insert(0, "epoch", np.arange(1, len(history) + 1))
    return keras.models.load_model(path), history, seconds


def evaluate(model: keras.Model, x: np.ndarray, y: np.ndarray):
    """Accuracy, cross-entropy loss, macro F1 and error count, with augmentation and dropout off."""
    probs = model.predict(x, batch_size=1000, verbose=0)
    pred = probs.argmax(axis=1)
    loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-7, 1.0)).mean()
    scores = {
        "accuracy": round(float((pred == y).mean()), 4),
        "loss": round(float(loss), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "errors": int((pred != y).sum()),
    }
    return scores, probs


def summarize_variant(key: str, model: keras.Model, history: pd.DataFrame, seconds: float, splits: dict):
    """Scores on every split and one row for the comparison table."""
    scores, probs = {}, {}
    for split, (x, y) in splits.items():
        scores[split], probs[split] = evaluate(model, x, y)
    best_epoch = int(history["val_loss"].idxmin()) + 1
    summary = {"label": VARIANTS[key], "parameters": model.count_params(), "best_epoch": best_epoch,
               "train_time_s": round(seconds), **scores}
    row = {
        "model": VARIANTS[key],
        "parameters": model.count_params(),
        "epochs": len(history),
        "best_epoch": best_epoch,
        "train_accuracy": scores["train"]["accuracy"],
        "val_accuracy": scores["validation"]["accuracy"],
        "train_val_gap": round(scores["train"]["accuracy"] - scores["validation"]["accuracy"], 4),
        "min_val_loss": round(float(history["val_loss"].min()), 4),
        "final_val_loss": round(float(history["val_loss"].iloc[-1]), 4),
        "test_accuracy": scores["test"]["accuracy"],
        "test_loss": scores["test"]["loss"],
        "test_macro_f1": scores["test"]["macro_f1"],
        "test_errors": scores["test"]["errors"],
        "train_time_s": round(seconds),
    }
    log.info(f"{VARIANTS[key]:<24} | best epoch {best_epoch:>2} | train acc {row['train_accuracy']:.4f} | "
             f"val acc {row['val_accuracy']:.4f} | test acc {row['test_accuracy']:.4f} | "
             f"test errors {row['test_errors']} of {len(splits['test'][1]):,} | {row['train_time_s']} s")
    return summary, row, probs["test"]


def save_results(results_dir: Path, run_info: dict, summaries: dict, rows: list, y_test, test_probs: dict) -> dict:
    """Write metrics.json, model_comparison.csv and the per-digit report of the final model."""
    final = min(summaries, key=lambda key: summaries[key]["validation"]["loss"])  # chosen on validation data
    report = classification_report(y_test, test_probs[final].argmax(axis=1), target_names=CLASS_NAMES,
                                   digits=4, output_dict=True)
    report.pop("accuracy")
    metrics = {**run_info, "models": summaries, "final_model": final,
               "tensorflow_version": tf.__version__, "keras_version": keras.__version__}
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(results_dir / "model_comparison.csv", index=False)
    (pd.DataFrame(report).T.round(4).astype({"support": int}).rename_axis("class")
     .to_csv(results_dir / "classification_report.csv"))
    log.info(f"Final model (lowest validation loss): {VARIANTS[final]}")
    return metrics


def main(argv=None) -> dict:
    args = parse_args(argv)
    setup_logging(args.results_dir / "train.log")
    tf.config.experimental.enable_op_determinism()  # repeatable results for a given seed

    (x_pool, y_pool), (x_test, y_test) = load_mnist(args.data)
    if args.limit:
        x_pool, y_pool, x_test, y_test = (a[:args.limit] for a in (x_pool, y_pool, x_test, y_test))
    log.info(f"Loaded {len(x_pool):,} training and {len(x_test):,} test images "
             f"({x_pool.shape[1]} x {x_pool.shape[2]} pixels, 10 digit classes) from {relative(args.data)}")
    splits = split_data(x_pool, y_pool, x_test, y_test, args.val_size, args.seed)
    log.info(f"Split: {len(splits['train'][1]):,} training / {len(splits['validation'][1]):,} validation "
             f"(stratified, seed {args.seed}) / {len(splits['test'][1]):,} test images")

    summaries, rows, test_probs = {}, [], {}
    for key in VARIANTS:
        model, history, seconds = train_variant(key, splits, args.epochs, args.batch_size, args.models_dir, args.seed)
        history.round(5).to_csv(args.results_dir / f"history_{key}.csv", index=False)
        summaries[key], row, test_probs[key] = summarize_variant(key, model, history, seconds, splits)
        rows.append(row)

    run_info = {
        "task": "Progree ML Task 3: CNN image classification on MNIST",
        "n_train": len(splits["train"][1]),
        "n_validation": len(splits["validation"][1]),
        "n_test": len(splits["test"][1]),
        "random_state": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "Adam, learning rate 0.001",
        "augmentation": "rotation up to 10 degrees, shift up to 10%, zoom up to 10%, training batches only",
        "checkpoint": "weights from the epoch with the lowest validation loss",
    }
    metrics = save_results(args.results_dir, run_info, summaries, rows, splits["test"][1], test_probs)
    gap = {key: summaries[key]["train"]["accuracy"] - summaries[key]["validation"]["accuracy"] for key in VARIANTS}
    log.info(f"Train-validation accuracy gap: {gap['baseline'] * 100:.2f} points without augmentation, "
             f"{gap['augmented'] * 100:.2f} with augmentation")
    log.info(f"Saved results to {relative(args.results_dir)}/ and models to {relative(args.models_dir)}/")
    return metrics


if __name__ == "__main__":
    main()
