import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .data import compute_mean_std, scan_dataset, split_dataset
from .metrics import accuracy, evaluate, print_confusion_matrix
from .model import SimpleMLP


def sgd_step(model, grads, lr):
    """Vanilla SGD update."""
    model.W1 -= lr * grads["W1"]
    model.b1 -= lr * grads["b1"]
    model.W2 -= lr * grads["W2"]
    model.b2 -= lr * grads["b2"]


def train_model(args):
    image_paths, labels, class_names = scan_dataset(args.data_dir)
    num_classes = len(class_names)
    splits = split_dataset(
        image_paths,
        labels,
        num_classes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )

    print("Number of classes:", num_classes)
    print(
        "Train:",
        len(splits["train_paths"]),
        "Val:",
        len(splits["val_paths"]),
        "Test:",
        len(splits["test_paths"]),
    )

    print("Computing training mean/std ...")
    mean, std = compute_mean_std(splits["train_paths"], image_size=args.image_size)
    input_dim = args.image_size * args.image_size * 3

    model = SimpleMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        activation=args.activation,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.npz"
    history_path = output_dir / "history.json"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        current_lr = args.lr * (args.lr_decay ** (epoch - 1))
        total_loss = 0.0
        total_num = 0
        train_true = []
        train_pred = []

        from .data import make_batches

        batches = make_batches(
            splits["train_paths"],
            splits["train_labels"],
            args.batch_size,
            args.image_size,
            mean,
            std,
            shuffle=True,
            seed=args.seed + epoch,
        )

        for features, targets, _ in batches:
            loss, grads, scores = model.loss_and_grad(features, targets, weight_decay=args.weight_decay)
            pred = np.argmax(scores, axis=1)
            sgd_step(model, grads, current_lr)

            total_loss += loss * len(targets)
            total_num += len(targets)
            train_true.append(targets)
            train_pred.append(pred)

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_loss = total_loss / total_num
        train_acc = accuracy(train_true, train_pred)

        val_result = evaluate(
            model,
            splits["val_paths"],
            splits["val_labels"],
            args.batch_size,
            args.image_size,
            mean,
            std,
            num_classes,
            seed=args.seed,
        )

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_result["loss"]))
        history["val_acc"].append(float(val_result["accuracy"]))
        history["lr"].append(float(current_lr))

        print(
            "Epoch %02d/%02d | lr %.5f | train loss %.4f acc %.4f | val loss %.4f acc %.4f"
            % (
                epoch,
                args.epochs,
                current_lr,
                train_loss,
                train_acc,
                val_result["loss"],
                val_result["accuracy"],
            )
        )

        if val_result["accuracy"] > best_val_acc:
            best_val_acc = val_result["accuracy"]
            best_epoch = epoch
            meta = {
                "data_dir": str(Path(args.data_dir).resolve()),
                "class_names": class_names,
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "num_classes": num_classes,
                "activation": args.activation,
                "image_size": args.image_size,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "max_per_class": args.max_per_class,
                "best_epoch": best_epoch,
                "best_val_acc": float(best_val_acc),
            }
            model.save(best_path, meta)

        with open(history_path, "w", encoding="utf-8") as file_obj:
            json.dump(history, file_obj, ensure_ascii=False, indent=2)

    print("\nTraining complete. Best val acc %.4f at epoch %d." % (best_val_acc, best_epoch))
    print("Best checkpoint saved to:", best_path)
    print("Training history saved to:", history_path)

    return {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "best_path": str(best_path),
        "history_path": str(history_path),
    }


def test_model(args):
    model, meta = SimpleMLP.load(args.checkpoint)
    data_dir = args.data_dir if args.data_dir else meta["data_dir"]

    image_paths, labels, class_names = scan_dataset(data_dir)
    splits = split_dataset(
        image_paths,
        labels,
        num_classes=len(class_names),
        train_ratio=float(meta["train_ratio"]),
        val_ratio=float(meta["val_ratio"]),
        seed=int(meta["seed"]),
        max_per_class=int(meta["max_per_class"]),
    )

    split_name = args.split
    result = evaluate(
        model,
        splits[split_name + "_paths"],
        splits[split_name + "_labels"],
        args.batch_size,
        int(meta["image_size"]),
        np.array(meta["mean"], dtype=np.float32),
        np.array(meta["std"], dtype=np.float32),
        len(class_names),
        seed=int(meta["seed"]),
    )

    print("%s accuracy: %.4f" % (split_name, result["accuracy"]))
    print("%s loss: %.4f" % (split_name, result["loss"]))
    print_confusion_matrix(result["confusion_matrix"], class_names)


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def grid_search(args):
    hidden_list = parse_int_list(args.hidden_dims)
    lr_list = parse_float_list(args.lrs)
    wd_list = parse_float_list(args.weight_decays)
    act_list = [part.strip() for part in args.activations.split(",") if part.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "search_results.csv"

    rows = []
    for hidden_dim, lr, weight_decay, activation in itertools.product(
        hidden_list, lr_list, wd_list, act_list
    ):
        run_name = "h%d_lr%s_wd%s_%s" % (hidden_dim, str(lr), str(weight_decay), activation)
        print("\n========== Start:", run_name, "==========")

        one_args = argparse.Namespace(**vars(args))
        one_args.mode = "train"
        one_args.hidden_dim = hidden_dim
        one_args.lr = lr
        one_args.weight_decay = weight_decay
        one_args.activation = activation
        one_args.output_dir = str(output_dir / run_name)

        result = train_model(one_args)
        row = {
            "hidden_dim": hidden_dim,
            "lr": lr,
            "weight_decay": weight_decay,
            "activation": activation,
            "best_val_acc": result["best_val_acc"],
            "best_epoch": result["best_epoch"],
            "checkpoint": result["best_path"],
        }
        rows.append(row)

        with open(result_file, "w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)

    rows = sorted(rows, key=lambda row: row["best_val_acc"], reverse=True)
    print("\nGrid search complete. Results saved to:", result_file)
    print("Best configuration:")
    print(rows[0])

