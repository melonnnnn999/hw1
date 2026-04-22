import json
import os
from pathlib import Path

import numpy as np

from .data import read_image, scan_dataset, split_dataset
from .metrics import evaluate
from .model import SimpleMLP


def get_plt():
    mpl_dir = Path("runs") / "matplotlib_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_history(args):
    plt = get_plt()
    with open(args.history, "r", encoding="utf-8") as file_obj:
        history = json.load(file_obj)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train acc")
    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print("Training curves saved to:", args.output)


def visualize_weights(args):
    plt = get_plt()
    model, meta = SimpleMLP.load(args.checkpoint)
    image_size = int(meta["image_size"])
    count = min(args.num, model.hidden_dim)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for index in range(rows * cols):
        axes[index].axis("off")
        if index >= count:
            continue
        weights = model.W1[:, index].reshape(image_size, image_size, 3)
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        axes[index].imshow(weights)
        axes[index].set_title("h%d" % index, fontsize=8)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print("First-layer weight visualization saved to:", args.output)


def save_error_examples(args):
    plt = get_plt()
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

    result = evaluate(
        model,
        splits["test_paths"],
        splits["test_labels"],
        args.batch_size,
        int(meta["image_size"]),
        np.array(meta["mean"], dtype=np.float32),
        np.array(meta["std"], dtype=np.float32),
        len(class_names),
        seed=int(meta["seed"]),
        return_details=True,
    )

    wrong_indices = np.where(result["y_true"] != result["y_pred"])[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "wrong_examples.txt"
    with open(txt_path, "w", encoding="utf-8") as file_obj:
        for index in wrong_indices[: args.num]:
            true_name = class_names[int(result["y_true"][index])]
            pred_name = class_names[int(result["y_pred"][index])]
            file_obj.write("%s | true=%s | pred=%s\n" % (result["paths"][index], true_name, pred_name))

    if len(wrong_indices) == 0:
        print("No wrong predictions found on the test split.")
        print("Saved list to:", txt_path)
        return

    show_num = min(args.num, len(wrong_indices))
    cols = min(4, show_num)
    rows = int(np.ceil(show_num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for panel_index in range(rows * cols):
        axes[panel_index].axis("off")
        if panel_index >= show_num:
            continue
        wrong_index = wrong_indices[panel_index]
        image = read_image(result["paths"][wrong_index], image_size=int(meta["image_size"]))
        true_name = class_names[int(result["y_true"][wrong_index])]
        pred_name = class_names[int(result["y_pred"][wrong_index])]
        axes[panel_index].imshow(image)
        axes[panel_index].set_title("T:%s\nP:%s" % (true_name, pred_name), fontsize=8)

    fig.tight_layout()
    image_path = output_dir / "wrong_examples.png"
    fig.savefig(image_path, dpi=160)
    print("Wrong-example figure saved to:", image_path)
    print("Wrong-example list saved to:", txt_path)

