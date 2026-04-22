import argparse
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def scan_dataset(data_dir):
    data_dir = Path(data_dir)
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError("没有在数据目录中找到类别文件夹: %s" % data_dir)

    image_paths = []
    labels = []
    class_names = [p.name for p in class_dirs]

    for label, class_dir in enumerate(class_dirs):
        files = sorted(class_dir.glob("*.jpg"))
        for f in files:
            image_paths.append(str(f))
            labels.append(label)

    return image_paths, np.array(labels, dtype=np.int64), class_names


def split_dataset(image_paths, labels, num_classes, train_ratio=0.7, val_ratio=0.15, seed=42, max_per_class=0):
    rng = np.random.default_rng(seed)
    splits = {
        "train_paths": [],
        "train_labels": [],
        "val_paths": [],
        "val_labels": [],
        "test_paths": [],
        "test_labels": [],
    }

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)

        if max_per_class and max_per_class > 0:
            idx = idx[:max_per_class]

        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        for i in train_idx:
            splits["train_paths"].append(image_paths[i])
            splits["train_labels"].append(labels[i])
        for i in val_idx:
            splits["val_paths"].append(image_paths[i])
            splits["val_labels"].append(labels[i])
        for i in test_idx:
            splits["test_paths"].append(image_paths[i])
            splits["test_labels"].append(labels[i])

    for name in ["train", "val", "test"]:
        paths = splits[name + "_paths"]
        y = np.array(splits[name + "_labels"], dtype=np.int64)
        order = rng.permutation(len(paths))
        splits[name + "_paths"] = [paths[i] for i in order]
        splits[name + "_labels"] = y[order]

    return splits


def read_image(path, image_size=64):
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (image_size, image_size):
            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:
                resample = Image.BILINEAR
            img = img.resize((image_size, image_size), resample)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compute_mean_std(train_paths, image_size=64):
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in train_paths:
        img = read_image(path, image_size=image_size)
        channel_sum += img.sum(axis=(0, 1))
        channel_square_sum += (img ** 2).sum(axis=(0, 1))
        pixel_count += img.shape[0] * img.shape[1]

    mean = channel_sum / pixel_count
    var = channel_square_sum / pixel_count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def make_batches(paths, labels, batch_size, image_size, mean, std, shuffle=True, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    if shuffle:
        rng.shuffle(idx)

    mean = mean.reshape(1, 1, 3)
    std = std.reshape(1, 1, 3)

    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        batch_paths = [paths[i] for i in batch_idx]
        y = labels[batch_idx]
        x = np.zeros((len(batch_idx), image_size * image_size * 3), dtype=np.float32)

        for j, path in enumerate(batch_paths):
            img = read_image(path, image_size=image_size)
            img = (img - mean) / std
            x[j] = img.reshape(-1)

        yield x, y, batch_paths


class SimpleMLP:
    def __init__(self, input_dim, hidden_dim, num_classes, activation="relu", seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.activation = activation

        rng = np.random.default_rng(seed)

        if activation == "relu":
            scale1 = np.sqrt(2.0 / input_dim)
        else:
            scale1 = np.sqrt(1.0 / input_dim)

        self.W1 = rng.normal(0, scale1, size=(input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, num_classes)).astype(np.float32)
        self.b2 = np.zeros(num_classes, dtype=np.float32)

    def activate(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "sigmoid":
            z = np.clip(z, -50, 50)
            return 1.0 / (1.0 + np.exp(-z))
        if self.activation == "tanh":
            return np.tanh(z)
        raise ValueError("不支持的激活函数: %s" % self.activation)

    def activation_grad(self, z, a):
        if self.activation == "relu":
            return (z > 0).astype(np.float32)
        if self.activation == "sigmoid":
            return a * (1.0 - a)
        if self.activation == "tanh":
            return 1.0 - a ** 2
        raise ValueError("不支持的激活函数: %s" % self.activation)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self.activate(z1)
        scores = a1 @ self.W2 + self.b2
        cache = (x, z1, a1)
        return scores, cache

    def softmax_loss(self, scores, y):
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        n = y.shape[0]
        loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
        return loss, probs

    def loss(self, x, y, weight_decay=0.0):
        scores, _ = self.forward(x)
        data_loss, _ = self.softmax_loss(scores, y)
        reg_loss = 0.5 * weight_decay * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return data_loss + reg_loss

    def loss_and_grad(self, x, y, weight_decay=0.0):
        scores, cache = self.forward(x)
        x, z1, a1 = cache
        loss, probs = self.softmax_loss(scores, y)
        loss += 0.5 * weight_decay * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))

        n = x.shape[0]
        dscores = probs.copy()
        dscores[np.arange(n), y] -= 1.0
        dscores /= n

        dW2 = a1.T @ dscores + weight_decay * self.W2
        db2 = np.sum(dscores, axis=0)

        da1 = dscores @ self.W2.T
        dz1 = da1 * self.activation_grad(z1, a1)

        dW1 = x.T @ dz1 + weight_decay * self.W1
        db1 = np.sum(dz1, axis=0)

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
        return loss, grads, scores

    def predict(self, x):
        scores, _ = self.forward(x)
        return np.argmax(scores, axis=1)

    def save(self, save_path, meta):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            meta=json.dumps(meta, ensure_ascii=False),
        )

    @staticmethod
    def load(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        model = SimpleMLP(
            input_dim=int(meta["input_dim"]),
            hidden_dim=int(meta["hidden_dim"]),
            num_classes=int(meta["num_classes"]),
            activation=meta["activation"],
            seed=0,
        )
        model.W1 = data["W1"].astype(np.float32)
        model.b1 = data["b1"].astype(np.float32)
        model.W2 = data["W2"].astype(np.float32)
        model.b2 = data["b2"].astype(np.float32)
        return model, meta


def sgd_step(model, grads, lr):
    model.W1 -= lr * grads["W1"]
    model.b1 -= lr * grads["b1"]
    model.W2 -= lr * grads["W2"]
    model.b2 -= lr * grads["b2"]


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def print_confusion_matrix(cm, class_names):
    print("\n类别编号：")
    for i, name in enumerate(class_names):
        print("%2d -> %s" % (i, name))

    print("\n混淆矩阵（行是真实类别，列是预测类别）：")
    header = "true\\pred " + " ".join(["%5d" % i for i in range(len(class_names))])
    print(header)
    for i in range(len(class_names)):
        row = "    %2d    " % i + " ".join(["%5d" % v for v in cm[i]])
        print(row)


def evaluate(model, paths, labels, batch_size, image_size, mean, std, num_classes, seed=0, return_details=False):
    all_true = []
    all_pred = []
    all_paths = []
    total_loss = 0.0
    total_num = 0

    batches = make_batches(paths, labels, batch_size, image_size, mean, std, shuffle=False, seed=seed)
    for x, y, batch_paths in batches:
        loss = model.loss(x, y, weight_decay=0.0)
        pred = model.predict(x)
        total_loss += loss * len(y)
        total_num += len(y)
        all_true.append(y)
        all_pred.append(pred)
        if return_details:
            all_paths.extend(batch_paths)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm = confusion_matrix(y_true, y_pred, num_classes)

    result = {
        "loss": total_loss / total_num,
        "accuracy": accuracy(y_true, y_pred),
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if return_details:
        result["paths"] = all_paths
    return result


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

    print("类别数:", num_classes)
    print("训练集:", len(splits["train_paths"]), "验证集:", len(splits["val_paths"]), "测试集:", len(splits["test_paths"]))

    print("正在统计训练集 mean/std ...")
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

        for x, y, _ in batches:
            loss, grads, scores = model.loss_and_grad(x, y, weight_decay=args.weight_decay)
            pred = np.argmax(scores, axis=1)
            sgd_step(model, grads, current_lr)

            total_loss += loss * len(y)
            total_num += len(y)
            train_true.append(y)
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
            % (epoch, args.epochs, current_lr, train_loss, train_acc, val_result["loss"], val_result["accuracy"])
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

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n训练完成。最佳验证集准确率 %.4f，出现在第 %d 轮。" % (best_val_acc, best_epoch))
    print("最佳模型已保存到:", best_path)
    print("训练曲线数据已保存到:", history_path)

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


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def grid_search(args):
    hidden_list = parse_int_list(args.hidden_dims)
    lr_list = parse_float_list(args.lrs)
    wd_list = parse_float_list(args.weight_decays)
    act_list = [x.strip() for x in args.activations.split(",") if x.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "search_results.csv"

    rows = []
    for hidden_dim, lr, wd, act in itertools.product(hidden_list, lr_list, wd_list, act_list):
        run_name = "h%d_lr%s_wd%s_%s" % (hidden_dim, str(lr), str(wd), act)
        print("\n========== 开始实验:", run_name, "==========")

        one_args = argparse.Namespace(**vars(args))
        one_args.mode = "train"
        one_args.hidden_dim = hidden_dim
        one_args.lr = lr
        one_args.weight_decay = wd
        one_args.activation = act
        one_args.output_dir = str(output_dir / run_name)

        result = train_model(one_args)
        row = {
            "hidden_dim": hidden_dim,
            "lr": lr,
            "weight_decay": wd,
            "activation": act,
            "best_val_acc": result["best_val_acc"],
            "best_epoch": result["best_epoch"],
            "checkpoint": result["best_path"],
        }
        rows.append(row)

        with open(result_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)

    rows = sorted(rows, key=lambda r: r["best_val_acc"], reverse=True)
    print("\n超参数搜索完成，结果保存到:", result_file)
    print("验证集最好的一组:")
    print(rows[0])


def get_plt():
    mpl_dir = Path("outputs") / "matplotlib_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_history(args):
    plt = get_plt()
    with open(args.history, "r", encoding="utf-8") as f:
        history = json.load(f)

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
    print("训练曲线图片已保存到:", args.output)


def visualize_weights(args):
    plt = get_plt()
    model, meta = SimpleMLP.load(args.checkpoint)
    image_size = int(meta["image_size"])
    n = min(args.num, model.hidden_dim)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        axes[i].axis("off")
        if i >= n:
            continue
        w = model.W1[:, i].reshape(image_size, image_size, 3)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        axes[i].imshow(w)
        axes[i].set_title("h%d" % i, fontsize=8)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print("第一层权重可视化已保存到:", args.output)


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

    wrong = np.where(result["y_true"] != result["y_pred"])[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "wrong_examples.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for k in wrong[:args.num]:
            true_name = class_names[int(result["y_true"][k])]
            pred_name = class_names[int(result["y_pred"][k])]
            f.write("%s | true=%s | pred=%s\n" % (result["paths"][k], true_name, pred_name))

    if len(wrong) == 0:
        print("测试集中没有找到分类错误的样本。")
        print("记录文件:", txt_path)
        return

    show_num = min(args.num, len(wrong))
    cols = min(4, show_num)
    rows = int(np.ceil(show_num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        axes[i].axis("off")
        if i >= show_num:
            continue
        k = wrong[i]
        img = read_image(result["paths"][k], image_size=int(meta["image_size"]))
        true_name = class_names[int(result["y_true"][k])]
        pred_name = class_names[int(result["y_pred"][k])]
        axes[i].imshow(img)
        axes[i].set_title("T:%s\nP:%s" % (true_name, pred_name), fontsize=8)

    fig.tight_layout()
    fig_path = output_dir / "wrong_examples.png"
    fig.savefig(fig_path, dpi=160)
    print("错例图片已保存到:", fig_path)
    print("错例列表已保存到:", txt_path)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "search", "plot", "weights", "errors"])

    parser.add_argument("--data-dir", default="EuroSAT_RGB")
    parser.add_argument("--output-dir", default="outputs/run1")
    parser.add_argument("--checkpoint", default="outputs/run1/best_model.npz")
    parser.add_argument("--history", default="outputs/run1/history.json")
    parser.add_argument("--output", default="outputs/run1/figure.png")

    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=0)

    parser.add_argument("--split", choices=["train", "val", "test"], default="test")

    parser.add_argument("--hidden-dims", default="64,128")
    parser.add_argument("--lrs", default="0.01,0.005")
    parser.add_argument("--weight-decays", default="0,0.0001")
    parser.add_argument("--activations", default="relu,tanh")

    parser.add_argument("--num", type=int, default=12)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)
    elif args.mode == "search":
        grid_search(args)
    elif args.mode == "plot":
        plot_history(args)
    elif args.mode == "weights":
        visualize_weights(args)
    elif args.mode == "errors":
        save_error_examples(args)


if __name__ == "__main__":
    main()
