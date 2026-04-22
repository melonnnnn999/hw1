from pathlib import Path

import numpy as np
from PIL import Image


def scan_dataset(data_dir):
    """Scan the EuroSAT_RGB directory and return paths, labels, and class names."""
    data_dir = Path(data_dir)
    class_dirs = sorted([path for path in data_dir.iterdir() if path.is_dir()])
    if not class_dirs:
        raise ValueError("No class folders found in data directory: %s" % data_dir)

    image_paths = []
    labels = []
    class_names = [path.name for path in class_dirs]

    for label, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.glob("*.jpg")):
            image_paths.append(str(image_path))
            labels.append(label)

    return image_paths, np.array(labels, dtype=np.int64), class_names


def split_dataset(
    image_paths,
    labels,
    num_classes,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    max_per_class=0,
):
    """Create stratified train/val/test splits."""
    rng = np.random.default_rng(seed)
    splits = {
        "train_paths": [],
        "train_labels": [],
        "val_paths": [],
        "val_labels": [],
        "test_paths": [],
        "test_labels": [],
    }

    for class_index in range(num_classes):
        indices = np.where(labels == class_index)[0]
        rng.shuffle(indices)

        if max_per_class and max_per_class > 0:
            indices = indices[:max_per_class]

        total = len(indices)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        if total >= 3:
            train_count = max(1, train_count)
            val_count = max(1, val_count)
            if train_count + val_count >= total:
                train_count = max(1, total - 2)
                val_count = 1

        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count + val_count]
        test_indices = indices[train_count + val_count:]

        for idx in train_indices:
            splits["train_paths"].append(image_paths[idx])
            splits["train_labels"].append(labels[idx])
        for idx in val_indices:
            splits["val_paths"].append(image_paths[idx])
            splits["val_labels"].append(labels[idx])
        for idx in test_indices:
            splits["test_paths"].append(image_paths[idx])
            splits["test_labels"].append(labels[idx])

    for split_name in ["train", "val", "test"]:
        paths = splits[split_name + "_paths"]
        split_labels = np.array(splits[split_name + "_labels"], dtype=np.int64)
        order = rng.permutation(len(paths))
        splits[split_name + "_paths"] = [paths[i] for i in order]
        splits[split_name + "_labels"] = split_labels[order]

    return splits


def read_image(path, image_size=64):
    """Read an RGB image, resize if needed, and normalize to [0, 1]."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:
                resample = Image.BILINEAR
            image = image.resize((image_size, image_size), resample)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def compute_mean_std(train_paths, image_size=64):
    """Compute per-channel mean and std on the training split only."""
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in train_paths:
        image = read_image(path, image_size=image_size)
        channel_sum += image.sum(axis=(0, 1))
        channel_square_sum += (image ** 2).sum(axis=(0, 1))
        pixel_count += image.shape[0] * image.shape[1]

    mean = channel_sum / pixel_count
    variance = channel_square_sum / pixel_count - mean ** 2
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def make_batches(paths, labels, batch_size, image_size, mean, std, shuffle=True, seed=0):
    """Yield normalized mini-batches without loading the full dataset into memory."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(paths))
    if shuffle:
        rng.shuffle(indices)

    mean = mean.reshape(1, 1, 3)
    std = std.reshape(1, 1, 3)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_paths = [paths[i] for i in batch_indices]
        targets = labels[batch_indices]
        features = np.zeros((len(batch_indices), image_size * image_size * 3), dtype=np.float32)

        for row, path in enumerate(batch_paths):
            image = read_image(path, image_size=image_size)
            image = (image - mean) / std
            features[row] = image.reshape(-1)

        yield features, targets, batch_paths

