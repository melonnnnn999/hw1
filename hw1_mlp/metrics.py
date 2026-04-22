import numpy as np

from .data import make_batches


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def print_confusion_matrix(matrix, class_names):
    print("\nClass indices:")
    for index, name in enumerate(class_names):
        print("%2d -> %s" % (index, name))

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "true\\pred " + " ".join(["%5d" % i for i in range(len(class_names))])
    print(header)
    for index in range(len(class_names)):
        row = "    %2d    " % index + " ".join(["%5d" % value for value in matrix[index]])
        print(row)


def evaluate(
    model,
    paths,
    labels,
    batch_size,
    image_size,
    mean,
    std,
    num_classes,
    seed=0,
    return_details=False,
):
    all_true = []
    all_pred = []
    all_paths = []
    total_loss = 0.0
    total_num = 0

    batches = make_batches(
        paths,
        labels,
        batch_size,
        image_size,
        mean,
        std,
        shuffle=False,
        seed=seed,
    )

    for features, targets, batch_paths in batches:
        loss = model.loss(features, targets, weight_decay=0.0)
        pred = model.predict(features)
        total_loss += loss * len(targets)
        total_num += len(targets)
        all_true.append(targets)
        all_pred.append(pred)
        if return_details:
            all_paths.extend(batch_paths)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    matrix = confusion_matrix(y_true, y_pred, num_classes)

    result = {
        "loss": total_loss / total_num,
        "accuracy": accuracy(y_true, y_pred),
        "confusion_matrix": matrix,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if return_details:
        result["paths"] = all_paths
    return result

