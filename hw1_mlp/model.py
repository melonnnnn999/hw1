import json
from pathlib import Path

import numpy as np


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
        self.W2 = rng.normal(0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, num_classes)).astype(
            np.float32
        )
        self.b2 = np.zeros(num_classes, dtype=np.float32)

    def activate(self, values):
        if self.activation == "relu":
            return np.maximum(0, values)
        if self.activation == "sigmoid":
            values = np.clip(values, -50, 50)
            return 1.0 / (1.0 + np.exp(-values))
        if self.activation == "tanh":
            return np.tanh(values)
        raise ValueError("Unsupported activation: %s" % self.activation)

    def activation_grad(self, z1, a1):
        if self.activation == "relu":
            return (z1 > 0).astype(np.float32)
        if self.activation == "sigmoid":
            return a1 * (1.0 - a1)
        if self.activation == "tanh":
            return 1.0 - a1 ** 2
        raise ValueError("Unsupported activation: %s" % self.activation)

    def forward(self, features):
        z1 = features @ self.W1 + self.b1
        a1 = self.activate(z1)
        scores = a1 @ self.W2 + self.b2
        cache = (features, z1, a1)
        return scores, cache

    def softmax_loss(self, scores, targets):
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        batch_size = targets.shape[0]
        loss = -np.mean(np.log(probs[np.arange(batch_size), targets] + 1e-12))
        return loss, probs

    def loss(self, features, targets, weight_decay=0.0):
        scores, _ = self.forward(features)
        data_loss, _ = self.softmax_loss(scores, targets)
        reg_loss = 0.5 * weight_decay * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return data_loss + reg_loss

    def loss_and_grad(self, features, targets, weight_decay=0.0):
        scores, cache = self.forward(features)
        features, z1, a1 = cache
        loss, probs = self.softmax_loss(scores, targets)
        loss += 0.5 * weight_decay * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))

        batch_size = features.shape[0]
        dscores = probs.copy()
        dscores[np.arange(batch_size), targets] -= 1.0
        dscores /= batch_size

        dW2 = a1.T @ dscores + weight_decay * self.W2
        db2 = np.sum(dscores, axis=0)

        da1 = dscores @ self.W2.T
        dz1 = da1 * self.activation_grad(z1, a1)

        dW1 = features.T @ dz1 + weight_decay * self.W1
        db1 = np.sum(dz1, axis=0)

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
        return loss, grads, scores

    def predict(self, features):
        scores, _ = self.forward(features)
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

