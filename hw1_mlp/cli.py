from .runner import grid_search, test_model, train_model
from .visualize import plot_history, save_error_examples, visualize_weights

import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "search", "plot", "weights", "errors"])

    parser.add_argument("--data-dir", default="EuroSAT_RGB")
    parser.add_argument("--output-dir", default="runs/run1")
    parser.add_argument("--checkpoint", default="runs/run1/best_model.npz")
    parser.add_argument("--history", default="runs/run1/history.json")
    parser.add_argument("--output", default="runs/run1/figure.png")

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

