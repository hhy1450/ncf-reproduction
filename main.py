"""
NCF (Neural Collaborative Filtering) Reproduction
Paper: He et al., "Neural Collaborative Filtering", WWW 2017
Dataset: MovieLens-1M

Usage:
    python main.py --model GMF        # Train GMF
    python main.py --model MLP        # Train MLP
    python main.py --model NeuMF      # Train NeuMF (random init)
    python main.py --model NeuMF --pretrain  # NeuMF with pre-training
"""

import argparse
import torch
import numpy as np

from data_loader import load_data, leave_one_out_split, negative_sampling
from model import GMF, MLP, NeuMF
from evaluate import evaluate
from train import train


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data():
    """Load and preprocess MovieLens-1M data."""
    print("\nLoading MovieLens-1M...")
    df = load_data()
    train, val, test, num_users, num_items = leave_one_out_split(df)

    # Build train user_items for negative sampling exclusion
    train_user_items = {uid: set(items) for uid, items in train.items()}

    # Generate training samples with negative sampling
    print("Generating negative samples for training...")
    train_users, train_items, train_labels = negative_sampling(train, num_items, num_neg=4)

    return train_users, train_items, train_labels, train_user_items, val, test, num_users, num_items


def main():
    parser = argparse.ArgumentParser(description="NCF Reproduction")
    parser.add_argument("--model", type=str, default="GMF",
                        choices=["GMF", "MLP", "NeuMF"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gmf_dim", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=8)
    parser.add_argument("--mlp_layers", type=int, nargs="+", default=[32, 16, 8])
    parser.add_argument("--pretrain", action="store_true",
                        help="Pre-train GMF and MLP before NeuMF")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Fusion weight for pre-trained embeddings")
    parser.add_argument("--eval_k", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Device: {args.device}")

    # Load data once
    data = prepare_data()
    train_users, train_items, train_labels, train_user_items, val, test, num_users, num_items = data

    # ---- GMF ----
    if args.model == "GMF":
        model = GMF(num_users, num_items, embed_dim=args.gmf_dim)
        train(
            model, train_users, train_items, train_labels,
            val, test, train_user_items, num_items,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=args.device, eval_k=args.eval_k,
            early_stop_patience=args.patience,
        )

    # ---- MLP ----
    elif args.model == "MLP":
        model = MLP(num_users, num_items, embed_dim=args.mlp_dim,
                     layers=args.mlp_layers)
        train(
            model, train_users, train_items, train_labels,
            val, test, train_user_items, num_items,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=args.device, eval_k=args.eval_k,
            early_stop_patience=args.patience,
        )

    # ---- NeuMF ----
    elif args.model == "NeuMF":
        if args.pretrain:
            # Step 1: Pre-train GMF
            print("\n" + "=" * 60)
            print("Pre-training GMF...")
            print("=" * 60)
            gmf = GMF(num_users, num_items, embed_dim=args.gmf_dim)
            gmf_hr, _, _ = train(
                gmf, train_users, train_items, train_labels,
                val, test, train_user_items, num_items,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                device=args.device, eval_k=args.eval_k,
                early_stop_patience=args.patience,
            )

            # Step 2: Pre-train MLP
            print("\n" + "=" * 60)
            print("Pre-training MLP...")
            print("=" * 60)
            mlp = MLP(num_users, num_items, embed_dim=args.mlp_dim,
                       layers=args.mlp_layers)
            mlp_hr, _, _ = train(
                mlp, train_users, train_items, train_labels,
                val, test, train_user_items, num_items,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                device=args.device, eval_k=args.eval_k,
                early_stop_patience=args.patience,
            )

            # Step 3: NeuMF with pre-trained embeddings
            print("\n" + "=" * 60)
            print("Training NeuMF with pre-trained GMF + MLP...")
            print("=" * 60)
            neumf = NeuMF(num_users, num_items,
                          gmf_dim=args.gmf_dim, mlp_dim=args.mlp_dim,
                          mlp_layers=args.mlp_layers)
            neumf.load_pretrained(gmf, mlp, alpha=args.alpha)
            train(
                neumf, train_users, train_items, train_labels,
                val, test, train_user_items, num_items,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                device=args.device, eval_k=args.eval_k,
                early_stop_patience=args.patience,
            )
        else:
            # NeuMF from random initialization
            neumf = NeuMF(num_users, num_items,
                          gmf_dim=args.gmf_dim, mlp_dim=args.mlp_dim,
                          mlp_layers=args.mlp_layers)
            train(
                neumf, train_users, train_items, train_labels,
                val, test, train_user_items, num_items,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                device=args.device, eval_k=args.eval_k,
                early_stop_patience=args.patience,
            )


if __name__ == "__main__":
    main()
