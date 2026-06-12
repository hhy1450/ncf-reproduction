"""Run experiment groups: comparison, cold-start analysis."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from data_loader import MovieLensDataset, LastfmDataset, negative_sampling
from recall.cf import ItemCF
from recall.svd import SVDRecall
from recall.nmf import NMFRecall
from model import GMF, MLP, NeuMF
from rank.wide_deep import WideAndDeep
from train import train as train_fn
from evaluate import evaluate
from cold_start import ColdStartHandler


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(dataset_name="movielens", data_dir="data"):
    if dataset_name == "movielens":
        ds = MovieLensDataset(data_dir)
    elif dataset_name == "lastfm":
        ds = LastfmDataset(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    df = ds.load()
    train, val, test, num_users, num_items = ds.split(df)
    user_items = {uid: set(items) for uid, items in train.items()}
    return train, val, test, user_items, num_users, num_items


def run_comparison_experiment(dataset_name="movielens", epochs=10, device="cpu"):
    """Experiment 1: Traditional vs Deep comparison."""
    print(f"\n{'='*60}")
    print(f"Experiment 1: Model Comparison on {dataset_name}")
    print(f"{'='*60}\n")

    train, val, test, user_items, nu, ni = prepare_data(dataset_name)
    users, items_, labels = negative_sampling(user_items, ni, num_neg=4)

    results = {}

    print("--- Traditional Methods ---")

    # ItemCF
    icf = ItemCF(top_k_neighbors=50)
    icf.fit(train, nu, ni)
    hrs, ndcgs = [], []
    for uid, true_item in test.items():
        recs = icf.recommend(uid, k=10)
        hrs.append(1.0 if true_item in recs else 0.0)
        if true_item in recs:
            rank = list(recs).index(true_item)
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            ndcgs.append(0.0)
    results["ItemCF"] = {"HR@10": np.mean(hrs), "NDCG@10": np.mean(ndcgs)}
    print(f"ItemCF    HR@10={np.mean(hrs):.4f} NDCG@10={np.mean(ndcgs):.4f}")

    # SVD
    svd = SVDRecall(embed_dim=64, epochs=epochs, device=device)
    svd.fit(train, nu, ni)
    hrs, ndcgs = [], []
    for uid, true_item in test.items():
        recs = svd.recommend(uid, k=10)
        hrs.append(1.0 if true_item in recs else 0.0)
        if true_item in recs:
            rank = list(recs).index(true_item)
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            ndcgs.append(0.0)
    results["SVD"] = {"HR@10": np.mean(hrs), "NDCG@10": np.mean(ndcgs)}
    print(f"SVD       HR@10={np.mean(hrs):.4f} NDCG@10={np.mean(ndcgs):.4f}")

    # NMF
    nmf = NMFRecall(embed_dim=64, epochs=epochs, device=device)
    nmf.fit(train, nu, ni)
    hrs, ndcgs = [], []
    for uid, true_item in test.items():
        recs = nmf.recommend(uid, k=10)
        hrs.append(1.0 if true_item in recs else 0.0)
        if true_item in recs:
            rank = list(recs).index(true_item)
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            ndcgs.append(0.0)
    results["NMF"] = {"HR@10": np.mean(hrs), "NDCG@10": np.mean(ndcgs)}
    print(f"NMF       HR@10={np.mean(hrs):.4f} NDCG@10={np.mean(ndcgs):.4f}")

    print("\n--- Deep Methods ---")

    # GMF
    gmf = GMF(nu, ni, embed_dim=8)
    gmf_hr, gmf_ndcg, _ = train_fn(gmf, users, items_, labels, val, test, user_items, ni,
                                  epochs=epochs, batch_size=256, lr=0.001,
                                  device=device, eval_k=10, early_stop_patience=5)
    results["GMF"] = {"HR@10": gmf_hr, "NDCG@10": gmf_ndcg}

    # MLP
    mlp = MLP(nu, ni, embed_dim=8, layers=[32, 16, 8])
    mlp_hr, mlp_ndcg, _ = train_fn(mlp, users, items_, labels, val, test, user_items, ni,
                                  epochs=epochs, batch_size=256, lr=0.001,
                                  device=device, eval_k=10, early_stop_patience=5)
    results["MLP"] = {"HR@10": mlp_hr, "NDCG@10": mlp_ndcg}

    # NeuMF
    neumf = NeuMF(nu, ni, gmf_dim=8, mlp_dim=8, mlp_layers=[32, 16, 8])
    neumf_hr, neumf_ndcg, _ = train_fn(neumf, users, items_, labels, val, test, user_items, ni,
                                      epochs=epochs, batch_size=256, lr=0.001,
                                      device=device, eval_k=10, early_stop_patience=5)
    results["NeuMF"] = {"HR@10": neumf_hr, "NDCG@10": neumf_ndcg}

    # Wide&Deep
    wd = WideAndDeep(nu, ni, embed_dim=16, mlp_layers=[64, 32])
    wd_hr, wd_ndcg, _ = train_fn(wd, users, items_, labels, val, test, user_items, ni,
                                epochs=epochs, batch_size=256, lr=0.001,
                                device=device, eval_k=10, early_stop_patience=5)
    results["Wide&Deep"] = {"HR@10": wd_hr, "NDCG@10": wd_ndcg}

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Model':<15} {'HR@10':<10} {'NDCG@10':<10}")
    print("-" * 35)
    for model, metrics in results.items():
        print(f"{model:<15} {metrics['HR@10']:<10.4f} {metrics['NDCG@10']:<10.4f}")

    return results


def run_cold_start_experiment(dataset_name="movielens", device="cpu"):
    """Experiment 2: Cold-start analysis by user group."""
    print(f"\n{'='*60}")
    print(f"Experiment 2: Cold-Start Analysis on {dataset_name}")
    print(f"{'='*60}\n")

    train, val, test, user_items, nu, ni = prepare_data(dataset_name)
    handler = ColdStartHandler(ni)
    handler.compute_popularity(train)

    sorted_pop = sorted(handler.item_popularity.items(), key=lambda x: -x[1])
    top_k_items = [iid for iid, _ in sorted_pop[:200]]

    def popular_recommend(uid, k):
        return top_k_items[:k]

    results = handler.evaluate_groups(test, train, popular_recommend)

    print(f"{'Group':<15} {'Users':<8} {'HR@10':<10} {'NDCG@10':<10}")
    print("-" * 43)
    for group, metrics in results.items():
        print(f"{group:<15} {metrics['count']:<8} {metrics['HR@10']:<10.4f} {metrics['NDCG@10']:<10.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recommendation experiments")
    parser.add_argument("--dataset", type=str, default="movielens",
                        choices=["movielens", "lastfm"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["comparison", "cold_start", "all"])
    args = parser.parse_args()

    set_seed(42)

    if args.experiment in ("comparison", "all"):
        run_comparison_experiment(args.dataset, args.epochs, args.device)

    if args.experiment in ("cold_start", "all"):
        run_cold_start_experiment(args.dataset, args.device)
