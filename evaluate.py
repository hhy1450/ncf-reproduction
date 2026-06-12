import torch
import numpy as np


def hit_ratio_at_k(pred_items, true_item, k=10):
    """Hit Ratio @ K: 1 if true_item in top-K predictions, else 0."""
    return int(true_item in pred_items[:k])


def ndcg_at_k(pred_items, true_item, k=10):
    """NDCG @ K: DCG / IDCG for a single true item (ideal DCG=1 at position 0)."""
    if true_item in pred_items[:k]:
        rank = list(pred_items[:k]).index(true_item)
        return 1.0 / np.log2(rank + 2)  # position 0 → log2(2)=1
    return 0.0


def evaluate(model, test_dict, user_items_train, num_items, num_neg=99, k=10):
    """Evaluate HR@K and NDCG@K on all test users.

    For each test user, rank the test item against num_neg random negatives.

    Args:
        model: PyTorch model (eval mode)
        test_dict: {user_id: test_item_id}
        user_items_train: {user_id: set of train items} (for excluding negatives)
        num_items: total item count
        num_neg: negative samples per evaluation (default 99 → rank among 100)
        k: top-K cutoff

    Returns:
        hr, ndcg (averaged over all test users)
    """
    model.eval()
    hrs, ndcgs = [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for uid, true_item in test_dict.items():
            # Candidate list: test item + num_neg random negatives
            negatives = set()
            train_items = user_items_train.get(uid, set())
            if true_item in train_items:
                train_items = train_items - {true_item}
            while len(negatives) < num_neg:
                neg = np.random.randint(0, num_items)
                if neg != true_item and neg not in train_items and neg not in negatives:
                    negatives.add(neg)
            candidates = [true_item] + list(negatives)

            user_tensor = torch.tensor([uid] * len(candidates), device=device)
            item_tensor = torch.tensor(candidates, device=device)

            scores = model(user_tensor, item_tensor).cpu().numpy()
            # Sort by score descending
            ranked_indices = np.argsort(-scores)
            ranked_items = np.array(candidates)[ranked_indices]

            hrs.append(hit_ratio_at_k(ranked_items, true_item, k))
            ndcgs.append(ndcg_at_k(ranked_items, true_item, k))

    model.train()
    return np.mean(hrs), np.mean(ndcgs)


def map_at_k(pred_items, true_item, k=10):
    """Mean Average Precision @ K (AP for single user)."""
    if true_item in pred_items[:k]:
        rank = list(pred_items[:k]).index(true_item)
        return 1.0 / (rank + 1)
    return 0.0


def diversity_at_k(pred_items, item_similarity_matrix, k=10):
    """Diversity @ K: 1 - average pairwise similarity of top-K items. Higher = more diverse."""
    items = pred_items[:k]
    if len(items) < 2:
        return 1.0
    total_sim = 0.0
    count = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] < item_similarity_matrix.shape[0] and items[j] < item_similarity_matrix.shape[1]:
                total_sim += item_similarity_matrix[items[i], items[j]]
                count += 1
    return 1.0 - (total_sim / count if count > 0 else 0)


def coverage_at_k(all_recommendations, num_items):
    """Item coverage: fraction of unique items across all users."""
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)
    return len(unique_items) / num_items


def novelty_at_k(pred_items, item_popularity, k=10):
    """Novelty @ K: -avg log popularity. Higher = more novel."""
    items = pred_items[:k]
    if not items:
        return 0.0
    total_users = max(item_popularity.values()) if item_popularity else 1
    novelties = [-np.log(p / total_users + 1e-8) for p in [item_popularity.get(i, 1) for i in items]]
    return np.mean(novelties)


def compute_item_popularity(train):
    """Compute {item_id: interaction_count} from training data."""
    pop = {}
    for uid, items in train.items():
        for iid in items:
            pop[iid] = pop.get(iid, 0) + 1
    return pop


def compute_item_similarity(item_embeddings):
    """Compute pairwise item cosine similarity matrix from (num_items, dim) embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(item_embeddings)


def full_evaluate(model, test_dict, user_items_train, num_items, num_neg=99, k=10):
    """Extended evaluation: HR, NDCG, MAP."""
    model.eval()
    hrs, ndcgs, maps, all_recs = [], [], [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for uid, true_item in test_dict.items():
            negatives = set()
            train_items = user_items_train.get(uid, set())
            if true_item in train_items:
                train_items = train_items - {true_item}
            while len(negatives) < num_neg:
                neg = np.random.randint(0, num_items)
                if neg != true_item and neg not in train_items and neg not in negatives:
                    negatives.add(neg)
            candidates = [true_item] + list(negatives)

            user_tensor = torch.tensor([uid] * len(candidates), device=device)
            item_tensor = torch.tensor(candidates, device=device)
            scores = model(user_tensor, item_tensor).cpu().numpy()
            ranked_indices = np.argsort(-scores)
            ranked_items = np.array(candidates)[ranked_indices]

            hrs.append(hit_ratio_at_k(ranked_items, true_item, k))
            ndcgs.append(ndcg_at_k(ranked_items, true_item, k))
            maps.append(map_at_k(ranked_items, true_item, k))
            all_recs.append(ranked_items[:k].tolist())

    model.train()
    return {
        f"HR@{k}": np.mean(hrs),
        f"NDCG@{k}": np.mean(ndcgs),
        f"MAP@{k}": np.mean(maps),
    }
