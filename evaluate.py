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
