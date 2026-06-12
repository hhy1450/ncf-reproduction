from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

from recall.base import RecallBase


class SVDModel(nn.Module):
    """SVD matrix factorization: user_embed · item_embed^T + biases."""

    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        u = self.user_embed(user)
        i = self.item_embed(item)
        dot = (u * i).sum(dim=-1)
        bias = self.user_bias(user).squeeze(-1) + self.item_bias(item).squeeze(-1) + self.global_bias
        return torch.sigmoid(dot + bias)


class SVDRecall(RecallBase):
    """SVD-based recall using PyTorch with BCE loss and negative sampling."""

    def __init__(self, embed_dim=64, epochs=20, lr=0.001, batch_size=512,
                 reg=0.0, device="cpu"):
        super().__init__(name="SVD")
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.reg = reg
        self.device = device
        self.model = None
        self.item_vectors = None

    def fit(self, train: Dict[int, List[int]], num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items
        self.model = SVDModel(num_users, num_items, self.embed_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                      weight_decay=self.reg)
        criterion = nn.BCELoss()

        # Build positive pairs + negative sampling (4 negs per positive)
        user_items_set = {uid: set(items) for uid, items in train.items()}
        users_list, items_list, labels_list = [], [], []
        for uid, pos_items in train.items():
            for pos in pos_items:
                users_list.append(uid)
                items_list.append(pos)
                labels_list.append(1.0)
                for _ in range(4):
                    neg = np.random.randint(0, num_items)
                    while neg in user_items_set[uid]:
                        neg = np.random.randint(0, num_items)
                    users_list.append(uid)
                    items_list.append(neg)
                    labels_list.append(0.0)

        users_arr = np.array(users_list, dtype=np.int64)
        items_arr = np.array(items_list, dtype=np.int64)
        labels_arr = np.array(labels_list, dtype=np.float32)
        n = len(users_arr)

        self.model.train()
        for epoch in range(self.epochs):
            indices = np.random.permutation(n)
            total_loss, batches = 0, 0
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]
                u = torch.tensor(users_arr[idx], device=self.device)
                i = torch.tensor(items_arr[idx], device=self.device)
                y = torch.tensor(labels_arr[idx], device=self.device)

                preds = self.model(u, i)
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1

            if (epoch + 1) % 5 == 0:
                print(f"  SVD Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/batches:.4f}")

        self.model.eval()
        with torch.no_grad():
            all_items = torch.arange(num_items, device=self.device)
            self.item_vectors = self.model.item_embed(all_items).cpu().numpy()

    def recommend(self, user_id: int, k: int = 200) -> np.ndarray:
        if self.model is None or user_id >= self.num_users:
            return np.array([], dtype=np.int64)
        self.model.eval()
        with torch.no_grad():
            u_tensor = torch.tensor([user_id], device=self.device)
            u_vec = self.model.user_embed(u_tensor).cpu().numpy()
            u_bias = self.model.user_bias(u_tensor).cpu().item()
            all_items = torch.arange(self.num_items, device=self.device)
            i_vecs = self.model.item_embed(all_items).cpu().numpy()
            i_biases = self.model.item_bias(all_items).cpu().numpy().ravel()
            scores = (u_vec @ i_vecs.T).ravel() + u_bias + i_biases + self.model.global_bias.cpu().item()
            scores = 1.0 / (1.0 + np.exp(-scores))  # sigmoid
        top_k = min(k, self.num_items)
        top_items = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_items = top_items[np.argsort(-scores[top_items])]
        return top_items
