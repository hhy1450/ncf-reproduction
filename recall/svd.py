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
        return dot + bias


class SVDRecall(RecallBase):
    """SVD-based recall using PyTorch."""

    def __init__(self, embed_dim=64, epochs=20, lr=0.005, batch_size=512,
                 reg=0.02, device="cpu"):
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

        users, items = [], []
        for uid, iids in train.items():
            for iid in iids:
                users.append(uid)
                items.append(iid)
        users = np.array(users, dtype=np.int64)
        items = np.array(items, dtype=np.int64)
        n = len(users)

        self.model.train()
        for epoch in range(self.epochs):
            indices = np.random.permutation(n)
            total_loss = 0
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]
                u = torch.tensor(users[idx], device=self.device)
                i = torch.tensor(items[idx], device=self.device)

                pos_scores = self.model(u, i)
                pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()

                neg_items = torch.randint(0, num_items, (len(idx),), device=self.device)
                neg_scores = self.model(u, neg_items)
                neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-8).mean()

                loss = pos_loss + neg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  SVD Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

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
            scores = (u_vec @ self.item_vectors.T).ravel()
        top_k = min(k, self.num_items)
        top_items = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_items = top_items[np.argsort(-scores[top_items])]
        return top_items
