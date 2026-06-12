from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

from recall.base import RecallBase


class NMFModel(nn.Module):
    """Non-negative Matrix Factorization using exponential activations."""

    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.user_embed.weight, 0, 0.1)
        nn.init.uniform_(self.item_embed.weight, 0, 0.1)

    def forward(self, user, item):
        u = torch.exp(self.user_embed(user))
        i = torch.exp(self.item_embed(item))
        return (u * i).sum(dim=-1)


class NMFRecall(RecallBase):
    """NMF-based recall using PyTorch with non-negative constraint."""

    def __init__(self, embed_dim=64, epochs=20, lr=0.005, batch_size=512, device="cpu"):
        super().__init__(name="NMF")
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def fit(self, train: Dict[int, List[int]], num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items
        self.model = NMFModel(num_users, num_items, self.embed_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        users, items = [], []
        for uid, iids in train.items():
            for iid in iids:
                users.append(uid)
                items.append(iid)
        users = np.array(users, dtype=np.int64)
        items = np.array(items, dtype=np.int64)
        n = len(users)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            indices = np.random.permutation(n)
            total_loss = 0
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]
                u = torch.tensor(users[idx], device=self.device)
                i = torch.tensor(items[idx], device=self.device)
                scores = self.model(u, i)
                loss = criterion(scores, torch.ones_like(scores))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"  NMF Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

    def recommend(self, user_id: int, k: int = 200) -> np.ndarray:
        if self.model is None or user_id >= self.num_users:
            return np.array([], dtype=np.int64)
        self.model.eval()
        with torch.no_grad():
            u_tensor = torch.tensor([user_id], device=self.device)
            u_vec = torch.exp(self.model.user_embed(u_tensor)).cpu().numpy()
            all_items = torch.arange(self.num_items, device=self.device)
            i_vecs = torch.exp(self.model.item_embed(all_items)).cpu().numpy()
            scores = (u_vec @ i_vecs.T).ravel()
        top_k = min(k, self.num_items)
        top_items = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_items = top_items[np.argsort(-scores[top_items])]
        return top_items
