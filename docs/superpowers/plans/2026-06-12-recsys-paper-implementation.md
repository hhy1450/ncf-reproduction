# 个性化推荐系统 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ncf_reproduction 基础上构建完整推荐系统（召回+排序+冷启动+API+UI）并撰写小论文

**Architecture:** 四层架构 — 数据层（多数据集统一加载）→ 召回层（CF/SVD/NMF 多路召回）→ 排序层（NCF/Wide&Deep 精排）→ 服务层（FastAPI + Gradio），每层独立可测试

**Tech Stack:** PyTorch, Pandas, NumPy, FastAPI, Gradio, scikit-learn (SVD reference)

---

## File Structure

```
ncf_reproduction/
├── data_loader.py           # 扩展: BaseDataset + Last.fm/MIND + 困难负采样
├── model.py                 # 不变: NCF (GMF/MLP/NeuMF) — 通过 rank/__init__.py 重导出
├── train.py                 # 不变: 训练循环
├── evaluate.py              # 扩展: +MAP, Diversity, Coverage, Novelty
├── pipeline.py              # 新增: 两阶段推荐管道
├── cold_start.py            # 新增: 冷启动策略
├── main.py                  # 重构: 统一 CLI 入口
├── recall/
│   ├── __init__.py          # 重导出
│   ├── base.py              # RecallBase 抽象接口
│   ├── cf.py                # UserCF, ItemCF
│   ├── svd.py               # SVD 矩阵分解 (PyTorch)
│   └── nmf.py               # NMF 非负矩阵分解 (PyTorch)
├── rank/
│   ├── __init__.py          # 重导出: model.py 的类 + wide_deep
│   ├── base.py              # BaseRanker 抽象接口
│   └── wide_deep.py         # Wide&Deep 模型
├── api/
│   ├── __init__.py
│   ├── server.py            # FastAPI 推荐服务
│   └── app.py               # Gradio 可视化界面
├── paper/
│   └── 小论文.md             # 论文草稿
├── experiments/
│   └── run_experiments.py   # 三组实验脚本
└── requirements.txt         # 扩展: fastapi, gradio, uvicorn
```

---

## Phase 1: 数据层

### Task 1.1: 重构 data_loader.py — 添加 BaseDataset 基类

**Files:**
- Modify: `data_loader.py`

- [ ] **Step 1: 在文件顶部添加 BaseDataset 抽象基类**

在现有 `data_loader.py` 顶部（所有函数之前）插入：

```python
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Set, Optional
import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile


class BaseDataset(ABC):
    """Unified dataset interface for recommendation systems.

    Subclasses implement download() and load_raw().
    Base provides split, negative sampling, and cold-start utilities.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    @abstractmethod
    def download(self):
        """Download raw data to self.data_dir."""
        ...

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Load raw data → DataFrame with columns [user_id, item_id, timestamp]."""
        ...

    def load(self) -> pd.DataFrame:
        """Download if needed, then load."""
        self.download()
        return self.load_raw()

    def split(self, df: pd.DataFrame = None) -> Tuple[
        Dict[int, list], Dict[int, int], Dict[int, int], int, int
    ]:
        """Leave-one-out split by timestamp. Returns train, val, test, num_users, num_items."""
        if df is None:
            df = self.load()
        return leave_one_out_split(df)

    def build_user_items(self, train: Dict[int, list]) -> Dict[int, Set[int]]:
        """Build user → set(items) dict for negative sampling exclusion."""
        return {uid: set(items) for uid, items in train.items()}

    def neg_sample(
        self, train: Dict[int, list], num_items: int, num_neg: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random negative sampling. Returns users, items, labels arrays."""
        user_items = self.build_user_items(train)
        return negative_sampling(user_items, num_items, num_neg)

    def hard_neg_sample(
        self,
        model,
        train: Dict[int, list],
        num_items: int,
        num_neg: int = 4,
        sample_pool: int = 100,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hard negative sampling: pick negatives with highest model scores."""
        return hard_negative_sampling(
            model, train, num_items, num_neg, sample_pool, device
        )

    def cold_start_split(
        self, train: Dict[int, list], threshold: int = 5
    ) -> Tuple[Dict[int, list], Dict[int, list]]:
        """Split users into cold (<threshold interactions) and warm (>=threshold)."""
        cold, warm = {}, {}
        for uid, items in train.items():
            if len(items) < threshold:
                cold[uid] = items
            else:
                warm[uid] = items
        return cold, warm
```

- [ ] **Step 2: 保留现有函数不变** — `download_and_extract()`, `load_data()`, `build_interactions()`, `leave_one_out_split()`, `negative_sampling()`, `batch_generator()` 全部保留

- [ ] **Step 3: 确认现有 MovieLensDataset 子类**

在文件末尾（`batch_generator` 之后）添加：

```python
class MovieLensDataset(BaseDataset):
    """MovieLens-1M dataset wrapper."""

    DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def download(self):
        zip_path = os.path.join(self.data_dir, "ml-1m.zip")
        extract_dir = os.path.join(self.data_dir, "ml-1m")
        if os.path.exists(os.path.join(extract_dir, "ratings.dat")):
            return
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(zip_path):
            print("Downloading MovieLens-1M...")
            urllib.request.urlretrieve(self.DATA_URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(self.data_dir)

    def load_raw(self) -> pd.DataFrame:
        ratings_path = os.path.join(self.data_dir, "ml-1m", "ratings.dat")
        df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        return df[["user_id", "item_id", "timestamp"]]
```

- [ ] **Step 4: 验证现有代码仍可运行**

```bash
cd /c/Users/黄海亦/Desktop/ncf_reproduction
python main.py --model GMF --epochs 1
```
Expected: 正常运行，无 import 错误

- [ ] **Step 5: Commit**

```bash
git add data_loader.py
git commit -m "feat: add BaseDataset abstract class and MovieLensDataset wrapper"
```

---

### Task 1.2: 添加 Last.fm 数据集支持

**Files:**
- Modify: `data_loader.py` (追加)
- Create: (无，数据自动下载到 data/lastfm/)

- [ ] **Step 1: 添加 LastfmDataset 类**

在 `data_loader.py` 末尾追加：

```python
class LastfmDataset(BaseDataset):
    """Last.fm music listening dataset.

    Uses the HetRec 2011 Last.fm dataset (user_artists.dat).
    """

    DATA_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"

    def download(self):
        zip_path = os.path.join(self.data_dir, "lastfm-2k.zip")
        extract_dir = os.path.join(self.data_dir, "lastfm-2k")
        if os.path.exists(os.path.join(extract_dir, "user_artists.dat")):
            return
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(zip_path):
            print("Downloading Last.fm dataset...")
            urllib.request.urlretrieve(self.DATA_URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(self.data_dir)

    def load_raw(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "lastfm-2k", "user_artists.dat")
        df = pd.read_csv(path, sep="\t")
        df = df.rename(columns={"userID": "user_id", "artistID": "item_id", "weight": "play_count"})
        # Use play_count as implicit weight; create synthetic timestamp from row order
        df = df.sort_values(["user_id", "play_count"], ascending=[True, False])
        df["timestamp"] = df.groupby("user_id").cumcount()
        return df[["user_id", "item_id", "timestamp"]]
```

- [ ] **Step 2: 验证 Last.fm 加载**

```bash
python -c "
from data_loader import LastfmDataset
ds = LastfmDataset()
df = ds.load()
print(f'Last.fm users: {df.user_id.nunique()}, items: {df.item_id.nunique()}, interactions: {len(df)}')
train, val, test, nu, ni = ds.split(df)
print(f'Train: {sum(len(v) for v in train.values())}, Val: {len(val)}, Test: {len(test)}')
"
```
Expected: 用户和交互数正确输出

- [ ] **Step 3: Commit**

```bash
git add data_loader.py
git commit -m "feat: add LastfmDataset for music recommendation"
```

---

### Task 1.3: 添加困难负采样

**Files:**
- Modify: `data_loader.py` (追加)

- [ ] **Step 1: 添加 hard_negative_sampling 函数**

在 `batch_generator` 函数之后追加：

```python
def hard_negative_sampling(model, user_items, num_items, num_neg=4,
                           sample_pool=100, device="cpu"):
    """Hard negative sampling: for each positive, sample `sample_pool` random
    negatives, score them with the model, and keep the top `num_neg` highest-scoring.

    This forces the model to learn harder decision boundaries.
    """
    import torch

    model.eval()
    model = model.to(device)
    users_list, items_list, labels_list = [], [], []

    with torch.no_grad():
        for uid, pos_items in user_items.items():
            for pos in pos_items:
                # Positive sample
                users_list.append(uid)
                items_list.append(pos)
                labels_list.append(1)

                # Candidate negatives (exclude known positives)
                cands = set()
                while len(cands) < sample_pool:
                    neg = np.random.randint(0, num_items)
                    if neg not in user_items[uid] and neg not in cands:
                        cands.add(neg)
                cands = list(cands)

                # Score candidates
                u_tensor = torch.tensor([uid] * len(cands), device=device)
                i_tensor = torch.tensor(cands, device=device)
                scores = model(u_tensor, i_tensor)

                # Keep top num_neg hardest (highest scoring = hardest negative)
                _, top_idx = torch.topk(scores, min(num_neg, len(cands)))
                for idx in top_idx:
                    users_list.append(uid)
                    items_list.append(cands[idx.item()])
                    labels_list.append(0)

    model.train()
    return (
        np.array(users_list, dtype=np.int64),
        np.array(items_list, dtype=np.int64),
        np.array(labels_list, dtype=np.float32),
    )
```

- [ ] **Step 2: 验证困难负采样生成**

```bash
python -c "
from data_loader import MovieLensDataset, hard_negative_sampling
from model import GMF
ds = MovieLensDataset()
df = ds.load()
train, val, test, nu, ni = ds.split(df)
model = GMF(nu, ni, embed_dim=8)
users, items, labels = hard_negative_sampling(model, {k: set(v) for k,v in train.items()}, ni, num_neg=2, sample_pool=20)
print(f'Generated {len(users)} samples, pos ratio: {labels.mean():.2%}')
"
```
Expected: 生成了样本，正负比约为 1:2

- [ ] **Step 3: Commit**

```bash
git add data_loader.py
git commit -m "feat: add hard negative sampling for improved model training"
```

---

## Phase 2: 召回层

### Task 2.1: 创建 recall/base.py — 召回基类

**Files:**
- Create: `recall/__init__.py`
- Create: `recall/base.py`

- [ ] **Step 1: 创建 recall/__init__.py**

```python
from recall.cf import UserCF, ItemCF
from recall.svd import SVDRecall
from recall.nmf import NMFRecall
from recall.base import RecallBase
```

- [ ] **Step 2: 创建 recall/base.py**

```python
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class RecallBase(ABC):
    """Base class for recall methods.

    Each recall method implements fit() and recommend().
    """

    def __init__(self, name: str = "RecallBase"):
        self.name = name
        self.num_users = 0
        self.num_items = 0

    @abstractmethod
    def fit(self, train: Dict[int, List[int]], num_users: int, num_items: int):
        """Train the recall model on user-item interaction data."""
        ...

    @abstractmethod
    def recommend(self, user_id: int, k: int = 200) -> np.ndarray:
        """Return top-k item indices for a given user."""
        ...

    def recommend_batch(self, user_ids: List[int], k: int = 200) -> Dict[int, np.ndarray]:
        """Return {user_id: top-k item array} for multiple users."""
        return {uid: self.recommend(uid, k) for uid in user_ids}
```

- [ ] **Step 3: 验证**

```bash
python -c "from recall import RecallBase; print('Import OK')"
```

- [ ] **Step 4: Commit**

```bash
git add recall/
git commit -m "feat: add recall layer base class"
```

---

### Task 2.2: 创建 recall/cf.py — 协同过滤召回

**Files:**
- Create: `recall/cf.py`

- [ ] **Step 1: 实现 UserCF 和 ItemCF**

```python
from typing import Dict, List
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from recall.base import RecallBase


class UserCF(RecallBase):
    """User-based Collaborative Filtering."""

    def __init__(self, top_k_neighbors: int = 50):
        super().__init__(name="UserCF")
        self.top_k_neighbors = top_k_neighbors
        self.user_sim = None
        self.user_items = None
        self.item_users = None
        self.user_item_matrix = None

    def fit(self, train: Dict[int, List[int]], num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items
        self.user_items = {uid: set(items) for uid, items in train.items()}

        # Build sparse user-item matrix
        rows, cols, data = [], [], []
        for uid, items in train.items():
            for iid in items:
                rows.append(uid)
                cols.append(iid)
                data.append(1.0)
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)), shape=(num_users, num_items)
        )
        # Compute user similarity (may be large — approximate if needed)
        self.user_sim = cosine_similarity(self.user_item_matrix, dense_output=False)

    def recommend(self, user_id: int, k: int = 200) -> np.ndarray:
        if user_id >= self.num_users:
            return np.array([], dtype=np.int64)
        # Find similar users
        sim_row = self.user_sim[user_id].toarray().ravel()
        sim_row[user_id] = -1  # exclude self
        neighbor_indices = np.argpartition(-sim_row, min(self.top_k_neighbors, len(sim_row) - 1))[:self.top_k_neighbors]
        neighbors = neighbor_indices[sim_row[neighbor_indices] > 0]

        # Aggregate scores from neighbor items
        scores = np.zeros(self.num_items)
        interacted = self.user_items.get(user_id, set())
        for nb in neighbors:
            sim = sim_row[nb]
            for iid in self.user_items.get(int(nb), set()):
                if iid not in interacted:
                    scores[iid] += sim

        # Top-k
        top_k = min(k, self.num_items)
        top_items = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        # Sort by score descending
        top_items = top_items[np.argsort(-scores[top_items])]
        return top_items


class ItemCF(RecallBase):
    """Item-based Collaborative Filtering."""

    def __init__(self, top_k_neighbors: int = 50):
        super().__init__(name="ItemCF")
        self.top_k_neighbors = top_k_neighbors
        self.item_sim = None
        self.user_items = None

    def fit(self, train: Dict[int, List[int]], num_users: int, num_items: int):
        self.num_users = num_users
        self.num_items = num_items
        self.user_items = {uid: set(items) for uid, items in train.items()}

        # Build item-user matrix (transpose of user-item)
        rows, cols, data = [], [], []
        for uid, items in train.items():
            for iid in items:
                rows.append(iid)
                cols.append(uid)
                data.append(1.0)
        item_user_matrix = csr_matrix(
            (data, (rows, cols)), shape=(num_items, num_users)
        )
        self.item_sim = cosine_similarity(item_user_matrix, dense_output=False)

    def recommend(self, user_id: int, k: int = 200) -> np.ndarray:
        if user_id >= self.num_users:
            return np.array([], dtype=np.int64)
        interacted = self.user_items.get(user_id, set())
        if not interacted:
            return np.array([], dtype=np.int64)

        scores = np.zeros(self.num_items)
        for iid in interacted:
            sim_row = self.item_sim[iid].toarray().ravel()
            sim_row[iid] = -1
            top_neighbors = np.argpartition(-sim_row, min(self.top_k_neighbors, len(sim_row) - 1))[:self.top_k_neighbors]
            for nb in top_neighbors:
                if nb not in interacted:
                    scores[nb] += sim_row[nb]

        top_k = min(k, self.num_items)
        top_items = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_items = top_items[np.argsort(-scores[top_items])]
        return top_items
```

- [ ] **Step 2: 验证 CF 召回**

```bash
python -c "
from data_loader import MovieLensDataset
from recall.cf import UserCF, ItemCF
ds = MovieLensDataset()
df = ds.load()
train, val, test, nu, ni = ds.split(df)
ucf = UserCF(top_k_neighbors=30)
ucf.fit(train, nu, ni)
recs = ucf.recommend(0, k=20)
print(f'UserCF: {len(recs)} recommendations for user 0')
icf = ItemCF(top_k_neighbors=30)
icf.fit(train, nu, ni)
recs = icf.recommend(0, k=20)
print(f'ItemCF: {len(recs)} recommendations for user 0')
"
```
Expected: 输出 20 个推荐结果

- [ ] **Step 3: Commit**

```bash
git add recall/cf.py
git commit -m "feat: add UserCF and ItemCF collaborative filtering recall"
```

---

### Task 2.3: 创建 recall/svd.py — SVD 矩阵分解召回

**Files:**
- Create: `recall/svd.py`

- [ ] **Step 1: 实现 SVDRecall**

```python
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

        # Build training pairs
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

                # Positive loss
                pos_scores = self.model(u, i)
                pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()

                # Negative sampling (in-batch)
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

        # Pre-compute item vectors for fast recommendation
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
            scores = u_vec @ self.item_vectors.T  # (1, num_items)
            scores = scores.ravel()
        top_k = min(k, self.num_items)
        top_items = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_items = top_items[np.argsort(-scores[top_items])]
        return top_items
```

- [ ] **Step 2: 验证 SVD 召回**

```bash
python -c "
from data_loader import MovieLensDataset
from recall.svd import SVDRecall
ds = MovieLensDataset()
df = ds.load()
train, val, test, nu, ni = ds.split(df)
svd = SVDRecall(embed_dim=32, epochs=5, device='cpu')
svd.fit(train, nu, ni)
recs = svd.recommend(0, k=20)
print(f'SVD: {len(recs)} recommendations for user 0')
"
```
Expected: 完成训练并输出 20 个推荐

- [ ] **Step 3: Commit**

```bash
git add recall/svd.py
git commit -m "feat: add SVD matrix factorization recall"
```

---

### Task 2.4: 创建 recall/nmf.py — NMF 非负矩阵分解召回

**Files:**
- Create: `recall/nmf.py`

- [ ] **Step 1: 实现 NMFRecall**

```python
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
        u = torch.exp(self.user_embed(user))  # ensure non-negative
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
```

- [ ] **Step 2: 验证 NMF 召回**

```bash
python -c "
from data_loader import MovieLensDataset
from recall.nmf import NMFRecall
ds = MovieLensDataset()
df = ds.load()
train, val, test, nu, ni = ds.split(df)
nmf = NMFRecall(embed_dim=32, epochs=5, device='cpu')
nmf.fit(train, nu, ni)
recs = nmf.recommend(0, k=20)
print(f'NMF: {len(recs)} recommendations for user 0')
"
```
Expected: 输出 20 个推荐

- [ ] **Step 3: Commit**

```bash
git add recall/nmf.py
git commit -m "feat: add NMF non-negative matrix factorization recall"
```

---

## Phase 3: 排序层

### Task 3.1: 创建 rank/base.py 和 rank/__init__.py

**Files:**
- Create: `rank/__init__.py`
- Create: `rank/base.py`

- [ ] **Step 1: 创建 rank/__init__.py**

```python
# Re-export existing NCF models
from model import GMF, MLP, NeuMF

# New model
from rank.wide_deep import WideAndDeep

from rank.base import BaseRanker
```

- [ ] **Step 2: 创建 rank/base.py**

```python
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch


class BaseRanker(ABC):
    """Unified interface for ranking models."""

    def __init__(self, name: str = "BaseRanker", device: str = "cpu"):
        self.name = name
        self.device = device

    @abstractmethod
    def predict(self, user: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Score user-item pairs. Returns scores tensor of shape (N,)."""
        ...

    def rank(self, user_id: int, candidates: List[int], top_k: int = 10) -> np.ndarray:
        """Rank candidates for a user and return top-k items."""
        if not candidates:
            return np.array([], dtype=np.int64)
        user = torch.tensor([user_id] * len(candidates), device=self.device)
        items = torch.tensor(candidates, device=self.device)
        with torch.no_grad():
            scores = self.predict(user, items).cpu().numpy()
        top_k = min(top_k, len(candidates))
        top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return np.array(candidates)[top_indices]
```

- [ ] **Step 3: 验证导入**

```bash
python -c "from rank import GMF, MLP, NeuMF, BaseRanker; print('Import OK')"
```

- [ ] **Step 4: Commit**

```bash
git add rank/
git commit -m "feat: add ranking layer base class and re-exports"
```

---

### Task 3.2: 创建 rank/wide_deep.py — Wide&Deep 模型

**Files:**
- Create: `rank/wide_deep.py`

- [ ] **Step 1: 实现 WideAndDeep**

```python
import torch
import torch.nn as nn


class WideAndDeep(nn.Module):
    """Wide & Deep Learning for Recommender Systems.

    Wide part:  Linear cross-product features → memorization
    Deep part:  Embedding → MLP → generalization
    Combined:   σ(Wide_out + Deep_out)
    """

    def __init__(self, num_users, num_items, embed_dim=32,
                 mlp_layers=None, wide_dim=32):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        # Deep part
        self.user_embed_deep = nn.Embedding(num_users, embed_dim)
        self.item_embed_deep = nn.Embedding(num_items, embed_dim)
        mlp_blocks = []
        in_dim = embed_dim * 2
        for out_dim in mlp_layers:
            mlp_blocks.append(nn.Linear(in_dim, out_dim))
            mlp_blocks.append(nn.ReLU())
            mlp_blocks.append(nn.Dropout(0.2))
            in_dim = out_dim
        mlp_blocks.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_blocks)

        # Wide part (cross-product of user/item embeddings)
        self.user_embed_wide = nn.Embedding(num_users, wide_dim)
        self.item_embed_wide = nn.Embedding(num_items, wide_dim)
        self.wide_output = nn.Linear(wide_dim * 2, 1)

        # Combined output
        self.output_bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_embed_deep, self.item_embed_deep,
                     self.user_embed_wide, self.item_embed_wide]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.wide_output.weight)

    def forward(self, user, item):
        # Deep: concatenate embeddings → MLP
        u_deep = self.user_embed_deep(user)
        i_deep = self.item_embed_deep(item)
        deep_out = self.mlp(torch.cat([u_deep, i_deep], dim=-1))

        # Wide: cross-product of wide embeddings
        u_wide = self.user_embed_wide(user)
        i_wide = self.item_embed_wide(item)
        wide_cross = torch.cat([u_wide * i_wide, u_wide, i_wide], dim=-1)
        wide_out = self.wide_output(wide_cross)

        return torch.sigmoid(deep_out + wide_out + self.output_bias).squeeze(-1)
```

- [ ] **Step 2: 验证 Wide&Deep 训练**

```bash
python -c "
import torch
from data_loader import MovieLensDataset
from rank.wide_deep import WideAndDeep
from train import train
ds = MovieLensDataset()
df = ds.load()
train_dict, val, test, nu, ni = ds.split(df)
user_items = {uid: set(items) for uid, items in train_dict.items()}
from data_loader import negative_sampling
users, items, labels = negative_sampling(user_items, ni, num_neg=4)
model = WideAndDeep(nu, ni, embed_dim=16, mlp_layers=[64, 32])
train(model, users, items, labels, val, test, user_items, ni,
      epochs=3, batch_size=256, lr=0.001, device='cpu', eval_k=10)
print('Wide&Deep training complete')
"
```
Expected: 训练完成，输出 HR 和 NDCG

- [ ] **Step 3: Commit**

```bash
git add rank/wide_deep.py
git commit -m "feat: add Wide&Deep ranking model"
```

---

## Phase 4: 管道、评估 & 实验

### Task 4.1: 创建 pipeline.py — 两阶段推荐管道

**Files:**
- Create: `pipeline.py`

- [ ] **Step 1: 实现 RecommendationPipeline**

```python
from typing import List, Dict, Optional
import numpy as np

from recall.base import RecallBase
from rank.base import BaseRanker


class RecommendationPipeline:
    """Two-stage recommendation pipeline: multi-recall → rank → Top-N."""

    def __init__(self, recalls: List[RecallBase], ranker: Optional[BaseRanker] = None,
                 recall_k: int = 200, top_n: int = 10):
        self.recalls = recalls
        self.ranker = ranker
        self.recall_k = recall_k
        self.top_n = top_n

    def recall(self, user_id: int) -> np.ndarray:
        """Multi-recall fusion: intersect all recall results, deduplicate."""
        all_items = set()
        for rec in self.recalls:
            items = rec.recommend(user_id, k=self.recall_k)
            all_items.update(items.tolist())
        return np.array(sorted(all_items))

    def recommend(self, user_id: int, top_n: int = None) -> Dict:
        """Full pipeline: recall → (optional rank) → top-n."""
        if top_n is None:
            top_n = self.top_n

        candidates = self.recall(user_id)
        if len(candidates) == 0:
            return {"user_id": user_id, "items": [], "scores": [], "source": "empty"}

        if self.ranker is not None:
            ranked = self.ranker.rank(user_id, candidates.tolist(), top_k=top_n)
            return {
                "user_id": user_id,
                "items": ranked.tolist(),
                "scores": None,
                "source": "ranked",
            }
        else:
            result = candidates[:top_n]
            return {
                "user_id": user_id,
                "items": result.tolist(),
                "scores": None,
                "source": "recall-only",
            }

    def recommend_batch(self, user_ids: List[int], top_n: int = None) -> List[Dict]:
        return [self.recommend(uid, top_n) for uid in user_ids]
```

- [ ] **Step 2: 端到端验证**

```bash
python -c "
from data_loader import MovieLensDataset
from recall.cf import ItemCF
from recall.svd import SVDRecall
from pipeline import RecommendationPipeline
ds = MovieLensDataset()
df = ds.load()
train, val, test, nu, ni = ds.split(df)
icf = ItemCF(top_k_neighbors=30)
icf.fit(train, nu, ni)
svd = SVDRecall(embed_dim=32, epochs=3, device='cpu')
svd.fit(train, nu, ni)
pipe = RecommendationPipeline(recalls=[icf, svd], ranker=None, recall_k=100, top_n=10)
result = pipe.recommend(0)
print(f'Pipeline result: {result}')
"
```
Expected: 输出推荐结果字典

- [ ] **Step 3: Commit**

```bash
git add pipeline.py
git commit -m "feat: add two-stage recommendation pipeline"
```

---

### Task 4.2: 扩展 evaluate.py — 添加新指标

**Files:**
- Modify: `evaluate.py`

- [ ] **Step 1: 追加 MAP, Diversity, Coverage, Novelty 函数**

在 `evaluate.py` 末尾追加：

```python
def map_at_k(pred_items, true_item, k=10):
    """Mean Average Precision @ K (AP for single user)."""
    if true_item in pred_items[:k]:
        rank = list(pred_items[:k]).index(true_item)
        return 1.0 / (rank + 1)
    return 0.0


def diversity_at_k(pred_items, item_similarity_matrix, k=10):
    """Diversity @ K: 1 - average pairwise similarity of top-K items.
    Higher = more diverse.
    """
    items = pred_items[:k]
    if len(items) < 2:
        return 1.0
    total_sim = 0.0
    count = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            total_sim += item_similarity_matrix[items[i], items[j]]
            count += 1
    return 1.0 - (total_sim / count if count > 0 else 0)


def coverage_at_k(all_recommendations, num_items):
    """Item coverage: fraction of unique items recommended across all users."""
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)
    return len(unique_items) / num_items


def novelty_at_k(pred_items, item_popularity, k=10):
    """Novelty @ K: -average log popularity of top-K items.
    Higher = more novel (less popular items recommended).
    """
    items = pred_items[:k]
    if not items:
        return 0.0
    pops = [item_popularity.get(i, 1) for i in items]
    # Normalize: log(pop) inverted
    total_users = max(item_popularity.values()) if item_popularity else 1
    novelties = [-np.log(p / total_users + 1e-8) for p in pops]
    return np.mean(novelties)


def compute_item_popularity(train):
    """Compute item popularity from training data.
    Returns: dict {item_id: interaction_count}
    """
    pop = {}
    for uid, items in train.items():
        for iid in items:
            pop[iid] = pop.get(iid, 0) + 1
    return pop


def compute_item_similarity(item_embeddings):
    """Compute pairwise item cosine similarity matrix.
    item_embeddings: (num_items, dim) numpy array
    Returns: (num_items, num_items) similarity matrix
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(item_embeddings)


def full_evaluate(model, test_dict, user_items_train, num_items, num_neg=99, k=10):
    """Extended evaluation: HR, NDCG, MAP, and posterior metrics."""
    model.eval()
    hrs, ndcgs, maps = [], [], []
    all_recs = []
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
```

- [ ] **Step 2: 验证新指标**

```bash
python -c "
from evaluate import map_at_k, diversity_at_k, novelty_at_k, compute_item_popularity
import numpy as np
print(f'MAP: {map_at_k([0,1,2,3,4], 2, k=5):.4f}')
pop = {0: 100, 1: 50, 2: 200, 3: 10, 4: 5}
print(f'Novelty: {novelty_at_k([0,1,2], pop, k=3):.4f}')
print('New metrics OK')
"
```
Expected: 指标计算成功

- [ ] **Step 3: Commit**

```bash
git add evaluate.py
git commit -m "feat: add MAP, Diversity, Coverage, Novelty evaluation metrics"
```

---

### Task 4.3: 创建 cold_start.py

**Files:**
- Create: `cold_start.py`

- [ ] **Step 1: 实现冷启动策略**

```python
from typing import Dict, List, Tuple
import numpy as np


class ColdStartHandler:
    """Cold-start recommendation strategies for users with few interactions."""

    def __init__(self, num_items: int):
        self.num_items = num_items
        self.item_popularity = None  # item_id → interaction count
        self.user_groups = None

    def compute_popularity(self, train: Dict[int, List[int]]):
        """Compute global item popularity from training data."""
        self.item_popularity = {}
        for uid, items in train.items():
            for iid in items:
                self.item_popularity[iid] = self.item_popularity.get(iid, 0) + 1

    def split_users(self, train: Dict[int, List[int]]) -> Dict[str, List[int]]:
        """Group users by interaction count.
        Returns: {'extreme_cold': [...], 'cold': [...], 'warm': [...], 'hot': [...]}
        """
        groups = {"extreme_cold": [], "cold": [], "warm": [], "hot": []}
        for uid, items in train.items():
            n = len(items)
            if n <= 1:
                groups["extreme_cold"].append(uid)
            elif n <= 3:
                groups["cold"].append(uid)
            elif n <= 5:
                groups["warm"].append(uid)
            else:
                groups["hot"].append(uid)
        self.user_groups = groups
        return groups

    def popular_recommend(self, k: int = 10, diversity_weight: float = 0.3) -> np.ndarray:
        """Recommend top-K popular items with diversity injection."""
        if self.item_popularity is None:
            return np.array([])
        sorted_items = sorted(self.item_popularity.items(), key=lambda x: -x[1])
        top_items = [iid for iid, _ in sorted_items[:k * 2]]

        if diversity_weight > 0 and len(top_items) > k:
            # Simple diversity: pick top items with some randomness
            scores = np.array([self.item_popularity[i] for i in top_items], dtype=float)
            scores = scores / scores.max()
            noise = np.random.uniform(0, diversity_weight, len(top_items))
            scores = scores + noise
            top_indices = np.argsort(-scores)[:k]
            return np.array(top_items)[top_indices]
        return np.array(top_items[:k])

    def group_recommend(self, user_id: int, train: Dict[int, List[int]],
                        recall_model=None, ranker=None, top_n: int = 10) -> np.ndarray:
        """Recommend based on user's interaction group."""
        user_interactions = train.get(user_id, [])
        n_interact = len(user_interactions)

        if n_interact <= 1:
            # Extreme cold: popular + diversity
            return self.popular_recommend(k=top_n, diversity_weight=0.5)
        elif n_interact <= 3:
            # Cold: popular with less randomness
            return self.popular_recommend(k=top_n, diversity_weight=0.2)
        elif n_interact <= 5:
            # Warm: lightweight CF (popular personalized)
            if recall_model is not None:
                return recall_model.recommend(user_id, k=top_n)
            return self.popular_recommend(k=top_n, diversity_weight=0.1)
        else:
            # Hot: full pipeline (handled externally)
            return None  # signal to use full pipeline

    def evaluate_groups(self, test: Dict[int, int], train: Dict[int, List[int]],
                        recommend_fn) -> Dict[str, Dict[str, float]]:
        """Evaluate recommendation performance by user group.
        recommend_fn: (user_id, k) → list of recommended item IDs
        """
        groups = self.split_users(train)
        results = {}
        for group_name, users in groups.items():
            if not users:
                results[group_name] = {"count": 0, "HR@10": 0, "NDCG@10": 0}
                continue
            hrs, ndcgs = [], []
            for uid in users:
                if uid not in test:
                    continue
                recs = recommend_fn(uid, 10)
                true_item = test[uid]
                hrs.append(1.0 if true_item in recs else 0.0)
                if true_item in recs:
                    rank = list(recs).index(true_item)
                    ndcgs.append(1.0 / np.log2(rank + 2))
                else:
                    ndcgs.append(0.0)
            results[group_name] = {
                "count": len(hrs),
                "HR@10": np.mean(hrs) if hrs else 0,
                "NDCG@10": np.mean(ndcgs) if ndcgs else 0,
            }
        return results
```

- [ ] **Step 2: 验证冷启动策略**

```bash
python -c "
from data_loader import MovieLensDataset
from cold_start import ColdStartHandler
ds = MovieLensDataset()
df = ds.load()
train, val, test, nu, ni = ds.split(df)
cs = ColdStartHandler(ni)
cs.compute_popularity(train)
groups = cs.split_users(train)
for name, users in groups.items():
    print(f'{name}: {len(users)} users')
recs = cs.popular_recommend(k=10)
print(f'Popular recs: {recs[:5].tolist()}...')
"
```
Expected: 各组分组的用户数 + 热门推荐

- [ ] **Step 3: Commit**

```bash
git add cold_start.py
git commit -m "feat: add cold-start recommendation strategies"
```

---

### Task 4.4: 创建实验脚本

**Files:**
- Create: `experiments/run_experiments.py`

- [ ] **Step 1: 创建实验脚本**（完整代码）

```python
"""Run all three experiment groups: comparison, ablation, cold-start."""
import argparse
import numpy as np
import torch
from data_loader import MovieLensDataset, LastfmDataset
from recall.cf import UserCF, ItemCF
from recall.svd import SVDRecall
from recall.nmf import NMFRecall
from model import GMF, MLP, NeuMF
from rank.wide_deep import WideAndDeep
from train import train
from evaluate import evaluate, full_evaluate, compute_item_popularity
from cold_start import ColdStartHandler


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(dataset_name="movielens"):
    if dataset_name == "movielens":
        ds = MovieLensDataset()
    elif dataset_name == "lastfm":
        ds = LastfmDataset()
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
    from data_loader import negative_sampling
    users, items_, labels = negative_sampling(user_items, ni, num_neg=4)

    results = {}

    # Traditional methods
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

    # Deep methods
    print("--- Deep Methods ---")

    # GMF
    gmf = GMF(nu, ni, embed_dim=8)
    gmf_hr, gmf_ndcg, _ = train(gmf, users, items_, labels, val, test, user_items, ni,
                                  epochs=epochs, batch_size=256, lr=0.001,
                                  device=device, eval_k=10, early_stop_patience=3)
    results["GMF"] = {"HR@10": gmf_hr, "NDCG@10": gmf_ndcg}

    # MLP
    mlp = MLP(nu, ni, embed_dim=8, layers=[32, 16, 8])
    mlp_hr, mlp_ndcg, _ = train(mlp, users, items_, labels, val, test, user_items, ni,
                                  epochs=epochs, batch_size=256, lr=0.001,
                                  device=device, eval_k=10, early_stop_patience=3)
    results["MLP"] = {"HR@10": mlp_hr, "NDCG@10": mlp_ndcg}

    # NeuMF
    neumf = NeuMF(nu, ni, gmf_dim=8, mlp_dim=8, mlp_layers=[32, 16, 8])
    neumf_hr, neumf_ndcg, _ = train(neumf, users, items_, labels, val, test, user_items, ni,
                                      epochs=epochs, batch_size=256, lr=0.001,
                                      device=device, eval_k=10, early_stop_patience=3)
    results["NeuMF"] = {"HR@10": neumf_hr, "NDCG@10": neumf_ndcg}

    # Wide&Deep
    wd = WideAndDeep(nu, ni, embed_dim=16, mlp_layers=[64, 32])
    wd_hr, wd_ndcg, _ = train(wd, users, items_, labels, val, test, user_items, ni,
                                epochs=epochs, batch_size=256, lr=0.001,
                                device=device, eval_k=10, early_stop_patience=3)
    results["Wide&Deep"] = {"HR@10": wd_hr, "NDCG@10": wd_ndcg}

    # Print results table
    print(f"\n{'Model':<15} {'HR@10':<10} {'NDCG@10':<10}")
    print("-" * 35)
    for model, metrics in results.items():
        print(f"{model:<15} {metrics['HR@10']:<10.4f} {metrics['NDCG@10']:<10.4f}")

    return results


def run_cold_start_experiment(dataset_name="movielens", device="cpu"):
    """Experiment 3: Cold-start performance by user group."""
    print(f"\n{'='*60}")
    print(f"Experiment 3: Cold-Start Analysis on {dataset_name}")
    print(f"{'='*60}\n")

    train, val, test, user_items, nu, ni = prepare_data(dataset_name)
    handler = ColdStartHandler(ni)
    handler.compute_popularity(train)

    # Evaluate popular baseline by group
    groups = handler.split_users(train)
    pop = compute_item_popularity(train)
    sorted_pop = sorted(pop.items(), key=lambda x: -x[1])
    top_k = [iid for iid, _ in sorted_pop[:200]]

    def popular_recommend(uid, k):
        return top_k[:k]

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
```

- [ ] **Step 2: 运行对比实验**

```bash
python experiments/run_experiments.py --dataset movielens --epochs 3 --experiment comparison
```
Expected: 输出所有模型在 MovieLens 上的对比表格

- [ ] **Step 3: 运行冷启动实验**

```bash
python experiments/run_experiments.py --dataset movielens --experiment cold_start
```
Expected: 输出各用户分组的 HR 和 NDCG

- [ ] **Step 4: Commit**

```bash
git add experiments/
git commit -m "feat: add experiment scripts for comparison and cold-start analysis"
```

---

## Phase 5: 服务层

### Task 5.1: 创建 api/server.py — FastAPI 推荐服务

**Files:**
- Create: `api/__init__.py`
- Create: `api/server.py`

- [ ] **Step 1: 创建 api/__init__.py**

```python
"""API and UI layer for the recommendation system."""
```

- [ ] **Step 2: 创建 api/server.py**

```python
"""FastAPI recommendation service."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="个性化推荐系统 API",
    description="Two-stage recommendation system (Recall + Ranking)",
    version="1.0.0",
)

# --- Global state (set by loader) ---
pipeline = None
available_models = {}
cold_start_handler = None


class RecommendResponse(BaseModel):
    user_id: int
    items: List[int]
    scores: Optional[List[float]] = None
    model_used: str


class ColdStartRequest(BaseModel):
    interactions: List[int]  # existing item IDs
    top_k: int = 10


class ErrorResponse(BaseModel):
    error: str
    detail: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    return {"models": list(available_models.keys())}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=100),
    model: str = Query(default=None),
):
    if pipeline is None:
        return {"error": "Pipeline not initialized"}

    try:
        result = pipeline.recommend(user_id, top_n=top_k)
        return RecommendResponse(
            user_id=user_id,
            items=result["items"],
            scores=result.get("scores"),
            model_used=model or "default",
        )
    except Exception as e:
        return {"error": str(e)}


@app.post("/recommend/cold-start", response_model=RecommendResponse)
async def cold_start_recommend(req: ColdStartRequest):
    if cold_start_handler is None:
        return {"error": "Cold-start handler not initialized"}

    try:
        items = cold_start_handler.popular_recommend(
            k=req.top_k, diversity_weight=0.5
        )
        return RecommendResponse(
            user_id=-1,  # new user
            items=items.tolist(),
            model_used="cold-start-popular",
        )
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 2: 验证 FastAPI 启动**

```bash
cd /c/Users/黄海亦/Desktop/ncf_reproduction
pip install fastapi uvicorn -q 2>&1
python -c "from api.server import app; print(f'FastAPI app: {app.title}')"
```
Expected: 打印 "FastAPI app: 个性化推荐系统 API"

- [ ] **Step 3: Commit**

```bash
git add api/__init__.py api/server.py
git commit -m "feat: add FastAPI recommendation service"
```

---

### Task 5.2: 创建 api/app.py — Gradio 可视化界面

**Files:**
- Create: `api/app.py`

- [ ] **Step 1: 创建 api/app.py**

```python
"""Gradio visualization interface for the recommendation system."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from data_loader import MovieLensDataset
from pipeline import RecommendationPipeline
from recall.cf import ItemCF
from recall.svd import SVDRecall
from recall.nmf import NMFRecall
from rank.base import BaseRanker
from cold_start import ColdStartHandler


class GradioApp:
    """Gradio-based UI for the recommendation system."""

    def __init__(self, pipeline: RecommendationPipeline = None,
                 cs_handler: ColdStartHandler = None,
                 metrics_data: dict = None):
        self.pipeline = pipeline
        self.cs_handler = cs_handler
        self.metrics = metrics_data or {}

    def recommend_tab(self, user_id, model_name, top_k):
        """Tab 1: Get recommendations for a user."""
        try:
            user_id = int(user_id)
            top_k = int(top_k)
            result = self.pipeline.recommend(user_id, top_n=top_k)
            items = result.get("items", [])
            if not items:
                return "No recommendations found for this user."
            lines = [f"### Top-{top_k} Recommendations for User {user_id}\n"]
            for rank, iid in enumerate(items, 1):
                lines.append(f"{rank}. Item **{iid}** (score from {result.get('source', 'N/A')})")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    def metrics_tab(self):
        """Tab 2: Show model comparison chart."""
        if not self.metrics:
            return None
        models = list(self.metrics.keys())
        hr_vals = [self.metrics[m].get("HR@10", 0) for m in models]
        ndcg_vals = [self.metrics[m].get("NDCG@10", 0) for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

        ax1.bar(models, hr_vals, color=colors)
        ax1.set_title("HR@10 Comparison")
        ax1.set_ylabel("HR@10")
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(models, ndcg_vals, color=colors)
        ax2.set_title("NDCG@10 Comparison")
        ax2.set_ylabel("NDCG@10")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig

    def cold_start_tab(self, item1, item2, item3):
        """Tab 3: Cold-start test with few interactions."""
        try:
            interactions = []
            for i in [item1, item2, item3]:
                if i and i.strip():
                    interactions.append(int(i.strip()))
            if not interactions:
                return "Please enter at least one item ID."
            n = len(interactions)
            if n <= 1:
                recs = self.cs_handler.popular_recommend(k=10, diversity_weight=0.5)
            elif n <= 3:
                recs = self.cs_handler.popular_recommend(k=10, diversity_weight=0.2)
            else:
                recs = self.cs_handler.popular_recommend(k=10, diversity_weight=0.1)
            return f"With {n} interactions, recommended: {recs.tolist()}"
        except Exception as e:
            return f"Error: {e}"

    def build(self):
        with gr.Blocks(title="个性化推荐系统 Demo") as demo:
            gr.Markdown("# 个性化推荐系统")
            gr.Markdown("Two-stage architecture: Recall → Ranking → Top-N")

            with gr.Tab("🎯 Top-N 推荐"):
                gr.Markdown("输入用户ID和参数，获取个性化推荐")
                with gr.Row():
                    user_input = gr.Textbox(label="User ID", value="0")
                    model_input = gr.Dropdown(
                        label="Ranking Model",
                        choices=["NeuMF", "Wide&Deep", "GMF", "MLP"],
                        value="NeuMF",
                    )
                    k_input = gr.Slider(label="Top-K", minimum=5, maximum=50, value=10, step=5)
                btn = gr.Button("获取推荐", variant="primary")
                output = gr.Markdown()
                btn.click(self.recommend_tab, [user_input, model_input, k_input], output)

            with gr.Tab("📊 模型对比"):
                gr.Markdown("各模型在测试集上的指标对比")
                if self.metrics:
                    plot_btn = gr.Button("生成对比图")
                    plot_output = gr.Plot()
                    plot_btn.click(self.metrics_tab, [], plot_output)
                else:
                    gr.Markdown("*请先运行实验脚本获取指标数据*")

            with gr.Tab("❄️ 冷启动测试"):
                gr.Markdown("模拟新用户：输入少量已交互物品ID")
                with gr.Row():
                    i1 = gr.Textbox(label="已交互物品1")
                    i2 = gr.Textbox(label="已交互物品2")
                    i3 = gr.Textbox(label="已交互物品3")
                cold_btn = gr.Button("测试冷启动推荐")
                cold_output = gr.Markdown()
                cold_btn.click(self.cold_start_tab, [i1, i2, i3], cold_output)

        return demo


def create_app(pipeline=None, cs_handler=None, metrics_data=None):
    app = GradioApp(pipeline, cs_handler, metrics_data)
    return app.build()
```

- [ ] **Step 2: 验证 Gradio App 可创建**

```bash
pip install gradio -q 2>&1
python -c "
from api.app import create_app
demo = create_app()
print(f'Gradio app created: {demo}')
"
```
Expected: 输出 "Gradio app created: ..."

- [ ] **Step 3: Commit**

```bash
git add api/app.py
git commit -m "feat: add Gradio visualization interface"
```

---

## Phase 6: 论文撰写

### Task 6.1: 基于模板和实验结果撰写论文

**Files:**
- Create: `paper/小论文.md`

- [ ] **Step 1: 更新 requirements.txt**

```bash
echo "fastapi>=0.100.0" >> requirements.txt
echo "gradio>=4.0.0" >> requirements.txt
echo "uvicorn>=0.23.0" >> requirements.txt
echo "scikit-learn>=1.3.0" >> requirements.txt
echo "scipy>=1.11.0" >> requirements.txt
echo "matplotlib>=3.7.0" >> requirements.txt
```

- [ ] **Step 2: 撰写论文草稿** — 按照模板章节结构，基于实验数据填充

论文大纲（等阶段1-5实验跑完后，用真实数据填充）：
```
1. 摘要 + 关键词 (4个: 个性化推荐;深度学习;协同过滤;冷启动)
2. 引言 (~800字)
3. 研究现状 (~1000字)
4. 相关理论与技术 (~1200字)
5. 本文方法 (~1500字) — 两阶段架构核心
6. 实验与分析 (~1500字) — 用实验脚本的实际输出
7. 结束语 (~400字)
参考文献 (GB/T 7714格式，15-20篇)
```

- [ ] **Step 3: 论文格式按模板要求** — 中文标题、作者信息、中英文摘要、图表编号、公式

- [ ] **Step 4: Commit**

```bash
git add paper/ requirements.txt
git commit -m "feat: add paper draft based on template"
```

---

## Final: Push All

所有阶段完成后：

```bash
cd /c/Users/黄海亦/Desktop/ncf_reproduction
git push origin master
```
