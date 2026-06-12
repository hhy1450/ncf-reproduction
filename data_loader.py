import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile
from abc import ABC, abstractmethod


# ──────────────────────────────────────────────────
#  BaseDataset: abstract base class for all datasets
# ──────────────────────────────────────────────────

class BaseDataset(ABC):
    """Abstract dataset with download / load / split / negative sampling.

    Subclasses must implement ``download()`` and ``load_raw()``.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    @abstractmethod
    def download(self):
        """Download and extract the dataset to ``self.data_dir``."""
        ...

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Read raw files and return a DataFrame with columns
        [user_id, item_id, timestamp].
        """
        ...

    def load(self) -> pd.DataFrame:
        """Download (if needed) then load raw interactions."""
        self.download()
        return self.load_raw()

    def split(self, df):
        """Leave-one-out split.

        Returns:
            train, val, test, num_users, num_items
        """
        return leave_one_out_split(df)

    @staticmethod
    def build_user_items(train):
        """Convert train dict (list values) to a dict of sets (for O(1) lookup).

        Args:
            train: dict {user_id: list_of_item_ids}

        Returns:
            dict {user_id: set_of_item_ids}
        """
        return {uid: set(items) for uid, items in train.items()}

    def neg_sample(self, train, num_items, num_neg=4):
        """Wrap ``negative_sampling``."""
        return negative_sampling(train, num_items, num_neg)

    def hard_neg_sample(self, model, train, num_items, num_neg, sample_pool, device):
        """Placeholder — wraps ``hard_negative_sampling`` (added in Task 1.3)."""
        # Import here so the module still works before Task 1.3 is done.
        try:
            from data_loader import hard_negative_sampling
        except ImportError:
            raise NotImplementedError(
                "hard_negative_sampling is not yet implemented (Task 1.3)."
            )
        return hard_negative_sampling(
            model, train, num_items, num_neg, sample_pool, device
        )

    @staticmethod
    def cold_start_split(train, threshold=5):
        """Split users into cold-start (few interactions) and warm.

        Args:
            train: dict {user_id: list_of_item_ids}
            threshold: minimum number of interactions to be "warm"

        Returns:
            cold_users: list of user ids with < threshold interactions
            warm_users: list of user ids with >= threshold interactions
        """
        cold_users, warm_users = [], []
        for uid, items in train.items():
            if len(items) < threshold:
                cold_users.append(uid)
            else:
                warm_users.append(uid)
        return cold_users, warm_users


# ──────────────────────────────────────────────────
#  Module-level helpers (preserved for backward compat)
# ──────────────────────────────────────────────────

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATA_DIR, "ml-1m.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-1m")


def download_and_extract():
    """Download and extract MovieLens-1M dataset."""
    if os.path.exists(os.path.join(EXTRACT_DIR, "ratings.dat")):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print("Downloading MovieLens-1M...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as f:
        f.extractall(DATA_DIR)
    # The zip contains a "ml-1m" folder; rename if needed
    inner = os.path.join(DATA_DIR, "ml-1m")
    if os.path.isdir(inner) and not os.path.exists(os.path.join(EXTRACT_DIR, "ratings.dat")):
        # Extracted to inner ml-1m/ml-1m/ — nothing to do (files are there)
        pass


def load_data():
    """Load ratings and return as implicit feedback DataFrame.

    Returns:
        df: DataFrame with columns [user_id, item_id, timestamp]
    """
    download_and_extract()
    ratings_path = os.path.join(EXTRACT_DIR, "ratings.dat")
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    # Treat all ratings as positive (implicit feedback)
    df = df[["user_id", "item_id", "timestamp"]]
    return df


def build_interactions(df):
    """Build user-item interaction dictionary.

    Returns:
        user_items: dict {user_id: set of item_ids}
    """
    user_items = {}
    for uid, g in df.groupby("user_id"):
        user_items[uid] = set(g["item_id"].tolist())
    return user_items


def leave_one_out_split(df):
    """Leave-one-out split: latest interaction as test, second-latest as val.

    Returns:
        train: dict {user_id: list of item_ids}
        val: dict {user_id: item_id}
        test: dict {user_id: item_id}
        num_users, num_items
    """
    # Re-index to 0-based contiguous IDs
    df = df.copy()
    user_ids = {uid: i for i, uid in enumerate(df["user_id"].unique())}
    item_ids = {iid: i for i, iid in enumerate(df["item_id"].unique())}
    df["user_idx"] = df["user_id"].map(user_ids)
    df["item_idx"] = df["item_id"].map(item_ids)

    num_users = len(user_ids)
    num_items = len(item_ids)

    train, val, test = {}, {}, {}

    for uid, g in df.groupby("user_idx"):
        # Sort by timestamp
        g = g.sort_values("timestamp")
        items = g["item_idx"].tolist()
        test[uid] = items[-1]
        if len(items) >= 2:
            val[uid] = items[-2]
            train_items = items[:-2]
        else:
            val[uid] = items[-1]
            train_items = items[:-1]
        train[uid] = train_items

    print(f"Users: {num_users}, Items: {num_items}")
    print(
        f"Train interactions: {sum(len(v) for v in train.values())}, "
        f"Val: {len(val)}, Test: {len(test)}"
    )
    return train, val, test, num_users, num_items


def negative_sampling(user_items, num_items, num_neg=4):
    """Sample negative items for each positive item in a batch.

    Args:
        user_items: dict {user_id: set of interacted items}
        num_items: total number of items
        num_neg: number of negative samples per positive

    Returns:
        users, items, labels (all np.arrays)
    """
    users, items, labels = [], [], []
    for uid, pos_items in user_items.items():
        for pos in pos_items:
            # Positive sample
            users.append(uid)
            items.append(pos)
            labels.append(1)
            # Negative samples
            for _ in range(num_neg):
                neg = np.random.randint(0, num_items)
                while neg in user_items[uid]:
                    neg = np.random.randint(0, num_items)
                users.append(uid)
                items.append(neg)
                labels.append(0)
    return (
        np.array(users, dtype=np.int64),
        np.array(items, dtype=np.int64),
        np.array(labels, dtype=np.float32),
    )


def batch_generator(users, items, labels, batch_size=256, shuffle=True):
    """Yield minibatches."""
    n = len(users)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        yield users[idx], items[idx], labels[idx]


# ──────────────────────────────────────────────────
#  Concrete dataset implementations
# ──────────────────────────────────────────────────

class MovieLensDataset(BaseDataset):
    """MovieLens-1M dataset wrapper.

    Usage::

        ds = MovieLensDataset("data")
        df = ds.load()
        train, val, test, num_users, num_items = ds.split(df)
        train_ui = ds.build_user_items(train)
        users, items, labels = ds.neg_sample(train, num_items, num_neg=4)
    """

    DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def download(self):
        """Download and extract ml-1m.zip into ``self.data_dir``."""
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
        """Read ratings.dat → DataFrame with [user_id, item_id, timestamp]."""
        ratings_path = os.path.join(self.data_dir, "ml-1m", "ratings.dat")
        df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        return df[["user_id", "item_id", "timestamp"]]


class LastfmDataset(BaseDataset):
    """Last.fm music listening dataset (HetRec 2011)."""

    DATA_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        super().__init__(data_dir)

    def download(self):
        import urllib.request, zipfile
        zip_path = os.path.join(self.data_dir, "lastfm-2k.zip")
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.data_dir, "user_artists.dat")):
            return
        if not os.path.exists(zip_path):
            print("Downloading Last.fm dataset...")
            urllib.request.urlretrieve(self.DATA_URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(self.data_dir)

    def load_raw(self):
        import pandas as pd
        path = os.path.join(self.data_dir, "user_artists.dat")
        df = pd.read_csv(path, sep="\t")
        df = df.rename(columns={"userID": "user_id", "artistID": "item_id", "weight": "play_count"})
        df = df.sort_values(["user_id", "play_count"], ascending=[True, False])
        df["timestamp"] = df.groupby("user_id").cumcount()
        return df[["user_id", "item_id", "timestamp"]]
