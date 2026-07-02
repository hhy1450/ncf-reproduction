"""
启动推荐系统界面。

用法:
    python launch.py                  # 默认 Streamlit 界面
    python launch.py --ui streamlit   # Streamlit 界面
    python launch.py --ui gradio      # Gradio 界面
"""
import sys
import os
import argparse
import pickle
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import MovieLensDataset, negative_sampling
from recall.cf import ItemCF
from recall.svd import SVDRecall
from recall.nmf import NMFRecall
from pipeline import RecommendationPipeline
from cold_start import ColdStartHandler
from rank.deep_ranker import DeepRanker, build_ranker, MODEL_NAMES
from train import train as train_model

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_state.pkl")
RANKER_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RANKER = "Wide&Deep"


def _ranker_path(model_name):
    return os.path.join(RANKER_DIR, f"ranker_{model_name}.pt")


def build_pipeline(dataset_name="movielens", data_dir="data"):
    """Train recall and ranking models, build the full recommendation pipeline."""
    print("=" * 50)
    print("  个性化推荐系统 -- 初始化中...")
    print("=" * 50)

    # 1. Load data
    print("\n[1/5] 加载数据集...")
    if dataset_name == "movielens":
        ds = MovieLensDataset(data_dir)
    else:
        from data_loader import LastfmDataset
        ds = LastfmDataset(data_dir)

    df = ds.load()
    train, val, test, nu, ni = ds.split(df)

    # Re-index item names: original IDs -> 0-based contiguous IDs
    raw_item_names = ds.load_item_names() if hasattr(ds, "load_item_names") else {}
    orig_to_new = {iid: i for i, iid in enumerate(df["item_id"].unique())}
    item_names = {}
    for orig_id, new_id in orig_to_new.items():
        name = raw_item_names.get(orig_id)
        if name:
            item_names[new_id] = name

    print(f"  用户: {nu}, 物品: {ni}, 训练交互: {sum(len(v) for v in train.values())}")

    # 2. Train recall models
    print("\n[2/5] 训练召回模型...")
    print("  - ItemCF...")
    icf = ItemCF(top_k_neighbors=50)
    icf.fit(train, nu, ni)

    print("  - SVD...")
    svd = SVDRecall(embed_dim=64, epochs=15, device="cpu")
    svd.fit(train, nu, ni)

    print("  - NMF...")
    nmf = NMFRecall(embed_dim=64, epochs=15, device="cpu")
    nmf.fit(train, nu, ni)

    # 3. Train all deep learning ranking models
    print("\n[3/5] 训练深度学习排序模型...")
    user_items_set = {uid: set(items) for uid, items in train.items()}
    train_users, train_items, train_labels = negative_sampling(
        user_items_set, ni, num_neg=4
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rankers = {}  # model_name -> DeepRanker
    for model_name in MODEL_NAMES:
        print(f"  - {model_name}...")
        model = build_ranker(model_name, nu, ni, device=device)
        hr, ndcg, best_epoch = train_model(
            model, train_users, train_items, train_labels,
            val, test, user_items_set, ni,
            epochs=15, batch_size=256, lr=0.001,
            device=device, eval_k=10, early_stop_patience=5,
        )
        ranker = DeepRanker(model, name=model_name, device=device)
        rankers[model_name] = ranker

        # Save weights
        weight_path = _ranker_path(model_name)
        torch.save(model.state_dict(), weight_path)
        print(f"    HR@10={hr:.4f}  NDCG@10={ndcg:.4f}  已保存: {weight_path}")

    default_ranker = rankers[DEFAULT_RANKER]

    # 4. Build pipeline
    print("\n[4/5] 构建推荐管道...")
    pipeline = RecommendationPipeline(
        recalls=[icf, svd, nmf],
        ranker=default_ranker,
        recall_k=200,
        top_n=10,
    )

    cs_handler = ColdStartHandler(ni)
    cs_handler.compute_popularity(train)

    # 5. Save state
    print("\n[5/5] 保存状态...")
    pipeline.ranker = None  # detach for safe pickling
    state = {
        "pipeline": pipeline,
        "cs_handler": cs_handler,
        "item_names": item_names,
        "train_data": train,
        "num_users": nu,
        "num_items": ni,
        "ranker_configs": {  # one config per model
            name: {"num_users": nu, "num_items": ni, "device": device}
            for name in MODEL_NAMES
        },
        "ranker_default": DEFAULT_RANKER,
    }
    with open(STATE_PATH, "wb") as f:
        pickle.dump(state, f)
    pipeline.ranker = default_ranker  # re-attach
    state["pipeline"] = pipeline
    print(f"  状态已保存至: {STATE_PATH}")
    print("=" * 50)

    return state


def load_state():
    """Load pre-built pipeline state from disk, attach default ranker.

    Handles both old format (ranker_config, single model) and new format
    (ranker_configs, multiple models).
    """
    if not os.path.exists(STATE_PATH):
        return None
    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)

    # Normalise: old format had "ranker_config" (singular), new has "ranker_configs"
    if "ranker_configs" not in state or not state["ranker_configs"]:
        if "ranker_config" in state and state["ranker_config"]:
            # Convert old single-model config to new multi-model format
            cfg = state["ranker_config"]
            state["ranker_configs"] = {cfg["model_name"]: cfg}
            state["ranker_default"] = cfg["model_name"]
        else:
            state["ranker_configs"] = {}
            state["ranker_default"] = None

    default_name = state.get("ranker_default")
    if not default_name:
        return state  # no ranker configured

    cfg = state["ranker_configs"].get(default_name, {})
    if not cfg:
        cfg = {"num_users": state["num_users"], "num_items": state["num_items"],
               "device": "cpu"}

    # Try new naming first, then old
    for path_candidate in [_ranker_path(default_name),
                           os.path.join(RANKER_DIR, "ranker_weights.pt")]:
        if os.path.exists(path_candidate):
            try:
                model = build_ranker(default_name, cfg["num_users"],
                                    cfg["num_items"], device=cfg.get("device", "cpu"))
                model.load_state_dict(torch.load(path_candidate,
                                                 map_location=cfg.get("device", "cpu")))
                state["pipeline"].ranker = DeepRanker(
                    model, name=default_name, device=cfg.get("device", "cpu")
                )
                break
            except Exception as e:
                print(f"  注意: 排序模型加载失败 ({e})，将仅使用召回结果。")

    return state


def launch_streamlit():
    """Launch Streamlit UI."""
    import subprocess

    if not os.path.exists(STATE_PATH):
        print("未找到已保存的状态，正在构建...")
        build_pipeline()

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "api", "streamlit_app.py")
    print("\n启动 Streamlit 界面...")
    print("浏览器打开 http://localhost:8501 即可使用。")
    print("按 Ctrl+C 停止服务。\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path,
                    "--server.port", "8501",
                    "--server.address", "127.0.0.1",
                    "--browser.serverAddress", "localhost"])


def launch_gradio():
    """Launch Gradio UI (original)."""
    state = load_state()
    if state is None:
        state = build_pipeline()

    from api.app import create_app

    demo = create_app(
        pipeline=state["pipeline"],
        cs_handler=state["cs_handler"],
        item_names=state["item_names"],
        train_data=state["train_data"],
    )

    print("\n启动 Gradio 界面...")
    print("浏览器打开后即可使用！")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


def main():
    parser = argparse.ArgumentParser(description="推荐系统界面启动器")
    parser.add_argument("--ui", type=str, default="streamlit",
                        choices=["streamlit", "gradio"],
                        help="界面框架 (默认: streamlit)")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重新训练模型（忽略缓存）")
    args = parser.parse_args()

    if args.rebuild:
        for p in [STATE_PATH] + [_ranker_path(n) for n in MODEL_NAMES]:
            if os.path.exists(p):
                os.remove(p)
        print("已清除缓存，将重新训练。")

    if args.ui == "streamlit":
        launch_streamlit()
    else:
        launch_gradio()


if __name__ == "__main__":
    main()
