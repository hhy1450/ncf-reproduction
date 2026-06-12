"""一键启动推荐系统 Gradio 界面。

用法: python launch.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import MovieLensDataset
from recall.cf import ItemCF
from recall.svd import SVDRecall
from recall.nmf import NMFRecall
from pipeline import RecommendationPipeline
from cold_start import ColdStartHandler
from api.app import create_app


def main():
    print("=" * 50)
    print("  个性化推荐系统 — 启动中...")
    print("=" * 50)

    # 1. 加载数据
    print("\n[1/4] 加载 MovieLens-1M 数据集...")
    ds = MovieLensDataset()
    df = ds.load()
    train, val, test, nu, ni = ds.split(df)
    print(f"  用户: {nu}, 物品: {ni}, 训练交互: {sum(len(v) for v in train.values())}")

    # 2. 训练召回模型
    print("\n[2/4] 训练召回模型...")
    print("  - ItemCF...")
    icf = ItemCF(top_k_neighbors=50)
    icf.fit(train, nu, ni)

    print("  - SVD...")
    svd = SVDRecall(embed_dim=32, epochs=5, device="cpu")
    svd.fit(train, nu, ni)

    print("  - NMF...")
    nmf = NMFRecall(embed_dim=32, epochs=5, device="cpu")
    nmf.fit(train, nu, ni)

    # 3. 构建管道 & 冷启动
    print("\n[3/4] 构建推荐管道...")
    pipeline = RecommendationPipeline(
        recalls=[icf, svd, nmf],
        ranker=None,
        recall_k=200,
        top_n=10
    )

    cs = ColdStartHandler(ni)
    cs.compute_popularity(train)

    # 4. 启动界面
    print("\n[4/4] 启动 Gradio 界面...")
    print("  浏览器打开后即可使用！")
    print("=" * 50)

    demo = create_app(pipeline=pipeline, cs_handler=cs)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
