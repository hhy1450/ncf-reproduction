"""Gradio visualization interface for the recommendation system."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pipeline import RecommendationPipeline
from cold_start import ColdStartHandler


class GradioApp:
    """Gradio-based UI for the recommendation system."""

    def __init__(self, pipeline: RecommendationPipeline = None,
                 cs_handler: ColdStartHandler = None,
                 metrics_data: dict = None,
                 item_names: dict = None,
                 train_data: dict = None):
        self.pipeline = pipeline
        self.cs_handler = cs_handler
        self.metrics = metrics_data or {}
        self.item_names = item_names or {}
        self.train_data = train_data or {}

    def _item_name(self, iid):
        """Return item name if available, otherwise item ID."""
        name = self.item_names.get(iid)
        return f"{name} (ID: {iid})" if name else f"物品 {iid}"

    def recommend_tab(self, user_id, top_k):
        """Tab 1: Get recommendations for a user."""
        try:
            user_id = int(user_id)
            top_k = int(top_k)
            if self.pipeline is None:
                return "推荐管道未加载。"

            # 用户历史
            user_history = self.train_data.get(user_id, [])
            if user_history:
                history_lines = ["### 📖 该用户看过的电影\n"]
                for iid in user_history[-10:]:  # 最近10部
                    history_lines.append(f"- {self._item_name(iid)}")
                history_text = "\n".join(history_lines) + "\n"
            else:
                history_text = "### 📖 该用户暂无历史记录（冷启动用户）\n"

            # 推荐结果
            result = self.pipeline.recommend(user_id, top_n=top_k)
            items = result.get("items", [])
            sources = result.get("sources", {})

            if not items:
                return history_text + "\n未找到推荐结果。"

            rec_lines = [f"### 🎯 推荐 Top-{top_k} 电影\n"]
            rec_lines.append("*系统通过以下算法为你推荐：ItemCF（基于物品相似度）、SVD（矩阵分解）、NMF（非负矩阵分解）*\n")
            for rank, iid in enumerate(items, 1):
                src_names = sources.get(iid, [])
                src_str = " + ".join(src_names) if src_names else "融合"
                rec_lines.append(f"{rank}. {self._item_name(iid)}  `← {src_str}`")

            return history_text + "\n" + "\n".join(rec_lines)

        except Exception as e:
            return f"出错了: {e}"

    def metrics_tab(self):
        """Tab 2: Model comparison chart."""
        if not self.metrics:
            return None
        models = list(self.metrics.keys())
        hr_vals = [self.metrics[m].get("HR@10", 0) for m in models]
        ndcg_vals = [self.metrics[m].get("NDCG@10", 0) for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

        ax1.bar(models, hr_vals, color=colors)
        ax1.set_title("HR@10 对比")
        ax1.set_ylabel("HR@10")
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(models, ndcg_vals, color=colors)
        ax2.set_title("NDCG@10 对比")
        ax2.set_ylabel("NDCG@10")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig

    def cold_start_tab(self, item1, item2, item3):
        """Tab 3: Cold-start test with few interactions."""
        try:
            interactions = []
            for i in [item1, item2, item3]:
                if i and str(i).strip():
                    interactions.append(int(str(i).strip()))
            if not interactions:
                return "请输入至少一个物品ID。"
            if self.cs_handler is None:
                return "冷启动处理器未加载。"
            n = len(interactions)
            if n <= 1:
                recs = self.cs_handler.popular_recommend(k=10, diversity_weight=0.5)
                strategy = "极冷启动（≤1条交互）：热门推荐 + 高多样性"
            elif n <= 3:
                recs = self.cs_handler.popular_recommend(k=10, diversity_weight=0.2)
                strategy = "冷启动（2-3条交互）：热门推荐 + 低多样性"
            else:
                recs = self.cs_handler.popular_recommend(k=10, diversity_weight=0.1)
                strategy = "温启动（4-5条交互）：热门推荐 + 最小随机"

            input_items = [self._item_name(i) for i in interactions]
            rec_items = [self._item_name(iid) for iid in recs]

            lines = [
                f"### ❄️ 冷启动推荐\n",
                f"**策略**: {strategy}",
                f"**你输入的物品**: {', '.join(input_items)}",
                f"\n**推荐结果**:\n",
            ]
            for i, name in enumerate(rec_items, 1):
                lines.append(f"{i}. {name}")

            return "\n".join(lines)

        except Exception as e:
            return f"出错了: {e}"

    def build(self):
        with gr.Blocks(title="个性化推荐系统演示") as demo:
            gr.Markdown("# 个性化推荐系统")
            gr.Markdown("两阶段架构：多路召回（ItemCF + SVD + NMF）→ 融合排序 → Top-N 推荐")

            with gr.Tab("🎯 Top-N 推荐"):
                gr.Markdown("输入用户ID，系统基于该用户的**历史观影记录**，通过三路召回算法生成个性化推荐。")
                with gr.Row():
                    user_input = gr.Textbox(label="用户ID (0-6039)", value="0")
                    k_input = gr.Slider(label="推荐数量", minimum=5, maximum=50, value=10, step=5)
                btn = gr.Button("获取推荐", variant="primary")
                output = gr.Markdown()
                btn.click(self.recommend_tab, [user_input, k_input], output)

            with gr.Tab("📊 模型对比"):
                gr.Markdown("各模型在测试集上的性能对比")
                if self.metrics:
                    plot_btn = gr.Button("生成对比图")
                    plot_output = gr.Plot()
                    plot_btn.click(self.metrics_tab, [], plot_output)
                else:
                    gr.Markdown("*请先运行实验脚本获取指标数据*")

            with gr.Tab("❄️ 冷启动测试"):
                gr.Markdown("模拟新用户场景：输入 1-3 个已交互物品ID，系统自动判断冷启动等级并生成推荐。")
                with gr.Row():
                    i1 = gr.Textbox(label="已交互物品1")
                    i2 = gr.Textbox(label="已交互物品2")
                    i3 = gr.Textbox(label="已交互物品3")
                cold_btn = gr.Button("测试冷启动推荐")
                cold_output = gr.Markdown()
                cold_btn.click(self.cold_start_tab, [i1, i2, i3], cold_output)

        return demo


def create_app(pipeline=None, cs_handler=None, metrics_data=None, item_names=None, train_data=None):
    app = GradioApp(pipeline, cs_handler, metrics_data, item_names, train_data)
    return app.build()
