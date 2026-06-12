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
                 metrics_data: dict = None):
        self.pipeline = pipeline
        self.cs_handler = cs_handler
        self.metrics = metrics_data or {}

    def recommend_tab(self, user_id, top_k):
        """Tab 1: Get recommendations for a user."""
        try:
            user_id = int(user_id)
            top_k = int(top_k)
            if self.pipeline is None:
                return "推荐管道未加载。"
            result = self.pipeline.recommend(user_id, top_n=top_k)
            items = result.get("items", [])
            if not items:
                return "未找到推荐结果。"
            lines = [f"### 为用户 {user_id} 推荐的 Top-{top_k} 物品\n"]
            for rank, iid in enumerate(items, 1):
                lines.append(f"{rank}. 物品 **{iid}**")
            return "\n".join(lines)
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
            elif n <= 3:
                recs = self.cs_handler.popular_recommend(k=10, diversity_weight=0.2)
            else:
                recs = self.cs_handler.popular_recommend(k=10, diversity_weight=0.1)
            return f"基于 {n} 条交互记录，推荐结果: {recs.tolist()}"
        except Exception as e:
            return f"出错了: {e}"

    def build(self):
        with gr.Blocks(title="个性化推荐系统演示") as demo:
            gr.Markdown("# 个性化推荐系统")
            gr.Markdown("两阶段架构：召回 → 排序 → Top-N 推荐")

            with gr.Tab("🎯 Top-N 推荐"):
                gr.Markdown("输入用户ID，获取个性化推荐结果")
                with gr.Row():
                    user_input = gr.Textbox(label="用户ID", value="0")
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
                gr.Markdown("模拟新用户场景：输入少量已交互物品ID")
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
