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
                return "Pipeline not loaded."
            result = self.pipeline.recommend(user_id, top_n=top_k)
            items = result.get("items", [])
            if not items:
                return "No recommendations found."
            lines = [f"### Top-{top_k} Recommendations for User {user_id}\n"]
            for rank, iid in enumerate(items, 1):
                lines.append(f"{rank}. Item **{iid}**")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

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
                if i and str(i).strip():
                    interactions.append(int(str(i).strip()))
            if not interactions:
                return "Please enter at least one item ID."
            if self.cs_handler is None:
                return "Cold-start handler not loaded."
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
        with gr.Blocks(title="Recommendation System Demo") as demo:
            gr.Markdown("# Personalized Recommendation System")
            gr.Markdown("Two-stage architecture: Recall -> Ranking -> Top-N")

            with gr.Tab("Top-N Recommendation"):
                gr.Markdown("Enter user ID to get recommendations")
                with gr.Row():
                    user_input = gr.Textbox(label="User ID", value="0")
                    k_input = gr.Slider(label="Top-K", minimum=5, maximum=50, value=10, step=5)
                btn = gr.Button("Get Recommendations", variant="primary")
                output = gr.Markdown()
                btn.click(self.recommend_tab, [user_input, k_input], output)

            with gr.Tab("Model Comparison"):
                gr.Markdown("Model performance comparison")
                if self.metrics:
                    plot_btn = gr.Button("Generate Chart")
                    plot_output = gr.Plot()
                    plot_btn.click(self.metrics_tab, [], plot_output)
                else:
                    gr.Markdown("*Run experiments first to see metrics*")

            with gr.Tab("Cold Start Test"):
                gr.Markdown("Simulate new user: enter a few item IDs")
                with gr.Row():
                    i1 = gr.Textbox(label="Item 1")
                    i2 = gr.Textbox(label="Item 2")
                    i3 = gr.Textbox(label="Item 3")
                cold_btn = gr.Button("Test Cold-Start")
                cold_output = gr.Markdown()
                cold_btn.click(self.cold_start_tab, [i1, i2, i3], cold_output)

        return demo


def create_app(pipeline=None, cs_handler=None, metrics_data=None):
    app = GradioApp(pipeline, cs_handler, metrics_data)
    return app.build()
