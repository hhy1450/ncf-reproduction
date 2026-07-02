"""
Streamlit UI for the recommendation system.
Clean, professional design with Chinese localization.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pipeline import RecommendationPipeline
from cold_start import ColdStartHandler

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="个性化推荐系统",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ───────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* Base */
    .stApp { background-color: #f8f9fa; }

    /* Headers */
    h1, h2, h3, h4 {
        color: #111827 !important;
        font-weight: 600 !important;
    }

    /* Cards */
    .rec-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
        transition: border-color 0.15s;
    }
    .rec-card:hover { border-color: #93c5fd; }

    .rec-rank {
        font-size: 22px;
        font-weight: 700;
        color: #1a56db;
    }
    .rec-rank-other {
        font-size: 22px;
        font-weight: 700;
        color: #9ca3af;
    }
    .rec-title {
        font-size: 15px;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 4px;
    }
    .rec-source {
        display: inline-block;
        font-size: 11px;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 3px;
        margin-right: 4px;
        margin-top: 4px;
    }
    .src-itemcf  { background: #dbeafe; color: #1e40af; }
    .src-svd     { background: #dcfce7; color: #166534; }
    .src-nmf     { background: #fef3c7; color: #92400e; }
    .src-default { background: #f3f4f6; color: #6b7280; }

    /* Strategy badge */
    .strategy-tag {
        display: inline-block;
        background: #eff6ff;
        color: #1e40af;
        font-size: 13px;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 4px;
        border: 1px solid #bfdbfe;
        margin-bottom: 12px;
    }

    /* History tags */
    .hist-tag {
        display: inline-block;
        background: #f3f4f6;
        color: #4b5563;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 2px 4px 2px 0;
    }

    /* Metric display */
    .metric-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a56db;
    }
    .metric-label {
        font-size: 13px;
        color: #6b7280;
        margin-top: 4px;
    }

    /* Section divider */
    .section-divider {
        height: 1px;
        background: #e5e7eb;
        margin: 24px 0;
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebarNav"] { display: none; }
    </style>
    """, unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────

SOURCE_CLASS = {
    "ItemCF": "src-itemcf",
    "SVD": "src-svd",
    "NMF": "src-nmf",
}


def _source_tag(name):
    cls = SOURCE_CLASS.get(name, "src-default")
    return f'<span class="rec-source {cls}">{name}</span>'


def _render_rec_card(rank, item_name, sources):
    rank_class = "rec-rank" if rank <= 3 else "rec-rank-other"
    tags = "".join(_source_tag(s) for s in sources) if sources else _source_tag("综合")
    return f"""
    <div class="rec-card">
        <span class="{rank_class}">{rank}</span>
        <span class="rec-title" style="margin-left:12px;">{item_name}</span>
        <div style="margin-top:4px;">{tags}</div>
    </div>"""


# ── Session init ─────────────────────────────────────────────

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_state.pkl")


def load_app_state():
    """Load pre-built pipeline state from pickle file."""
    import pickle
    if not os.path.exists(STATE_PATH):
        return None
    with open(STATE_PATH, "rb") as f:
        return pickle.load(f)


def init_session():
    defaults = {
        "pipeline": None,
        "cs_handler": None,
        "item_names": {},
        "train_data": {},
        "metrics_data": {},
        "num_users": 0,
        "num_items": 0,
        "ranker_configs": {},
        "ranker_default": "Wide&Deep",
        "active_ranker_name": "Wide&Deep",
        "app_ready": False,
        "state_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Load state from disk on first run
    if not st.session_state.state_loaded:
        state = load_app_state()
        if state is not None:
            st.session_state.pipeline = state["pipeline"]
            st.session_state.cs_handler = state["cs_handler"]
            st.session_state.item_names = state["item_names"]
            st.session_state.train_data = state["train_data"]
            st.session_state.num_users = state["num_users"]
            st.session_state.num_items = state["num_items"]

            # Normalise: old format had "ranker_config" (singular)
            configs = state.get("ranker_configs") or {}
            if not configs and state.get("ranker_config"):
                cfg = state["ranker_config"]
                configs = {cfg["model_name"]: cfg}
            st.session_state.ranker_configs = configs

            default_name = state.get("ranker_default") or \
                next(iter(configs.keys()), "Wide&Deep") if configs else "Wide&Deep"
            st.session_state.ranker_default = default_name
            st.session_state.active_ranker_name = default_name

            # Auto-load default ranker (pipeline was saved without one)
            switch_ranker(default_name, silent=True)

            st.session_state.app_ready = True
        st.session_state.state_loaded = True


def switch_ranker(model_name, silent=False):
    """Load a different ranking model and attach it to the pipeline."""
    import torch
    from rank.deep_ranker import build_ranker, DeepRanker

    # If already active and not forced, skip
    if model_name == st.session_state.active_ranker_name:
        pipeline = st.session_state.pipeline
        if pipeline is not None and pipeline.ranker is not None:
            return  # already loaded

    cfg = st.session_state.ranker_configs.get(model_name, {})
    if not cfg:
        if not silent:
            st.warning(f"模型 {model_name} 的配置不存在。")
        return

    weight_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f"ranker_{model_name}.pt"
    )
    # Fall back to old naming (single model file)
    if not os.path.exists(weight_path):
        old_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ranker_weights.pt"
        )
        if os.path.exists(old_path):
            weight_path = old_path

    if not os.path.exists(weight_path):
        if not silent:
            st.warning(f"模型 {model_name} 的权重文件不存在。")
        return

    try:
        model = build_ranker(model_name, cfg["num_users"],
                            cfg["num_items"], device=cfg.get("device", "cpu"))
        model.load_state_dict(torch.load(weight_path,
                                         map_location=cfg.get("device", "cpu")))
        st.session_state.pipeline.ranker = DeepRanker(
            model, name=model_name, device=cfg.get("device", "cpu")
        )
        st.session_state.active_ranker_name = model_name
    except Exception as e:
        if not silent:
            st.warning(f"模型 {model_name} 加载失败: {e}")


# ── Pages ────────────────────────────────────────────────────

def page_recommend():
    """Page 1: Recommendation query."""
    st.header("Top-N 个性化推荐")

    st.markdown(
        '<p style="color:#6b7280;font-size:14px;">'
        '三路召回（ItemCF + SVD + NMF）后由深度学习模型精排，输出最终推荐结果。'
        '</p>',
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 查询设置")

        # Model selector
        available = list(st.session_state.ranker_configs.keys())
        if available:
            current_idx = available.index(st.session_state.active_ranker_name) \
                if st.session_state.active_ranker_name in available else 0
            selected = st.selectbox(
                "排序模型", available, index=current_idx,
                on_change=None,
            )
            if selected != st.session_state.active_ranker_name:
                switch_ranker(selected)
                st.rerun()

        user_id = st.number_input(
            "用户 ID", min_value=0,
            max_value=st.session_state.num_users - 1,
            value=0, step=1,
        )
        top_k = st.slider("推荐数量", min_value=5, max_value=50, value=10, step=5)
        go_btn = st.button("开始推荐", type="primary", use_container_width=True)

    if not go_btn:
        st.info("请在左侧边栏设置用户 ID 并点击「开始推荐」按钮。")
        return

    pipeline = st.session_state.pipeline
    if pipeline is None:
        st.warning("推荐管道未加载，请检查系统初始化。")
        return

    # User history
    user_history = st.session_state.train_data.get(user_id, [])
    if user_history:
        st.markdown("#### 该用户的观影记录（最近 10 部）")
        tags = "".join(
            f'<span class="hist-tag">{st.session_state._item_name(iid)}</span>'
            for iid in user_history[-10:]
        )
        st.markdown(tags, unsafe_allow_html=True)
    else:
        st.markdown("#### 该用户暂无历史记录（冷启动用户）")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Recommendations
    result = pipeline.recommend(user_id, top_n=top_k)
    items = result.get("items", [])
    sources = result.get("sources", {})

    if not items:
        st.warning("未找到推荐结果。")
        return

    ranker_name = st.session_state.active_ranker_name or "召回融合"
    st.markdown(f"#### 推荐结果（共 {len(items)} 部，排序模型: {ranker_name}）")

    for rank, iid in enumerate(items, 1):
        name = st.session_state._item_name(iid)
        src = sources.get(iid, [])
        st.markdown(_render_rec_card(rank, name, src), unsafe_allow_html=True)


def page_comparison():
    """Page 2: Model comparison charts."""
    st.header("模型性能对比")

    metrics = st.session_state.metrics_data
    if not metrics:
        st.info("暂无实验数据。请先运行实验脚本获取各模型指标。")
        return

    models = list(metrics.keys())
    hr_vals = [metrics[m].get("HR@10", 0) for m in models]
    ndcg_vals = [metrics[m].get("NDCG@10", 0) for m in models]

    # Color palette
    bar_color_hr = "#1a56db"
    bar_color_ndcg = "#059669"

    col1, col2 = st.columns(2)

    with col1:
        fig_hr = go.Figure(
            data=[go.Bar(
                x=models, y=hr_vals,
                marker_color=bar_color_hr,
                text=[f"{v:.4f}" for v in hr_vals],
                textposition="outside",
                textfont=dict(size=12, color="#374151"),
            )]
        )
        fig_hr.update_layout(
            title=dict(text="HR@10", font=dict(size=16, color="#111827")),
            xaxis=dict(tickangle=30, title=None),
            yaxis=dict(title=None, gridcolor="#f3f4f6"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=60, l=40, r=20),
            height=380,
        )
        st.plotly_chart(fig_hr, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fig_ndcg = go.Figure(
            data=[go.Bar(
                x=models, y=ndcg_vals,
                marker_color=bar_color_ndcg,
                text=[f"{v:.4f}" for v in ndcg_vals],
                textposition="outside",
                textfont=dict(size=12, color="#374151"),
            )]
        )
        fig_ndcg.update_layout(
            title=dict(text="NDCG@10", font=dict(size=16, color="#111827")),
            xaxis=dict(tickangle=30, title=None),
            yaxis=dict(title=None, gridcolor="#f3f4f6"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=60, l=40, r=20),
            height=380,
        )
        st.plotly_chart(fig_ndcg, use_container_width=True, config={"displayModeBar": False})

    # Data table
    st.markdown("#### 详细数据")
    import pandas as pd
    df = pd.DataFrame({
        "模型": models,
        "HR@10": [f"{v:.4f}" for v in hr_vals],
        "NDCG@10": [f"{v:.4f}" for v in ndcg_vals],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_cold_start():
    """Page 3: Cold-start test."""
    st.header("冷启动推荐测试")

    st.markdown(
        '<p style="color:#6b7280;font-size:14px;">'
        '模拟新用户场景：输入 1-3 个已交互的物品 ID，系统自动判断冷启动等级并生成推荐。'
        '</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        item1 = st.text_input("已交互物品 ID 1", key="cs_i1", placeholder="例如: 1193")
    with col2:
        item2 = st.text_input("已交互物品 ID 2", key="cs_i2", placeholder="可选")
    with col3:
        item3 = st.text_input("已交互物品 ID 3", key="cs_i3", placeholder="可选")

    go_btn = st.button("开始测试", type="primary")

    if not go_btn:
        return

    # Parse inputs
    interactions = []
    for raw in [item1, item2, item3]:
        v = raw.strip() if raw else ""
        if v:
            try:
                interactions.append(int(v))
            except ValueError:
                st.warning(f"'{v}' 不是有效的物品 ID，已跳过。")

    if not interactions:
        st.warning("请至少输入一个物品 ID。")
        return

    cs = st.session_state.cs_handler
    if cs is None:
        st.warning("冷启动处理器未加载。")
        return

    n = len(interactions)
    if n <= 1:
        recs = cs.popular_recommend(k=10, diversity_weight=0.5)
        strategy = "极冷启动（不超过 1 条交互）：全局热门推荐 + 高多样性注入"
    elif n <= 3:
        recs = cs.popular_recommend(k=10, diversity_weight=0.2)
        strategy = "冷启动（2-3 条交互）：热门推荐 + 低多样性注入"
    else:
        recs = cs.popular_recommend(k=10, diversity_weight=0.1)
        strategy = "温启动（4-5 条交互）：热门推荐 + 最小随机扰动"

    st.markdown(f'<span class="strategy-tag">策略：{strategy}</span>', unsafe_allow_html=True)

    # Input items
    st.markdown("**已输入物品**")
    input_tags = "".join(
        f'<span class="hist-tag">{st.session_state._item_name(iid)}</span>'
        for iid in interactions
    )
    st.markdown(input_tags, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Results
    st.markdown("#### 推荐结果")
    for rank, iid in enumerate(recs, 1):
        name = st.session_state._item_name(iid)
        st.markdown(_render_rec_card(rank, name, sources=None), unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────

def _item_name(iid):
    """Resolve item ID to display name."""
    name = st.session_state.item_names.get(iid)
    return f"{name} (ID: {iid})" if name else f"物品 {iid}"


def main():
    inject_css()
    init_session()

    # Bind the helper so session_state can use it (for inline rendering)
    st.session_state._item_name = _item_name

    # ── Sidebar header ──
    with st.sidebar:
        st.markdown("""
        <div style="padding:12px 0 8px 0;">
            <span style="font-size:20px;font-weight:700;color:#111827;">个性化推荐系统</span>
        </div>
        <p style="color:#6b7280;font-size:13px;margin-bottom:20px;">
            两阶段架构：多路召回 + 融合排序
        </p>
        """, unsafe_allow_html=True)

    # ── Navigation ──
    page = st.sidebar.radio(
        "导航",
        ["Top-N 推荐", "模型对比", "冷启动测试"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # System info in sidebar
    if st.session_state.app_ready:
        ranker_models = " / ".join(st.session_state.ranker_configs.keys()) or "无"
        st.sidebar.markdown(f"""
        <div style="font-size:12px;color:#9ca3af;">
            用户数：{st.session_state.num_users}<br>
            物品数：{st.session_state.num_items}<br>
            召回：ItemCF / SVD / NMF<br>
            排序：{ranker_models}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("系统未初始化")
        st.sidebar.info("请先运行 `python launch.py` 来训练模型并构建推荐管道。")

    # ── Main content ──
    if not st.session_state.app_ready:
        st.title("个性化推荐系统")
        st.warning("系统尚未初始化，未找到已训练的推荐管道。")
        st.markdown("""
        请先执行以下命令来初始化系统：

        ```bash
        python launch.py
        ```

        该命令将自动完成以下步骤：
        1. 加载 MovieLens-1M 数据集
        2. 训练 ItemCF、SVD、NMF 三种召回模型
        3. 构建推荐管道并保存状态

        完成后刷新本页面即可使用。
        """)
        return

    if page == "Top-N 推荐":
        page_recommend()
    elif page == "模型对比":
        page_comparison()
    else:
        page_cold_start()


if __name__ == "__main__":
    main()
