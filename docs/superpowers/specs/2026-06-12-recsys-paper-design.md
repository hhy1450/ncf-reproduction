# 基于机器学习的个性化推荐系统 — 设计文档

> **日期**: 2026-06-12
> **状态**: 已确认
> **目标**: 在 ncf_reproduction 基础上扩展，完成推荐系统项目 + 小论文

---

## 1. 项目概述

### 1.1 目标

1. **项目代码**：构建完整的个性化推荐系统，包含召回层、排序层、冷启动策略、API 服务和 Gradio 界面
2. **小论文**：按模板格式撰写 3000-8000 字，基于真实实验数据

### 1.2 关键决策

| 决策项 | 选择 |
|--------|------|
| 项目方式 | 在 `ncf_reproduction` 上扩展 |
| 数据集 | 多数据集：MovieLens-1M + Last.fm + MIND |
| 模型范围 | CF + SVD + NCF (GMF/MLP/NeuMF) + Wide&Deep |
| 工作顺序 | 先代码后论文（方案 A：分层渐进式） |

---

## 2. 技术架构

### 2.1 四层架构

```
服务层  (Gradio UI + FastAPI)
   ↑
排序层  (NCF/Wide&Deep)
   ↑
召回层  (CF/SVD/NMF 多路召回)
   ↑
数据层  (多数据集加载/预处理)
```

### 2.2 推荐流程

```
用户 ID → 多路召回 (CF + SVD + NMF, 每路 Top-200)
       → 融合去重 (~500 候选)
       → 排序模型 (NCF/Wide&Deep 逐条打分)
       → Top-N 推荐 (Top 10/20)
```

---

## 3. 项目结构

```
ncf_reproduction/
├── data/                    # 多数据集存储
├── data_loader.py           # ✅ 已有 — 扩展多数据集支持
├── recall/                  # 🆕 召回层
│   ├── __init__.py
│   ├── cf.py               # UserCF / ItemCF 协同过滤
│   ├── svd.py              # SVD 矩阵分解
│   └── nmf.py              # NMF 非负矩阵分解
├── rank/                    # 🆕 排序层
│   ├── __init__.py
│   ├── model.py            # ✅ 已有 NCF (GMF/MLP/NeuMF)
│   └── wide_deep.py        # 🆕 Wide&Deep
├── evaluate.py             # ✅ 已有 — 扩展指标
├── train.py                # ✅ 已有
├── cold_start.py           # 🆕 冷启动策略
├── pipeline.py             # 🆕 两阶段推荐管道
├── api/                     # 🆕 服务层
│   ├── __init__.py
│   ├── server.py           # FastAPI 推荐服务
│   └── app.py              # Gradio 可视化界面
├── paper/                   # 🆕 论文目录
│   └── 小论文.md            # 论文草稿
└── main.py                  # ✅ 已有 — 重构为统一入口
```

---

## 4. 六阶段实施计划

### 阶段 1：数据层

- **目标**：多数据集统一加载接口
- **任务**：
  - BaseDataset 基类设计（download/load_raw 子类实现，split/neg_sample/cold_start_split 基类提供）
  - MovieLens-1M ✅ 已有，Last.fm 🆕，MIND 🆕
  - 数据格式统一：`(user_id, item_id, timestamp)`
  - 时间序列 leave-one-out 划分
  - 负采样：训练 4:1，评估 99:1
  - 困难负采样策略

### 阶段 2：召回层

- **目标**：多路召回策略
- **任务**：
  - UserCF / ItemCF 协同过滤（余弦相似度）
  - SVD 矩阵分解（PyTorch 实现）
  - NMF 非负矩阵分解（PyTorch 实现）
  - 多路召回融合策略（各路 Top-200 → 合并去重 ~500）
  - 统一召回接口 `RecallBase.recommend(user_id, k)`

### 阶段 3：排序层

- **目标**：深度学习排序模型
- **任务**：
  - NCF 系列 ✅ 已有（GMF/MLP/NeuMF）
  - Wide&Deep 🆕（Wide 线性交叉 + Deep MLP 嵌入，联合输出）
  - 统一排序接口 `BaseRanker.predict(user, items) → scores`
  - 复用 NCF 嵌入层代码

### 阶段 4：实验评估

- **目标**：全面指标评估 + 对比/消融/冷启动实验
- **评估指标**：Recall@K, NDCG@K, MAP, Diversity, Coverage, Novelty
- **三组实验**：
  1. 对比实验：传统方法 vs 深度方法，多数据集 × 多指标
  2. 消融实验：NeuMF 分支贡献、Wide&Deep Wide/Deep 贡献、负采样策略对比
  3. 冷启动 & 长尾：少交互用户分组评估、热门 vs 长尾物品分析

### 阶段 5：服务层

- **目标**：可演示的推荐系统
- **API 接口**（FastAPI）：
  - `GET /recommend/{user_id}?top_k=10` — 基础推荐
  - `POST /recommend/cold-start` — 冷启动推荐
  - `GET /models` — 模型信息
  - `GET /health` — 健康检查
- **Gradio 界面**（3 个 Tab）：
  - Tab 1：Top-N 推荐（选择用户/模型 → 推荐列表）
  - Tab 2：模型对比图表（雷达图/柱状图）
  - Tab 3：冷启动测试

### 阶段 6：论文撰写

- **目标**：按模板格式撰写，基于真实实验数据
- **论文结构**：
  1. 摘要 + 关键词（4 个）
  2. 引言（~800 字）
  3. 研究现状（~1000 字）
  4. 相关理论与技术（~1200 字）
  5. 本文方法 ⭐（~1500 字）— 两阶段架构、冷启动策略
  6. 实验与分析 ⭐（~1500 字）— 三组实验
  7. 结束语（~400 字）

---

## 5. 冷启动策略

按用户交互数分组评估：

| 分组 | 交互数 | 策略 |
|------|--------|------|
| 极冷 | 0-1 | 热门 + 多样性加权 |
| 冷 | 2-3 | 人口统计聚类 + 热门 |
| 温 | 4-5 | 轻量协同过滤 |
| 热 | 5+ | 完整推荐管道 |

---

## 6. 技术选型

| 层 | 技术 | 理由 |
|----|------|------|
| 深度学习框架 | PyTorch | NCF 已基于 PyTorch |
| 传统 ML | scikit-learn | SVD/NMF 参考实现 |
| API 服务 | FastAPI | 异步高性能、自动文档 |
| 可视化 | Gradio | 快速搭建 demo UI |
| 数据处理 | Pandas + NumPy | 已有依赖 |
| 模型存储 | PyTorch .pt | 原生格式 |

---

## 7. 论文亮点设计

- **两阶段架构**（召回+排序）为工程化亮点
- **传统 vs 深度**完整对比链条
- **冷启动专项实验**解实际问题
- **多数据集**增强泛化性论证
- **多样性/覆盖率**指标体现推荐质量思考

---

## 8. 待确定项（代码跑完后）

- 英文标题翻译
- 实验具体数字和表格
- 消融实验关键结论
- Wide&Deep 是否作为主要贡献点
- 论文格式最终排版
