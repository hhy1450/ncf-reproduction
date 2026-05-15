# NCF 复现

**Neural Collaborative Filtering** (He et al., WWW 2017) 论文复现，基于 PyTorch，使用 MovieLens-1M 数据集。

## 模型简介

| 模型 | 说明 |
|------|------|
| **GMF** | 广义矩阵分解 — 用户/物品嵌入的逐元素乘积 |
| **MLP** | 多层感知机 — 拼接嵌入后经过多层全连接网络 |
| **NeuMF** | 神经矩阵分解 — GMF 与 MLP 分支融合，支持预训练 |

## 快速开始

```bash
pip install -r requirements.txt
```

首次运行时会自动下载 MovieLens-1M 数据集。

## 使用方法

```bash
# 训练 GMF
python main.py --model GMF

# 训练 MLP
python main.py --model MLP

# 训练 NeuMF（随机初始化）
python main.py --model NeuMF

# 训练 NeuMF（使用预训练的 GMF + MLP）
python main.py --model NeuMF --pretrain
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `GMF` | 模型选择：`GMF`、`MLP`、`NeuMF` |
| `--epochs` | `20` | 训练轮数 |
| `--batch_size` | `256` | 批次大小 |
| `--lr` | `0.001` | 学习率 |
| `--gmf_dim` | `8` | GMF 嵌入维度 |
| `--mlp_dim` | `8` | MLP 嵌入维度 |
| `--mlp_layers` | `32 16 8` | MLP 隐藏层大小（塔式结构） |
| `--pretrain` | `False` | 是否预训练 GMF 和 MLP |
| `--alpha` | `0.5` | 预训练嵌入融合权重 |
| `--eval_k` | `10` | Top-K 评估截断值 |
| `--patience` | `5` | 早停耐心值 |

## 评估指标

采用 **HR@K**（命中率）和 **NDCG@K**（归一化折损累计增益）。

数据划分采用 leave-one-out：每个用户最后一次交互作为测试集，倒数第二次作为验证集，其余作为训练集。训练时负采样比例为 4:1，评估时每个用户随机采样 99 个负样本。

## 参考文献

> He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural Collaborative Filtering. *WWW 2017*.
