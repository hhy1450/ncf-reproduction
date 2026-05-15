# NCF Reproduction

A PyTorch reproduction of **Neural Collaborative Filtering** (He et al., WWW 2017) on the MovieLens-1M dataset.

## Models

| Model | Description |
|-------|-------------|
| **GMF** | Generalized Matrix Factorization — element-wise product of user/item embeddings |
| **MLP** | Multi-Layer Perceptron — concatenated embeddings through stacked linear layers |
| **NeuMF** | Neural Matrix Factorization — fusion of GMF and MLP branches, with optional pre-training |

## Quick Start

```bash
pip install -r requirements.txt
```

The MovieLens-1M dataset is downloaded automatically on first run.

## Usage

```bash
# Train GMF
python main.py --model GMF

# Train MLP
python main.py --model MLP

# Train NeuMF (random initialization)
python main.py --model NeuMF

# Train NeuMF with pre-trained GMF + MLP
python main.py --model NeuMF --pretrain
```

### Key Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `GMF` | Model: `GMF`, `MLP`, or `NeuMF` |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `256` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--gmf_dim` | `8` | GMF embedding dimension |
| `--mlp_dim` | `8` | MLP embedding dimension |
| `--mlp_layers` | `32 16 8` | MLP hidden layer sizes (tower structure) |
| `--pretrain` | `False` | Pre-train GMF and MLP before NeuMF |
| `--alpha` | `0.5` | Fusion weight for pre-trained embeddings |
| `--eval_k` | `10` | Top-K cutoff for HR and NDCG |
| `--patience` | `5` | Early stopping patience |

## Evaluation

Metrics reported: **HR@K** (Hit Ratio) and **NDCG@K** (Normalized Discounted Cumulative Gain).

Leave-one-out split: latest interaction per user → test, second-latest → validation, rest → train. Negative sampling (4:1 ratio) for training; 99 random negatives per user for evaluation.

## Reference

> He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural Collaborative Filtering. *WWW 2017*.
