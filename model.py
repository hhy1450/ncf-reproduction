import torch
import torch.nn as nn


class GMF(nn.Module):
    """Generalized Matrix Factorization.

    ŷ_ui = σ(h^T · (p_u ⊙ q_i))
    """

    def __init__(self, num_users, num_items, embed_dim=8):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.output = nn.Linear(embed_dim, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.kaiming_uniform_(self.output.weight, nonlinearity="sigmoid")

    def forward(self, user, item):
        u = self.user_embed(user)  # (B, embed_dim)
        i = self.item_embed(item)  # (B, embed_dim)
        x = u * i                   # element-wise product
        return torch.sigmoid(self.output(x)).squeeze(-1)


class MLP(nn.Module):
    """Multi-Layer Perceptron.

    ŷ_ui = σ(W_L^T · φ_{L-1}(... φ_1([p_u; q_i])))
    """

    def __init__(self, num_users, num_items, embed_dim=8, layers=None):
        super().__init__()
        if layers is None:
            layers = [32, 16, 8]  # tower structure
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        mlp_layers = []
        in_dim = embed_dim * 2  # concatenated user + item
        for out_dim in layers:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU())
            in_dim = out_dim
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, user, item):
        u = self.user_embed(user)   # (B, embed_dim)
        i = self.item_embed(item)   # (B, embed_dim)
        x = torch.cat([u, i], dim=-1)  # (B, embed_dim*2)
        x = self.mlp(x)
        return torch.sigmoid(x).squeeze(-1)


class NeuMF(nn.Module):
    """Neural Matrix Factorization = GMF + MLP fusion.

    With optional pre-training (α-controlled gradient scaling).
    """

    def __init__(self, num_users, num_items, gmf_dim=8, mlp_dim=8, mlp_layers=None):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [32, 16, 8]

        # GMF branch
        self.gmf_user_embed = nn.Embedding(num_users, gmf_dim)
        self.gmf_item_embed = nn.Embedding(num_items, gmf_dim)

        # MLP branch
        self.mlp_user_embed = nn.Embedding(num_users, mlp_dim)
        self.mlp_item_embed = nn.Embedding(num_items, mlp_dim)
        mlp_blocks = []
        in_dim = mlp_dim * 2
        for out_dim in mlp_layers:
            mlp_blocks.append(nn.Linear(in_dim, out_dim))
            mlp_blocks.append(nn.ReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_blocks)

        # Fusion
        self.output = nn.Linear(gmf_dim + mlp_layers[-1], 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gmf_user_embed.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embed.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embed.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embed.weight, std=0.01)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.output.weight, nonlinearity="sigmoid")

    def load_pretrained(self, gmf, mlp, alpha=0.5):
        """Load pre-trained GMF and MLP weights with α-weighted fusion."""
        self.gmf_user_embed.weight.data.copy_(gmf.user_embed.weight.data)
        self.gmf_item_embed.weight.data.copy_(gmf.item_embed.weight.data)
        self.mlp_user_embed.weight.data.copy_(mlp.user_embed.weight.data)
        self.mlp_item_embed.weight.data.copy_(mlp.item_embed.weight.data)

        # Weight fusion for the output layer: α * GMF + (1-α) * MLP
        pre_gmf_weight = gmf.output.weight.data
        pre_mlp_weight = mlp.mlp[-1].weight.data
        pre_gmf_bias = gmf.output.bias.data
        pre_mlp_bias = mlp.mlp[-1].bias.data

        self.output.weight.data.copy_(
            torch.cat([pre_gmf_weight * alpha, pre_mlp_weight * (1 - alpha)], dim=-1)
        )
        self.output.bias.data.copy_(pre_gmf_bias * alpha + pre_mlp_bias * (1 - alpha))

    def forward(self, user, item):
        gmf_u = self.gmf_user_embed(user)
        gmf_i = self.gmf_item_embed(item)
        mlp_u = self.mlp_user_embed(user)
        mlp_i = self.mlp_item_embed(item)

        gmf_out = gmf_u * gmf_i
        mlp_out = self.mlp(torch.cat([mlp_u, mlp_i], dim=-1))

        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return torch.sigmoid(self.output(fused)).squeeze(-1)
