import torch
import torch.nn as nn


class WideAndDeep(nn.Module):
    """Wide & Deep Learning for Recommender Systems.

    Wide part:  cross-product of user/item wide embeddings → memorization
    Deep part:  concatenated embeddings → MLP → generalization
    Combined:   σ(Wide_out + Deep_out + bias)
    """

    def __init__(self, num_users, num_items, embed_dim=32,
                 mlp_layers=None, wide_dim=32):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        # Deep part
        self.user_embed_deep = nn.Embedding(num_users, embed_dim)
        self.item_embed_deep = nn.Embedding(num_items, embed_dim)
        mlp_blocks = []
        in_dim = embed_dim * 2
        for out_dim in mlp_layers:
            mlp_blocks.append(nn.Linear(in_dim, out_dim))
            mlp_blocks.append(nn.ReLU())
            mlp_blocks.append(nn.Dropout(0.2))
            in_dim = out_dim
        mlp_blocks.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_blocks)

        # Wide part (cross-product of wide embeddings)
        self.user_embed_wide = nn.Embedding(num_users, wide_dim)
        self.item_embed_wide = nn.Embedding(num_items, wide_dim)
        self.wide_output = nn.Linear(wide_dim * 3, 1)

        # Combined output bias
        self.output_bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_embed_deep, self.item_embed_deep,
                     self.user_embed_wide, self.item_embed_wide]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.wide_output.weight)

    def forward(self, user, item):
        # Deep: concatenate embeddings → MLP
        u_deep = self.user_embed_deep(user)
        i_deep = self.item_embed_deep(item)
        deep_out = self.mlp(torch.cat([u_deep, i_deep], dim=-1))

        # Wide: cross-product of wide embeddings + individual embeddings
        u_wide = self.user_embed_wide(user)
        i_wide = self.item_embed_wide(item)
        wide_cross = torch.cat([u_wide * i_wide, u_wide, i_wide], dim=-1)
        wide_out = self.wide_output(wide_cross)

        return torch.sigmoid(deep_out + wide_out + self.output_bias).squeeze(-1)
