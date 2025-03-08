import torch
import torch.nn as nn


class NeRFNetwork(nn.Module):
    """
    Implemented as in https://arxiv.org/abs/2003.08934
    NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
    """

    def __init__(self, hiddem_dim=256, embedding_dim_pos=10, embedding_dim_dir=4, dropout_rate=0.2):
        super().__init__()

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_dir

        alpha = 0.02

        self.blk1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hiddem_dim),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiddem_dim, hiddem_dim),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiddem_dim, hiddem_dim),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiddem_dim, hiddem_dim),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Dropout(p=dropout_rate),
        )

        self.blk2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3 + hiddem_dim, hiddem_dim),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiddem_dim, hiddem_dim),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiddem_dim, hiddem_dim),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiddem_dim, hiddem_dim),
        )

        self.blk3 = nn.Sequential(
            nn.Linear(embedding_dim_dir * 6 + 3 + hiddem_dim, hiddem_dim // 2), nn.LeakyReLU(negative_slope=alpha)
        )

        self.rgb_layer = nn.Sequential(nn.Linear(hiddem_dim // 2, 3))
        self.sig_layer = nn.Sequential(nn.Linear(hiddem_dim, 1))

    def init_weights(self):
        """
        This is based on the Glorot and Bengio method
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, o, d):
        # Forward pass assumes that positional encoding was done out side of the model
        h = self.blk1(o)

        tmp = self.blk2(torch.cat((h, o), dim=1))
        sigma = self.sig_layer(tmp)

        h = self.blk3(torch.cat((h, d), dim=1))
        rgb = self.rgb_layer(h)

        return torch.cat([rgb, sigma], dim=1)
