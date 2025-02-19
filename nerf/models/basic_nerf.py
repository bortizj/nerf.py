import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRFNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4]):
        super().__init__()
        self.D = D  # Depth of the network
        self.W = W  # Width of the network
        self.input_ch = input_ch  # Number of input channels for position
        self.input_ch_views = input_ch_views  # Number of input channels for view direction
        self.output_ch = output_ch  # Number of output channels (RGBA)
        self.skips = skips  # Skip connections

        # These layers learn to extract features from the 3D coordinates.
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)])
        for ii in range(D - 1):
            if ii in skips:
                self.pts_linears.append(nn.Linear(W + input_ch, W))
            else:
                self.pts_linears.append(nn.Linear(W, W))

        # These layers learn extract features from the viewing direction
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        for ii in range(D // 2):
            self.views_linears.append(nn.Linear(W // 2, W // 2))

        self.feature_linear = nn.Linear(W, W)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # These layers learn to extract features from the 3D coordinates.
        pts_ = input_pts
        for ii, layer in enumerate(self.pts_linears):
            if ii in self.skips:
                pts_ = torch.cat([pts_, input_pts], -1)
            pts_ = layer(pts_)
            pts_ = F.leaky_relu(pts_, negative_slope=0.2)

        feature = self.feature_linear(pts_)

        # These layers learn extract features from the viewing direction
        views_ = input_views
        for layer in self.views_linears:
            views_ = layer(torch.cat([views_, feature], -1))
            views_ = F.relu(views_)

        rgb = self.rgb_linear(views_)
        rgb = F.relu(rgb)

        return rgb
