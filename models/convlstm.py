"""
Spatiotemporal ConvLSTM Cell and Encoder-Decoder Architecture
"""
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur

class CrossAttentionFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, visual_feat, thermo_feat):
        B, C, H, W = visual_feat.shape
        proj_q = self.query_conv(visual_feat).view(B, -1, H * W).permute(0, 2, 1)
        proj_k = self.key_conv(thermo_feat).view(B, -1, H * W)
        energy = torch.bmm(proj_q, proj_k)
        attention = torch.softmax(energy, dim=-1)
        proj_v = self.value_conv(thermo_feat).view(B, -1, H * W)
        out = torch.bmm(proj_v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return visual_feat + self.gamma * out

class AeroCastBackbone(nn.Module):
    def __init__(self, in_channels: int = 10, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.cell1 = ConvLSTMCell(in_channels, hidden_dims[0])
        self.cell2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1])
        self.cell3 = ConvLSTMCell(hidden_dims[1], hidden_dims[2])
        self.downsample = nn.MaxPool2d(2, 2)
        self.cross_attn = CrossAttentionFusion(hidden_dims[2])
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec_conv = nn.Conv2d(hidden_dims[2], hidden_dims[0], kernel_size=3, padding=1)

    def forward(self, x_seq):
        # x_seq: [Batch, Time=12, Channels=10, H=256, W=256]
        B, T, C, H, W = x_seq.shape
        h1, c1 = torch.zeros(B, 64, H, W, device=x_seq.device), torch.zeros(B, 64, H, W, device=x_seq.device)
        h2, c2 = torch.zeros(B, 128, H // 2, W // 2, device=x_seq.device), torch.zeros(B, 128, H // 2, W // 2, device=x_seq.device)
        h3, c3 = torch.zeros(B, 256, H // 4, W // 4, device=x_seq.device), torch.zeros(B, 256, H // 4, W // 4, device=x_seq.device)

        for t in range(T):
            xt = x_seq[:, t]
            h1, c1 = self.cell1(xt, (h1, c1))
            h2, c2 = self.cell2(self.downsample(h1), (h2, c2))
            h3, c3 = self.cell3(self.downsample(h2), (h3, c3))

        # Bottleneck Cross-Attention
        h3_fused = self.cross_attn(h3, h3)
        # Decoder Upsampling
        dec = self.upsample(self.upsample(h3_fused))
        shared_features = torch.relu(self.dec_conv(dec))
        return shared_features
