import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import math

# 尝试导入核心算子
try:
    from .ss2d import SS2D
    from .csms6s import CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4
    from .csms6s import CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4
except ImportError:
    from ss2d import SS2D
    from csms6s import CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4
    from csms6s import CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class FourBranchSSAGate(nn.Module):

    def __init__(self, branch_dim):
        super().__init__()
        self.branch_dim = branch_dim
        self.total_dim = branch_dim * 4

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.gate = nn.Conv2d(
            in_channels=self.total_dim,
            out_channels=self.total_dim,
            kernel_size=1,
            groups=branch_dim,
            bias=True
        )
        self.act = nn.Sigmoid()

    def channel_shuffle(self, x):

        B, C, H, W = x.shape
        assert C == self.total_dim, f"expected {self.total_dim}, got {C}"

        x = x.view(B, 4, self.branch_dim, H, W)          # [B, 4, Cb, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()        # [B, Cb, 4, H, W]
        x = x.view(B, C, H, W)                           # [B, 4*Cb, H, W]
        return x

    def channel_unshuffle(self, x):

        B, C, H, W = x.shape
        assert C == self.total_dim, f"expected {self.total_dim}, got {C}"

        x = x.view(B, self.branch_dim, 4, H, W)          # [B, Cb, 4, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()        # [B, 4, Cb, H, W]
        x = x.view(B, C, H, W)                           # [B, 4*Cb, H, W]
        return x

    def forward(self, x1, x2, x3, x4):

        x_cat = torch.cat([x1, x2, x3, x4], dim=-1)      # [B, H, W, 4Cb]
        feat = x_cat.permute(0, 3, 1, 2).contiguous()    # [B, 4Cb, H, W]

        feat_shuf = self.channel_shuffle(feat)

        w = self.pool(feat_shuf)                         # [B, 4Cb, 1, 1]
        w = self.gate(w)                                # [B, 4Cb, 1, 1]
        w = self.act(w)

        w = self.channel_unshuffle(w)                   # [B, 4Cb, 1, 1]
        gated = feat * w
        out = gated.permute(0, 2, 3, 1).contiguous()                     # [B, H, W, 4Cb]
        return out

class GAMMBlock(nn.Module):

    def __init__(self, input_dim, output_dim, d_state=16, d_conv=3, expand=2):
        super().__init__()

        self.branch_dim = input_dim // 4

        self.in_norm = nn.LayerNorm(input_dim)
        self.out_norm = nn.LayerNorm(input_dim)
        self.mamba_g1 = SS2D(d_model=input_dim // 4, d_state=d_state, ssm_ratio=expand, d_conv=d_conv,depth_gate=True,
            depth_gate_b_hidden=32,
            depth_gate_c_hidden=32,
            depth_gate_b_gmin=0.1,
            depth_gate_c_gmin=0.1,
            )
        self.branch_fuse = FourBranchSSAGate(
            branch_dim=self.branch_dim
        )
        self.fuse_alpha = nn.Parameter(torch.zeros(1))
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    def forward(self, x, depth):
        if x.dtype == torch.float16: x = x.type(torch.float32)
        B, C, H, W = x.shape

        if depth.shape[-2:] != (H, W):
            depth_rs = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)
        else:
            depth_rs = depth
        d_mean = depth_rs.mean(dim=(2, 3), keepdim=True)
        d_std = depth_rs.std(dim=(2, 3), keepdim=True) + 1e-6
        depth_normalized = (depth_rs - d_mean) / d_std

        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.in_norm(x_perm)


        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=-1)
        x_mamba1 = self.mamba_g1(x1, CrossScan=CrossScan_1, CrossMerge=CrossMerge_1, depth=depth_normalized)
        x_mamba2 = self.mamba_g1(x2, CrossScan=CrossScan_2, CrossMerge=CrossMerge_2, depth=depth_normalized)
        x_mamba3 = self.mamba_g1(x3, CrossScan=CrossScan_3, CrossMerge=CrossMerge_3, depth=depth_normalized)
        x_mamba4 = self.mamba_g1(x4, CrossScan=CrossScan_4, CrossMerge=CrossMerge_4, depth=depth_normalized)
        x_cat = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=-1)
        x_gate = self.branch_fuse(x_mamba1, x_mamba2, x_mamba3, x_mamba4)
        x_mamba = (1-self.fuse_alpha)*x_cat + self.fuse_alpha * x_gate

        x_mamba = self.out_norm(x_mamba)
        x_out = self.proj(x_mamba)
        return x_out.permute(0, 3, 1, 2)

class GAMM(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.layer = GAMMBlock(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, depth):
        return x + self.drop_path(self.layer(x, depth))

class GAMM_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, hidden_dim=32, c_list=[64,128,348,448], drop_path_rate=0.2):
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 7, 2, 3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.stem_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, c_list[0], 3, 2, 1, bias=False),
            LayerNorm2d(c_list[0])
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]

        # --- Encoder Layers ---
        self.encoder1 = GAMM(c_list[0], drop_path=dpr[0])
        self.encoder2 = GAMM(c_list[1], drop_path=dpr[1])
        self.encoder3 = GAMM(c_list[2], drop_path=dpr[2])
        self.encoder4 = GAMM(c_list[3], drop_path=dpr[3])

        # --- Downsample Layers ---
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c_list[0], c_list[1], 1),
            LayerNorm2d(c_list[1])
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c_list[1], c_list[2], 1),
            LayerNorm2d(c_list[2])
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c_list[2], c_list[3], 1),
            LayerNorm2d(c_list[3])
        )

        # --- Decoder Layers ---
        self.decoder1 = GAMM(c_list[3], drop_path=dpr[3])
        self.decoder2 = GAMM(c_list[2], drop_path=dpr[2])
        self.decoder3 = GAMM(c_list[1], drop_path=dpr[1])
        self.decoder4 = GAMM(c_list[0], drop_path=dpr[0])

        # --- Upsample Layers ---
        self.up1 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[2], 1),
            LayerNorm2d(c_list[2])
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 1),
            LayerNorm2d(c_list[1])
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 1),
            LayerNorm2d(c_list[0])
        )

        self.final_conv = nn.Conv2d(c_list[0], num_classes, 1)

        # --- Norms ---
        self.ebn1 = LayerNorm2d(c_list[0])
        self.ebn2 = LayerNorm2d(c_list[1])
        self.ebn3 = LayerNorm2d(c_list[2])
        self.ebn4 = LayerNorm2d(c_list[3])

        self.dbn1 = LayerNorm2d(c_list[3])
        self.dbn2 = LayerNorm2d(c_list[2])
        self.dbn3 = LayerNorm2d(c_list[1])
        self.dbn4 = LayerNorm2d(c_list[0])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, depth):
        # 1. Stem
        x = self.stem(x)
        x = self.stem_proj(x)
        #x = self.norm(x)
        # --- Encoder ---
        t1 = self.ebn1(self.encoder1(x, depth)) # (B, c0, H/4, W/4)
        x = self.down1(t1)               # (B, c1, H/8, W/8)

        t2 = self.ebn2(self.encoder2(x, depth)) # (B, c1, H/8, W/8)
        x = self.down2(t2)               # (B, c2, H/16, W/16)

        t3 = self.ebn3(self.encoder3(x, depth)) # (B, c2, H/16, W/16)
        x = self.down3(t3)               # (B, c3, H/32, W/32)

        # Bottleneck
        out = self.ebn4(self.encoder4(x, depth))# (B, c3, H/32, W/32)
        t4 = out
        # --- Decoder ---

        # Up 1
        out = self.dbn1(self.decoder1(out, depth)) # (B, c3, H/32, W/32)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.up1(out)
        out = out + t3

        # Up 2
        out = self.dbn2(self.decoder2(out, depth))
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.up2(out)
        out = out + t2


        # Up 3
        out = self.dbn3(self.decoder3(out, depth))
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.up3(out)
        out = out + t1
        # Final
        out = self.dbn4(self.decoder4(out, depth))
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.final_conv(out)

        if self.num_classes == 1: return torch.sigmoid(out)
        else: return out