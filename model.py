import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPUPreprocessor(nn.Module):
    """
    在 GPU 上动态生成 HP, Lap, Grid 通道。
    输入: [B, 1, H, W]
    输出: [B, 4, H, W] (如果全开启)
    优点: 利用 GPU 并行计算能力，释放 CPU 压力，减少 PCIe 传输。
    """
    def __init__(self, add_hp=True, add_lap=True, add_grid=True, h=128, w=128):
        super().__init__()
        self.add_hp = add_hp
        self.add_lap = add_lap
        self.add_grid = add_grid
        
        # Laplacian Kernel
        self.register_buffer("lap_kernel", torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3))

        # Grid Prior
        self.register_buffer("grid", self._make_grid_prior(h, w))

    @staticmethod
    def _make_grid_prior(h: int, w: int) -> torch.Tensor:
        g = torch.zeros((1, h, w), dtype=torch.float32)
        g[:, 7::8, :] = 1.0
        g[:, :, 7::8] = 1.0
        g = g * 2.0 - 1.0
        return g.unsqueeze(0) # [1, 1, H, W]

    def forward(self, x):
        # x: [B, 1, H, W]
        chs = [x]
        
        if self.add_hp:
            # GPU Average Pooling
            blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            hp = x - blur
            chs.append(hp)

        if self.add_lap:
            # GPU Convolution
            lap = F.conv2d(x, self.lap_kernel, padding=1)
            lap = torch.tanh(lap * 0.5)
            chs.append(lap)

        if self.add_grid:
            B, _, H, W = x.shape
            # 动态扩展 Batch 维度
            if self.grid.shape[2:] == (H, W):
                grid_batch = self.grid.expand(B, -1, -1, -1)
            else:
                # 应对可能的尺寸变化
                grid_batch = self._make_grid_prior(H, W).to(x.device).expand(B, -1, -1, -1)
            chs.append(grid_batch)
            
        return torch.cat(chs, dim=1)

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, groups=1, act=True):
        super().__init__()
        layers = [
            nn.Conv2d(c1, c2, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        hidden = max(8, c // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, groups=1, base_width=64, dilation=1):
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = ConvBNAct(in_planes, width, k=1, s=1, p=0)
        # 3x3 conv with optional dilation
        self.conv2 = ConvBNAct(width, width, k=3, s=stride, p=dilation, groups=groups)
        if dilation > 1:
            self.conv2.block[0].dilation = (dilation, dilation)

        self.conv3 = ConvBNAct(width, planes * self.expansion, k=1, s=1, p=0, act=False)
        
        self.se = SEBlock(planes * self.expansion)
        self.act = nn.SiLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = ConvBNAct(
                in_planes, planes * self.expansion, k=1, s=stride, p=0, act=False
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)
        out += self.shortcut(x)
        out = self.act(out)
        return out

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, kernel_size=x.shape[-2:])
        return x.pow(1.0 / self.p)

class SpatialBackbone(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.in_planes = 64
        
        # Stem (Uses Convolution for downsampling, better than MaxPool)
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, 64, 3, 2, 1),   # 128 -> 64
            ConvBNAct(64, 64, 3, 1, 1),
            ConvBNAct(64, 128, 3, 1, 1),
        )
        self.in_planes = 128

        self.layer1 = self._make_layer(Bottleneck, 64,  3, stride=1) # 64
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2) # 64 -> 32
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2) # 32 -> 16
        
        # Layer 4: Stride=1, Dilation=2.
        # This is optimal for QF detection (preserves 16x16 resolution).
        # Do not change to stride=2.
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1, dilation=2)
        
        self.pool = GeM()

    def _make_layer(self, block, planes, num_blocks, stride, dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s, dilation=dilation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return x  # [B, 2048]

class DCT8x8(nn.Module):
    def __init__(self):
        super().__init__()
        N = 8
        mat = torch.zeros(N, N)
        for u in range(N):
            for x in range(N):
                val = math.cos((2 * x + 1) * u * math.pi / (2 * N))
                if u == 0:
                    val *= math.sqrt(1 / N)
                else:
                    val *= math.sqrt(2 / N)
                mat[u, x] = val
        self.register_buffer("dct_mat", mat)

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 8 != 0 or W % 8 != 0:
            x = F.pad(x, (0, (8 - W % 8) % 8, 0, (8 - H % 8) % 8))
            H, W = x.shape[2:]

        blocks = x.unfold(2, 8, 8).unfold(3, 8, 8)
        h_blocks, w_blocks = blocks.shape[2], blocks.shape[3]
        
        blocks = blocks.contiguous().view(-1, 8, 8)
        
        dct_coeffs = torch.matmul(self.dct_mat, blocks)
        dct_coeffs = torch.matmul(dct_coeffs, self.dct_mat.t())
        
        dct_coeffs = dct_coeffs.view(B, C, h_blocks, w_blocks, 64)
        dct_coeffs = dct_coeffs.permute(0, 1, 4, 2, 3).contiguous()
        dct_coeffs = dct_coeffs.view(B, 64, h_blocks, w_blocks)
        return dct_coeffs

class FreqBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.dct = DCT8x8()
        self.net = nn.Sequential(
            ConvBNAct(64, 128, 3, 1, 1),
            ConvBNAct(128, 128, 3, 1, 1),
            ConvBNAct(128, 256, 3, 2, 1),
            ConvBNAct(256, 256, 3, 1, 1),
            ConvBNAct(256, 256, 3, 1, 1),
            ConvBNAct(256, 512, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x1: torch.Tensor):
        feat = self.dct(x1) # [B, 64, 16, 16]
        feat = torch.log1p(torch.abs(feat))
        return self.net(feat).flatten(1)  # [B, 512]

class JPEGQFRegNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 4, 
        dropout: float = 0.3,
        out_temp: float = 2.0,
        sigma_min: float = 0.15,
        sigma_max: float = 8.0,
        # Preprocessor toggles
        add_hp: bool = True,
        add_lap: bool = True,
        add_grid: bool = True,
    ):
        super().__init__()
        
        # GPU Preprocessing
        self.preprocessor = GPUPreprocessor(add_hp, add_lap, add_grid)
        
        self.spatial = SpatialBackbone(in_ch=in_ch)
        self.freq = FreqBackbone()

        fusion_dim = 2048 + 512
        
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
        )

        self.mu_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.out_temp = float(out_temp)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def _bounded_qf(self, raw_mu: torch.Tensor) -> torch.Tensor:
        return 1.0 + 98.0 * torch.sigmoid(raw_mu / self.out_temp)

    def forward(self, x, return_dict: bool = True):
        # x input is [B, 1, 128, 128] (Single Channel from DataLoader)
        
        # Expand channels on GPU
        x_processed = self.preprocessor(x) # -> [B, 4, 128, 128]
        
        y = x_processed[:, :1]
        fs = self.spatial(x_processed)
        ff = self.freq(y)
        feat = self.fuse(torch.cat([fs, ff], dim=1))

        raw_mu = self.mu_head(feat).squeeze(1)
        pred_qf = self._bounded_qf(raw_mu)

        raw_sigma = self.sigma_head(feat).squeeze(1)
        pred_sigma = self.sigma_min + F.softplus(raw_sigma)
        pred_sigma = torch.clamp(pred_sigma, self.sigma_min, self.sigma_max)

        if return_dict:
            return {
                "pred_qf": pred_qf,
                "pred_sigma": pred_sigma,
                "raw_mu": raw_mu,
            }
        return pred_qf, pred_sigma

class JPEGQFHybridNet(JPEGQFRegNet):
    pass