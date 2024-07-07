import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


def backward_warp(img, flow):
    """
    Args:
        img: [B, C, H, W]
        flow: [B, 2, H, W]
    """
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FlowAttention(nn.Module):
    def __init__(self,
                 dim,
                 bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.scale = qk_scale or dim**-0.5

        self.norm1 = norm_layer(dim)
        # define the projection layer
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim + 2, dim + 2, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim+2, dim+2)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * 2)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

        self.grids = {}

    def generate_grid(self, B, H, W, normalize=True):
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W))
        if normalize:
            yy = yy / (H - 1)
            xx = xx / (W - 1)
        grid = torch.stack([xx, yy], dim=0)
        grid = grid[None].expand(B, -1, -1, -1)
        return grid

    def forward(self, x, tgt, return_attn=False):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        B, C, H, W = x.shape
        grid = self.grids.get(f"{H}_{W}")
        if grid is None:
            grid = self.generate_grid(B, H, W).to(x)
            # grid = self.generate_grid(B, H, W, normalize=False).to(x)
            self.grids[f"{H}_{W}"] = grid.clone()
        grid = grid.flatten(2).permute(0, 2, 1)

        x = x.flatten(2).permute(0, 2, 1)
        k = tgt.flatten(2).permute(0, 2, 1)
        v = torch.cat([tgt.flatten(2).permute(0, 2, 1), grid], dim=-1)

        x = self.norm1(x)
        shortcut = x

        q = self.q_proj(x)  # [B, N, C]
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)  # [B, H, N, N]

        x = attn @ v
        x = self.proj_drop(self.proj(x))  # .view(B, H, W, 2).permute(0, 3, 1, 2)

        # mlp
        flow = x[..., -2:] - grid
        x = x[..., :-2]
        x = shortcut + x  # [B, N, :-2], global warped features, [B, N, -2:]: correspondence
        x = x + self.mlp(self.norm2(x))

        x = torch.cat([x, flow], dim=-1)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, C+2, H, W]
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, act=nn.LeakyReLU(negative_slope=0.1), stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.act = act
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        identity = x

        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))

        out = out + identity
        # out = self.act(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channels=3, inner_channels=[16, 24, 32, 48], num_blocks=3) -> None:
        """
        inner_channels: [C0, C1, ..., Cn]
        """
        super().__init__()
        self.convs = nn.ModuleList()

        for i, channel in enumerate(inner_channels):
            if i == 0:
                self.convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, channel, 7, 1, 3),
                    # nn.LeakyReLU(0.1),
                    nn.PReLU(channel),
                    *[BasicBlock(channel, nn.PReLU(channel)) for _ in range(num_blocks)]
                ))
                in_channels = channel
                continue

            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, channel, 3, 2, 1),
                # nn.LeakyReLU(0.1),
                nn.PReLU(channel),
                *[BasicBlock(channel, nn.PReLU(channel)) for _ in range(num_blocks)]
            ))
            in_channels = channel

    def forward(self, x, return_ms_feats=True):
        out_feats = []
        for conv in self.convs:
            x = conv(x)
            out_feats.append(x)

        if return_ms_feats:
            return out_feats[::-1]

        return x


class InitialDecoderStage(nn.Module):
    def __init__(self,
                 in_channels,
                 inner_channels,
                 out_channels,
                 num_res_blocks=3,
                 num_frames=2) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.pred_feats_t = self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(inner_channels),
            BasicBlock(inner_channels, nn.PReLU(inner_channels)),
        )
        self.convblock = nn.Sequential(
            nn.Conv2d((inner_channels + 2) * num_frames, inner_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(inner_channels),
            *[BasicBlock(inner_channels, nn.PReLU(inner_channels)) for _ in range(3)],
        )
        self.flow_attn = FlowAttention(inner_channels)

        self.flow_ch = 2
        self.out_conv = nn.Conv2d(inner_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t_embedding=None, split=True):
        """
        Args
            x: [B, N, C, H, W], N is the number of frames
            t_embedding: [B, 1, 1, H, W]
            split: if True, split the out feature into flow_t and feats_t
        """
        B, N, C, H, W = x.shape
        identity = x
        x = x.flatten(1, 2)

        if t_embedding is not None:  # for initial flow prediction
            if t_embedding.dim() == 5:
                t_embedding = t_embedding[:, 0]
            if t_embedding.shape[-2:] != (H, W):
                t_embedding = F.interpolate(t_embedding, (H, W))
                t_embedding = t_embedding.reshape(B, 1, H, W)
            x = torch.cat([x, t_embedding], dim=1)

        feats_t = self.pred_feats_t(x)
        flow_t_feats_t = self.flow_attn(
            feats_t.unsqueeze(1).repeat(1, N, 1, 1, 1).flatten(0, 1),
            identity.flatten(0, 1)
        ).view(B, N, -1, H, W)
        x = self.convblock(flow_t_feats_t.flatten(1, 2))
        x = self.out_conv(x)
        if split:
            flow_t = x[:, :self.flow_ch*N]
            feats_t = x[:, self.flow_ch*N:]
            flow_t = flow_t.view(B, N, 2, H, W)
            feats_t = feats_t.view(B, 1, -1, H, W)
            return flow_t, feats_t
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, num_res_blocks, upsample=False) -> None:
        super().__init__()
        self.upsample = upsample
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.1),
            nn.PReLU(inner_channels),
            *[BasicBlock(inner_channels, nn.PReLU(inner_channels)) for _ in range(num_res_blocks)],
        )
        self.out_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(inner_channels, out_channels, kernel_size=3, stride=1, padding=1),
        ) if upsample else nn.Conv2d(inner_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, flow_t=None, feats_t=None, t_embedding=None, split=True):
        """
        Args
            x: [B, N, C, H, W], N is the number of frames
            t_embedding: [B, 1, 1, H, W]
            flow_t [optional]: [B, N, 2, H, W]
            feats_t [optional]: [B, 1, C, H, W], features corresponds to t_embedding
            split: if True, split the out feature into flow_t and feats_t
        """
        B, N, C, H, W = x.shape
        if flow_t is not None:
            x = torch.stack([
                backward_warp(x[:, i], flow_t[:, i]) for i in range(N)
            ], dim=1)
        x = x.flatten(1, 2)

        if t_embedding is not None:  # for initial flow prediction
            if t_embedding.dim() == 5:
                t_embedding = t_embedding[:, 0]
            if t_embedding.shape[-2:] != (H, W):
                t_embedding = F.interpolate(t_embedding, (H, W))
                t_embedding = t_embedding.reshape(B, 1, H, W)
            # t_embedding = t_embedding.expand(1, N, 1, 1, 1)
            x = torch.cat([x, t_embedding], dim=1)

        # except the first stage, we need to concat the features from the previous stage
        if feats_t is not None and flow_t is not None:
            x = torch.cat([feats_t.flatten(1, 2), x, flow_t.flatten(1, 2)], dim=1)

        x = self.convblock(x)
        x = self.out_conv(x)
        if split:
            flow_t = x[:, :2*N]
            feats_t = x[:, 2*N:]
            if self.upsample:
                H *= 2
                W *= 2
            flow_t = flow_t.view(B, N, 2, H, W)
            feats_t = feats_t.view(B, 1, -1, H, W)
            return flow_t, feats_t
        else:
            return x


class MultiFlowDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 inner_channels,
                 out_channels=3,
                 num_res_blocks=3,
                 num_frames=2,
                 num_flows=1,
                 with_mask=False) -> None:
        super().__init__()
        self.num_flows = num_flows
        self.flow_ch = 3 if with_mask else 2

        self.pred_multi_flow = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(inner_channels),
            BasicBlock(inner_channels, nn.PReLU(inner_channels)),
            nn.Conv2d(inner_channels, num_frames * num_flows * self.flow_ch, kernel_size=3, stride=1, padding=1)
        )
        self.fusion_conv = nn.Conv2d(num_flows * inner_channels, inner_channels, kernel_size=1, stride=1, padding=0)

        self.convblock = nn.Sequential(
            nn.Conv2d(inner_channels * (num_frames + 1) + num_frames * num_flows * self.flow_ch, inner_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(inner_channels),
            *[BasicBlock(inner_channels, nn.PReLU(inner_channels)) for _ in range(num_res_blocks)],
        )

        self.pred_img = nn.Conv2d(inner_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rs_imgs, flow_t=None, feats_t=None):
        """
        Args
            x: [B, N, C, H, W], N is the number of frames
            rs_imgs: [B, N, C, H, W]
            flow_t [optional]: [B, N, 2, H, W]
            feats_t [optional]: [B, 1, C, H, W], features corresponds to t_embedding
        """
        B, N, C, H, W = x.shape
        rs_feats = x
        if flow_t is not None:
            x = torch.stack([
                backward_warp(x[:, i], flow_t[:, i]) for i in range(N)
            ], dim=1)
        x = x.flatten(1, 2)

        # except the first stage, we need to concat the features from the previous stage
        if feats_t is not None and flow_t is not None:
            x = torch.cat([feats_t.flatten(1, 2), x, flow_t.flatten(1, 2)], dim=1)

        res_flows = self.pred_multi_flow(x)  # residule multiple flows [B, N*NUM_FLOW*FLOW_CH, H, w]
        flows = res_flows.reshape(B, N, self.num_flows, self.flow_ch, H, W)  # + flow_t.unsqueeze(2)
        # rs_imgs = torch.stack([rs_imgs] * self.num_flows, dim=2)  # [B, N, NUM_FLOWs, 3, H, W]
        # rs_warped = backward_warp(rs_imgs.flatten(0, 2), flows.flatten(0, 2)).reshape(B, -1, H, W)
        rs_feats = torch.stack([rs_feats] * self.num_flows, dim=2)  # [B, N, NUM_FLOWs, C, H, W]
        rs_warped = backward_warp(rs_feats.flatten(0, 2), flows.flatten(0, 2)).reshape(B*N, self.num_flows * C, H, W)
        rs_warped = self.fusion_conv(rs_warped).reshape(B, -1, H, W)
        if feats_t is not None:
            rs_warped = torch.cat([feats_t.flatten(1, 2), rs_warped, flows.flatten(1, 3)], dim=1)
        out_feats = self.convblock(rs_warped)

        out_img = self.pred_img(out_feats)
        return out_img


@ARCH_REGISTRY.register()
class DFRSC(nn.Module):
    def __init__(self,
                 in_channels=3,
                 inner_channels=[16, 24, 32],
                 out_channels=3,
                 num_frames=2,
                 num_flows=1,
                 num_blocks=3,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.out_channels = out_channels
        self.num_frames = num_frames

        self.encoder = Encoder(in_channels, inner_channels, num_blocks=num_blocks)

        self.pred_init_flow = InitialDecoderStage(
            inner_channels[-1] * num_frames + 1,
            inner_channels[-1],
            inner_channels[-1] + 2 * num_frames,
            5,
            num_frames=num_frames)

        self.decoder4 = DecoderStage(
            inner_channels[-1] * (num_frames + 1) + 2 * num_frames,
            inner_channels[-2],
            inner_channels[-2] + 2 * num_frames,
            num_blocks,
            upsample=True)
        self.decoder3 = DecoderStage(
            inner_channels[-2] * (num_frames + 1) + 2 * num_frames,
            inner_channels[-3],
            inner_channels[-3] + 2 * num_frames,
            num_blocks,
            upsample=True)
        self.decoder2 = DecoderStage(
            inner_channels[-3] * (num_frames + 1) + 2 * num_frames,
            inner_channels[-4],
            inner_channels[-4] + 2 * num_frames,
            num_blocks,
            upsample=True)
        self.decoder1 = DecoderStage(
            inner_channels[-4] * (num_frames + 1) + 2 * num_frames,
            inner_channels[-5],
            inner_channels[-5] + 2 * num_frames,
            num_blocks,
            upsample=True)

        self.multi_flow_decoder = MultiFlowDecoder(
            inner_channels[-5] * (num_frames + 1) + 2 * num_frames,
            inner_channels[-5],
            out_channels,
            num_blocks,
            num_frames=num_frames,
            num_flows=num_flows)

    def forward(self, x, time_map=None, **kwargs):
        """
        Args:
            x: [B, N, 3, H, W]
            time_map: [B, 1, 1, H, W]
        """
        B, N, C, H, W = x.shape
        if C > 3:
            time_map = x[:, :, -1:]
            x = x[:, :, :3]
        # encode the frames
        feats_4, feats_3, feats_2, feats_1, feats_0 = self.encoder(x.flatten(0, 1), return_ms_feats=True)
        feats_4 = feats_4.reshape(B, N, -1, H//16, W//16)
        feats_3 = feats_3.reshape(B, N, -1, H//8, W//8)
        feats_2 = feats_2.reshape(B, N, -1, H//4, W//4)
        feats_1 = feats_1.reshape(B, N, -1, H//2, W//2)
        feats_0 = feats_0.reshape(B, N, -1, H, W)

        # predict the inter-frame distortion flow
        flow_t_4, feats_t_4 = self.pred_init_flow(feats_4, t_embedding=time_map, split=True)
        flow_t_3, feats_t_3 = self.decoder4(feats_4, flow_t_4, feats_t_4, split=True)
        flow_t_2, feats_t_2 = self.decoder3(feats_3, flow_t_3, feats_t_3, split=True)
        flow_t_1, feats_t_1 = self.decoder2(feats_2, flow_t_2, feats_t_2, split=True)
        flow_t_0, feats_t_0 = self.decoder1(feats_1, flow_t_1, feats_t_1, split=True)
        # out_img = self.decoder0(feats_0, flow_t_0, feats_t_0, split=False)
        out_img = self.multi_flow_decoder(feats_0, x, flow_t_0, feats_t_0)

        return out_img, [flow_t_0, flow_t_1, flow_t_2, flow_t_3, flow_t_4]