import torch
from torch import nn
import itertools


class ConcatLayer(nn.Module):
    def forward(self, x, y):
        return torch.cat((x, y), dim=1)
        
class InterleavedLayer(nn.Module):
    def __init__(self,x_channel,y_channel):
        super().__init__()
        self.x_channel=x_channel
        self.y_channel=y_channel
    def forward(self,x,y):
        # Tách các channel
        c1= x.size(1)
        c2=y.size(1)
        radio=c1//c2
        out1_chunks = torch.chunk(x, c1//radio, dim=1)
        out2_chunks = torch.chunk(y, c2, dim=1)

        # Xen kẽ các channel
        interleaved = []
        for i in range(c2):
            interleaved.append(out1_chunks[i])
            interleaved.append(out2_chunks[i])

        return torch.cat(interleaved, dim=1)

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1) # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x



class SqueezeExitationBlock(nn.Module):
    def __init__(self, in_channels: int):
        """Constructor for SqueezeExitationBlock.

        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(
            in_channels, in_channels // 4
        )  # divide by 4 is mentioned in the paper, 5.3. Large squeeze-and-excite
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(in_channels // 4, in_channels)
        self.act2 = nn.Hardsigmoid()

    def forward(self, x):
        """Forward pass for SqueezeExitationBlock."""

        identity = x

        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)

        x = identity * x[:, :, None, None]
        return x

class ConvNormActivationBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size:int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            bias: bool = False,
            activation: torch.nn = nn.Hardswish,
    ):
        """Constructs a block containing a convolution, batch normalization and activation layer

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (list): size of the convolutional kernel
            stride (int, optional): stride of the convolutional kernel. Defaults to 1.
            padding (int, optional): padding of the convolutional kernel. Defaults to 0.
            groups (int, optional): number of groups for depthwise seperable convolution. Defaults to 1.
            bias (bool, optional): whether to use bias. Defaults to False.
            activation (torch.nn, optional): activation function. Defaults to nn.Hardswish.
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation()

    def forward(self, x):
        """Perform forward pass."""

        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class InverseResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            expansion_size: int = 6,
            stride: int = 1,
            squeeze_exitation: bool = True,
            activation: nn.Module = nn.Hardswish,
    ):

        """Constructs a inverse residual block

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of the convolutional kernel
            expansion_size (int, optional): size of the expansion factor. Defaults to 6.
            stride (int, optional): stride of the convolutional kernel. Defaults to 1.
            squeeze_exitation (bool, optional): whether to add squeeze and exitation block or not. Defaults to True.
            activation (nn.Module, optional): activation function. Defaults to nn.Hardswish.
        """

        super().__init__()

        self.residual = in_channels == out_channels and stride == 1
        self.squeeze_exitation = squeeze_exitation

        self.conv1 = (
            ConvNormActivationBlock(
                in_channels, expansion_size, 1, activation=activation
            )
            if in_channels != expansion_size
            else nn.Identity()
        )  # If it's not the first layer, then we need to add a 1x1 convolutional layer to expand the number of channels
        self.depthwise_conv = ConvNormActivationBlock(
            expansion_size,
            expansion_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expansion_size,
            activation=activation,
        )
        if self.squeeze_exitation:
            self.se = SqueezeExitationBlock(expansion_size)

        self.conv2 = nn.Conv2d(expansion_size, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Perform forward pass."""

        identity = x

        x = self.conv1(x)
        x = self.depthwise_conv(x)

        if self.squeeze_exitation:
            x = self.se(x)
        x = self.conv2(x)
        x = self.norm(x)

        if self.residual:
            x = x + identity

        return x