# xresnet1d implementation, taken and adapted from https://github.com/helme/ecg_ptbxl_benchmarking

from enum import Enum

import torch

from .network import AdaptiveConcatPool1d, Flatten, bn_drop_lin, encoder_kwargs, init_weights
from .network import add_heads as _add_heads

NormType = Enum("NormType", "Batch BatchZero")


def batch_norm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    """BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."""
    return _get_norm("BatchNorm", nf, ndim, zero=norm_type == NormType.BatchZero, **kwargs)


def avg_pool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    """nn.AvgPool layer for `ndim`"""
    assert 1 <= ndim <= 3
    return getattr(torch.nn, f"AvgPool{ndim}d")(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)


def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    """Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."""
    assert 1 <= ndim <= 3
    bn = getattr(torch.nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0.0 if zero else 1.0)
    return bn


def _conv_func(ndim=2, transpose=False):
    """Return the proper conv `ndim` function, potentially `transposed`."""
    assert 1 <= ndim <= 3
    return getattr(torch.nn, f"Conv{'Transpose' if transpose else ''}{ndim}d")


class ConvLayer(torch.nn.Sequential):
    """Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."""

    def __init__(
        self,
        ni,
        nf,
        ks=3,
        stride=1,
        padding=None,
        bias=None,
        ndim=2,
        norm_type=NormType.Batch,
        bn_1st=True,
        act_cls=torch.nn.ReLU,
        transpose=False,
        xtra=None,
        **kwargs,
    ):
        if padding is None:
            padding = (ks - 1) // 2 if not transpose else 0
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        if bias is None:
            bias = not bn
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        layers = [conv]
        act_bn = []
        if act_cls is not None:
            act_bn.append(act_cls())
        if bn:
            act_bn.append(batch_norm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st:
            act_bn.reverse()
        layers += act_bn
        if xtra:
            layers.append(xtra)
        super().__init__(*layers)


class ResBlock(torch.nn.Module):
    """Resnet block from `ni` to `nh` with `stride`"""

    def __init__(
        self,
        expansion,
        ni,
        nf,
        stride=1,
        kernel_size=3,
        groups=1,
        nh1=None,
        nh2=None,
        dw=False,
        g2=1,
        sym=False,
        norm_type=NormType.Batch,
        act_cls=torch.nn.ReLU,
        ndim=2,
        pool=avg_pool,
        pool_first=True,
        **kwargs,
    ):
        super().__init__()
        norm2 = NormType.BatchZero if norm_type == NormType.Batch else norm_type
        if nh2 is None:
            nh2 = nf
        if nh1 is None:
            nh1 = nh2
        nf, ni = nf * expansion, ni * expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        layers = (
            [
                ConvLayer(ni, nh2, kernel_size, stride=stride, groups=ni if dw else groups, **k0),
                ConvLayer(nh2, nf, kernel_size, groups=g2, **k1),
            ]
            if expansion == 1
            else [
                ConvLayer(ni, nh1, 1, **k0),
                ConvLayer(nh1, nh2, kernel_size, stride=stride, groups=nh1 if dw else groups, **k0),
                ConvLayer(nh2, nf, 1, groups=g2, **k1),
            ]
        )
        self.convs = torch.nn.Sequential(*layers)
        self.convpath = torch.nn.Sequential(self.convs)  # redundant, but needed for backward compatibility
        idpath = []
        if ni != nf:
            idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride != 1:
            idpath.insert((1, 0)[pool_first], pool(2, ndim=ndim, ceil_mode=True))
        self.idpath = torch.nn.Sequential(*idpath)
        self.act = torch.nn.ReLU(inplace=True) if act_cls is torch.nn.ReLU else act_cls()

    def forward(self, x):
        return self.act(self.convpath(x) + self.idpath(x))


def make_block(
    expansion: int, ni: int, nf: int, blocks: int, stride: int, kernel_size: int, act_cls=torch.nn.ReLU, **kwargs
):
    return torch.nn.Sequential(
        *[
            ResBlock(
                expansion,
                ni if i == 0 else nf,
                nf,
                stride=stride if i == 0 else 1,
                kernel_size=kernel_size,
                act_cls=act_cls,
                **kwargs,
            )
            for i in range(blocks)
        ]
    )


def make_encoder(
    expansion: int | None = None,
    layers: list | None = None,
    p: float = 0.0,
    input_channels: int = 3,
    stem_szs=(32, 32, 64),
    kernel_size: int = 5,
    kernel_size_stem: int = 5,
    act_cls=torch.nn.ReLU,
    widen: float = 1.0,
    concat_pooling: bool = True,
    latent_dim=None,
    bn_head: bool = True,
    layout: str | None = None,
    **kwargs,
) -> tuple[torch.nn.Sequential, int]:
    if layout is None:
        # default values of `expansion` and `layers` correspond to XResNet1d101
        if expansion is None:
            expansion = 4
        if layers is None:
            layers = [3, 4, 23, 3]
    else:
        assert expansion is None
        assert layers is None
        layout = _get_layout(layout)
        expansion = layout["expansion"]
        layers = layout["layers"]

    stem_szs = [input_channels, *stem_szs]
    stem = [
        ConvLayer(stem_szs[i], stem_szs[i + 1], ks=kernel_size_stem, stride=2 if i == 0 else 1, act_cls=act_cls, ndim=1)
        for i in range(len(stem_szs) - 1)
    ]

    if len(layers) > 4:
        block_szs = [int(64 * widen)] * 4 + [int(32 * widen)] * (len(layers) - 4)
    else:
        block_szs = [int(64 * widen)] * len(layers)
    block_szs = [stem_szs[-1] // expansion] + block_szs
    blocks = [
        make_block(
            expansion,
            ni=block_szs[i],
            nf=block_szs[i + 1],
            blocks=l,
            stride=1 if i == 0 else 2,
            kernel_size=kernel_size,
            act_cls=act_cls,
            ndim=1,
            **kwargs,
        )
        for i, l in enumerate(layers)
    ]
    blocks = [torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)] + blocks

    n = block_szs[-1] * expansion
    if concat_pooling:
        n *= 2
    layers = [*stem, *blocks, AdaptiveConcatPool1d() if concat_pooling else torch.nn.MaxPool1d(2), Flatten()]
    if latent_dim is None:
        latent_dim = n
    elif latent_dim != n:
        layers += bn_drop_lin(n, latent_dim, bn_head, p, torch.nn.ReLU(inplace=True))

    return torch.nn.Sequential(*layers), latent_dim


def add_heads(
    module: torch.nn.Module,
    input_size: int,
    num_binary: int = 0,
    lin_ftrs_head=None,
    ps_head: float = 0.5,
    bn_final_head: bool = False,
    bn_head: bool = True,
    act_head="relu",
    **kwargs,
):
    # this is merely for defining suitable default values for the keyword arguments
    _add_heads(
        module,
        input_size,
        num_binary=num_binary,
        lin_ftrs_head=lin_ftrs_head,
        ps_head=ps_head,
        bn_head=bn_head,
        bn_final_head=bn_final_head,
        act_head=act_head,
    )


class XResNet1d(torch.nn.Module):
    def __init__(self, num_add_input: int = 0, **kwargs):
        super().__init__()

        self.encoder, latent_dim = make_encoder(**encoder_kwargs(kwargs))
        add_heads(self, latent_dim + num_add_input, **kwargs)
        init_weights("network", self)

    def forward(self, x: torch.Tensor, *args):
        x_enc = self.encoder(x)
        if args:
            x_enc = torch.cat((x_enc, *args), dim=1)
        return self.binary_head(x_enc)


def _get_layout(layout: str) -> dict:
    layout = layout.lower()
    layout = layout.removeprefix("xresnet1d")
    if layout == "18_small":
        return {"expansion": 1, "layers": [2, 2], "stem_szs": (16, 32)}
    elif layout == "18":
        return {"expansion": 1, "layers": [2, 2, 2, 2]}
    elif layout == "34":
        return {"expansion": 1, "layers": [3, 4, 6, 3]}
    elif layout == "50":
        return {"expansion": 4, "layers": [3, 4, 6, 3]}
    elif layout == "101":
        return {"expansion": 4, "layers": [3, 4, 23, 3]}
    elif layout == "152":
        return {"expansion": 4, "layers": [3, 8, 36, 3]}
    elif layout == "18_deep":
        return {"expansion": 1, "layers": [2, 2, 2, 2, 1, 1]}
    elif layout == "34_deep":
        return {"expansion": 1, "layers": [3, 4, 6, 3, 1, 1]}
    elif layout == "50_deep":
        return {"expansion": 4, "layers": [3, 4, 6, 3, 1, 1]}
    elif layout == "18_deeper":
        return {"expansion": 1, "layers": [2, 2, 1, 1, 1, 1, 1, 1]}
    elif layout == "34_deeper":
        return {"expansion": 1, "layers": [3, 4, 6, 3, 1, 1, 1, 1]}
    elif layout == "50_deeper":
        return {"expansion": 4, "layers": [3, 4, 6, 3, 1, 1, 1, 1]}
    else:
        raise ValueError("Unknown XResNet1d layout: " + layout)


xresnet1d = XResNet1d


def xresnet1d18_small(**kwargs):
    kwargs.update(_get_layout("18_small"))
    return XResNet1d(**kwargs)


def xresnet1d18(**kwargs):
    kwargs.update(_get_layout("18"))
    return XResNet1d(**kwargs)


def xresnet1d34(**kwargs):
    kwargs.update(_get_layout("34"))
    return XResNet1d(**kwargs)


def xresnet1d50(**kwargs):
    kwargs.update(_get_layout("50"))
    return XResNet1d(**kwargs)


def xresnet1d101(**kwargs):
    kwargs.update(_get_layout("101"))
    return XResNet1d(**kwargs)


def xresnet1d152(**kwargs):
    kwargs.update(_get_layout("152"))
    return XResNet1d(**kwargs)


def xresnet1d18_deep(**kwargs):
    kwargs.update(_get_layout("18_deep"))
    return XResNet1d(**kwargs)


def xresnet1d34_deep(**kwargs):
    kwargs.update(_get_layout("34_deep"))
    return XResNet1d(**kwargs)


def xresnet1d50_deep(**kwargs):
    kwargs.update(_get_layout("50_deep"))
    return XResNet1d(**kwargs)


def xresnet1d18_deeper(**kwargs):
    kwargs.update(_get_layout("18_deeper"))
    return XResNet1d(**kwargs)


def xresnet1d34_deeper(**kwargs):
    kwargs.update(_get_layout("34_deeper"))
    return XResNet1d(**kwargs)


def xresnet1d50_deeper(**kwargs):
    kwargs.update(_get_layout("50_deeper"))
    return XResNet1d(**kwargs)
