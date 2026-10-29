from collections import OrderedDict

import torch

from .network import add_heads as _add_heads
from .network import encoder_kwargs, init_weights


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    pool_size: int | None = 4,
    n_layers: int = 2,
    dropout: float = 0.2,
) -> torch.nn.Sequential:
    if isinstance(kernel_size, int):
        kernel_size = (1, kernel_size)
    if isinstance(pool_size, int):
        pool_size = (1, pool_size)

    layers = []
    for i in range(n_layers):
        layers += [
            torch.nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size,
                padding=(0, kernel_size[1] // 2),
                padding_mode="reflect",
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(out_channels),
        ]

    if pool_size is not None:
        layers.append(torch.nn.MaxPool2d(kernel_size=pool_size))
    if dropout > 0:
        layers.append(torch.nn.Dropout(dropout))
    return torch.nn.Sequential(*layers)


def make_encoder(
    input_channels: int = 12, n_timesteps: int = 2400, fs: int = 240, ps: float = 0.2
) -> tuple[torch.nn.Module, int]:
    # input shape: (N, n_channels, n_timesteps)

    kernel_mult = 1 if fs < 300 else 2
    kernel_sub = 1 if kernel_mult == 2 else 0

    latent_dim = 128
    if input_channels > 1:
        layers = [
            conv_block(1, 63, 12 * kernel_mult - 1, dropout=ps, pool_size=None if input_channels > 6 else 4),
            conv_block(63, 32, 9 * kernel_mult - kernel_sub, dropout=ps),
            conv_block(
                32,
                32,
                9 * kernel_mult - kernel_sub,
                dropout=ps if input_channels > 6 else 0,
                pool_size=3 if n_timesteps <= 3000 else 4,
            ),
            conv_block(32, 32, 9 * kernel_mult - kernel_sub, dropout=ps),
            conv_block(
                32,
                64,
                9 * kernel_mult - kernel_sub,
                dropout=ps,
                pool_size=(2, 4 if n_timesteps <= 3000 or input_channels > 6 else 5),
            ),
        ]

        if input_channels > 6:
            layers += [
                conv_block(64, 128, 3, dropout=0, pool_size=None, n_layers=1),
                conv_block(128, 256, 3, dropout=0, pool_size=None, n_layers=1),
                conv_block(256, 512, 3, dropout=0, pool_size=(2, 4), n_layers=1),
            ]
            latent_dim = 512 * (input_channels // 4) * (n_timesteps // 1280)
        else:
            layers.append(conv_block(64, 128, 2 * kernel_mult + 1, dropout=ps, pool_size=None, n_layers=3))
            latent_dim *= (input_channels // 2) * (n_timesteps // 1280)
    else:
        layers = [
            conv_block(1, 3, 8 * kernel_mult - 1, dropout=ps),
            conv_block(3, 32, 3 * kernel_mult - kernel_sub, dropout=ps),
            conv_block(32, 32, 3 * kernel_mult - kernel_sub, dropout=0, pool_size=3 if n_timesteps <= 3000 else 4),
            conv_block(32, 32, 3 * kernel_mult - kernel_sub, dropout=ps),
            conv_block(32, 64, 3 * kernel_mult - kernel_sub, dropout=ps, pool_size=4 if n_timesteps <= 3000 else 5),
            conv_block(64, 128, kernel_mult + kernel_sub, dropout=ps, pool_size=None, n_layers=3),
        ]
        latent_dim *= n_timesteps // 1280

    # backward compatibility
    dct = OrderedDict(unsqueeze=Unsqueeze(1))
    for i, layer in enumerate(layers + [torch.nn.Flatten()]):
        dct[str(i)] = layer

    return torch.nn.Sequential(dct), latent_dim


def add_heads(
    module: torch.nn.Module,
    input_size: int,
    num_binary: int = 0,
    lin_ftrs_head=None,
    ps_head: float = 0,
    bn_final_head: bool = False,
    bn_head: bool = True,
    act_head="relu",
    input_channels: int = 12,
    **kwargs,
):
    # this is merely for defining suitable default values for the keyword arguments
    if lin_ftrs_head is None:
        lin_ftrs_head = [256 if input_channels > 6 else 128, 128]
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


class Kwon(torch.nn.Module):
    """Good (albeit not perfect) approximation of the hemoglobin prediction model by Kwon et al. _[1].
    The number of parameters for 6 and 1 input channels is a bit larger compared to [1], for 12 input channels it is a
    bit smaller.

    Input
    -----
    * `C` is arbitrary. In _[1], it is 1 (I), 6 (limb leads) and 12 (full 12-lead ECG).
    * `L` is arbitrary. In _[1], it is 4000, corresponding to 8 seconds at a sample rate of 500 Hz.
    * In _[1], the input is bandpass filtered from 0.05 to 150 Hz.

    References
    ----------
    .. [1] https://doi.org/10.1016/S2589-7500(20)30108-4
    """

    def __init__(self, num_add_input: int = 2, **kwargs):
        super().__init__()

        self.encoder, latent_dim = make_encoder(**encoder_kwargs(kwargs))
        add_heads(self, latent_dim + num_add_input, **kwargs)
        init_weights("network", self)

    def forward(self, x: torch.Tensor, *args):
        # `x` has shape `(N, C, L)`
        x_enc = self.encoder(x)
        if args:
            x_enc = torch.cat((x_enc, *args), dim=1)
        return self.binary_head(x_enc)
