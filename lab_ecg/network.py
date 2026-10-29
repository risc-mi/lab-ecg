from collections.abc import Iterable

import torch


def listify(p) -> list:
    if p is None:
        p = []
    elif isinstance(p, str) or not isinstance(p, Iterable):
        p = [p]
    else:
        # rank-0 tensors in PyTorch are iterable but don't have a length
        try:
            len(p)
        except:  # noqa: E722
            p = [p]
    return list(p)


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0.0, actn: torch.nn.Module | None = None):
    """Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."""
    # copied from `fastai.layers`
    layers = [torch.nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(torch.nn.Dropout(p))
    layers.append(torch.nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


class Flatten(torch.nn.Module):
    """Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"""

    # copied from `fastai.layers`

    def __init__(self, full: bool = False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class AdaptiveConcatPool1d(torch.nn.Module):
    """Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."""

    def __init__(self, sz=None):
        """
        Output will be 2*sz or 2 if sz is None
        """
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = torch.nn.AdaptiveAvgPool1d(sz), torch.nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def encoder_kwargs(kwargs: dict) -> dict:
    return {
        k: v
        for k, v in kwargs.items()
        if k not in ("input_size", "num_binary", "lin_ftrs_head", "ps_head", "bn_head", "bn_final_head", "act_head")
    }


def create_head(
    nf: int,
    nc: int,
    lin_ftrs=None,
    ps: float = 0,
    bn_final: bool = False,
    bn: bool = True,
    act="relu",
    concat_pooling=True,
    pool=True,
):
    if lin_ftrs is None:
        lin_ftrs = []
    lin_ftrs = [2 * nf if concat_pooling and pool else nf] + lin_ftrs + [nc]  # was [nf, 512,nc]
    ps = listify(ps)
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps

    actns = [torch.nn.ReLU(inplace=True) if act == "relu" else torch.nn.ELU(inplace=True)] * (len(lin_ftrs) - 2) + [
        None
    ]
    layers = [AdaptiveConcatPool1d() if concat_pooling else torch.nn.MaxPool1d(2), Flatten()] if pool else []
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, bn, p, actn)

    if bn_final:
        layers.append(torch.nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))

    return layers[0] if len(layers) == 1 else torch.nn.Sequential(*layers)


def add_heads(
    module: torch.nn.Module,
    input_size: int,
    num_binary: int = 0,
    lin_ftrs_head=None,
    ps_head: float = 0,
    bn_head: bool = True,
    bn_final_head: bool = False,
    act_head="relu",
):
    assert num_binary > 0
    module.binary_head = create_head(
        input_size,
        nc=num_binary,
        lin_ftrs=lin_ftrs_head,
        ps=ps_head,
        bn_final=bn_final_head,
        bn=bn_head,
        act=act_head,
        pool=False,
    )


def init_weights(name: str, m, encoder_name: str = "network.encoder", init_fn=None):
    if isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
        # skip LSTM layers
        if init_fn is not None:
            init_fn(m)
    else:
        if init_fn is None:
            if getattr(m, "bias", None) is not None:
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight)
        else:
            init_fn(m)

        for n, c in m.named_children():
            init_weights(name + "." + n, c, encoder_name=encoder_name, init_fn=init_fn)
