from collections.abc import Callable
from typing import Self

import numpy as np
import torch


class BinaryOutputTarget:
    def __init__(self, positive: int = True, sigmoid: bool = False, multilabel: int = 0, output_idx: int = 0):
        self.sign = 1 if positive else -1
        self.multilabel = multilabel
        self.sigmoid = sigmoid
        self.output_idx = output_idx

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        x = args[self.output_idx]
        if self.sigmoid:
            x = torch.sigmoid(x)
        if len(x.shape) == 1:
            return self.sign * x
        else:
            assert len(x.shape) == 2
            return self.sign * x[:, self.multilabel]


class Gradient:
    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor], device=None):
        if device is None:
            if isinstance(model, torch.nn.Module):
                device = next(model.parameters()).device
            else:
                raise ValueError("Unable to infer device. Please pass explicitly.")

        self.model = model
        self.device = device

    def forward(
        self, *args: torch.Tensor, target: Callable[[torch.Tensor], torch.Tensor] | None = None, absolute: bool = True
    ) -> tuple[list[torch.Tensor], list[np.ndarray]]:
        args = [a.to(device=self.device) for a in args]
        for a in args:
            a.requires_grad_(True)

        output = self.model(*args)
        if not isinstance(output, (list, tuple)):
            output = [output]

        if target is None:
            target = BinaryOutputTarget()

        for a in args:
            a.grad = None
        loss: torch.Tensor = target(*output).sum()
        loss.backward(retain_graph=True)

        grads = [a.grad.cpu().data.numpy() for a in args]

        return output, scale(grads, absolute)

    def release(self):
        torch.cuda.empty_cache()

    def __call__(
        self, *args: torch.Tensor, target: Callable[[torch.Tensor], torch.Tensor] | None = None, absolute: bool = True
    ) -> tuple[list[torch.Tensor], list[np.ndarray]]:
        """Apply the Gradient Method _[1] to a given list of input tensors, to obtain saliency maps.
        These are simply the (absolute values of) the gradients of the model output wrt. the inputs.

        Parameters
        ----------
        *args
            Input arguments passed to `model`, tensors of shape `(N, ...)`.
        target : callable
            Must map model output (tensor(s) of shape `(N, ...)`) to single tensor of shape `(N,)`.
            Must be differentiable.
        absolute : bool
            Take the absolute value of the gradient, as proposed in the original publication _[1].

        Returns
        -------
        model_output : list of tensor
            Model output.
        saliency_map : list of array
            Saliency maps computed by the Gradient Method, list of arrays with the exact same shape as the
            corresponding input arguments.

        Notes
        -----
        In _[1], the per-pixel and per-channel saliency map is turned into a per-pixel saliency map by taking the
        maximum along the channel axis.

        References
        ----------
        .. [1] K Simonyan, A Vedaldi, A Zisserman. Deep Inside Convolutional Networks: Visualising Image Classification
           Models and Saliency Maps. arXiv:1312.6034, 2013. http://arxiv.org/abs/1312.6034.
        """
        return self.forward(*args, target=target, absolute=absolute)

    def __del__(self):
        self.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.release()


def scale(xs: list[np.ndarray], absolute: bool) -> list[np.ndarray]:
    axes = [tuple(range(1, x.ndim)) for x in xs]
    if absolute:
        xs = [np.abs(x) for x in xs]
        a = np.stack([x.min(axis=ax) for x, ax in zip(xs, axes)], axis=1).min(axis=1)  # (N,)
        for x, ax in zip(xs, axes):
            x -= np.expand_dims(a, axis=ax)
        a = np.maximum(np.stack([x.max(axis=ax) for x, ax in zip(xs, axes)], axis=1).max(axis=1), 1e-7)  # (N,)
        for x, ax in zip(xs, axes):
            x /= np.expand_dims(a, axis=ax)
        return xs
    else:
        a = np.maximum(
            np.stack(
                [np.maximum(-x.min(axis=ax, keepdims=True), x.max(axis=ax, keepdims=True)) for x, ax in zip(xs, axes)],
                axis=1,
            ).max(axis=1),
            1e-7,
        )
        return [x / np.expand_dims(a, axis=ax) for x, ax in zip(xs, axes)]


def plot_saliency_map(
    signal: np.ndarray,
    saliency_map: np.ndarray | None,
    pos_range=(0.2, 1),
    neg_range=(-1, -0.2),
    s: float = 100,
    channels=None,
    grid=None,
    axis=None,
    sharey: bool = False,
    figsize=None,
    ylog: bool = False,
    yticks: bool = True,
    channel_order=None,
    **kwargs,
):
    if signal.ndim == 1:
        signal = signal[np.newaxis]
    assert signal.ndim == 2  # (channels, length)

    if saliency_map is None:
        pos_range = neg_range = None
    else:
        if saliency_map.ndim == 1:
            saliency_map = saliency_map[np.newaxis]
        assert saliency_map.ndim == 2  # (channels, length)
        assert saliency_map.shape[0] in (1, signal.shape[0])

    if axis is None:
        from matplotlib import pyplot as plt

        if grid is None:
            grid = (signal.shape[0], 1)
        _, axis = plt.subplots(nrows=grid[0], ncols=grid[1], sharex=True, sharey=sharey, squeeze=False, figsize=figsize)
        axis = list(axis.reshape(-1))
    elif isinstance(axis, str) and axis == "current":
        from matplotlib import pyplot as plt

        axis = plt.gcf().axes
    elif not isinstance(axis, (list, tuple, np.ndarray)):
        axis = [axis]

    if pos_range is not None and neg_range is not None:
        assert pos_range[0] >= neg_range[1]

    if sharey:
        a = signal.min()
        b = signal.max()
        if ylog:
            ylim = (0.67 * a, 1.5 * b)
        else:
            ylim = (a - 0.05 * (b - a), b + 0.05 * (b - a))
    else:
        ylim = None

    if channel_order is None:
        channel_order = np.arange(len(signal))
    else:
        assert len(channel_order) == len(signal)

    for i, ax in enumerate(axis):
        sig = signal[channel_order[i]]
        j = 0 if saliency_map is None or saliency_map.shape[0] == 1 else channel_order[i]
        if pos_range is not None:
            a, b = pos_range
            assert a < b
            mask = saliency_map[j] >= a
            if mask.any():
                c = np.zeros((mask.sum(), 4), dtype=np.float32)
                c[:, 0] = 1
                c[:, 3] = np.minimum(1, (saliency_map[j, mask] - a) / (b - a))
                ax.scatter(np.flatnonzero(mask), sig[mask], s=s, c=c, marker="o", linewidths=0)

        if neg_range is not None:
            a, b = neg_range
            assert a < b
            mask = saliency_map[j] <= b
            if mask.any():
                c = np.zeros((mask.sum(), 4), dtype=np.float32)
                c[:, 2] = 1
                c[:, 3] = np.minimum(1, (b - saliency_map[j, mask]) / (b - a))
                ax.scatter(np.flatnonzero(mask), sig[mask], s=s, c=c, marker="o", linewidths=0)

        ax.plot(sig, **kwargs)
        if channels is not None:
            ax.set_ylabel(channels[channel_order[i]])
        if ylim is not None:
            ax.set_ylim(ylim)
        if ylog:
            ax.set_yscale("log")
        if not yticks:
            ax.set_yticks([])
        if i + 1 < len(axis):
            ax.tick_params(bottom=False)
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)
