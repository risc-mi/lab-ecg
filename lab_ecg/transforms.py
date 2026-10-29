import warnings

import numpy as np
import torch
from scipy import interpolate, signal


def _iter_axes(a: tuple, *axes):
    if axes:
        ax: int = axes[0]
        if len(axes) == 1:
            idx: list = [slice(None)] * len(a)
            for i in range(a[ax]):
                idx[ax] = i
                yield idx
        else:
            for i in range(a[ax]):
                for t in _iter_axes(a, *axes[1:]):
                    t[ax] = i
                    yield t
    else:
        yield []


def fancy(a, dct) -> tuple:
    try:
        a = len(a.shape)
    except AttributeError:
        assert isinstance(a, int)
    out = [slice(None)] * a
    for ax, ind in dct.items():
        out[ax] = ind
    return tuple(out)


def iter_axes_except(a, *axes):
    # for technical reasons, `iter_axes_except` yields lists that must be converted into tuples manually
    try:
        a = a.shape
    except AttributeError:
        assert isinstance(a, tuple)

    mask = np.ones((len(a),), dtype=bool)
    mask[axes] = False
    for t in _iter_axes(a, *np.flatnonzero(mask)):
        yield tuple(t)


def get_axis(sample: dict, ax):
    n = sample.get("n_new_axes", 0)
    if isinstance(ax, int):
        return ax if ax < 0 else (ax + n)
    else:
        return tuple(a if a < 0 else (a + n) for a in ax)


class ToTensor:
    """Converts the ECG waveform and label numpy arrays to FloatTensors."""

    def __call__(self, sample):
        for k in list(sample):
            v = sample[k]
            if isinstance(v, np.ndarray):
                if any(s in v.dtype.kind for s in "SaUO"):
                    sample[k] = v.tolist()
                else:
                    sample[k] = torch.from_numpy(v).type(torch.float)
            elif k == "label":
                # legacy
                sample[k] = torch.from_numpy(np.array(v)).type(torch.float)

        return sample

    def is_deterministic(self) -> bool:
        return True


class Impute:
    def __init__(self, method: str = "value", value: float = 0.0, interpolation_axis: int = -1):
        if method not in ("value", "interpolate"):
            raise NotImplementedError
        self.method = method
        self.value = value
        self.interpolation_axis = interpolation_axis

    def __call__(self, sample):
        if self.method == "value":
            for k in ("waveform", "secondary_waveform"):
                x = sample.get(k)
                if x is not None:
                    mask = np.isfinite(x)
                    if not mask.all():
                        x[~mask] = self.value
                        sample[k] = x
        elif self.method == "interpolate":
            axis = get_axis(sample, self.interpolation_axis)
            for k in ("waveform", "secondary_waveform"):
                x = sample.get(k)
                if x is not None:
                    y = None
                    for i in iter_axes_except(x, axis):
                        mask = np.isfinite(x[i])
                        if not mask.all():
                            if mask.any():
                                idx = np.flatnonzero(mask)
                                f = interpolate.interp1d(
                                    x.astype(np.float32),
                                    x[i][idx],
                                    copy=False,
                                    fill_value=x[i][idx].mean(),
                                    bounds_error=False,
                                )
                                if y is None:
                                    y = np.arange(x.shape[axis], dtype=np.float32)
                                x[i] = f(y)
                            else:
                                # all values are invalid
                                x[i] = self.value

        return sample

    def is_deterministic(self) -> bool:
        return True


class Normalize:
    def __init__(self, method: str = "z-score", eps: float = 1e-5, axis: int = -1):
        self.method = method
        self.eps = eps
        self.axis = axis

    def __call__(self, sample):
        axis = get_axis(sample, self.axis)
        if self.method == "z-score":
            for k in ("waveform", "secondary_waveform"):
                x = sample.get(k)
                if x is not None:
                    x = x - x.mean(axis=axis, keepdims=True)
                    x /= np.maximum(x.std(axis=axis, keepdims=True), self.eps)
                    sample[k] = x
        elif self.method == "min-max":
            for k in ("waveform", "secondary_waveform"):
                x = sample.get(k)
                if x is not None:
                    x = x - x.min(axis=axis, keepdims=True)
                    x /= np.maximum(x.max(axis=axis, keepdims=True), self.eps)
                    sample[k] = x

        return sample

    def is_deterministic(self) -> bool:
        return True


class SelectChannels:
    """Select signal individual channels from a multichannel input signal. Assumes a list `channel_idx` whose elements
    are lists of integers. Whenever this transformation is applied to a signal, one index is randomly selected from
    each of these lists.
    """

    def __init__(self, channel_idx: list, channel_names=None, channel_axis: int = -2):
        if channel_idx is None:
            self.channel_idx = None
        else:
            self.channel_idx = []
            for cs in channel_idx:
                cs_new = []
                for c in cs:
                    if isinstance(c, int):
                        assert c >= 0
                        cs_new.append(c)
                    else:
                        assert channel_names is not None
                        try:
                            cs_new.append(channel_names.index(c))
                        except ValueError:
                            pass
                assert len(cs_new) > 0
                self.channel_idx.append(cs_new)
        self.channel_axis = channel_axis

    def __call__(self, sample):
        if self.channel_idx is not None:
            idx = np.empty((len(self.channel_idx),), dtype=np.int32)
            for i, cs in enumerate(self.channel_idx):
                if len(cs) == 1:
                    idx[i] = cs[0]
                else:
                    idx[i] = np.random.choice(cs)

            axis = get_axis(sample, self.channel_axis)
            for k in ("waveform", "secondary_waveform"):
                x = sample.get(k)
                if x is not None:
                    sample[k] = np.take(x, idx, axis=axis)

        return sample

    def waveform_output(self, channels=None, **kwargs) -> dict:
        if self.channel_idx is None:
            return dict(channels=channels, **kwargs)
        elif channels is None:
            return dict(channels=[None] * len(self.channel_idx), **kwargs)
        else:
            assert len(channels) >= max(j for cs in self.channel_idx for j in cs)
            channels_out = []
            for cs in self.channel_idx:
                if len(cs) == 1:
                    channels_out.append(channels[cs[0]])
                else:
                    channels_out.append(None)
            return dict(channels=channels_out, **kwargs)

    def is_deterministic(self) -> bool:
        return self.channel_idx is None


class Resample:
    """Resamples the waveform to the specified sampling frequency.

    Attributes
    ----------
    sample_freq : int
        The required sample frequency to resample to.
    """

    def __init__(self, sample_freq, axis: int = -1):
        """Initializes the resample transformation."""
        self.sample_freq = int(sample_freq)
        self.axis = axis

    def _resample(self, waveform, samplebase, axis: int):
        length = int(waveform.shape[axis])
        out_length = int((length / samplebase) * self.sample_freq)

        if type(waveform) == np.ndarray:
            x = np.linspace(0, length / samplebase, num=length)
            f = interpolate.interp1d(x, waveform, axis=axis)
            xnew = np.linspace(0, length / samplebase, num=out_length)
            return f(xnew)
        elif type(waveform) == torch.Tensor:
            return torch.nn.functional.interpolate(
                waveform.unsqueeze(0), size=out_length, mode="linear", align_corners=False
            ).squeeze(0)

    def __call__(self, sample):
        samplebase = int(sample["samplebase"])
        waveform = sample["waveform"]
        secondary_waveform = None
        if "secondary_waveform" in sample:
            secondary_waveform = sample["secondary_waveform"]

        axis = get_axis(sample, self.axis)
        if samplebase != self.sample_freq:
            sample["waveform"] = self._resample(waveform, samplebase, axis)

            # Do the same for secondary waveform
            if secondary_waveform is not None:
                sample["secondary_waveform"] = self._resample(secondary_waveform, samplebase, axis)

            sample["samplebase"] = self.sample_freq

        return sample

    def waveform_output(self, length=None, fs=None, **kwargs) -> dict:
        if length is None or fs is None:
            return dict(fs=self.sample_freq, **kwargs)
        else:
            return dict(length=int((length / fs) * self.sample_freq), fs=self.sample_freq, **kwargs)

    def is_deterministic(self) -> bool:
        return True


class Filter:
    def __init__(self, type: str = "butter", axis: int = -1, low: float = 0.05, high: float = 150, order: int = 3):
        assert type == "butter"
        self.low = low
        self.high = high
        self.order = order
        self.axis = axis

    def __call__(self, sample):
        fs = sample["samplebase"]
        if self.low > 0:
            if self.high * 2 < fs:
                # band-pass filter
                sos = signal.butter(self.order, [self.low, self.high], btype="bandpass", output="sos", fs=fs)
            else:
                # high-pass filter
                sos = signal.butter(self.order, self.low, btype="high", output="sos", fs=fs)
        elif self.high * 2 < fs:
            # low-pass filter
            sos = signal.butter(self.order, self.high, btype="low", output="sos", fs=fs)
        else:
            return sample

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            axis = get_axis(sample, self.axis)
            for k in ("waveform", "secondary_waveform"):
                if k in sample:
                    x = sample[k]
                    x_new = np.empty_like(x)
                    for i in iter_axes_except(x, axis):
                        x_new[i] = signal.sosfiltfilt(sos, x[i])
                    sample[k] = x_new

        return sample

    def is_deterministic(self) -> bool:
        return True


class RandomSnippet:
    def __init__(self, min_len: int = 100, max_len: int = 10000, center: bool = False, axis: int = -1):
        self.min_len = min_len
        self.max_len = max_len
        self.center = center
        self.axis = axis

    def __call__(self, sample):
        wf = sample["waveform"]
        axis = get_axis(sample, self.axis)
        L = wf.shape[axis]
        if self.min_len < L:
            max_len = min(self.max_len, L)
            if self.min_len < max_len:
                L_new = np.random.randint(self.min_len, max_len + 1)
            else:
                L_new = self.min_len

            if self.center:
                a = (L - L_new) // 2
            else:
                a = np.random.randint(L - L_new + 1)

            sample["waveform"] = wf[fancy(wf, {axis: slice(a, a + L_new)})]

            if "secondary_waveform" in sample:
                wf = sample["secondary_waveform"]
                sample["secondary_waveform"] = wf[fancy(wf, {axis: slice(a, a + L_new)})]

            sample["snippet_start"] = a
            sample["snippet_len"] = L_new

        return sample

    def waveform_output(self, length=None, **kwargs) -> dict:
        if length is None:
            if self.min_len >= self.max_len:
                return dict(length=self.min_len, **kwargs)
            else:
                return kwargs
        else:
            if self.min_len < length:
                length = None if self.min_len < self.max_len else self.min_len
            return dict(length=length, **kwargs)

    def is_deterministic(self) -> bool:
        return False


class SlidingWindow:
    def __init__(self, length: int = 100, offset: int | None = None, axis: int = -1):
        self.length = length
        self.offset = self.length if offset is None else offset
        assert self.length > 0
        assert self.offset > 0
        if axis != -1:
            raise NotImplementedError()

    def __call__(self, sample):
        for k in ("waveform", "secondary_waveform"):
            if k in sample:
                wf = sample[k]
                L = wf.shape[-1]
                if self.length < L:
                    wfs = np.empty(((L - self.length) // self.offset + 1, *wf.shape[:-1], self.length), dtype=wf.dtype)
                    i = j = 0
                    while i + self.length <= L:
                        wfs[j] = wf[..., i : i + self.length]
                        i += self.offset
                        j += 1
                    sample[k] = wfs
                    sample["n_new_axes"] = sample.get("n_new_axes", 0) + 1

        return sample

    def waveform_output(self, length=None, **kwargs) -> dict:
        if length is None:
            return dict(length=self.length, **kwargs)
        else:
            length = min(length, self.length)
            return dict(length=length, **kwargs)

    def is_deterministic(self) -> bool:
        return True


def split_deterministic(t, return_none: bool = False) -> tuple:
    """Split a transformation into a deterministic and a non-deterministic part.

    Parameters
    ----------
    t
        The transformation to split.
    return_none : bool, default=False
        Represent the identity transformation by None. Otherwise, it is represented by `Compose([])`.

    Returns
    -------
    det
        Determinstic transformations at the beginning of `t`.
    non_det
        Non-deterministic transformations after `det`.

    Notes
    -----
    * `t` is identical to `non_det o det`.
    * A shallow copy of `t` is created, i.e., the output refers to transformation instances present in `t`,
        or `t` itself.
    * If `t` contains `Compose` instances at multiple levels of nesting, some levels might be flattened in the output.
    """
    from torchvision.transforms import Compose

    if isinstance(t, Compose):
        non_det = None if return_none else Compose([])

        if len(t.transforms) == 0:
            return t, non_det

        all_det = []
        for i, t0 in enumerate(t.transforms):
            det, non_det = split_deterministic(t0, return_none=return_none)
            if isinstance(det, Compose):
                all_det += det.transforms
            elif det is not None:
                all_det.append(det)
            if isinstance(non_det, Compose):
                if len(non_det.transforms):
                    non_det.transforms += t.transforms[i + 1 :]
                    break
            elif non_det is not None:
                if i + 1 < len(t.transforms):
                    non_det = Compose([non_det] + t.transforms[i + 1 :])
                break
        return (None if return_none and len(all_det) == 0 else Compose(all_det)), non_det
    elif t.is_deterministic() if hasattr(t, "is_deterministic") else False:  # to be on the safe side
        return t, (None if return_none else Compose([]))
    else:
        return (None if return_none else Compose([])), t
