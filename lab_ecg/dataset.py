from pathlib import Path
from typing import Self

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .transforms import split_deterministic


def get_memory_limit_mb(value: int | str) -> int:
    if isinstance(value, str):
        if value.lower().endswith("gb"):
            value = int(value[:-2]) * 1024
        elif value.lower().endswith("mb"):
            value = int(value[:-2])
        else:
            value = int(value) * 1024
    return value


class HDFSignalDatastore:
    def __init__(
        self,
        samples,
        src: Path,
        src_channels=None,
        label_cols=None,
        channels=None,
        fs: int = 500,
        memory_limit: int | str = 0,
        binary_thresholds=None,
    ):
        """
        Parameters
        ----------
        samples : DataFrame or path-like
            Samples DataFrame, with one row per sample and column "sample_id", in addition to an arbitrary number of
            other columns.
        src : path-like
            HDF5 file with signals.
        src_channels : iterable, optional
            Names of the channels in `src`, of length `C`.
        label_cols : iterable, optional
        channels : iterable, optional
            Requested channels of each signal. If given, must be a subset of `src_channels`. None defaults to all
            source channels.
        fs : int
            Sample rate of the waveform signals in `src`, in Hz.
        memory_limit : int or str
            Memory limit, either int or str (optionally with suffix "MB" or "GB"; default is "GB"). If the size of the
            HDF5 file(s) containing the waveform data is less than this value, all data are loaded into main memory,
            which may speed up subsequent computations considerably.
            `samples` and `channels` are taken into account when computing the size of the HDF file(s), i.e., if only
            half of all samples and 1/4 of all channels are requested, the file size is divided by 8.
        """
        self._src = Path(src)
        self._fs = fs
        self._src_channels = src_channels
        self._binary_thresholds = binary_thresholds
        if isinstance(samples, pd.DataFrame):
            self._samples = None
            self._df = samples.copy()
        else:
            self._samples = Path(samples)
            self._df: pd.DataFrame = pd.read_hdf(self._samples, key="table")
        assert self._df.index.is_unique
        assert self._df["sample_id"].is_unique
        self._idx_name = "sample_id"
        self._shape = None  # shape of an individual sample
        self._dtype = None

        total_size = self._src.stat().st_size
        with h5py.File(self._src, mode="r") as h5:
            n_keys = len(h5)
            mask = self._df[self._idx_name].isin(h5.keys())
            if mask.any():
                ds = h5[self._df.loc[mask, self._idx_name].iloc[0]]
                self._shape = ds.shape
                self._dtype = ds.dtype

        assert self._shape is not None
        assert self._dtype is not None
        assert 2 <= len(self._shape) <= 3

        if not mask.all():
            print(f"Restricting to {mask.sum()} samples for which waveform data are available.")
        self._df = self._df[mask]

        self._make_labels(label_cols, binary_thresholds=binary_thresholds)
        self._feature_cols = [c for c in self._df.columns if self._df[c].dtype.kind in "uifbOMm"]

        mask = mask.reindex(self._df.index, fill_value=False)

        if channels is None:
            self._channels = self._src_channels
            self._channel_idx = None
        else:
            assert self._src_channels is not None
            cs = []
            for c in channels:
                if isinstance(c, int):
                    cs.append(c)
                else:
                    try:
                        cs.append(self._src_channels.index(c))
                    except ValueError:
                        pass
            self._channels = np.array(self._src_channels)[cs].tolist()  # `tolist()` ensures native Python data types
            self._channel_idx = cs

            if len(self._shape) == 2:
                self._shape = (len(self._channel_idx),) + self._shape[1:]
            else:
                self._shape = self._shape[:1] + (len(self._channel_idx),) + self._shape[2:]

        self._data = None
        if (
            total_size * len(self._channels) * self._df.loc[mask, self._idx_name].nunique()
            < get_memory_limit_mb(memory_limit) * (1024**2) * len(self._src_channels) * n_keys
        ):
            self._data = {}
            print("Loading data into memory", end="")
            n = 0
            n_total = self._df[self._idx_name].nunique()
            tic = pd.Timestamp.now()
            with h5py.File(self._src, mode="r") as h5:
                for i in self._df[self._idx_name].unique():
                    self._data[i] = self._fetch_from_h5(h5[i])
                    n += 1
                    toc = pd.Timestamp.now()
                    if tic + pd.Timedelta(100, unit="ms") < toc:
                        print(f"\rLoading data into memory ({n}/{n_total})", end="")
                        tic = toc
            print(f"\rLoading data into memory ({n_total}/{n_total})")

    @property
    def src(self) -> Path:
        return self._src

    @property
    def fs(self) -> int:
        return self._fs

    @property
    def idx_name(self) -> str:
        return self._idx_name

    @property
    def in_memory(self) -> bool:
        return self._data is not None

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def shape(self) -> tuple | None:
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def channels(self):
        return None if self._channels is None else list(self._channels)

    @property
    def feature_cols(self) -> list:
        return self._feature_cols

    @feature_cols.setter
    def feature_cols(self, value: list | None):
        if value is None:
            self._feature_cols = [c for c in self.df.columns if self.df[c].dtype.kind in "uifbOmM"]
        else:
            self._feature_cols = [c for c in self.df.columns if self.df[c].dtype.kind in "uifbOmM" and c in value]

    @property
    def binary_labels(self) -> list:
        return self._binary_labels

    @property
    def binary_labels_raw(self) -> list:
        return self._binary_labels_raw

    def __getitem__(self, item: int) -> dict:
        return self.get(item)

    def __len__(self) -> int:
        return len(self.df)

    def balance(self, indices=None, add_to_feature_cols: bool = True):
        """Assign sample weights based on binary labels. First, the weight of each label is determined as the
        reciprocal frequency of the respective positive class. Then, the weight of a sample is the sum of the class
        weights of its positive labels, plus 1 for each negative label. Finally, the sample weights are rescaled to
        have a mean of 1.

        These sample weights are saved in column "sample_weight" of the underlying DataFrame, replacing any existing
        values.

        Parameters
        ----------
        indices : array_like, optional
            iloc-indices of samples over which to compute class weights. If None, class weights are computed over
            all samples.
        add_to_feature_cols : bool
            Automatically add "sample_weight" to the feature columns, if new sample weights are assigned.
        """
        if self._binary_labels:
            sub = self._labels_df
            if indices is not None:
                sub = sub[sub.index.isin(self.df.index[indices])]

            class_weights = 1 / (np.maximum(sub[self._binary_labels].mean(), 1e-3)) - 1
            sample_weights = (
                (self._labels_df[self._binary_labels] > 0) * class_weights.values[np.newaxis]
                + ~(self._labels_df[self._binary_labels] > 0)
            ).sum(axis=1)
            sample_weights /= sample_weights.mean()
            self.df["sample_weight"] = sample_weights.reindex(self.df.index, fill_value=1.0)
            if add_to_feature_cols and "sample_weight" not in self._feature_cols:
                self._feature_cols.append("sample_weight")

    def get_dict(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "binary_thresholds": self._binary_thresholds,
            "channels": self.channels,
            "fs": self._fs,
            "label_cols": None if self._labels_df is None else list(self._labels_df.columns),
            "samples": None if self._samples is None else self._samples.as_posix(),
            "src": self._src.as_posix(),
            "src_channels": self._src_channels,
            "feature_cols": self._feature_cols,
        }

    def get(self, item: int, handle=None) -> dict:
        # if `self._df` is large, the following is *way* faster than `self._df[self._feature_cols].iloc[item].to_dict()`
        sample = {k: self._sanitize_value(self._df[k].iloc[item]) for k in self._feature_cols}
        sample["samplebase"] = self._fs

        key = self._df[self._idx_name].iloc[item]
        if self._data is None:
            waveform = self._fetch_from_h5(handle.get(key))
        else:
            try:
                waveform = self._data[key]
            except KeyError:
                waveform = np.full(self._shape, fill_value=np.nan, dtype=self._dtype)

        sample["waveform"] = waveform

        return self._add_labels_to_sample_dict(sample, item)

    def get_handle(self):
        return h5py.File(self._src, mode="r") if self._data is None else 0

    def close_handle(self, handle):
        if isinstance(handle, h5py.File):
            handle.close()
        elif isinstance(handle, dict):
            for v in handle.values():
                self.close_handle(v)

    def _fetch_from_h5(self, ds, indices: np.ndarray | None = None) -> np.ndarray:
        if ds is None:
            return np.full(
                self._shape if indices is None else ((len(indices),) + self._shape),
                fill_value=np.nan,
                dtype=self._dtype,
            )
        else:
            if indices is not None:
                # dataset slicing only works with sorted indices
                if (indices[:-1] < indices[1:]).all():
                    ds = ds[indices]
                else:
                    s = np.argsort(indices)
                    ds = ds[indices[s]][np.argsort(s)]

            if self._channel_idx is None:
                if indices is None:
                    ds = ds[()]
            else:
                if indices is None:
                    # load all channels into memory, since dataset slicing only works with sorted indices
                    ds = ds[()]
                if ds.ndim == 2:
                    ds = ds[self._channel_idx]
                elif ds.ndim == 3:
                    ds = ds[:, self._channel_idx]
                else:
                    ds = ds[:, :, self._channel_idx]

            return ds

    def _make_labels(self, label_cols, binary_thresholds: list | None = None):
        if binary_thresholds is None:
            binary_thresholds = []
        n = len(self._df)
        self._binary_labels = []
        self._binary_labels_raw = []  # raw values before applying threshold

        self._labels_df = self._df[label_cols].copy()
        self._df = self._df.drop(label_cols, axis=1)

        if self._labels_df is not None:
            for c in list(self._labels_df.columns):
                s = self._labels_df[c]
                if not s.isna().all():
                    add = True
                    for c0, th in binary_thresholds:
                        if c0 == c:
                            c_new = c + "_" + th
                            if th.startswith("<"):
                                self._labels_df[c_new] = (s < float(th[1:])).astype(np.float32)
                            elif th.startswith(">"):
                                self._labels_df[c_new] = (s > float(th[1:])).astype(np.float32)
                            elif th.startswith("["):
                                assert th.endswith("]")
                                th = th[1:-1].split(",")
                                assert len(th) == 2
                                # if an interval is given, positive class is outside of interval
                                self._labels_df[c_new] = 1 - s.between(float(th[0]), float(th[1])).astype(np.float32)
                            else:
                                assert False
                            self._labels_df.loc[s.isna(), c_new] = np.nan
                            self._binary_labels.append(c_new)
                            self._binary_labels_raw.append(c)
                            add = False

                    if not add:
                        pass
                    elif s.dtype.kind == "f":
                        if (s.isin((0, 1)) | s.isna()).all():
                            self._binary_labels.append(c)
                    elif s.dtype.kind == "b":
                        self._binary_labels.append(c)
                        self._labels_df[c] = s.astype(np.float32)
                    elif s.dtype.kind in "ui":
                        if (s.isin((0, 1)) | s.isna()).all():
                            self._binary_labels.append(c)
                            self._labels_df[c] = s.astype(np.float32)
                    elif s.dtype.name == "category" and len(s.cat.categories) == 2:
                        self._binary_labels.append(c)
                        self._labels_df[c] = s.cat.codes.astype(np.float32)
                        self._labels_df.loc[s.isna(), c] = np.nan

            if len(self._binary_labels_raw) != len(self._binary_labels):
                # if there are other binary labels, we have a mismatch and better set `binary_labels_raw` to []
                self._binary_labels_raw = []

            mask = self._labels_df[self._binary_labels].notna().any(axis=1)
            if not mask.all():
                self._labels_df = self._labels_df[mask]
                self._df = self._df[self._df.index.isin(self._labels_df.index)]

        if len(self._df) < n:
            print(f"Restricting to {len(self._df)} labeled samples.")

    def _sanitize_value(self, value):
        if isinstance(value, pd.Timestamp):
            value = (value - pd.Timestamp(0)) / pd.Timedelta(1, unit="h")
        elif isinstance(value, pd.Timedelta):
            value = value / pd.Timedelta(1, unit="h")
        return value

    def _add_labels_to_sample_dict(self, sample: dict, item: int) -> dict:
        for coll, key in [(self._binary_labels, "binary_labels"), (self._binary_labels_raw, "binary_labels_raw")]:
            if coll:
                try:
                    sample[key] = self._labels_df.loc[self._df.index[item], coll].values
                except KeyError:
                    sample[key] = np.full((len(coll),), fill_value=np.nan, dtype=np.float32)

        return sample


class SignalDataset(Dataset):
    def __init__(
        self,
        store: HDFSignalDatastore,
        mask=None,
        indices=None,
        transform=None,
        auto_open: bool = False,
        max_cache_size: int | None = 0,
    ):
        """
        Parameters
        ----------
        mask, indices : array_like, optional
            Always refer to individual samples in `store.df`, even if `store` returns sequences. In the latter case,
            the last individual sample in a sequence determines whether the sequence is contained in this data set.
        max_cache_size : int, optional, default=0
            Maximum cache size in number of samples. 0 disables caching, None allows the cache to grow indefinitely.
            In contrast to caching implemented in SignalDataStore classes, this class caches
            deterministically-transformed samples, such that these transformations do not need to be applied repeatedly.
        """
        self._store = store
        if indices is None:
            if mask is None:
                self._indices = np.arange(len(self._store))
            else:
                assert len(mask) == len(self._store.df)
                self._indices = np.flatnonzero(mask)
        else:
            assert mask is None
            assert (indices >= 0).all()
            assert (indices < len(self._store)).all()
            self._indices = indices

        self._transform = transform
        if self._transform is None:
            self._det_transform = self._non_det_transform = None
        else:
            self._det_transform, self._non_det_transform = split_deterministic(self._transform, return_none=True)
        self._handle = None
        self.auto_open = auto_open
        self._cache = {}
        self.max_cache_size = max_cache_size

    @property
    def store(self) -> HDFSignalDatastore:
        return self._store

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    @property
    def dataset(self) -> pd.DataFrame:
        return self._store.df.iloc[self._indices]

    @property
    def transform(self):
        return self._transform

    @property
    def sample_weight(self) -> np.ndarray | None:
        if "sample_weight" in self._store.df.columns:
            return self._store.df["sample_weight"].values[self._indices]
        return None

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, item: int) -> dict:
        if self.auto_open:
            self.open()

        sample = self._cache.get(item)
        if sample is None:
            sample = self._store.get(self._indices[item], self._handle)
            if self._det_transform is not None:
                sample = self._det_transform(sample)

            if self.max_cache_size is None or self.max_cache_size > len(self._cache):
                self._cache[item] = sample
            elif self.max_cache_size == 0:
                self._cache.clear()
            elif self.max_cache_size == len(self._cache):
                del self._cache[next(iter(self._cache))]
                self._cache[item] = sample

        if self._non_det_transform is not None:
            sample = self._non_det_transform(sample)
        return sample

    def dataloader(self, sampler=None, num_workers="auto", drop_last=False, **kwargs) -> DataLoader:
        if isinstance(sampler, str):
            if sampler in ("random", "shuffle"):
                kwargs["shuffle"] = True
            elif sampler == "weighted":
                sample_weight = self.sample_weight
                if sample_weight is None:
                    print("No sample weights found, using random sampler.")
                    kwargs["shuffle"] = True
                else:
                    if "sample_weight" in self._store.feature_cols:
                        print(
                            'WARNING: "sample_weight" is used for sampling and may be used additionally for'
                            " weighting the loss. Is this intended?"
                        )
                    kwargs["sampler"] = WeightedRandomSampler(sample_weight, len(self))
            elif sampler not in ("seq", "sequential"):
                raise ValueError(f'Unknown sampler: "{sampler}". Choose among "sequential", "random", "weighted".')
        else:
            kwargs["sampler"] = sampler

        if num_workers == "auto":
            num_workers = 0 if self._store.in_memory else 8
        else:
            num_workers = int(num_workers)
            if num_workers != 0 and self._store.in_memory:
                print(
                    f"WARNING: num_workers should be 0 if all data are loaded into main memory, but got {num_workers}"
                )
        if num_workers > 0 and self.max_cache_size != 0:
            print(f"WARNING: Caching might not work properly with num_workers={num_workers}")

        if drop_last in ("if_one", "if_1"):
            drop_last = len(self) % kwargs["batch_size"] == 1

        return DataLoader(self, num_workers=num_workers, drop_last=drop_last, **kwargs)

    def clear_cache(self):
        self._cache.clear()

    def open(self) -> Self:
        if self._handle is None:
            self._handle = self._store.get_handle()
        return self

    def close(self):
        self._store.close_handle(self._handle)
        self._handle = None
        self.clear_cache()

    def __del__(self):
        self.close()

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
