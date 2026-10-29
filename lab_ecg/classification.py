from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities.exceptions import TorchMetricsUserError


class BCELoss(pl.LightningModule):
    def __init__(self, gamma: float = 0):
        """Binary cross-entropy with focal loss (optional) _[1].

        Parameters
        ----------
        gamma : float
            Non-negative exponent of focal loss; set to 0 for default cross entropy loss. Gamma is set to 2 in _[1],
            so 2 seems to be a reasonable default value.

        Notes
        -----
        `BinaryFocalLoss` has another parameter, `alpha`, which also appears in _[1]. This parameter is meant to
        represent class weights, and should better be provided via "sample_weight" in this implementation.
        Applying both class weights (via "sample_weight") and focal loss is perfectly fine and may lead to even better
        results.

        References
        ----------
        .. [1] https://arxiv.org/pdf/1708.02002.pdf
        """
        super().__init__()
        self.gamma = gamma

    @property
    def input_args(self):
        return ["y_binary", "y_binary_hat"]

    @property
    def batch_args(self):
        return []

    def __call__(self, args: dict, reduce: bool = True):
        y, y_hat, sample_weight = args["y_binary"], args["y_binary_hat"], args.get("sample_weight")
        mask = torch.isfinite(y).float()

        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, torch.nan_to_num(y, 0), reduction="none")

        if self.gamma > 0:
            pt = torch.exp(-loss)
            loss = (1 - pt) ** self.gamma * loss

        loss = torch.mean(mask * loss, dim=1)

        if sample_weight is not None:
            loss = loss * sample_weight

        return (loss.mean() if reduce else loss), {}


class BinaryRanked(Metric):
    def __init__(self, auroc: str = "average", ap: str = "average"):
        assert auroc in ("average", "per_label", "none", None)
        assert ap in ("average", "per_label", "none", None)
        super().__init__()
        self.auroc = auroc
        self.ap = ap
        self.add_state("preds", [], "cat")
        self.add_state("targets", [], "cat")

    @property
    def input_args(self):
        return ["y_binary", "y_binary_hat"]

    @property
    def batch_args(self):
        return []

    @torch.jit.unused
    def forward(self, *args, **kwargs):
        if self._is_synced:
            raise TorchMetricsUserError(
                "The Metric shouldn't be synced when performing `update`. HINT: Did you forget to call `unsync`?"
            )
        with torch.no_grad():
            self.update(*args, **kwargs)

    def update(self, args: dict) -> None:
        y_true = args["y_binary"].detach() > 0  # (N, C)
        y_hat = args["y_binary_hat"].detach()  # (N, C)

        assert y_true.shape == y_hat.shape

        self.preds.append(y_hat)
        self.targets.append(y_true)

    def compute(self):
        if isinstance(self.preds, torch.Tensor):
            preds = self.preds.cpu().numpy()
        elif isinstance(self.preds, list) and len(self.preds) > 0:
            preds = np.concatenate([p.cpu().numpy() for p in self.preds])
        else:
            preds = None
        if isinstance(self.targets, torch.Tensor):
            targets = self.targets.cpu().numpy()
        elif isinstance(self.targets, list) and len(self.targets) > 0:
            targets = np.concatenate([t.cpu().numpy() for t in self.targets])
        else:
            targets = None

        if preds is not None and targets is not None:
            mask = np.isfinite(targets) & np.isfinite(preds)

            if self.auroc in ("average", "per_label"):
                aurocs = np.full((preds.shape[1],), dtype=np.float32, fill_value=np.nan)
                for i in range(preds.shape[1]):
                    try:
                        aurocs[i] = roc_auc_score(targets[mask[:, i], i], preds[mask[:, i], i])
                    except:  # noqa
                        pass
            else:
                aurocs = None

            if self.ap in ("average", "per_label"):
                aps = np.full((preds.shape[1],), dtype=np.float32, fill_value=np.nan)
                for i in range(preds.shape[1]):
                    try:
                        aps[i] = average_precision_score(targets[mask[:, i], i], preds[mask[:, i], i])
                    except:  # noqa
                        pass
            else:
                aps = None
        else:
            aurocs = np.full((1,), dtype=np.float32, fill_value=np.nan)
            aps = aurocs

        if self.auroc == "average":
            if np.isnan(aurocs).all():
                out = {"AUROC": torch.scalar_tensor(0, dtype=torch.float32, device="cpu")}
            else:
                out = {"AUROC": torch.scalar_tensor(np.nanmean(aurocs), dtype=torch.float32, device="cpu")}
        elif self.auroc == "per_label":
            out = {"AUROC_" + str(i): a for i, a in enumerate(aurocs) if not np.isnan(a)}
        else:
            out = {}

        if self.ap == "average":
            if np.isnan(aps).all():
                out["AP"] = torch.scalar_tensor(0, dtype=torch.float32, device="cpu")
            else:
                out["AP"] = torch.scalar_tensor(np.nanmean(aps), dtype=torch.float32, device="cpu")
        elif self.ap == "per_label":
            out.update({"AP_" + str(i): a for i, a in enumerate(aps) if not np.isnan(a)})

        return out

    @property
    def is_differentiable(self) -> bool:
        return False


class Classification(pl.LightningModule):
    def __init__(
        self,
        network,
        labels=None,
        add_input=(),
        loss=None,
        lr: float = 0.01,
        weight_decay: float = 0,
        epochs: int = 150,
        train_metrics: MetricCollection = None,
        val_metrics: MetricCollection = None,
        test_metrics: MetricCollection = None,
        cp=None,
    ):
        super().__init__()

        labels = labels or {}
        self.n_binary = labels.get("num_binary", 0)

        self._check_loss(loss)

        self.train_metrics = train_metrics.clone(prefix="train_") if train_metrics else None
        self.valid_metrics = val_metrics.clone(prefix="val_") if val_metrics else None
        self.test_metrics = test_metrics.clone(prefix="test_") if test_metrics else None

        self._check_metrics_input(train_metrics, prefix="(train)")
        self._check_metrics_input(val_metrics, prefix="(val)")
        self._check_metrics_input(test_metrics, prefix="(test)")

        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.checkpoint = cp
        self.network = network
        self.add_input = add_input

    def forward(self, x: torch.Tensor, *args):
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, dim=0)
        return self.network.forward(x, *args)

    @property
    def step_args(self):
        return ["y_binary", "y_binary_hat"] if self.n_binary > 0 else []

    @property
    def checkpoint(self) -> Path | None:
        return None if self._checkpoint is None else Path(self._checkpoint)

    @checkpoint.setter
    def checkpoint(self, value):
        if isinstance(value, Path):
            value = value.as_posix()
        self._checkpoint = value

    def get_args(self, batch: dict, add_input=None):
        x = batch["waveform"]
        n_new_axes = int(max(batch.get("n_new_axes", [0])))
        if n_new_axes > 0:
            m = x.shape[0]
            x = x.reshape(-1, *x.shape[n_new_axes + 1 :])
            m = x.shape[0] // m
        else:
            m = 1

        if add_input is None:
            add_input = self.add_input
        return [x] + [to_rank2(maybe_repeat(batch[k].float(), m)) for k, _ in add_input], m

    def _step(self, batch: dict, predict: bool = False):
        args, m0 = self.get_args(batch)
        m = None
        y = self.forward(*args)
        if not isinstance(y, (list, tuple)):
            y = [y]

        dct = {}
        if self.n_binary > 0:
            y_binary = batch["binary_labels"]
            y_binary_hat = y[0]
            if m is None:
                m = y_binary_hat.shape[0] // y_binary.shape[0]
            y_binary = maybe_repeat(y_binary, m)
            y_binary_hat, y_binary_hat_aux = maybe_split_auxiliary(y_binary_hat, y_binary.shape[1])
            dct.update(y_binary=y_binary, y_binary_hat=y_binary_hat)
            if y_binary_hat_aux is not None:
                dct["y_binary_hat_aux"] = y_binary_hat_aux

        if m is None:
            m = m0
        for k in ("sample_weight", "beat_idx", "signal_idx", "sample_id"):
            if k in batch:
                dct[k] = maybe_repeat(batch[k], m)
        return dct

    def _check_loss(self, loss):
        for loss_arg in loss.input_args:
            if loss_arg not in self.step_args:
                raise KeyError(
                    f"Loss of type '{type(loss).__name__}' requires input_arg {loss_arg} but it is not defined as"
                    " a return argument of the current system."
                )

    def _check_metrics_input(self, metrics, prefix=""):
        k_map = {"(train)": self.train_metrics, "(val)": self.valid_metrics, "(test)": self.test_metrics}

        if not k_map[prefix]:
            return

        for metric_name, metric in metrics.items():
            for metric_arg in metric.input_args:
                if metric_arg not in self.step_args:
                    raise KeyError(
                        f"{prefix} Metric '{metric_name}' of type {type(metric).__name__} requires input_arg"
                        f" {metric_arg} but it is not defined as a returned argument of the current system"
                    )

    def _retrieve_loss_batch_args(self, batch, step_dict):
        for loss_arg in self.loss.batch_args:
            if loss_arg not in batch:
                raise KeyError(
                    f"Loss '{type(self.loss).__name__}' requires batch_arg {loss_arg}"
                    " but it is not found in the given batch"
                )
            step_dict[loss_arg] = batch[loss_arg]
        return step_dict

    def _retrieve_metrics_batch_args(self, metrics, batch, step_dict, prefix=""):
        if not metrics:
            return step_dict

        for metric_name, metric in metrics.items():
            for metric_arg in metric.batch_args:
                if metric_arg not in batch:
                    raise KeyError(
                        f"{prefix} Metric '{metric_name}' of type {type(metric).__name__} requires batch_arg"
                        f" {metric_arg} but it is not found in the given batch"
                    )
                step_dict[metric_arg] = batch[metric_arg]
        return step_dict

    def _log_metrics(
        self, metrics: MetricCollection, batch, step_dict, prefix: str = "", on_step: bool = False
    ) -> dict:
        if metrics:
            if step_dict is None:
                metric_results = {k: v.compute() for k, v in metrics.items() if not getattr(v, "compute_on_step", True)}
            else:
                step_dict = self._retrieve_metrics_batch_args(metrics, batch, step_dict, prefix=prefix)
                metric_results = {k: v for k, v in metrics(step_dict).items() if v is not None}
            # flatten nested dicts, e.g., if a metric returns a dict
            metric_results = {
                (k + s): v
                for k, vs in metric_results.items()
                for s, v in ([(f"_{k0}", v0) for k0, v0 in vs.items()] if isinstance(vs, dict) else [("", vs)])
            }
            self.log_dict(metric_results, on_step=on_step, on_epoch=True, sync_dist=True)
        else:
            metric_results = {}
        return metric_results

    def configure_optimizers(self):
        if self.weight_decay > 0:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            # `optim.Adam` also has a `weight_decay` parameter, but this in fact corresponds to L2 penalty and differs
            # from `optim.AdamW`
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {"optimizer": self.optimizer}

    def training_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        step_dict = self._retrieve_loss_batch_args(batch, step_dict)

        loss, logs = self.loss(step_dict)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, sync_dist=True)
        self._log_metrics(self.train_metrics, batch, step_dict, on_step=True, prefix="(train)")

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        step_dict = self._retrieve_loss_batch_args(batch, step_dict)

        loss, logs = self.loss(step_dict)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        self._log_metrics(self.valid_metrics, batch, step_dict, prefix="(val)")

        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        step_dict = self._step(batch)
        step_dict = self._retrieve_loss_batch_args(batch, step_dict)

        loss, logs = self.loss(step_dict)
        self.log_dict({f"test_{k}": v for k, v in logs.items()}, sync_dist=True)
        self._log_metrics(self.test_metrics, batch, step_dict, prefix="(test)")

        self.log("test_loss", loss, sync_dist=True)

        return step_dict

    def predict_step(self, batch, batch_idx):
        return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self._step(batch, predict=True).items()}

    def on_train_epoch_start(self):
        if self.train_metrics:
            self.train_metrics.reset()

    def on_train_epoch_end(self, unused=None):
        if self.train_metrics:
            self._log_metrics(self.train_metrics, None, None, prefix="(train)")

    def on_validation_epoch_start(self):
        if self.valid_metrics:
            self.valid_metrics.reset()

    def on_validation_epoch_end(self):
        if self.valid_metrics:
            self._log_metrics(self.valid_metrics, None, None, prefix="(val)")

    def on_test_epoch_start(self):
        if self.test_metrics:
            self.test_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_metrics:
            self._log_metrics(self.test_metrics, None, None, prefix="(test)")


def maybe_repeat(t: torch.Tensor, repeats: int = 1, dim=0) -> torch.Tensor:
    if repeats == 1:
        return t
    elif isinstance(t, torch.Tensor):
        return torch.repeat_interleave(t, repeats, dim=dim)
    else:
        return np.repeat(t, repeats, axis=dim)


def maybe_split_auxiliary(arg, n_output: int):
    if isinstance(arg, torch.Tensor) and arg.ndim > 1 and arg.shape[1] > n_output:
        arg, aux = arg[:, -n_output:], arg[:, :-n_output]
        return arg, aux
    return arg, None


def to_rank2(t: torch.Tensor) -> torch.Tensor:
    if t.ndim != 2:
        return getattr(t, "view", t.resize)(int(np.prod(t.shape[:1])), int(np.prod(t.shape[1:])))
    return t


def tuple_constructor(loader, node):
    values = loader.construct_sequence(node)
    return tuple(values)


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)


def make(params: dict | None = None, weights: str | Path | None = None, train_dataset=None) -> Classification:
    if params is None:
        assert weights is not None
        weights = Path(weights)
        params = load_params(weights.parent / "params.yaml")
        if params is None:
            raise FileNotFoundError("No params file found.")

    network = params.get("network", {}).copy()
    labels = params.get("labels", {})
    architecture = network.pop("architecture", "xresnet1d101")

    add_input = network.pop("add_input", [])
    add_input = [k if isinstance(k, (list, tuple)) else [k, 1] for k in add_input]  # pairs `(key, dim)`
    num_add_input = sum(d for _, d in add_input)
    network.update(num_add_input=num_add_input)

    fs = network.pop("fs", None)
    n_timesteps = network.pop("n_timesteps", None)

    if architecture == "kwon":
        from .kwon import Kwon

        network.update(fs=fs, n_timesteps=n_timesteps)
        network = Kwon(**network, **labels)
    elif architecture.startswith("xresnet1d"):
        from . import xresnet1d

        network = getattr(xresnet1d, architecture)(**network, **labels)

    # metrics
    metrics = {}
    n_binary = labels.get("num_binary", 0)
    if n_binary > 0:
        metrics["ranked"] = BinaryRanked(
            auroc="per_label" if n_binary in (2, 3) else "average", ap="per_label" if n_binary in (2, 3) else "average"
        )
    metrics = MetricCollection(metrics) if metrics else None

    training = params.get("training", {})
    weight_decay = training.get("weight_decay", 0)
    if weight_decay == 0:
        weight_decay = training.get("lambda_norm", 0)
        if train_dataset is None:
            if weight_decay > 0:
                print("WARNING: Interpreting lambda_norm as weight_decay, since no training dataset is provided.")
        else:
            weight_decay *= (training.get("batch_size", 128) / (training["epochs"] * len(train_dataset))) ** 0.5
    elif training.get("lambda_norm", 0) > 0:
        print("WARNING: Ignoring lambda_norm, since weight_decay is specified.")
    kwargs = {
        "network": network,
        "labels": labels,
        "loss": BCELoss(gamma=params.get("training", {}).get("focal_loss", 0)),
        "lr": training.get("learning_rate", 1e-3),
        "weight_decay": weight_decay,
        "epochs": training["epochs"],
        "val_metrics": metrics,
        "add_input": add_input,
        "cp": weights,
    }

    return (
        Classification(**kwargs)
        if weights is None
        else Classification.load_from_checkpoint(checkpoint_path=weights, **kwargs)
    )


def load_params(file: Path):
    if file.exists():
        with open(file, mode="rt") as stream:
            params = yaml.safe_load(stream)
        return params
    return None
