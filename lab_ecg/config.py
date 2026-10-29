from torchvision.transforms import Compose

from . import dataset, transforms


def get_default_config(architecture: str, checkpoint_path: str) -> dict:
    if architecture == "kwon":
        out = {
            "network": {
                "add_input": ["male", "age"],
                "architecture": architecture,
            },
            "preprocessing": {
                "center_snippets": True,
                "filter": {"type": "butter", "high": 150, "low": 0.05},
                "normalize": None,
                "snippet_duration": 8,
                "target_fs": 500,
            },
            "training": {
                "batch_size": 48,
                "epochs": 200,
                "gradient_clip": 1,
                "lambda_norm": 0,
                "learning_rate": 0.001,
                "patience": 10,
            },
        }
    else:
        out = {
            "network": {
                "add_input": ["male", "age"],
                "architecture": architecture,
                "kernel_size": 5,
                "latent_dim": 128,
                "lin_ftrs_head": [],
                "ps_head": 0.5,
            },
            "preprocessing": {
                "center_snippets": False,
                "filter": None,
                "normalize": None,
                "snippet_duration": 2.5,
                "target_fs": 100,
            },
            "training": {
                "batch_size": 128,
                "epochs": 200,
                "learning_rate": 0.01,
                "patience": 20,
            },
        }

    out["paths"] = {"checkpoints": checkpoint_path}
    return out


def build_pipeline(params: dict, store: dataset.HDFSignalDatastore, **kwargs) -> tuple[Compose, Compose]:
    transform, test_transform = get_transform(params, store, **kwargs)
    waveform_output = get_waveform_output(transform, store=store)
    params["network"].update(
        input_channels=len(waveform_output["channels"]),
        fs=waveform_output.get("fs"),
        n_timesteps=waveform_output.get("length"),
    )
    params["labels"] = {"num_binary": len(store.binary_labels)}
    params["data_store"] = dict(params.get("data_store", {}), **store.get_dict())

    return transform, test_transform


def get_transform(params: dict, store, channel_axis: int = -2, temporal_axis: int = -1) -> tuple[Compose, Compose]:
    snippet_len = int(params["preprocessing"]["target_fs"] * params["preprocessing"]["snippet_duration"])
    transform = [
        transforms.SelectChannels(
            params["preprocessing"].get("select_channels"), channel_names=store.channels, channel_axis=channel_axis
        ),
        transforms.Impute(method="value", value=0.0, interpolation_axis=temporal_axis),
        transforms.Resample(params["preprocessing"]["target_fs"], axis=temporal_axis),
    ]
    filt = params["preprocessing"].get("filter")
    if filt is not None:
        transform.append(transforms.Filter(**{k: v for k, v in filt.items()}, axis=temporal_axis))
    normalize = params["preprocessing"].get("normalize")
    if isinstance(normalize, str) and normalize.endswith("-uniform"):
        normalize = normalize[:-8]
        normalize_axis = (channel_axis, temporal_axis)
    else:
        normalize_axis = temporal_axis
    transform += [
        transforms.Normalize(normalize, axis=normalize_axis),
        transforms.RandomSnippet(
            min_len=snippet_len,
            max_len=snippet_len,
            center=params["preprocessing"].get("center_snippets", False),
            axis=temporal_axis,
        ),
    ]
    transform.append(transforms.ToTensor())

    test_transform = []
    for tr in transform:
        if isinstance(tr, transforms.RandomSnippet) and not tr.center:
            test_transform.append(
                transforms.SlidingWindow(
                    length=(tr.min_len + tr.max_len) // 2, offset=(tr.min_len + tr.max_len) // 4, axis=tr.axis
                )
            )
        else:
            test_transform.append(tr)

    return Compose(transform), Compose(test_transform)


def get_waveform_output(transform, store: dataset.HDFSignalDatastore | None = None, **kwargs) -> dict:
    # known keys: "channels" (list), "fs" (float), "length" (int)

    if store is not None:
        if "channels" not in kwargs:
            if getattr(store, "multichannel", True) or len(store.channels) <= 1:
                kwargs["channels"] = store.channels
            else:
                kwargs["channels"] = [None]
        if "fs" not in kwargs:
            kwargs["fs"] = store.fs
        if "length" not in kwargs:
            kwargs["length"] = store.shape[-1]

    if hasattr(transform, "waveform_output"):
        kwargs = transform.waveform_output(**kwargs)
    elif isinstance(transform, Compose):
        for t in transform.transforms:
            kwargs = get_waveform_output(t, **kwargs)

    return kwargs
