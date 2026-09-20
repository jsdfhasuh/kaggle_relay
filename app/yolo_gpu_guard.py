# Embedded by the gateway; executed in Kaggle immediately before model.train.
def _relay_yolo_train(train, *args, **kwargs):
    requested = kwargs.get("device")
    if isinstance(requested, str) and requested.strip().lower() in {"cpu", "mps"}:
        return train(*args, **kwargs)
    if isinstance(requested, (tuple, list)):
        devices = list(requested)
    elif requested is None or str(requested).strip().lower() in {"", "auto"}:
        # The generated client resolves auto before this call; Ultralytics
        # itself defaults to one GPU when the device argument is omitted.
        return train(*args, **kwargs)
    else:
        devices = [part.strip() for part in str(requested).split(",") if part.strip()]
    if len(devices) < 2:
        return train(*args, **kwargs)
    task = getattr(getattr(train, "__self__", None), "task", "detect")
    if task not in {"detect", "segment", "pose", "obb"}:
        return train(*args, **kwargs)

    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    # Run after client archive extraction/YAML repair, including cache hits.
    # Use training's own image/label validator, so corrupt samples don't count.
    data = check_det_dataset(kwargs["data"], autodownload=False)
    counts = {}
    for split in ("train", "val"):
        dataset = YOLODataset(
            img_path=data[split], data=data, task=task,
            imgsz=kwargs.get("imgsz", 640), augment=False, cache=False, rect=False,
            single_cls=kwargs.get("single_cls", False), classes=kwargs.get("classes"),
            fraction=kwargs.get("fraction", 1.0) if split == "train" else 1.0,
            prefix=f"[RELAY GPU POLICY] {split}: ",
        )
        counts[split] = len(dataset)
        del dataset
    if min(counts.values()) < len(devices):
        kwargs["device"] = str(devices[0])
        decision = f"using single GPU {devices[0]} to avoid empty DDP shards"
    else:
        decision = "keeping requested multi-GPU training"
    print(
        f"[RELAY GPU POLICY] train={counts['train']} val={counts['val']} "
        f"requested_gpus={len(devices)}; {decision}", flush=True,
    )
    return train(*args, **kwargs)
