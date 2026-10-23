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
    if task not in {"detect", "segment", "pose", "obb", "classify"}:
        return train(*args, **kwargs)

    if task == "classify":
        from pathlib import Path
        from PIL import Image
        from torchvision.datasets import ImageFolder
        import importlib
        import os

        root = Path(kwargs["data"])
        if root.is_file():
            root = root.parent
        counts = {}
        mapping = None
        for split in ("train", "val"):
            dataset = ImageFolder(root / split)
            if mapping is not None and dataset.class_to_idx != mapping:
                raise ValueError("Classification train/val class mappings differ")
            mapping = dataset.class_to_idx
            for filename, _ in dataset.samples:
                with Image.open(filename) as image:
                    image.verify()
            counts[split] = len(dataset)
        trainer = kwargs.get("trainer")
        if trainer is not None:
            module = importlib.import_module(trainer.__module__)
            if getattr(module, trainer.__name__, None) is not trainer:
                raise ValueError("Custom classification trainer must be importable by torchrun")
            module_file = Path(module.__file__).resolve()
            import_root = module_file.parents[len(trainer.__module__.split(".")) - 1]
            os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(import_root), os.environ.get("PYTHONPATH", "")]))
        if min(counts.values()) < len(devices):
            kwargs["device"] = str(devices[0])
        print(f"[RELAY GPU POLICY] classification train={counts['train']} val={counts['val']} "
              f"requested_gpus={len(devices)} selected_device={kwargs['device']}", flush=True)
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
