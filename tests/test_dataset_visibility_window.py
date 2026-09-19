from types import SimpleNamespace

import pytest

from app.kaggle_adapter import DatasetUploadReceipt, KaggleAdapter, KaggleAdapterError
from test_relay_api import make_settings


def test_late_private_dataset_visibility_preserves_version_and_inventory_checks(tmp_path, monkeypatch):
    adapter = KaggleAdapter(make_settings(tmp_path), lambda _: None)
    times = iter([0, 600, 620])
    monkeypatch.setattr('app.kaggle_adapter.time.time', lambda: next(times))
    monkeypatch.setattr(adapter, '_sleep', lambda _: None)
    replies = iter([
        SimpleNamespace(returncode=1, stdout='403 Forbidden'),
        SimpleNamespace(returncode=0, stdout='{"status":"ready","current_version_number":1}'),
    ])
    monkeypatch.setattr(adapter, '_run', lambda *args, **kwargs: next(replies))
    checked = []
    def inventory(ref, version_number):
        checked.append((ref, version_number))
        return {'data.yaml': 20}
    monkeypatch.setattr(adapter, '_dataset_file_inventory', inventory)
    cancels = []
    adapter.dataset_cancel_check = lambda: cancels.append(True)
    adapter.wait_dataset('owner/private-data', permission_grace_seconds=900,
                         upload_receipt=DatasetUploadReceipt(1, (('data.yaml', 20),)))
    assert checked == [('owner/private-data', 1)]
    assert len(cancels) == 2


def test_private_dataset_forbidden_still_fails_after_bounded_grace(tmp_path, monkeypatch):
    adapter = KaggleAdapter(make_settings(tmp_path), lambda _: None)
    times = iter([0, 901])
    monkeypatch.setattr('app.kaggle_adapter.time.time', lambda: next(times))
    monkeypatch.setattr(adapter, '_run', lambda *a, **k: SimpleNamespace(returncode=1, stdout='403 Forbidden'))
    with pytest.raises(KaggleAdapterError, match='403 Forbidden'):
        adapter.wait_dataset('owner/private-data', permission_grace_seconds=900,
                             upload_receipt=DatasetUploadReceipt(1, (('data.yaml', 20),)))
