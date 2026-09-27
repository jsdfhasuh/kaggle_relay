"""P6 transport fixtures are not executable models or cloud acceptance."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.dinov2_251_artifacts import FORMAT, CONTRACT, canonical_hash, sha256, validate_result, package_result


@pytest.fixture
def result(tmp_path):
    root = tmp_path/'p6_result'
    (root/'training').mkdir(parents=True)
    (root/'reports').mkdir()
    identity = {'dataset_id': 'a'*64, 'identity_sha256': 'b'*64, 'run_id': 'run', 'run_identity_sha256': 'c'*64}
    for name in ('model.ckpt', 'threshold.json', 'metrics.json'):
        (root/'training'/name).write_bytes(b'transport fixture')
    (root/'training'/'threshold.json').write_text(json.dumps({'threshold': .1,
        'model_sha256': sha256(root/'training'/'model.ckpt'), 'score_kind': 'patchcore_dinov2_l2_nnmax_raw_v1',
        'decision_rule': 'score_gte_threshold', 'calibration_owner': 'training'}))
    def rows(folder):
        return [{'path': p.relative_to(folder).as_posix(), 'size': p.stat().st_size, 'sha256': sha256(p)}
                for p in sorted(folder.rglob('*')) if p.is_file()]
    (root/'training/runtime_result.json').write_text(json.dumps({'format': 'vision_workshop_dinov2_251_training_stage_v1',
        'training_status': 'completed', 'run_identity': identity, 'files': rows(root/'training')}))
    (root/'reports/export.json').write_text('{"onnx_status":"FAIL"}')
    files = rows(root)
    document = {'format': FORMAT, 'artifact_contract': CONTRACT, 'identity': identity,
        'task_sha256': 'd'*64, 'training_status': 'PASS', 'onnx_status': 'FAIL', 'windows_status': 'PENDING',
        'files': files, 'file_set_sha256': canonical_hash(files)}
    (root/'result_manifest.json').write_text(json.dumps(document))
    return root, identity, document


def test_partial_failure_pack_and_receipt(result, tmp_path):
    root, identity, document = result
    receipt = package_result(root, tmp_path/'archive.zip', expected_identity=identity, expected_task_sha256='d'*64)
    assert receipt['training_status'] == 'PASS' and receipt['onnx_status'] == 'FAIL'
    assert receipt['archive_sha256'] == sha256(tmp_path/'archive.zip')
    assert receipt['deployment_manifest_sha256'] == ''


@pytest.mark.parametrize('mutation', ['same_length', 'missing', 'identity', 'task', 'claim_pass', 'duplicate'])
def test_negative_transport(result, mutation):
    root, identity, document = result
    task_hash = 'd'*64
    if mutation == 'same_length':
        (root/'training/model.ckpt').write_bytes(b'Transport fixture')
    elif mutation == 'missing':
        (root/'training/threshold.json').unlink()
    elif mutation == 'identity':
        identity = {**identity, 'run_id': 'other'}
    elif mutation == 'task':
        task_hash = 'e'*64
    elif mutation == 'claim_pass':
        document['onnx_status'] = 'PASS'
    else:
        document['files'].append(document['files'][0])
    (root/'result_manifest.json').write_text(json.dumps(document))
    with pytest.raises((ValueError, OSError)):
        validate_result(root, expected_identity=identity, expected_task_sha256=task_hash)


def test_p6_schema_survives_database_reopen(tmp_path):
    from app.database import RelayDb
    from app.schemas import CreateJobRequest
    from test_relay_api import job_request_body
    values = job_request_body(b'dataset', b'kernel')
    values.update(artifact_contract=CONTRACT, dataset_id='a'*64, identity_sha256='b'*64,
                  run_id='run', run_identity_sha256='c'*64)
    request = CreateJobRequest(**values)
    db = RelayDb(tmp_path/'db.sqlite')
    assert request.artifact_contract == CONTRACT
    db.create_job({'job_id': 'job-p6', **request.model_dump()})
    assert RelayDb(tmp_path/'db.sqlite').get_job('job-p6')['artifact_contract'] == CONTRACT


@pytest.mark.parametrize('advance_after_download', [False, True])
def test_output_guard_rejects_head_advance(tmp_path, monkeypatch, advance_after_download):
    from app.dinov2_kaggle_output import download_version
    from app.dinov2_251_artifacts import DOWNLOAD_PATTERN
    requests = []
    observations = []
    class Client:
        def get_kernel(self, request):
            observations.append(request)
            version = 8 if advance_after_download and len(observations) > 1 else 7
            return SimpleNamespace(metadata=SimpleNamespace(current_version_number=version,
                ref='owner/run', dataset_data_sources=['owner/data/3']), blob=SimpleNamespace(source='pass\n'))
        def list_kernel_session_output(self, request):
            requests.append(request)
            return SimpleNamespace(files=[SimpleNamespace(file_name='p6_result/reports/export.json',
                url='https://example.invalid/signed')], next_page_token='')
    class Service:
        kernels = SimpleNamespace(kernels_api_client=Client())
        def __enter__(self): return self
        def __exit__(self, typ, value, traceback): pass
    class Api:
        def build_kaggle_client(self): return Service()
    class Download:
        def __enter__(self): return self
        def __exit__(self, typ, value, traceback): pass
        def raise_for_status(self): pass
        def iter_content(self, size): yield b'{}'
    def get(url, stream, timeout):
        assert stream is True
        return Download()
    monkeypatch.setattr('requests.get', get)
    def download():
        return download_version(Api(), {'kernel_ref': 'owner/run', 'kernel_version': 7,
            'source_sha256': __import__('hashlib').sha256(b'pass\n').hexdigest(),
            'dataset_sources': ['owner/data/3']}, tmp_path/'out', pattern=DOWNLOAD_PATTERN, max_bytes=10)
    if advance_after_download:
        with pytest.raises(ValueError, match='advanced'):
            download()
    else:
        assert download()['kernel_version'] == 7
    assert len(observations) == 2
    assert requests[0].version_label == ''
    assert requests[0].kernel_slug == 'run'


@pytest.mark.parametrize('observed_dataset', ['owner/data/3', 'owner/data'])
def test_observation_requires_frozen_candidate_at_current_head(observed_dataset):
    from app.dinov2_kaggle_output import observe_kernel
    class Client:
        def get_kernel(self, request):
            assert request.kernel_slug == 'run'
            assert request.version_label == ''
            return SimpleNamespace(metadata=SimpleNamespace(current_version_number=7,
                ref='owner/run', dataset_data_sources=[observed_dataset]), blob=SimpleNamespace(source='pass\n'))
    class Service:
        kernels = SimpleNamespace(kernels_api_client=Client())
        def __enter__(self): return self
        def __exit__(self, typ, value, traceback): pass
    class Api:
        def build_kaggle_client(self): return Service()
    observation = observe_kernel(Api(), 'owner/run', 'pass\n', ['owner/data/3'], version=7)
    assert observation['kernel_version'] == 7
    with pytest.raises(ValueError, match='binding'):
        observe_kernel(Api(), 'owner/run', 'changed', ['owner/data/3'], version=7)

    with pytest.raises(ValueError, match='version'):
        observe_kernel(Api(), 'owner/run', 'pass\n', ['owner/data/3'], version=1)

    for invalid in ['owner/other/3', 'other/data/3', 'owner/data', 'owner/data/0']:
        with pytest.raises(ValueError, match='binding'):
            observe_kernel(Api(), 'owner/run', 'pass\n', [invalid], version=7)


def test_explicit_wrong_dataset_version_is_never_accepted():
    from app.dinov2_kaggle_output import _dataset_binding
    assert not _dataset_binding(['owner/data/4'], ['owner/data/3'])
    assert not _dataset_binding([], ['owner/data/3'])
