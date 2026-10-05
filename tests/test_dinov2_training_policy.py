import json

import pytest

from app.dinov2_training_policy import validate_training_request, GPU_POLICY, GPU_PROFILE, HOST_GPU_POLICY
from app.dinov2_251_artifacts import CONTRACT_V2


@pytest.mark.parametrize('enabled', [True, 'true'])
@pytest.mark.parametrize('policy', [GPU_POLICY, HOST_GPU_POLICY])
def test_gpu_policy_preserved_and_advertised(tmp_path, enabled, policy):
    from app.schemas import HealthResponse
    task = {'format': 'dino_cloud_task_p6_v2', 'execution': {'device': 'cpu'}, 'training_execution': policy}
    (tmp_path/'p6_task.json').write_text(json.dumps(task))
    (tmp_path/'kernel-metadata.json').write_text(json.dumps({'enable_gpu': enabled}))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    validate_training_request(tmp_path, CONTRACT_V2)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    health = HealthResponse(status='ok', version='fixture', storage_dir='fixture', free_bytes=1)
    assert CONTRACT_V2 in health.artifact_contracts and policy['profile'] in health.training_profiles


@pytest.mark.parametrize('case', ['cpu', 'index', 'precision', 'deployment', 'old_contract'])
@pytest.mark.parametrize('policy', [GPU_POLICY, HOST_GPU_POLICY])
def test_gpu_invalid_policy_rejected(tmp_path, case, policy):
    task = {'format': 'dino_cloud_task_p6_v2', 'execution': {'device': 'cpu'}, 'training_execution': dict(policy)}
    enabled = True
    contract = CONTRACT_V2
    if case == 'cpu': enabled = False
    if case == 'index': task['training_execution']['torch_index'] = 'cpu'
    if case == 'precision': task['training_execution']['precision'] = 'fp16'
    if case == 'deployment': task['execution']['device'] = 'cuda'
    if case == 'old_contract': contract = 'patchcore_dinov2_251_onnx_v1'
    (tmp_path/'p6_task.json').write_text(json.dumps(task))
    (tmp_path/'kernel-metadata.json').write_text(json.dumps({'enable_gpu': enabled}))
    with pytest.raises(ValueError): validate_training_request(tmp_path, contract)


def test_old_cpu_and_unrelated_contracts_unchanged(tmp_path):
    validate_training_request(tmp_path, 'yolo')
    validate_training_request(tmp_path, 'patchcore_dinov2_v3')
    (tmp_path/'p6_task.json').write_text(json.dumps({'format': 'dino_cloud_task_p6_v1', 'execution': {'device': 'cpu'}}))
    (tmp_path/'kernel-metadata.json').write_text(json.dumps({'enable_gpu': 'false'}))
    validate_training_request(tmp_path, 'patchcore_dinov2_251_onnx_v1')
