"""Validate the bound P6 training device without rewriting uploaded snapshots."""
from app.dinov2_251_artifacts import read_json, plain, CONTRACTS, CONTRACT_V2

GPU_PROFILE = 'dino_cuda_features_cpu_coreset_v1'
GPU_POLICY = {'profile': GPU_PROFILE, 'device': 'cuda', 'coreset_device': 'cpu',
              'precision': 'fp32', 'torch_index': 'cu126', 'batch': 1, 'workers': 0}
HOST_GPU_POLICY = {**GPU_POLICY, 'profile': 'dino_cuda_host_torch210_cu128_cpu_coreset_v2',
                   'torch_index': 'host_cu128', 'environment': 'verified_host_overlay_v1'}


def validate_training_request(kernel_dir, artifact_contract):
    if artifact_contract not in CONTRACTS:
        return
    task = read_json(plain(kernel_dir, 'p6_task.json'))
    metadata = read_json(plain(kernel_dir, 'kernel-metadata.json'))
    enabled = metadata.get('enable_gpu')
    if type(enabled) is str and enabled in {'true', 'false'}:
        enabled = enabled == 'true'
    profile = task.get('training_execution')
    if profile is not None:
        if (artifact_contract != CONTRACT_V2 or task.get('format') != 'dino_cloud_task_p6_v2'
                or profile not in (GPU_POLICY, HOST_GPU_POLICY)):
            raise ValueError('unsupported DINO training execution policy')
        if enabled is not True:
            raise ValueError('DINO CUDA task requires bound Kernel enable_gpu=true')
    elif enabled is not False:
        raise ValueError('DINO CPU task must retain Kernel enable_gpu=false')
    if task.get('execution', {}).get('device') != 'cpu':
        raise ValueError('DINO deployment execution must remain CPU')
