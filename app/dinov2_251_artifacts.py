"""Bounded P6 transport validation shared byte-for-byte with Relay; never loads models."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile

CONTRACT = 'patchcore_dinov2_251_onnx_v1'
FORMAT = 'dino_cloud_result_p6_v1'
CONTRACT_V2 = 'patchcore_dinov2_251_onnx_v2'
CONTRACTS = (CONTRACT, CONTRACT_V2)
FORMATS = {FORMAT: (CONTRACT, 'dino_onnx_deployment_p6_v1', 'dino_cloud_task_p6_v1'),
           'dino_cloud_result_p6_v2': (CONTRACT_V2, 'dino_onnx_deployment_p6_v2', 'dino_cloud_task_p6_v2')}
MANIFEST = 'result_manifest.json'
IDENTITY_FIELDS = ('dataset_id', 'identity_sha256', 'run_id', 'run_identity_sha256')
TRAINING_FILES = {'training/model.ckpt', 'training/threshold.json', 'training/metrics.json', 'training/runtime_result.json'}
DEPLOYMENT_FILES = {'deployment/deployment.json', 'deployment/model.onnx', 'deployment/threshold.json',
                    'deployment/verification_report.json', 'deployment/runtime_requirements.txt', 'deployment/README_deployment.md'}
DOWNLOAD_PATTERN = r'^p6_result[/\\](?:result_manifest\.json|training[/\\](?:model\.ckpt|threshold\.json|metrics\.json|runtime_result\.json)|reports[/\\][a-zA-Z0-9_-]+\.json|validation[/\\](?:reference\.json|reference[/\\][a-zA-Z0-9_-]+\.(?:npz|npy))|deployment[/\\](?:deployment\.json|training_candidate_threshold\.json|threshold\.json|verification_report\.json|runtime_requirements\.txt|README_deployment\.md|model\.onnx(?:\.data(?:\.[0-9]+)?)?))$'


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def plain(root, name):
    if (not isinstance(name, str) or not name or ':' in name or '\\' in name
            or any(x in ('', '.', '..') or x.endswith((' ', '.')) for x in name.split('/'))):
        raise ValueError('unsafe P6 relative path')
    root = Path(root).absolute()
    path = root/name
    for item in [path, *path.parents]:
        if item.is_symlink() or (item.exists() and getattr(item.stat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('P6 links/reparse points forbidden')
    if not path.is_file():
        raise ValueError('missing P6 regular file: '+name)
    return path


def read_json(path):
    def pairs(values):
        result = {}
        for k, v in values:
            if k in result:
                raise ValueError('duplicate P6 JSON key')
            result[k] = v
        return result
    def invalid(value):
        raise ValueError('nonfinite P6 JSON')
    if Path(path).stat().st_size > 16*1024**2:
        raise ValueError('P6 JSON budget exceeded')
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError('P6 JSON object required')
    return value


def inventory(root, rows, max_bytes):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100000:
        raise ValueError('P6 file count budget exceeded')
    seen, total = set(), 0
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'path', 'size', 'sha256'}
                or type(row['size']) is not int or row['size'] < 1
                or not isinstance(row['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('invalid P6 file entry')
        name = row['path']
        path = plain(root, name)
        if name.casefold() in seen:
            raise ValueError('duplicate P6 file')
        seen.add(name.casefold())
        total += row['size']
        if total > max_bytes or path.stat().st_size != row['size'] or sha256(path) != row['sha256']:
            raise ValueError('P6 file budget/content mismatch')
    return {row['path']: row for row in rows}


def validate_result(root, *, expected_identity, expected_task_sha256, max_bytes=8*1024**3, expected_contract=None):
    document = read_json(plain(root, MANIFEST))
    versions = FORMATS.get(document.get('format'))
    if (versions is None or document.get('artifact_contract') != versions[0]
            or (expected_contract is not None and versions[0] != expected_contract)
            or document.get('training_status') != 'PASS' or document.get('windows_status') != 'PENDING'):
        raise ValueError('unsupported P6 result contract')
    if not re.fullmatch('[0-9a-f]{64}', str(expected_task_sha256)):
        raise ValueError('trusted frozen task digest required')
    if document.get('task_sha256') != expected_task_sha256:
        raise ValueError('P6 task mismatch')
    if not isinstance(expected_identity, dict) or any(not expected_identity.get(k) or
            document.get('identity', {}).get(k) != expected_identity[k] for k in IDENTITY_FIELDS):
        raise ValueError('P6 frozen identity mismatch')
    rows = inventory(root, document['files'], max_bytes)
    if document.get('file_set_sha256') != canonical_hash(document['files']) or not TRAINING_FILES <= rows.keys():
        raise ValueError('P6 training inventory incomplete')
    training = read_json(plain(root, 'training/runtime_result.json'))
    if training.get('format') != 'vision_workshop_dinov2_251_training_stage_v1' or training.get('training_status') != 'completed':
        raise ValueError('invalid P6 training stage')
    if any(training.get('run_identity', {}).get(k) != expected_identity[k] for k in IDENTITY_FIELDS):
        raise ValueError('P6 training identity mismatch')
    if {x.get('path') for x in training.get('files', [])} != {'model.ckpt', 'threshold.json', 'metrics.json'}:
        raise ValueError('P6 training subpackage incomplete')
    for row in training['files']:
        if rows.get('training/'+row['path']) != {**row, 'path': 'training/'+row['path']}:
            raise ValueError('P6 training subpackage inventory mismatch')
    threshold = read_json(plain(root, 'training/threshold.json'))
    if (type(threshold.get('threshold')) not in (float, int) or not math.isfinite(threshold['threshold'])
            or threshold.get('model_sha256') != rows['training/model.ckpt']['sha256']
            or threshold.get('score_kind') != 'patchcore_dinov2_l2_nnmax_raw_v1'
            or threshold.get('decision_rule') != 'score_gte_threshold'
            or threshold.get('calibration_owner') != 'training'):
        raise ValueError('P6 training threshold binding mismatch')
    status = document.get('onnx_status')
    if status not in {'PASS', 'FAIL', 'TIMEOUT', 'CANCELED', 'NOT_RUN'}:
        raise ValueError('invalid P6 format status')
    if not any(x.startswith('reports/') for x in rows):
        raise ValueError('P6 stage report required')
    allowed = set(TRAINING_FILES) | {x for x in rows if re.fullmatch(r'reports/[a-zA-Z0-9_-]+\.json', x)}
    deployment_digest = ''
    if status == 'PASS':
        if not DEPLOYMENT_FILES <= rows.keys():
            raise ValueError('P6 deployment incomplete')
        deployment = read_json(plain(root, 'deployment/deployment.json'))
        if (deployment.get('format') != versions[1]
                or deployment.get('task_contract', {}).get('format') != versions[2]
                or deployment.get('task_sha256') != expected_task_sha256
                or canonical_hash(deployment.get('task_contract')) != expected_task_sha256
                or deployment.get('cloud_qualification') != 'PASS'
                or deployment.get('windows_qualification') != 'PENDING'):
            raise ValueError('P6 deployment source/qualification mismatch')
        bound = deployment['graph']['files'] + [deployment['threshold_file'], deployment['verification_report']] + deployment['support_files']
        if versions[0] == CONTRACT_V2:
            bound += [deployment['training_candidate_file']]
            calibration = read_json(plain(root, 'deployment/threshold.json'))
            if (calibration.get('format') != 'dino_onnx_rgb_calibration_p6_v2'
                    or deployment.get('deployment_calibration_state') != 'FROZEN'
                    or calibration.get('task_sha256') != expected_task_sha256
                    or calibration.get('graph_files') != deployment['graph']['files']
                    or calibration.get('environment', {}).get('scope') != 'linux_cloud'
                    or calibration.get('threshold') != deployment['threshold']):
                raise ValueError('P6 v2 deployment calibration binding mismatch')
        for row in bound:
            if rows.get('deployment/'+row['path']) != {**row, 'path': 'deployment/'+row['path']}:
                raise ValueError('P6 deployment file binding mismatch')
            allowed.add('deployment/'+row['path'])
        allowed.add('deployment/deployment.json')
        deployment_digest = rows['deployment/deployment.json']['sha256']
        reference = read_json(plain(root, 'validation/reference.json'))
        if reference['graph'] != deployment['graph'] or reference['threshold'] != deployment['threshold']:
            raise ValueError('P6 training reference differs from deployment')
        allowed.add('validation/reference.json')
        if versions[0] == CONTRACT_V2 and reference.get('deployment_calibration_sha256') != deployment['threshold_file']['sha256']:
            raise ValueError('P6 v2 deployment reference calibration mismatch')
        for item in reference['items']:
            for key in (('reference', 'deployment_reference') if versions[0] == CONTRACT_V2 else ('reference',)):
                row = item[key]
                if rows.get('validation/'+row['path']) != {**row, 'path': 'validation/'+row['path']}:
                    raise ValueError('P6 validation reference file mismatch')
                allowed.add('validation/'+row['path'])
    if set(rows) != allowed or any(not re.fullmatch(DOWNLOAD_PATTERN, 'p6_result/'+name) for name in rows):
        raise ValueError('P6 undeclared/disallowed file')
    return {'format': ('dino_p6_verified_receipt_v2' if versions[0] == CONTRACT_V2 else 'dino_p6_verified_receipt_v1'), 'identity': {k: expected_identity[k] for k in IDENTITY_FIELDS},
            'task_sha256': expected_task_sha256, 'result_manifest_sha256': sha256(Path(root)/MANIFEST),
            'file_set_sha256': document['file_set_sha256'], 'deployment_manifest_sha256': deployment_digest,
            'training_status': 'PASS', 'onnx_status': status, 'windows_status': 'PENDING'}


def package_result(root, destination, *, expected_identity, expected_task_sha256, storage_budget=None, expected_contract=None):
    receipt = validate_result(root, expected_identity=expected_identity, expected_task_sha256=expected_task_sha256, expected_contract=expected_contract)
    rows = read_json(Path(root)/MANIFEST)['files']
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.p6-', suffix='.zip', dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for row in rows + [{'path': MANIFEST, 'sha256': receipt['result_manifest_sha256'],
                                 'size': (Path(root)/MANIFEST).stat().st_size}]:
                if storage_budget:
                    storage_budget.check_free(row['size'])
                total, value = 0, hashlib.sha256()
                with plain(root, row['path']).open('rb') as source, archive.open('p6_result/'+row['path'], 'w', force_zip64=True) as writer:
                    for block in iter(lambda: source.read(1024*1024), b''):
                        total += len(block)
                        if total > row['size']:
                            raise ValueError('P6 source grew during packaging')
                        value.update(block)
                        writer.write(block)
                if total != row['size'] or value.hexdigest() != row['sha256']:
                    raise ValueError('P6 source changed during packaging')
        receipt.update(archive_sha256=sha256(temporary), archive_size=Path(temporary).stat().st_size)
        os.replace(temporary, destination)
        return receipt
    finally:
        Path(temporary).unlink(missing_ok=True)
