"""P6 original-version transport; reject an advanced Kernel instead of using latest."""
from pathlib import Path
import hashlib
import re


def _dataset_binding(actual, expected):
    """SDK metadata omits versions; preserve frozen candidates, never query latest.

    The exact source embeds the frozen task and verifies all runtime/data bytes
    before training. Unversioned metadata alone is not a dataset version receipt.
    """
    if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected) or not expected:
        return False
    for observed, frozen in zip(actual, expected):
        if not isinstance(frozen, str) or not re.fullmatch(r'[^/]+/[^/]+/[1-9][0-9]*', frozen):
            return False
        if observed not in (frozen, frozen.rsplit('/', 1)[0]):
            return False
    return True


def observe_kernel(api, kernel_ref, expected_source, expected_datasets, *, version=1):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    owner, slug = kernel_ref.split('/')
    request = ApiGetKernelRequest()
    request.user_name = owner
    request.kernel_slug = slug
    if type(version) is not int or version < 1:
        raise ValueError('explicit original Kernel version required')
    # SDK 2.2.4 exposes version_label but the live service returns 404 for numeric
    # labels. The current head must equal the frozen candidate; no substitution.
    with api.build_kaggle_client() as service:
        response = service.kernels.kernels_api_client.get_kernel(request)
    metadata = response.metadata
    actual_version = metadata.current_version_number
    if type(actual_version) is not int or actual_version < 1 or (version is not None and actual_version != version):
        raise ValueError('exact Kernel version metadata unavailable/mismatched')
    if (metadata.ref != kernel_ref or response.blob.source != expected_source
            or not _dataset_binding(metadata.dataset_data_sources, expected_datasets)):
        raise ValueError('authenticated Kernel source/Dataset binding mismatch')
    return {'kernel_ref': kernel_ref, 'kernel_version': actual_version,
            'source_sha256': hashlib.sha256(expected_source.encode('utf-8')).hexdigest(),
            'dataset_sources': expected_datasets,
            'observed_dataset_sources': metadata.dataset_data_sources,
            'dataset_binding_method': 'frozen_submission_and_verified_runtime_content'}


def verify_observation(api, observation):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    owner, slug = observation['kernel_ref'].split('/')
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = owner, slug
    with api.build_kaggle_client() as service:
        response = service.kernels.kernels_api_client.get_kernel(request)
    if (response.metadata.ref != observation['kernel_ref']
            or response.metadata.current_version_number != observation['kernel_version']
            or hashlib.sha256(response.blob.source.encode('utf-8')).hexdigest() != observation['source_sha256']
            or not _dataset_binding(response.metadata.dataset_data_sources, observation['dataset_sources'])
            or response.metadata.dataset_data_sources != observation.get('observed_dataset_sources', observation['dataset_sources'])):
        raise ValueError('original Kernel version/source advanced or changed; refusing latest output')


def download_version(api, observation, destination, *, pattern, max_bytes):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    import requests
    verify_observation(api, observation)
    owner, slug = observation['kernel_ref'].split('/')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seen, tokens, total, token = set(), set(), 0, None
    with api.build_kaggle_client() as service:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            request.page_size, request.page_token = 100, token
            response = service.kernels.kernels_api_client.list_kernel_session_output(request)
            for remote in response.files:
                name = remote.file_name
                if not re.fullmatch(pattern, name):
                    continue
                if (name.casefold() in seen or '\\' in name or ':' in name
                        or any(x in ('', '.', '..') or x.endswith((' ', '.')) for x in name.split('/'))):
                    raise ValueError('unsafe exact-version output path')
                seen.add(name.casefold())
                if len(seen) > 100000:
                    raise ValueError('output file count budget exceeded')
                url = remote.url
                if not isinstance(url, str) or not url.startswith('https://'):
                    raise ValueError('authenticated output requires HTTPS')
                target = destination/name
                target.parent.mkdir(parents=True, exist_ok=True)
                if any(x.is_symlink() for x in [target, *target.parents]):
                    raise ValueError('output links forbidden')
                with requests.get(url, stream=True, timeout=(30, 120)) as download:
                    download.raise_for_status()
                    with target.open('xb') as stream:
                        for block in download.iter_content(1024*1024):
                            total += len(block)
                            if total > max_bytes:
                                raise ValueError('exact-version output budget exceeded')
                            stream.write(block)
            token = response.next_page_token
            if not token:
                break
            if token in tokens:
                raise ValueError('output pagination loop')
            tokens.add(token)
    if not seen:
        raise ValueError('no matching exact-version output')
    # Kernel versions are monotonic. Both observations must still equal the same
    # original candidate before downloaded bytes can receive a trusted receipt.
    verify_observation(api, observation)
    return {**observation, 'downloaded_bytes': total, 'downloaded_files': len(seen)}
