"""Verify an immutable Dataset version when Kaggle cannot serve its ZIP."""

import hashlib
from contextlib import closing

from app.payload_contract import upload_content_inventory


def archive_url_missing(exc: Exception) -> bool:
    """Only Kaggle's explicit missing archive URL permits the alternative read."""
    response = getattr(exc, "response", None)
    if response is None or response.status_code != 404:
        return False
    try:
        error = response.json().get("error", {})
        return isinstance(error, dict) and error.get("message") == "No gcs url found"
    except (ValueError, AttributeError):
        return False


def version_file_inventory(api, target_ref: str, check) -> dict[str, int]:
    inventory, names, tokens = {}, set(), set()
    token = None
    while True:
        check()
        response = api.dataset_list_files(target_ref, page_token=token, page_size=200)
        if getattr(response, "error_message", "") or getattr(response, "errorMessage", ""):
            raise ValueError("payload_file_listing_failed")
        files = getattr(response, "files", None)
        if files is None:
            files = getattr(response, "dataset_files", None)
        if files is None:
            raise ValueError("payload_file_listing_missing")
        for item in files:
            name, size = getattr(item, "name", None), getattr(item, "total_bytes", None)
            if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                    or any(p in ("", ".", "..") for p in name.split("/"))
                    or name.casefold() in names or type(size) is not int or size < 0):
                raise ValueError("payload_file_listing_invalid")
            names.add(name.casefold())
            inventory[name] = size
        if len(inventory) > 100000:
            raise ValueError("payload_file_listing_limit")
        token = getattr(response, "next_page_token", None)
        if not token:
            break
        if not isinstance(token, str) or token in tokens:
            raise ValueError("payload_file_listing_pagination_loop")
        tokens.add(token)
    if not inventory:
        raise ValueError("payload_file_listing_empty")
    return inventory


def verify_version_files(api, dataset_ref: str, version_number: int, dataset_dir, check, log):
    # Use the same SDK request as dataset_download_file, but hash its stream
    # directly. No remote paths or signed URLs are written to disk or logs.
    from kaggle.api.kaggle_api_extended import ApiDownloadDatasetRequest

    if type(version_number) is not int or version_number <= 0:
        raise ValueError("payload_version_invalid")
    owner, slug = dataset_ref.split("/")
    target_ref = f"{dataset_ref}/{version_number}"
    actual = version_file_inventory(api, target_ref, check)
    expected = upload_content_inventory(dataset_dir, actual)
    if set(actual) != set(expected):
        raise ValueError("payload_inventory_mismatch")
    if any(actual[name] != value[0] for name, value in expected.items()):
        raise ValueError("payload_size_mismatch")
    total = sum(actual.values())
    verified_bytes = 0
    log(f"Dataset version {version_number}: verifying {len(expected)} files ({total} bytes) individually")
    with api.build_kaggle_client() as service:
        for index, name in enumerate(sorted(expected), 1):
            check()
            request = ApiDownloadDatasetRequest()
            request.owner_slug = owner
            request.dataset_slug = slug
            request.dataset_version_number = version_number
            request.file_name = name
            digest, received = hashlib.sha256(), 0
            with closing(service.datasets.dataset_api_client.download_dataset(request)) as response:
                response.raise_for_status()
                for block in response.iter_content(chunk_size=1024 * 1024):
                    check()
                    received += len(block)
                    if received > expected[name][0]:
                        raise ValueError("payload_size_mismatch")
                    digest.update(block)
            if received != expected[name][0]:
                raise ValueError("payload_size_mismatch")
            if digest.hexdigest() != expected[name][1]:
                raise ValueError("payload_digest_mismatch")
            verified_bytes += received
            if index % 100 == 0 or index == len(expected):
                log(f"Dataset version {version_number}: verified {index}/{len(expected)} files, "
                    f"{verified_bytes}/{total} bytes")
    # Recheck the complete inventory before accepting the verification result.
    if version_file_inventory(api, target_ref, check) != actual:
        raise ValueError("payload_file_listing_changed")
