from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mercy.ingest import kaggle_motion as motion


def denied(status):
    error = RuntimeError(f'HTTP {status}')
    error.response = SimpleNamespace(status_code=status)
    return error


@pytest.mark.parametrize('status', [403, 404])
def test_new_private_dataset_waits_for_visibility(monkeypatch, status):
    monkeypatch.setattr(motion.time, 'sleep', lambda _: None)
    api = SimpleNamespace(dataset_status=Mock(side_effect=[denied(status), 'PROCESSING', 'READY']))
    motion._wait_dataset(api, 'owner/input')
    assert api.dataset_status.call_count == 3


def test_private_dataset_visibility_timeout_is_bounded(monkeypatch):
    monkeypatch.setattr(motion.time, 'sleep', lambda _: None)
    api = SimpleNamespace(dataset_status=Mock(side_effect=denied(403)))
    with pytest.raises(TimeoutError, match='dataset not ready'):
        motion._wait_dataset(api, 'owner/input')
    assert api.dataset_status.call_count == 90


def test_dataset_authentication_failure_is_not_retried(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(motion.time, 'sleep', sleep)
    api = SimpleNamespace(dataset_status=Mock(side_effect=denied(401)))
    with pytest.raises(RuntimeError, match='HTTP 401'):
        motion._wait_dataset(api, 'owner/input')
    assert api.dataset_status.call_count == 1
    sleep.assert_not_called()


def test_new_private_kernel_waits_for_visibility(monkeypatch, tmp_path):
    monkeypatch.setattr(motion.time, 'sleep', lambda _: None)
    api = SimpleNamespace(kernels_status=Mock(side_effect=[denied(403), SimpleNamespace(status='complete')]))
    motion._wait(api, 'owner/kernel', tmp_path)
    assert api.kernels_status.call_count == 2


def test_wrapped_sdk_permission_error_retains_visibility_status():
    try:
        raise denied(403)
    except RuntimeError as error:
        try:
            raise ValueError('Private kernel not yet visible') from error
        except ValueError as wrapped:
            assert motion._pending_visibility(wrapped)


def test_motion_cache_uses_content_and_case_instead_of_archive_timestamps(tmp_path):
    file=tmp_path/'frame.jpg';file.write_bytes(b'original frame')
    entries=[('data/cases/a/frames/0.jpg',file)];job={'case':'a','step':'recon'}
    before=motion._input_fingerprint(job,entries,'worker-v1')
    file.touch()
    assert motion._input_fingerprint(job,entries,'worker-v1')==before
    assert motion._input_fingerprint(dict(job,case='b'),entries,'worker-v1')!=before
    assert motion._input_fingerprint(job,entries,'worker-v2')!=before
    file.write_bytes(b'different frame')
    assert motion._input_fingerprint(job,entries,'worker-v1')!=before
