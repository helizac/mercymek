"""Public-release boundaries: paths, browser origins, audio keys and API version."""
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from mercy import __version__, config
from mercy.server import app as server


@pytest.mark.parametrize('case_id', ['../outside', '..', '.', '/tmp/case', 'C:\\private', 'a/b', 'a\\b', None])
def test_case_directory_rejects_escape(case_id, tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'CASES_DIR', tmp_path/'cases')
    with pytest.raises(ValueError):
        config.case_dir(case_id)


def test_named_case_stays_under_configured_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'CASES_DIR', tmp_path/'cases')
    assert config.case_dir('london_rearend') == tmp_path/'cases'/'london_rearend'


@pytest.mark.parametrize('key', ['not-a-cache-key', 'C:%5Cprivate', '..%5C..%5Caudio'])
def test_audio_endpoint_rejects_non_digest_paths(key):
    assert TestClient(server.app).get('/api/tts/'+key).status_code == 404


@pytest.mark.parametrize('origin', ['https://unrelated.example', 'null', 'http://testserver.attacker.example'])
def test_foreign_browser_cannot_issue_commands(origin, monkeypatch):
    monkeypatch.setattr(server.rt, 'case', None)
    with pytest.raises(WebSocketDisconnect) as exc:
        with TestClient(server.app).websocket_connect('/ws', headers={'origin': origin}):
            pass
    assert exc.value.code == 1008


@pytest.mark.parametrize('headers', [{}, {'origin': 'http://testserver'}])
def test_local_browser_and_native_clients_still_connect(headers, monkeypatch):
    monkeypatch.setattr(server.rt, 'case', None)
    with TestClient(server.app).websocket_connect('/ws', headers=headers):
        pass


def test_api_exposes_the_application_version():
    assert TestClient(server.app).get('/openapi.json').json()['info']['version'] == __version__


@pytest.mark.parametrize('url', ['/api/case/..%5Coutside', '/cases/..%5Coutside/meta.json'])
def test_invalid_case_paths_are_http_404_instead_of_server_errors(url):
    assert TestClient(server.app).get(url).status_code == 404
