import asyncio
import importlib

import httpx
import pytest

tts_module = importlib.import_module('mercy.voice.tts')


@pytest.fixture
def speaker(monkeypatch, tmp_path):
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'tts-only-test-key')
    monkeypatch.setattr(tts_module, 'CACHE', tmp_path)
    monkeypatch.setattr(tts_module, 'USAGE', tmp_path/'usage.json')
    return tts_module.TTS()


def test_successful_warm_up_keeps_speech_credentials(speaker, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=[])

    async def stop_after_one_probe(_):
        raise asyncio.CancelledError

    monkeypatch.setattr(tts_module.asyncio, 'sleep', stop_after_one_probe)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond),
                                     headers={'xi-api-key': speaker.key}) as client:
            speaker.client = client
            with pytest.raises(asyncio.CancelledError):
                await speaker.keep_warm()
            speech = client.build_request('POST', tts_module.URL.format(voice=speaker.voice))
            assert speech.headers['xi-api-key'] == speaker.key

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].url.path == '/v1/models'
    assert requests[0].headers['xi-api-key'] == speaker.key
    assert speaker.enabled


@pytest.mark.parametrize('status', [401, 403])
def test_denied_probe_stops_retrying_without_disabling_speech(speaker, status):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={'detail': {'status': 'missing_permissions'}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            speaker.client = client
            await asyncio.wait_for(speaker.keep_warm(), timeout=1)

    asyncio.run(run())
    assert len(requests) == 1
    assert speaker.enabled
