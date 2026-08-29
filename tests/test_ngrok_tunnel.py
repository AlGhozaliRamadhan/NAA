"""Tests for ngrok tunnel management."""

import pytest
from src.tunnel import ngrok


class FakeNgrokTunnel:
    def __init__(self, public_url):
        self.public_url = public_url


class FakeNgrokProcess:
    def __init__(self):
        self.proc = "fake-ngrok-subprocess"


def test_ngrok_tunnel_requires_token(monkeypatch):
    monkeypatch.delenv("NAA_NGROK_AUTHTOKEN", raising=False)
    monkeypatch.delenv("NGROK_AUTHTOKEN", raising=False)
    monkeypatch.delenv("NAA_NGROK_TOKEN", raising=False)
    monkeypatch.delenv("NGROK_TOKEN", raising=False)

    proc, url = ngrok.start_ngrok_tunnel(8000)
    assert proc is None
    assert url is None
    assert "NAA_NGROK_AUTHTOKEN" in ngrok.ngrok_log_tail(1)[0]


def test_ngrok_tunnel_starts_successfully(monkeypatch):
    monkeypatch.setenv("NAA_NGROK_AUTHTOKEN", "fake-test-token-12345")

    class FakePyngrokModule:
        def __init__(self):
            self.token_set = None

        def set_auth_token(self, token):
            self.token_set = token

        def get_tunnels(self):
            return []

        def disconnect(self, url):
            pass

        def connect(self, port, proto):
            return FakeNgrokTunnel("https://test-subdomain.ngrok-free.app")

        def get_ngrok_process(self):
            return FakeNgrokProcess()

    fake_pyngrok = FakePyngrokModule()
    import sys
    import types
    fake_pkg = types.ModuleType("pyngrok")
    fake_pkg.ngrok = fake_pyngrok
    monkeypatch.setitem(sys.modules, "pyngrok", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyngrok.ngrok", fake_pyngrok)

    proc, url = ngrok.start_ngrok_tunnel(8000)

    assert fake_pyngrok.token_set == "fake-test-token-12345"
    assert url == "https://test-subdomain.ngrok-free.app"
    assert proc == "fake-ngrok-subprocess"
