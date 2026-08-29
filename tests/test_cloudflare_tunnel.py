"""Cloudflare tunnel startup and stdout-drain regression tests."""

import time

from src.tunnel import cloudflare


class FakeStdout:
    def __init__(self, lines):
        self.lines = iter(lines)
        self.read_count = 0
        self.closed = False

    def readline(self):
        try:
            line = next(self.lines)
        except StopIteration:
            return ""
        self.read_count += 1
        return line

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, command, lines):
        self.command = command
        self.stdout = FakeStdout(lines)
        self.alive = True

    def poll(self):
        return None if self.alive else 1

    def terminate(self):
        self.alive = False


def _fake_popen_factory(captured, lines):
    def fake_popen(command, **kwargs):
        captured.append(command)
        return FakeProcess(command, lines)

    return fake_popen


def test_quick_tunnel_extracts_url_and_drains_remaining_output(monkeypatch):
    captured = []
    lines = [
        "INF starting\n",
        "INF https://bright-test.trycloudflare.com ready\n",
        "INF later diagnostic one\n",
        "INF later diagnostic two\n",
    ]
    monkeypatch.delenv("NAA_CF_TUNNEL_TOKEN", raising=False)
    monkeypatch.delenv("NAA_PUBLIC_URL", raising=False)
    monkeypatch.setattr(cloudflare, "download_cloudflared", lambda: "/tmp/cloudflared")
    monkeypatch.setattr(
        cloudflare.subprocess,
        "Popen",
        _fake_popen_factory(captured, lines),
    )

    proc, url = cloudflare.start_tunnel(8000)
    deadline = time.time() + 1
    while proc.stdout.read_count < len(lines) and time.time() < deadline:
        time.sleep(0.01)

    assert url == "https://bright-test.trycloudflare.com"
    assert "--url" in captured[0]
    assert proc.stdout.read_count == len(lines)
    assert proc.stdout.closed is True


def test_named_tunnel_uses_token_and_stable_public_url(monkeypatch):
    captured = []
    monkeypatch.setenv("NAA_CF_TUNNEL_TOKEN", "secret-test-token")
    monkeypatch.setenv("NAA_PUBLIC_URL", "https://naa.example.com/")
    monkeypatch.setattr(cloudflare, "download_cloudflared", lambda: "/tmp/cloudflared")
    monkeypatch.setattr(
        cloudflare.subprocess,
        "Popen",
        _fake_popen_factory(
            captured,
            ["INF Registered tunnel connection connIndex=0\n"],
        ),
    )

    proc, url = cloudflare.start_tunnel(8000)

    assert proc.poll() is None
    assert url == "https://naa.example.com"
    assert captured[0][-2:] == ["--token", "secret-test-token"]
    assert "--url" not in captured[0]


def test_named_tunnel_requires_public_url(monkeypatch):
    monkeypatch.setenv("NAA_CF_TUNNEL_TOKEN", "secret-test-token")
    monkeypatch.delenv("NAA_PUBLIC_URL", raising=False)
    monkeypatch.setattr(cloudflare, "download_cloudflared", lambda: "/tmp/cloudflared")

    proc, url = cloudflare.start_tunnel(8000)

    assert proc is None
    assert url is None
    assert "NAA_PUBLIC_URL is missing" in cloudflare.tunnel_log_tail(1)[0]
