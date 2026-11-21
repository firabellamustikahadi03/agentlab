"""CLI and OpenAI-compatible provider tests (network is always mocked)."""
import io
import json
import urllib.error

import pytest

from agentlab import cli
from agentlab.llm import Message, OpenAICompatProvider, ProviderError, make_provider


# --- provider ---------------------------------------------------------------
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok(content="hi", usage=True):
    body = {"choices": [{"message": {"content": content}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}
    return _Resp(json.dumps(body).encode())


def test_make_provider_specs(monkeypatch):
    assert make_provider("mock").name == "mock"
    assert make_provider("mock-flaky").fail_first
    monkeypatch.setenv("GROQ_API_KEY", "k")
    p = make_provider("groq:some-model")
    assert p.name == "groq" and p.model == "some-model"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        OpenAICompatProvider("groq")


def test_complete_parses_usage_and_cost(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _ok("hello"))
    p = OpenAICompatProvider("ollama", price_per_mtok=(1.0, 2.0))
    r = p.complete([Message("user", "x")])
    assert r.content == "hello" and r.usage.total == 15
    assert r.cost_usd == pytest.approx((10 * 1.0 + 5 * 2.0) / 1e6)


def test_complete_estimates_tokens_without_usage(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _ok("abcdefgh", usage=False))
    r = OpenAICompatProvider("ollama").complete([Message("user", "x" * 40)])
    assert r.usage.prompt_tokens == 10 and r.usage.completion_tokens == 2


def _http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(b"nope"))


def test_retries_on_429_then_succeeds(monkeypatch):
    calls = []

    def fake(req, timeout=0):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429)
        return _ok()

    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert OpenAICompatProvider("ollama").complete([Message("user", "x")]).content == "hi"
    assert len(calls) == 3


def test_non_retryable_http_error(monkeypatch):
    def fake(req, timeout=0):
        raise _http_error(401)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    with pytest.raises(ProviderError, match="HTTP 401"):
        OpenAICompatProvider("ollama").complete([Message("user", "x")])


def test_gives_up_after_retries(monkeypatch):
    def fake(req, timeout=0):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(ProviderError, match="after retries"):
        OpenAICompatProvider("ollama", max_retries=2).complete([Message("user", "x")])


# --- cli --------------------------------------------------------------------
def _cli(tmp_path, *args):
    return cli.main(["--db", str(tmp_path / "t.db"), *args])


def test_cli_run_traces_replay(tmp_path, capsys):
    assert _cli(tmp_path, "run", "What is 6*7?") == 0
    out = capsys.readouterr().out
    assert "answer: 42" in out
    run_id = next(line.split()[1] for line in out.splitlines() if line.startswith("run_id"))
    assert _cli(tmp_path, "traces", "list") == 0
    assert run_id in capsys.readouterr().out
    assert _cli(tmp_path, "traces", "show", run_id) == 0
    assert "tool:calculator" in capsys.readouterr().out
    assert _cli(tmp_path, "traces", "replay", run_id) == 0
    assert "divergences: none" in capsys.readouterr().out


def test_cli_resume_finished_run(tmp_path, capsys):
    _cli(tmp_path, "run", "What is 2+2?")
    run_id = next(line.split()[1] for line in capsys.readouterr().out.splitlines() if line.startswith("run_id"))
    assert _cli(tmp_path, "resume", run_id) == 0


def test_cli_eval_writes_reports_and_gates(tmp_path, capsys):
    out_md, out_json = tmp_path / "r.md", tmp_path / "r.json"
    args = ["eval", "--suite", "basic", "--providers", "mock", "--out", str(out_md), "--json", str(out_json)]
    assert _cli(tmp_path, *args) == 0
    assert "| mock |" in out_md.read_text()
    assert _cli(tmp_path, "eval", "--suite", "basic", "--providers", "mock", "--baseline", str(out_json)) == 0
    # a baseline the mock cannot meet triggers exit code 2
    base = json.loads(out_json.read_text())
    base["summary"]["mock"]["success_rate"] = 2.0
    out_json.write_text(json.dumps(base))
    assert _cli(tmp_path, "eval", "--suite", "basic", "--providers", "mock", "--baseline", str(out_json)) == 2
