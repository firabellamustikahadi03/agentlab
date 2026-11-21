# Contributing to AgentLab

Thanks for looking. The project is small on purpose, so changes are easy to review.

## Setup

```bash
git clone https://github.com/firabellamustikahadi03/agentlab && cd agentlab
pip install -e ".[dev]"
pytest -q --cov=agentlab        # 100+ tests, no API key or network; coverage gate is 85%
ruff check .
agentlab eval --suite basic --providers mock     # the offline eval CI also runs
```

Everything must pass offline. Tests never call a real provider or the real Supabase.

## Ground rules

- **The core package stays standard-library only.** Optional extras are fine; hard dependencies are not.
- **Free-tier friendly.** Anything that can call a model must work with the mock provider, and real-model paths must
  respect rate limits (`--rpm`).
- **Never commit secrets.** API keys come from environment variables only. `.env` and `*.db` are git-ignored; keep it that way.
- **Tests with the change.** New behaviour gets a test; a bug fix gets a regression test. Coverage must stay above 85%.
- **Match the surrounding code** (naming, comment density, idiom). `ruff check .` must be clean.
- Small, focused commits in [conventional style](https://www.conventionalcommits.org/): `feat:`, `fix:`, `test:`,
  `docs:`, `ci:`, `chore:`.

## Where things live

| I want to... | Look at |
|---|---|
| change the agent loop, budgets, resume | `src/agentlab/runtime.py` |
| add a tool | `src/agentlab/tools/` (`@registry.register` infers the schema from type hints) |
| add an LLM provider | `src/agentlab/llm/` (implement `complete(messages, temperature) -> LLMResponse`) |
| add a store backend | implement the `Store` protocol in `src/agentlab/tracing/base.py`, add it to the contract tests in `tests/test_stores.py` |
| add an eval suite or check type | `src/agentlab/eval/suites.py`, `scorers.py` (and `fixtures.py` for stateful tools) |
| change the dashboard | `src/agentlab/dashboard/` (single `index.html`; escape every dynamic string with `esc()`) |

## Adding an eval suite

1. Define `EvalTask`s with deterministic checks where you can (`numeric`, `contains`, `tool_used`, `tool_not_used`, ...).
2. If tasks need stateful tools, give `EvalTask.tools` a factory that returns a **fresh** registry per run.
3. Make sure the offline `mock` can pass it (or document why not), and add a flawed mock behaviour or a test proving the
   suite fails the agent it is meant to catch. An eval that cannot fail is not evidence.

## Pull requests

Describe what changed and why, and paste the output of `pytest -q` and `ruff check .`. Dashboard changes: attach a
screenshot (`agentlab demo`). Security issues: please open a private security advisory instead of a public issue.
