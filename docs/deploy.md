# Deploying the dashboard on a free host

The image serves the **offline demo** (mock agents, no API keys, no outbound calls). The demo data is generated
at build time, so the container starts in about a second. Pick one host.

> **No authentication.** The dashboard is read-only, but anyone with the URL can read every trace in the store.
> That is fine for the demo data. Before pointing it at real traces (for example a Supabase store), put it behind
> your host's access control or a reverse proxy with auth.

> **Not built in CI.** The Dockerfile follows each host's documented conventions (non-root uid 1000, `$PORT`,
> `0.0.0.0`), but it has not been built on the machine that wrote it (no Docker there). If the first build
> fails, the error will be a one-liner in the host's build log; please open an issue.

## Option A: Hugging Face Spaces (free CPU)

1. Create a Space at <https://huggingface.co/new-space>: SDK **Docker**, blank template, public.
2. Clone it and copy this repo's `Dockerfile`, `pyproject.toml`, `README.md`, `LICENSE` and `src/` into it.
   Replace the Space's `README.md` with the front matter below (Spaces reads its config from there):

   ```yaml
   ---
   title: AgentLab
   emoji: 🧪
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```
3. `git add . && git commit -m "deploy" && git push`. The Space builds and serves the dashboard on port 7860.

## Option B: Render (free web service)

1. Push this repo to GitHub (already done if you are reading it there).
2. In Render choose **New + > Blueprint**, select the repo. `render.yaml` configures a free Docker web service
   with `/api/health` as the health check.
3. Render injects `$PORT`; the container binds to it. Free services sleep when idle and take ~30 s to wake.

## Showing your own data (optional)

Run evals locally against a Supabase project (see `docs/supabase_schema.sql`), then deploy with these as the
host's **secret environment variables** (never commit them):

```
AGENTLAB_STORE=supabase
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_KEY=<service-role key>
```

and change the container command to `agentlab serve --host 0.0.0.0` (the `demo` command seeds mock data).
Mind the no-authentication warning above.

## Run it locally with Docker

```bash
docker build -t agentlab .
docker run --rm -p 7860:7860 agentlab      # http://localhost:7860
```
