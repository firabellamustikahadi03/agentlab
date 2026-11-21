# AgentLab demo dashboard. Runs on Hugging Face Spaces (Docker SDK) and Render with no changes.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PORT=7860

# Spaces runs the container as uid 1000; do the same everywhere so file permissions behave.
RUN useradd -m -u 1000 user
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

USER user
WORKDIR /home/user
# Bake the offline demo data into the image so the container starts instantly.
RUN agentlab demo --demo-db /home/user/demo.db --seed-only

EXPOSE 7860
CMD ["sh", "-c", "agentlab demo --demo-db /home/user/demo.db --host 0.0.0.0 --port ${PORT} --no-open"]
