FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN uv pip install --system -e ".[dev]"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/
COPY evals/ evals/

# Set Python path
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "intent_engine.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
