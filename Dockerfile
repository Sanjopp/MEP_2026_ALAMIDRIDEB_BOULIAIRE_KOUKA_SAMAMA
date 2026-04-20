# Stage 1 : builder 
FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY mep_2026_alamidrideb_bouliaire_kouka_samama/ ./mep_2026_alamidrideb_bouliaire_kouka_samama/

# Install all production dependencies into a virtual environment
RUN uv sync --frozen --no-dev

# Stage 2 : runtime
FROM python:3.13-slim AS runtime

WORKDIR /app

# Create non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY api/        ./api/
COPY models/     ./models/
COPY data/       ./data/
COPY config/     ./config/
COPY src/        ./src/

# Make sure the venv binaries are used
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Launch the API
CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
