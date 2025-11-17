# Multi-stage build for cURLinator API
# Stage 1: Build stage with all dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source code
COPY pyproject.toml README.md ./
COPY src ./src

# Install Python dependencies and package
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir . && \
    find /usr/local/lib/python3.11/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "*.pyo" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Stage 2: Runtime stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 curlinator && \
    chown -R curlinator:curlinator /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only necessary application files
COPY --chown=curlinator:curlinator src ./src
COPY --chown=curlinator:curlinator alembic ./alembic
COPY --chown=curlinator:curlinator alembic.ini ./
COPY --chown=curlinator:curlinator pyproject.toml ./
COPY --chown=curlinator:curlinator docker-entrypoint.sh ./

# Create directories for Chroma and logs, and set permissions
RUN mkdir -p /app/chroma_db /app/logs && \
    chown -R curlinator:curlinator /app/chroma_db /app/logs && \
    chmod +x /app/docker-entrypoint.sh

# Switch to non-root user
USER curlinator

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "curlinator.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

