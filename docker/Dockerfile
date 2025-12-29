# Quant Trading System Dockerfile
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim as runtime

# Labels
LABEL maintainer="Quant Trading System" \
      version="1.0.0" \
      description="Quantitative Trading System"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH" \
    # Application settings
    APP_ENV=production \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 trading && \
    useradd --uid 1000 --gid trading --shell /bin/bash --create-home trading

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=trading:trading . .

# Create directories for logs and models
RUN mkdir -p /app/logs /app/models_artifacts /app/data && \
    chown -R trading:trading /app/logs /app/models_artifacts /app/data

# Switch to non-root user
USER trading

# Expose ports
# 8000 - Dashboard/API
# 9090 - Prometheus metrics (internal)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run trading in paper mode
CMD ["python", "main.py", "trade", "--mode", "paper", "--log-format", "json"]

# Alternative entry points:
# Dashboard: python main.py dashboard --host 0.0.0.0 --port 8000
# Backtest: python main.py backtest --start-date 2024-01-01 --end-date 2024-12-31
