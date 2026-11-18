# =====================================================================
# FIFA World Cup 2026 Winner Prediction & Tournament Simulator
# =====================================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install minimal OS dependencies for health checking and building
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root application user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy and install python dependencies first for efficient layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and dataset assets
COPY . /app

# Ensure /app/data symlink or directory exists for direct path resolution
RUN if [ -d "/app/FIFA-2026-Winner-Prediction-main/FIFA Prediction/data" ] && [ ! -d "/app/data" ]; then \
        ln -s "/app/FIFA-2026-Winner-Prediction-main/FIFA Prediction/data" /app/data; \
    fi

# Set proper ownership for the non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command starts the interactive Streamlit dashboard
CMD ["streamlit", "run", "app.py"]
