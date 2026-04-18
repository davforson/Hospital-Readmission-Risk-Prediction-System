# Base Image
FROM python:3.11-slim

# Set up working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/list/*

# Copy requirments first
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY data/processed/ ./data/processed/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
    
#Start command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]