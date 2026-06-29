FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY artifacts/ ./artifacts/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python", "scripts/run_api.py"]