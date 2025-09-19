# Mochi-Moo Docker Container
# Author: Cazandra Aporbo MS

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mochi_moo/ ./mochi_moo/
COPY setup.py .
COPY README.md .

# Install Mochi-Moo
RUN pip install -e .

# Expose API port
EXPOSE 8000

# Run server
CMD ["mochi-server"]
