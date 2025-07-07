# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies including Redis and Supervisor
RUN apt-get update && \
    apt-get install -y build-essential libpoppler-cpp-dev pkg-config python3-dev \
    poppler-utils libreoffice redis-server supervisor && \
    rm -rf /var/lib/apt/lists/*

# Create supervisor log directory
RUN mkdir -p /var/log/supervisor

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Copy supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose FastAPI port
EXPOSE 8000

# Start supervisord
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]