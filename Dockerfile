# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (removed Redis)
RUN apt-get update && \
    apt-get install -y build-essential libpoppler-cpp-dev pkg-config python3-dev \
    poppler-utils libreoffice && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements (if you have requirements.txt)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI directly (no longer need supervisor for Redis)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]