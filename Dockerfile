# Multi-stage build for Fashion Detection System

# Base stage with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libnotify-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 fashion && \
    mkdir -p /app && \
    chown -R fashion:fashion /app

# Builder stage
FROM base AS builder

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base AS production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=fashion:fashion . .

# Install the package in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs cache models outputs data && \
    chown -R fashion:fashion logs cache models outputs data

# Switch to non-root user
USER fashion

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM production AS development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install -e ".[dev]"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to fashion user
USER fashion

# Override command for development
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]