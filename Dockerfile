# Use Python 3.12 base image
FROM python:3.12-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    binance-python \
    gym \
    numpy \
    pandas \
    matplotlib \
    schedule \
    opencv-python \
    python-dotenv

# Copy project files
COPY code/ ./code/
COPY weights/ ./weights/
COPY inference_images/ ./inference_images/
COPY plot.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "./code/live_r.py", "--model", "weights/bitcoin-ppo-True(m-raw)_agent0_best.pth", "--initial_balance", "10000"]