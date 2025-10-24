# Use a stable base image with better build support
FROM python:3.9-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies (for OpenCV, aiortc, numpy, etc.)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libavdevice-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    libsrtp2-dev \
    pkg-config \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip safely
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency list and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the Flask app port
EXPOSE 5005

# Default start command
CMD ["python", "app.py"]
