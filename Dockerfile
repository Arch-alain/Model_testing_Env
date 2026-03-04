# Use the official Ultralytics ARM64 optimized image
FROM ultralytics/ultralytics:latest-arm64

# Set working directory inside the container
WORKDIR /usr/src/app

# Install system dependencies for psutil and OpenCV
RUN apt-get update && apt-get install -y \
    gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script
COPY benchmark_mac.py ./benchmark_edge.py

# Create mount points for volumes
RUN mkdir models test_data results

# Run the benchmark script on startup
CMD ["python3", "benchmark_edge.py"]