# Base image
FROM python:3.9-slim

# Install system libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY . .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD streamlit health

# Run application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]