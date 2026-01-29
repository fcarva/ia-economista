🧪 Strategy Backtest Lab
No data in selected range.# Use slim python image for smaller footprint
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some scientific packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# Note: Installing torch-geometric might require extra steps for specific CUDA versions
# For CPU deployment (like Streamlit Cloud):
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Entrypoint
# We run the dashboard as the main interface
ENTRYPOINT ["streamlit", "run", "dashboard/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
