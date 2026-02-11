FROM python:3.10-slim-buster
WORKDIR /app

# Copy files first
COPY . /app

# Install system dependencies (awscli)
RUN apt-get update -y && \
    apt-get install -y awscli && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run your FastAPI app
CMD ["python3", "app.py"]
