# FROM python:3.10-slim-buster
# WORKDIR /app

# # Copy files first
# COPY . /app

# # Install system dependencies (awscli)
# RUN apt-get update -y && \
#     apt-get install -y awscli && \
#     rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Run your FastAPI app
# CMD ["python3", "app.py"]
FROM python:3.10-slim-buster

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy files first
COPY . /app

# Robust apt install (fixes exit code 100)
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        awscli \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["python3", "app.py", "--host", "0.0.0.0"]

