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

FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of app
COPY . .

EXPOSE 8000
CMD ["python3", "app.py", "--host", "0.0.0.0"]
