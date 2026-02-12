FROM python:3.10-slim-buster
WORKDIR /app

# Copy files first
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run your FastAPI app
CMD ["python3", "app.py"]


