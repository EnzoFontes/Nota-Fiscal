FROM python:3.11-slim

# Install system dependencies: Node.js 20, Tesseract OCR, libzbar
RUN apt-get update && \
    apt-get install -y curl tesseract-ocr tesseract-ocr-por libzbar0 libglib2.0-0 && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build React frontend
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# Install Python dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source
COPY backend/ ./backend/

WORKDIR /app/backend

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
