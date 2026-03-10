# Use a lightweight Python base
FROM python:3.9-slim

# Install system dependencies (Tesseract for OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Expose the port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app_groq.py", "--server.address=0.0.0.0"]