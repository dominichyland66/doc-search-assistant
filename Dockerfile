FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir faiss-cpu pymupdf sentence-transformers flask requests
RUN pip install gensim nltk

# Expose port for Flask app (optional but good practice)
EXPOSE 5000

# Default command when container starts
CMD ["bash"]

