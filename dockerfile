FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch and torchvision with matching CUDA versions
RUN pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install transformers after PyTorch to ensure compatibility
RUN pip3 install transformers sentence-transformers

# Create model directory and download models at build time
RUN mkdir -p /app/models
RUN python3 -c "from transformers import AutoTokenizer, T5ForConditionalGeneration; \
    model_name='mesolitica/nanot5-small-malaysian-translation-v2.1'; \
    tokenizer = AutoTokenizer.from_pretrained(model_name); \
    model = T5ForConditionalGeneration.from_pretrained(model_name); \
    tokenizer.save_pretrained('/app/models/translator'); \
    model.save_pretrained('/app/models/translator')"

# Copy Alembic files
COPY alembic alembic/
COPY alembic.ini .

# Copy application code
COPY . .

# Copy and set up entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/translator
ENV PORT=8080

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]