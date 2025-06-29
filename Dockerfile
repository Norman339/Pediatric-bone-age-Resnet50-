FROM python:3.9-slim

WORKDIR /app

# Force cache invalidation
ARG CACHEBUST=1

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy version file first to force cache invalidation
COPY version.txt .

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model file (this is the slowest part)
COPY bone_age_res50_epoch_101.pth ./

# Copy the rest of the application
COPY . .

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

# Run the new Gradio app
CMD ["python", "gradio_app.py"]