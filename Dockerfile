FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Copy the model file from parent directory (this will be done during build context)
# The model file should be copied to the build context or downloaded during runtime

# Critical Hugging Face settings
EXPOSE 7860
CMD ["python", "app.py"]