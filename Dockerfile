# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt --ignore-installed torch torchvision

# Copy the rest of the application code
COPY . /app/

# Make port 8000 available to the world outside this container for FastAPI
# Make port 8501 available for Streamlit
EXPOSE 8000

CMD uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8000}
