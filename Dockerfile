# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install packages specified in requirements.txt
# We use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Make port 8000 available to the world outside this container for FastAPI
# Make port 8501 available for Streamlit
EXPOSE 8000
EXPOSE 8501

# Default command: We can override this to run FastAPI or Streamlit
# By default, let's run the backend API
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
