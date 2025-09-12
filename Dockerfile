# Use the official Python 3.12 slim image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for Git and Python
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .env file and application code into the container
COPY .env .
COPY . .

# Expose the port used by Gradio 
EXPOSE 7860

# Set environment variables to ensure Python runs in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "ui.py"]