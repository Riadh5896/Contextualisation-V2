# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt /app/requirements.txt

# Install any necessary dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the current directory contents into the container
COPY . /app

# Copy the data folder to the container
COPY data /app/data

# Expose the port the app runs on
EXPOSE 8501

# Command to run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]