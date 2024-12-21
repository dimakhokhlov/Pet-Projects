# Start from a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install flask

# Copy the application code into the container
COPY best_xgb_model.bin main.py ./

# Run command
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8989", "main:app"]
