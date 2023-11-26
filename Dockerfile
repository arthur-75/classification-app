# Use the official Python image as a base image
FROM python:3.11
# Set the working directory
WORKDIR /app
# Install system packages
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
# Copy the requirements file to the working directory
COPY requirements.txt .
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy all files from the current directory to the container
COPY app.py .
COPY static/ static/
COPY templates/ templates/
COPY training/utils.py training/utils.py
COPY training/model.py training/model.py
COPY training/w_models/best_model.pth training/w_models/best_model.pth
# Expose port 5000 for the Flask app
EXPOSE 5001
# Define environment variable for Flask
ENV FLASK_APP=app.py
# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
