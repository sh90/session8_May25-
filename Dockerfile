# Use an official Python base image
FROM python:3.11-slim

# Expose Port
EXPOSE 8080

# Copy Files
COPY . /app

# Set Working Directory
WORKDIR /app

# Install Dependencies
RUN pip install -r requirements.txt

# Run Streamlit App
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS", "false"]
