# Use a Python base image
FROM python:3.9-slim

# Install git and any other dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY app/requirements.txt .
RUN echo "Installing dependencies..." \
    && pip install -r requirements.txt

# Copy main.py file and confirm
COPY app/main.py .
RUN echo "Copied main.py"

# Display directory contents for verification
RUN echo "Displaying contents of /app:" \
    && ls -al /app

# Expose port 8000
EXPOSE 8000

# Sleep to allow inspection before starting the app
CMD ["sh", "-c", "echo 'Sleeping for 10 seconds...' && sleep 10 && uvicorn main:app --host 0.0.0.0 --port 8000"]
