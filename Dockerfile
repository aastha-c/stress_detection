# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# EXPOSE port 8000 for Railway auto-detection
EXPOSE 8000

# Run the web service on container startup.
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --worker-class uvicorn.workers.UvicornWorker --threads 8 live_server:app
