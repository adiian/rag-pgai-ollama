#!/bin/bash

# Start the Ollama service in the background
ollama serve &

# Wait for the Ollama service to be ready
until ollama list >/dev/null 2>&1; do
  echo "Waiting for Ollama service to start..."
  sleep 2
done

# Pull necessary models once the service is ready
ollama pull mxbai-embed-large
ollama pull llama3.2

# Bring Ollama service back to the foreground
wait
