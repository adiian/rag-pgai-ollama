# Dockerfile
FROM ollama/ollama:latest

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint to run the script
ENTRYPOINT ["/entrypoint.sh"]

# Run the default command
CMD ["ollama", "serve"]
