# Compatible with Jetpack 6.2
FROM nvcr.io/nvidia/pytorch:25.02-py3-igpu

WORKDIR /app

# Install required system packages
RUN apt update && \
    apt install -y --no-install-recommends libportaudio2 curl && \
    apt clean

# Install Python dependencies
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
        transformers==4.49.0 \
        accelerate==1.5.2 \
        sounddevice \
        requests

# Copy your Python script into the container
COPY script.py .

# Set Hugging Face cache directory
ENV HF_HOME="/huggingface/"

# Run the script
ENTRYPOINT ["python", "script.py"]

