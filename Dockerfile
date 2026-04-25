FROM runpod/pytorch:2.1.0-base

# Set the working directory
WORKDIR /app

# Install additional Python dependencies
# The base image already has PyTorch, Transformers, and Datasets.
# We need to add TRL, PEFT, and bitsandbytes for our specific training task.
RUN pip install --no-cache-dir \
    trl==1.2.0 \
    peft==0.19.1 \
    bitsandbytes==0.49.2

# Copy the training script
COPY train_grpo.py /app/train_grpo.py

# Set the entry point
CMD ["python", "train_grpo.py"]