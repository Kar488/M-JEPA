FROM docker.io/pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Accept W&B API key at build time and expose at runtime
ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

# Install Python dependencies
WORKDIR /workspace
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 \
    --trusted-host download.pytorch.org \
    && pip install --no-cache-dir torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.2.1+cu121.html \
    --trusted-host data.pyg.org --trusted-host files.pythonhosted.org --trusted-host pypi.org \
    && pip install --no-cache-dir -r requirements.txt \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Copy repository code
COPY . /workspace

# Default command runs synthetic JEPA training demo; provide WANDB_API_KEY via
#   docker build --build-arg WANDB_API_KEY=YOUR_KEY .
#   docker run -e WANDB_API_KEY=YOUR_KEY <image>
CMD ["python", "scripts/train_jepa.py"]
