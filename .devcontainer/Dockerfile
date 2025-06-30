# Use micromamba base image for consistent conda environment
FROM mambaorg/micromamba:1.5.8

# Switch to root for system package installation
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    sudo \
    vim \
    nano \
    tree \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create vscode user with sudo privileges
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode \
    && echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# Set up micromamba for vscode user
USER vscode
WORKDIR /home/vscode

# Set environment variables so micromamba knows where to create its environments
ENV MAMBA_ROOT_PREFIX=/home/vscode/micromamba
ENV PATH=/home/vscode/micromamba/bin:$PATH

# Copy environment file
COPY binder/environment.yml /tmp/environment.yml

# Create conda environment from DemARK's environment.yml
RUN micromamba create -f /tmp/environment.yml -n DemARK && \
    micromamba clean --all --yes

# Initialize micromamba for bash
RUN micromamba shell init --shell bash --prefix $MAMBA_ROOT_PREFIX

# Set up shell initialization
RUN echo 'eval "$(micromamba shell hook --shell bash)"' >> ~/.bashrc && \
    echo 'micromamba activate DemARK' >> ~/.bashrc

# Install additional development tools in the environment
RUN eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate DemARK && \
    pip install --no-cache-dir \
        black \
        ruff \
        pre-commit \
        jupyterlab \
        ipywidgets

# Update PATH to include the new environment's bin directory
ENV PATH=$MAMBA_ROOT_PREFIX/envs/DemARK/bin:$PATH

# Set working directory to workspace
WORKDIR /workspace

# Default command
CMD ["sleep", "infinity"] 