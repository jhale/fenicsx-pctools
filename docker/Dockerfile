ARG IMAGE=dolfinx/dolfinx:v0.5.1-r1
FROM ${IMAGE} as dev
LABEL description="FEniCS PC Tools development image with all FEniCS components installed"

# Install required Python packages (via pip)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt
