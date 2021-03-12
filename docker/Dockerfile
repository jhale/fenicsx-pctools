ARG IMAGE=dolfinx/dolfinx-onbuild:latest
FROM ${IMAGE} as dev
LABEL description="FEniCS PC Tools development image with all FEniCS components installed"

# Install dependencies available via apt-get
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
        htop \
        screen \
        sudo \
        rsync && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install required Python packages (via pip)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# TODO: Add production image which will be used within CI pipeline!
# FROM dev as prod
# LABEL description="FEniCS PC Tools production image with complete installation"
