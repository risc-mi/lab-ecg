ARG VENV_PATH="/opt/venv"
ARG PYTHON_VERSION=None

# stable-20250317-slim
ARG FROM_IMAGE=debian@sha256:70b337e820bf51d399fa5bfa96a0066fbf22f3aa2c3307e2401b91e2207ac3c3
ARG FROM_IMAGE_RUNTIME=${FROM_IMAGE}
ARG FROM_IMAGE_DEV=build

ARG DEBIAN_FRONTEND=noninteractive

# ==============================================================================
# Build Stage
# ==============================================================================
FROM $FROM_IMAGE AS build
ARG VENV_PATH
ARG DEBIAN_FRONTEND
ARG PYTHON_VERSION

# uv version 0.10.2
COPY --from=ghcr.io/astral-sh/uv@sha256:94a23af2d50e97b87b522d3cea24aaf8a1faedec1344c952767434f69585cbf9 /uv /uvx /bin/

# Setup directory structure
# https://github.com/GoogleContainerTools/kaniko/issues/1508
# seems like remnant of an earlier kaniko implementation
# delete, to circumvent any side-effects
# can be removed after issue above is fixed
RUN rm -rf /workspace

# prepare workspace and dependencies
WORKDIR /workspace

COPY pyproject.toml *.lock README* ./

# ensure Python uses this virtual environment
ENV VIRTUAL_ENV=$VENV_PATH \
    UV_PROJECT_ENVIRONMENT=${VENV_PATH} \
    PATH="$VENV_PATH/bin:$PATH" \
    FORCE_CUDA="1"
RUN uv venv --python $PYTHON_VERSION \
    && uv sync \
    && uv cache prune --ci

# ==============================================================================
# Runtime Stage
# ==============================================================================
# TODO: build from FROM_IMAGE (or a plain python img), and just copy from build (without pyenv, uv)
FROM $FROM_IMAGE_RUNTIME AS runtime
ARG VENV_PATH
ARG DEBIAN_FRONTEND

# prepare workspace and dependencies
WORKDIR /workspace

# we don't need wheels anymore => remove
RUN rm -rf /wheels

# copy the virtual environment from the build stage - careful this won't work with the current setup
# see: https://github.com/astral-sh/uv/issues/6782
COPY --from=build $VENV_PATH $VENV_PATH

# Copy the rest of the repository
COPY . .

CMD ["bash"]

# ==============================================================================
# Development Stage
# ==============================================================================
FROM $FROM_IMAGE_DEV AS dev
ARG VENV_PATH
ARG DEBIAN_FRONTEND

RUN apt-get update && apt-get install -yq \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# install development dependencies
RUN uv sync --all-extras && uv cache prune --ci
