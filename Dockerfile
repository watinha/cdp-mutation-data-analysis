# Use a stable Python on Alpine
FROM python:3.14-alpine

# Set working directory
WORKDIR /app

# Install build-time and runtime dependencies needed to compile scientific packages on Alpine.
RUN apk add --no-cache --virtual .build-deps \
        build-base \
        gfortran \
        linux-headers \
        openblas-dev \
        lapack-dev \
        musl-dev \
        cmake \
    && apk add --no-cache \
        bash \
        openblas \
        jpeg-dev \
        zlib-dev

# Upgrade pip and wheel
RUN pip install --upgrade pip setuptools wheel

# Environment variables
ENV OPENBLAS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

# Install Python dependencies inferred from project files
RUN pip install --no-cache-dir \
        numpy \
        pandas \
        scikit-learn \
        imbalanced-learn \
        pillow

# Remove build dependencies to reduce image size
RUN apk del .build-deps || true

# Copy project files
COPY . /app

# Default command - change as needed
CMD ["python", "04-train-model.py"]
