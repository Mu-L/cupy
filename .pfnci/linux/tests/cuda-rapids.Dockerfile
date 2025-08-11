ARG BASE_IMAGE="rapidsai/base:25.02-cuda12.8-py3.11"
FROM ${BASE_IMAGE}

USER root

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install build-essential cuda-toolkit-12-8 ccache git curl

ENV CUDA_PATH "/usr/local/cuda"
ENV PATH "/usr/lib/ccache:${PATH}"
