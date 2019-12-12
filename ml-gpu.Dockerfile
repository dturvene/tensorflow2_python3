# Image for ML GPU work
# See ml.sh for usage

# use a tensorflow image, too much CUDA stuff to try
# and copy
FROM tensorflow/tensorflow:latest-gpu-py3 as base
ENV LANG C.UTF-8

# hack so apt-get does not prompt user
ENV DEBIAN_FRONTEND=noninteractive

ARG PYTHON=python3
ARG PIP=pip3

ARG USER=user1
ARG GROUP=user1
# These must be identical to the host user for shared volumes
ARG USERID=1000
ARG GROUPID=1000

# for X11 display add the python TK package
RUN apt-get update --fix-missing && \
    apt-get install -y \
       	    ${PYTHON}-tk \
    && rm -rf /var/lib/apt/lists/*

# install ML packages
RUN ${PIP} install numpy matplotlib pillow
# seaborn also pulls in pytz, pandas, scipy
RUN ${PIP} install seaborn

# this should already be installed by tensorflow/tensorflow:latest-gpu-py3
RUN ${PIP} install tensorflow-gpu
RUN ${PIP} install tensorflow_hub
RUN ${PIP} install tensorflow_datasets

COPY bashrc.docker /etc/bash.bashrc

# create user and group matching permissions for host volumes
RUN addgroup --gid ${GROUPID} ${GROUP} && \
    useradd --create-home \
    --shell /bin/bash \
    --uid ${USERID} \
    --gid ${GROUPID} \
    ${USER}

# stay as root for now. `su $USER` in guest.
# USER ${USER}
