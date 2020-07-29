# Image for ML work
# this is a hack of tensorflow/tensorflow:latest-py3 (cpu.Dockerfile)
# See ml.sh for usage

# Need Ubuntu for GUI mapping
FROM ubuntu:18.04 as base
# FROM tensorflow/tensorflow:latest-py3 as base
# FROM debian:10 as base
ENV LANG C.UTF-8
MAINTAINER dturvene@dahetral.com

# hack so apt-get does not prompt user
ENV DEBIAN_FRONTEND=noninteractive

ARG PYTHON=python3
ARG PIP=pip3

ARG USER=user1
ARG GROUP=user1
# These must be identical to the host user for shared volumes
ARG USERID=1000
ARG GROUPID=1000

# update base packages (based on cpu.Dockerfile)
# install python, pip, python-tk (for X11 display) and tools
# add miminal gstreamer debian packages
#  see https://gstreamer.freedesktop.org/documentation/installing/on-linux.html
# add gstreamer python3, introspection, plugin dev
RUN apt-get update --fix-missing && \
    apt-get install -y \
	    ${PYTHON} \
	    ${PYTHON}-pip \
	    ${PYTHON}-tk \
	    apt-utils \
	    curl \
	    wget \
	    sudo \
	    gstreamer1.0-0 \
	    gstreamer1.0-plugins-base \
	    gstreamer1.0-plugins-base-apps \
	    gstreamer1.0-plugins-good \
	    gstreamer1.0-plugins-bad \
	    gstreamer1.0-plugins-ugly \
	    gstreamer1.0-libav \
	    gstreamer1.0-doc \
	    gstreamer1.0-tools \
	    ${PYTHON}-gst-1.0 \
	    ${PYTHON}-gi \
	    python-gi-dev \
	    python-gobject \
	    libgstreamer-plugins-base1.0-dev

# don't do this in order to add packages at run-time
# otherwise need to do `apt-get update` to pull lists
# rm -rf /var/lib/apt/lists/*

# upgrade pip support
RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools
# tensorflow dockerfiles say to do this for some of their stuff    
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

# install ML packages
# seaborn also pulls in pytz, pandas, scipy
RUN ${PIP} install numpy \
    matplotlib \
    pillow \
    seaborn \
    websockets

# ~420M - tf2.1
# ut_ml.py
RUN ${PIP} install tensorflow
# ut_tf.py
RUN ${PIP} install tensorflow_datasets
# ut_hub.py
RUN ${PIP} install tensorflow_hub
# https://www.tensorflow.org/lite/guide/python
# to /usr/lib/python3.6/dist-packages
# test with ml.sh:guest_tflite_image
RUN ${PIP} install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl

# update system-wide bashrc before creating user
COPY bashrc.docker /etc/skel/.bashrc

# create user and group to have same ID as the host user
# add to sudoers for in-container work
# add to video group to use /dev/video0 webcam
RUN adduser --disabled-password --gecos '' ${USER} && \
 adduser ${USER} sudo && \
 echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
 usermod -a -G video ${USER}

# in container, set $USER on commandline or `su $USER`
