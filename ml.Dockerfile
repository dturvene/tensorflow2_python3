# Image for ML and python work
# tensorflow packages and requirements are based on 
#   docker pull tensorflow/tensorflow:latest
# at https://hub.docker.com/r/tensorflow/tensorflow
# See ml.sh for usage
# 220308: build with u20.04 3.37GB image
# 240106: tensorflow docker has become a lot more complex

# Need Ubuntu for GUI mapping
# FROM ubuntu:18.04 as base
# 220308 use 20.04 for updated python
# FROM ubuntu:20.04 as base
# 240106 use tensorflow (U22.04.3 LTS) as base
FROM tensorflow/tensorflow as base

ENV LANG C.UTF-8
MAINTAINER dturvene@gmail.com

# hack so apt-get does not prompt user
ENV DEBIAN_FRONTEND=noninteractive

# should be softlinked to /usr/bin/python3.11 /usr/local/bin/pip3.11
ARG PYTHON=python
ARG PIP=pip

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
    	    python3-tk \
	    vim \
	    sudo 
	    
# don't do this in order to add packages at run-time
# otherwise need to do `apt-get update` to pull lists
# rm -rf /var/lib/apt/lists/*

# update PIP
RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# install support python packages
RUN ${PIP} install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scipy

# install tensorflow packages
RUN ${PIP} install tensorflow_datasets
RUN ${PIP} install tensorflow_hub

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
