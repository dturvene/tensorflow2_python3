# Container for ML work
# See docker.sh ml_all
# This will put guest in python work area

FROM ubuntu:18.04 as base
ENV LANG C.UTF-8

# so apt-get does not prompt user
ENV DEBIAN_FRONTEND=noninteractive

ARG PYTHON=python3
ARG PIP=pip3

ARG USER=user1
ARG GROUP=user1
# These must be identical to the host user for shared volumes
ARG USERID=1000
ARG GROUPID=1000

# update base packages
# install tools
# From cpu.Dockerfile, install python and pip
# for X11 display add the python TK package
RUN apt-get update --fix-missing && \
    apt-get install -y \
	    ${PYTHON} \
	    ${PYTHON}-pip \
	    ${PYTHON}-tk \
    && rm -rf /var/lib/apt/lists/*

# install pip support
RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools
# tensorflow dockerfiles say to do this for some of their stuff    
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

# install ML packages
RUN ${PIP} install numpy matplotlib pillow
# seaborn also pulls in pytz, pandas, scipy
RUN ${PIP} install seaborn
RUN ${PIP} install tensorflow==2.0.0
RUN ${PIP} install tensorflow_hub
RUN ${PIP} install tensorflow_datasets

# create user and group matching permissions for host volumes
RUN addgroup --gid ${GROUPID} ${GROUP} && \
    useradd --create-home \
    --shell /bin/bash \
    --uid ${USERID} \
    --gid ${GROUPID} \
    ${USER}

# if update to system /etc/skel/.bashrc must do before creating user
COPY bashrc.docker /home/$USER/.bashrc

# stay as root for now. `su $USER` in guest.
# USER ${USER}
