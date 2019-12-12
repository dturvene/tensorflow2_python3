<!--
Review:
pandoc -f markdown -t html README.md > README.html
-->

# Abstract
I am working on several ML Tensorflow projects and found the existing docker
environments either too complex or too simple for my research. I created this repo
in an effort to streamline my experience.

This is a work-in-progress but serves to get me going on this tensorflow2 stuff...
 
# High level Container Requirements
So what are my basic requirements for the enivonment?  My host is Ubuntu 18.04
so I'm looking for an image that mimics that environment but doesn't polute
it. I messed around with some of the Kaggle docker builds before deciding they
were too complex.

This is a work-in-progress. If there is a package I need I'll add it, otherwise
it isn't in the image.

* Ubuntu 18.04
* Tensorflow 2
* Python 3
* Supplemental ML python packages as needed
* Host git repo mounted as guest volume
* Host dataset area mounted as guest volume
* X11 export for graphics (images, seaborn, matplotlib)
* a non-root user for typical work (user1)
* root access for exceptional work
* python unit tests for regression testing in containiner

## Tensorflow GPU Requirements
This is an extension to the requirements adding GPU support.  All the GPU cards
I have are NVIDIA so the default CUDA framework is very clean.  However,
inspecting the Tensorflow `gpu.Dockerfile` demonstrated that there are a number
of logistics to replicate including:

* load all the cuda apt packages,
* LD_LIBRARY_PATH and dynamic linking setup,
* load tensorflow-gpu

# Repo Layout

## ml.sh
This is a bash function library that illustrates how to build, start, stop,
etc. the containers.  Run `./ml.sh` to see the function list.  The function
library has only been run on Ubuntu 18.04.

Functions that begin with `host_` are to be run on the host (outside of the
container) and those with `guest_` are to be run inside a container.

The `host_cpu` functions must have the environment variable `ML_IMG` set and
are targeted for a CPU-only tensorflow environment.

The `host_gpu` functions must have the environment variable `GPU_IMG` set to
the docker image to use and are targeted for a GPU tensorflow environment.
The image should be capable of GPU support
(see `ml.sh:host_gpu_setup`)  If a GPU does not exist,
regression tests will run on the CPU.

For example to build an image with GPU-support and run a container based on it:

```
host> export ML_IMG=ml_gpu:latest
host> ./ml.sh host_gpu_build`
host> ./ml.sh host_gpu_run
```

This will start an interactive container with default bash prompt.  The
container will be destroyed on exit.

Once the container is running, go to `user1` and run the regression tests:

For CPU:
```
guest root> su user1
user1:1> ./ml.sh guest_test
```

For GPU:
```
guest root> su user1
user1:1> ./ml.sh guest_gpu_test
```

### ml.sh GPU setup
There are a number of functions in `ml.sh` to set up the nvidia CUDA libraries
and build a docker GPU image.  These only need to be done once.

* host_install_nvidia: install the lastest nvidia driver for the detected GPU,
  and switch to the intel display for graphics.  This requires a reboot.
* host_gpu_probe: test the nvidia GPU
* host_gpu_setup: install the necessary APT packages to run a GPU docker image
* host_gpu_build: build a custom GPU docker image use a base of
  `tensorflow/tensorflow:latest-gpu-py3`.  Name the image using the `GPU_IMG`
  environment variable.
  
### ml.sh GPU execution
There is one host-side function to start a container for `GPU_IMG`.  On exit,
the container will be removed.

* host_gpu_run: start a container for `GPU_IMG`.

There is one guest-side function to regresstion test GPU and X11 functionality.

* guest_gpu_test: regression test a GPU-enabled docker container. 

## ml.Dockerfile
This is the dockerfile used to build an image using only the CPU. See
`ml.sh:host_build` for its use.

This dockerfile creates a non-privileged user matching the user id/group of the
owner of the host volumes.

The container starts in root.  The first step is to run `su user1` to switch to
the non-privileged user1 and load the `.bashrc`.

## ml-gpu.Dockerfile
This is the dockerfile used to build an image that *CAN* use a GPU if found.
It has the same basic functionality as `ml.Dockerfile`.

## bashrc.docker
This is a simple `.bashrc` for the user.  Simple, nothing interesting.  I like
to run inside an emacs shell window so control characters and highlighting are
not desired.

## Regression Scripts
Simple python scripts to verify that all the expected functionality is present
and working.

### ut_gpu.py
This demonstrates GPU capability and acceleration versus CPU.  There *may* be an
initial staill of several minutes while the GPU is provisioned.

### ut_ml.py
This demonstrates that all the necessary python packages are installed and runs some
simple unit tests.

### ut_tf.py
This contains a set of simple keras and tf2 tests.  See
* [Keras Tutorials](https://www.tensorflow.org/tutorials/keras/regression)
* [TF2 Beginner MNIST](https://www.tensorflow.org/tutorials/quickstart/beginner)

### ut_hub.py
This is an extensive test using:
* the [flowers photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
  dataset
* transfer learning using `mobilenetv2`
* model based on `mobilenetv2` adding a couple layers
* Categorical Cross Entropy loss function

This will take ~20min to run on a CPU.  It fails on a GPU.  I'm still analyzing
this and I have removed it from the regression tests.

## Research Scripts
