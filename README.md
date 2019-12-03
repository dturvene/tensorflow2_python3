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

# Repo Layout

## ml.Dockerfile
This is the dockerfile used to build the image. See `ml.sh:host_build` for its
use.

This dockerfile creates a non-privileged user matching the user id/group of the
owner of the host volumes.

The container starts in root.  The first step is to run `su user1` to switch to
the non-privileged user1 and load the `.bashrc`.

## ml.sh
This is a bash library that illustrates how to build, start, stop, etc. the
container.  Run `./ml.sh` to see the function list.  Functions that begin with
`host_` are to be run on the host (outside of the container) and those with
`guest_` are to be run inside a container.  The `host_` functions must have the
environment variable `ML_IMG` set 
(`export ML_IMG=dturvene/tensorflow2_python3:hub_tfds`) 

For example to build the image and run a container based on it:

```
host> export ML_IMG=dturvene/tensorflow2_python3:hub_tfds`
host> ./ml.sh host_build`
host> ./ml.sh host_run
```

This will start an interactive container with default bash prompt that will be removed on exit.

Once the container is running, go to `user1` and run the regression tests:

```
guest root> su user1
user1:1> ./ml.sh guest_test
```

## bashrc.docker
This is a simple `.bashrc` for the user.  Simple, nothing interesting.  I like
to run inside an emacs shell window so control characters and highlighting are
not desired.

## Regression Scripts
See `ml.sh:guest_test` for the use of the regression scripts.  These attempt to
verify that all the expected functionality is present and working.

### ut_ml.py
This demonstrates that all the necessary python packages are installed and runs some
simple unit tests.

### ut_keras.py
This is a cut-and-paste from [Keras Tutorials](https://www.tensorflow.org/tutorials/keras/regression).
It uses the auto-mpg dataset and plots the predicted MPG regression against the
actual MPG feature values.

### ut_hub.py
This is an extensive test using:
* the [flowers photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
  dataset
* transfer learning using `mobilenetv2`
* model based on `mobilenetv2` adding a couple layers
* Categorical Cross Entropy loss function

This will take ~20min to run.


## Research Scripts
