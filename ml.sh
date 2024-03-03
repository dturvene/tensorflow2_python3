#!/bin/bash
# bash function library for ML docker support
#
# * .bashrc for current ML_IMG setting
# * host_cpu_r: start a docker container for ML testing
# * client_install.sh:linger_clone for cloning this repo
#
# 240106: restart work in repo
# * new_host_cpu_b : build a new image based on tensorflow/tensorflow
# * new_host_cpu_r : run the container
# * new_test_container: connect to container from secondary terminals
#
# connect to running container:
#   docker exec -e INSIDE_EMACS -it $(docker ps -q) bash

dstamp=$(date +%y%m%d)
tstamp='date +%H%M%S'
default_func=usage

# Defaults for options
ML_WORKPATH=$HOME/ggit/tf2_python3

usage() {

    printf "\nUsage: $0 [options] func [func]+"
    printf "\n Valid opts:\n"
    printf "\t-i ML_IMG: docker image (default ${ML_IMG})\n"
    printf "\t-e ML_WORKPATH: env var (default ${ML_WORKPATH})\n"
    printf "\n Valid funcs:\n"

    # display available function calls
    typeset -F | sed -e 's/declare -f \(.*\)/\t\1/' | grep -v -E "usage|parseargs"
}

parseargs() {

    while getopts "e:h" Option
    do
	case $Option in
	    e ) M_DEF=$OPTARG;;
	    h | * ) usage
		exit 1;;
	esac
    done

    shift $(($OPTIND - 1))
    
    # Remaining arguments are the functions to eval
    # If none, then call the default function
    EXECFUNCS=${@:-$default_func}
}

t_prompt() {

    printf "Continue [YN]?> " >> /dev/tty; read resp
    resp=${resp:-n}

    if ([ $resp = 'N' ] || [ $resp = 'n' ]); then 
	echo "Aborting" 
        exit 1
    fi
}

trapexit() {

    echo "Catch Ctrl-C"
    t_prompt
}

###########################################
# Operative functions
# ML docker - Machine learning using TF2
# ml.Dockerfile
#####################################################################

# build docker image
# example ML_IMG defs:
#   ml_cpu:latest
host_cpu_b()
{
    # need to be here for small directory cache size
    # and copy files to guest
    cd $ML_WORKPATH
    
    if [ -z "$ML_IMG" ]; then
	echo "\$ML_IMG needs to be set"
	docker images
	exit -1
    fi

    echo "$PWD: Building new ML_IMG=$ML_IMG"
    t_prompt

    echo "see what is running"
    docker ps -a
    # remove current container and image to build anew
    # docker rm $CID
    # docker rmi $ML_IMG
    
    echo "Building $ML_IMG in $PWD"
    docker build -f ml.Dockerfile -t $ML_IMG .

    echo "Show all images"
    docker images
}

# start docker image
host_cpu_r()
{
    # set necessary work areas
    DIR_REF=$ML_WORKPATH
    DIR_PYTHON=$HOME/ggit/python.git
    DIR_DATA=$HOME/ML_DATA
    DIR_ML=/opt/distros/ML
    DIR_GST=/opt/distros/GST
    USER=user1
    
    # host workspace, the git repo
    cd $DIR_REF
    printf "$PWD: starting\n$(docker images $ML_IMG)\n"
    t_prompt

    if [ -z "$ML_IMG" ]; then
	echo "\$ML_IMG needs to be set"	
	docker images
	exit -1
    fi

    # for video to work, must map device and --net=host
    # 220307: publish 8080, WARNING Published ports are discarded
    echo "Starting $ML_IMG in $DIR_REF"
    docker run \
	   --env="DISPLAY" \
	   --device=/dev/video0:/dev/video0 \
	   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	   --volume="$PWD:/home/ref" \
	   --volume="$DIR_PYTHON:/home/work" \
	   --volume="$DIR_DATA:/data" \
	   --volume="$DIR_ML:/home/ml" \
	   --volume="$DIR_GST:/home/gst" \
	   --workdir=/home/ref \
	   --net=host \
	   --rm -it $ML_IMG

    # run guest_cpu_test

}

host_info()
{
    if [ -z "$ML_IMG" ]; then
	echo "\$ML_IMG needs to be set"	
	docker images
	exit -1
    fi

    echo "### docker ps"
    docker ps
    echo "### docker history $ML_IMG"
    docker image history $ML_IMG
    echo "### docker inspect $ML_IMG"    
    docker image inspect $ML_IMG
}

host_cpu_commit()
{
    echo "commit container changes to a new image"

    ML_IMG_NEW=ml2:C1
   
    # should show current image tag
    docker images

    # should only be one running container
    CONTAINER_ID=$(docker ps -q)

    echo "ML_IMG=$ML_IMG ML_IMG_NEW=$ML_IMG_NEW"

    # https://docs.docker.com/engine/reference/commandline/container_commit/
    # -m: commit message
    # container id:
    # image repo:tag
    docker commit -m "running container updates" $CONTAINER_ID $ML_IMG_NEW

    # check new image is created
    docker images
}

host_info()
{
    docker ps
    
    docker image history $ML_IMG

    docker image inspect $ML_IMG
}

#####################################################################
# NVIDIA h/w setup for use as a GPU
#####################################################################
# https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/
host_install_nvidia()
{
    # detect nvidia h/w
    sudo lshw -C display

    echo "Remove existing nvidia drivers"
    sudo apt-get -y update
    sudo apt-get purge nvidia*

    # python3 script to check for nvidia drivers
    ubuntu-drivers devices

    # install the recommended nvidia driver
    # linger: GeForce GTX 1050 Ti Mobile
    # hoho: GeForce GTX 960M
    sudo ubuntu-drivers autoinstall

    # python script to check for, and switch graphics drivers
    prime-select query

    # switch to intel built-in graphics so the nvidia card is not used
    sudo prime-select intel

    # reboot
    sudo shutdown -r now
}

host_gpu_probe()
{
    # check this is intel
    sudo prime-select query

    # the nvidia drivers will not be loaded when intel is graphics
    # load the driver now
    sudo modprobe -v nvidia-uvm

    # check the driver and hw health
    nvidia-smi -a
    nvidia-debugdump -l
}

host_gpu_unload()
{
    sudo modprobe -r nvidia-uvm

    resp=$(nvidia-smi)
    if [ $? != 9 ]; then
	echo "nvidia driver: $resp"
    fi
}

#####################################################################
# GPU support needs for docker:
# * a special nvidia base image
# * a number of CUDA libraries
# * LD_LIBRARY_PATH update and library soft link update
# * a number of CUDA environment variables
# Best to use the TF gpu images for now
#
# See
# * tensorflow/tools/dockerfiles/dockerfiles/
#####################################################################
# https://github.com/NVIDIA/nvidia-docker/blob/master/README.md#quickstart
host_gpu_setup()
{
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    # ubuntu18.04
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    echo "host: install nvidia-container-toolkit"
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    systemctl status docker
}

#export GPU_IMG=tensorflow/tensorflow:devel-gpu-py3
#export GPU_IMG=tensorflow/tensorflow:nightly-gpu-py3
# this is the best
#export GPU_IMG=tensorflow/tensorflow:latest-gpu-py3
host_gpu_pull()
{

    echo "Use host_gpu_build"
    exit 0
    
    # need to be here for small directory cache size
    # and copy files to guest
    cd ~/GIT/tensorflow2_python3

    echo "Pull $GPU_IMG in $PWD"
    docker pull $GPU_IMG

    docker images
}

host_gpu_build()
{
    if [ -z "$GPU_IMG" ]; then
	echo "\$GPU_IMG needs to be set"
	docker images
	exit -1
    fi
    
    # need to be here for small directory cache size
    # and copy files to guest
    cd ~/GIT/tensorflow2_python3

    echo "Building $ML_IMG in $PWD"
    docker build -f ml-gpu.Dockerfile -t $GPU_IMG .

    docker images
}

host_gpu_run()
{
    DIR_REF=$ML_WORKPATH    
    DIR_PYTHON=$HOME/ggit/python.git
    DIR_DATA=$HOME/ML_DATA
    
    # host workspace, the git repo
    cd $DIR_REF

    if [ -z "$GPU_IMG" ]; then
	echo "\$GPU_IMG needs to be set"	
	docker images
	exit -1
    fi
    
    docker run \
	   --env="DISPLAY" \
	   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	   --volume="$PWD:/home/ref" \
	   --volume="$DIR_WORK:/home/work" \
	   --volume="$DIR_DATA:/data" \
	   --workdir=/home/work \
	   --gpus all \
	   --rm -it $GPU_IMG
}

###############################################################################
# Host config
###############################################################################
host_tflite_data()
{
    echo "get test image"

    cd /opt/distros/ML/google-coral/edgetpu/test_data

    ls -l grace_hopper.bmp

    echo "get tflite models"
    ls -l mobilenet_v1_1.0_224_quant.tflite
    ls -l mobilenet_v2_1.0_224_quant.tflite
    
    echo "get labels"
    # 1000
    ls -l imagenet_labels.txt
    # 90
    ls -l coco_labels.txt
}

host_tf_models()
{
    cd ~/ML_DATA/models
    wget https://tfhub.dev/google/imagenet/inception_v1/feature_vector/1?tf-hub-format=compressed -O inception_v1.tgz
    mkdir -p inception_v1/feature_vector/1
    tar -zxvf ../../../inception_v1.tgz

    # get tag_set and signature_def names for tfhub model
    saved_model_cli show --dir /data/models/inception_v1/feature_vector/1
    saved_model_cli show --dir /data/models/inception_v1/feature_vector/1 --tag_set train
    # saved_model_cli show --dir /data/models/inception_v1/feature_vector/1 --tag_set train --signature_def default
    saved_model_cli show --dir /data/models/inception_v1/feature_vector/1 --tag_set train --signature_def image_feature_vector

    echo "local saved, see flg1.py"
    saved_model_cli show --dir /data/dflg1/1 --tag_set serve --signature_def serving_default

    # works
    wget https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4?tf-hub-format=compressed -O mobilenet_v2_140_224.tgz
    wget https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4?tf-hub-format=compressed -O mobilenet_v1_025_224.tgz

    saved_model_cli show --dir /data/models/mobilenet_v2_035_128/4 --all
    # __saved_model_init_op is a noop placeholder

    saved_model_cli show --dir /data/models/mobilenet_v1_025_224/4 --all
}

# git clone the gstreamer python plugin source
# this needs to be built and installed in the guest ONCE
# after docker container start (guest_gst_config)
host_gstreamer()
{
    cd /opt/distros/ML
    
    # on host get python bindings and checkout tag
    git clone https://gitlab.freedesktop.org/gstreamer/gst-python.git
    git checkout 1.14.5
}

###############################################################################
# Guest config, setup and regression testing
###############################################################################
# set up gstreamer on guest
# We need to build and install the gstreamer python plugin glue
# then check it is installed (gst-inspect-1.0 python)
guest_gst_config()
{
    echo "build gst-python from source"
    cd /home/ml/gst-python
    export PYTHON=/usr/bin/python3
    ./autogen.sh --disable-gtk-doc --noconfigure
    # ./configure --with-libpython-dir=/usr/lib/x86_64-linux-gnu
    ./configure --prefix=/usr --with-libpython-dir=/usr/lib/x86_64-linux-gnu
    make
    sudo make install

    echo "check install"
    export GST_PLUGIN_PATH=/usr/lib/gstreamer-1.0
    gst-inspect-1.0 python
}

# after image build - run this in guest to validate functionality
# See docker.sh:conn_shell
guest_cpu_test()
{
    # manually check with $(id)
    if [[ $EUID -ne 1000 ]]; then
	echo "must be user1 for X11 display"
	exit 1
    fi

    # check guest distro - see docker.sh:guest_linux
    cat /proc/version
    cat /etc/os-release
    python --version

    # check for gstreamer 1.0 python bindings
    # python -c "import gi; gi.require_version('Gst', '1.0')"

    echo "$PWD: Run Python Tensorflow and Web tests"
    t_prompt
    
    # unit test python ML packages
    echo "Tensorflow regression testing..."
    # python ut_ml.py
    python ut_tf.py

    # tensorflow, hub and pre-trained model
    # comprehensive image training and validation on flower dataset
    # this takes 20min to download and train....
    FLOWERS_PREDICT_FILES=/data/flowers.predict
    if [ -d $FLOWERS_PREDICT_FILES  ]; then
	echo '$FLOWERS_PREDICT_FILES exists so training'
	python ut_hub.py
    fi

    # test asyncio
    cd /home/work

    # 220319: need to rebuild image with these packages...
    sudo pip3 install asyncio aiohttp websockets
    # t_cbp_feed appends to mltest.dat, best to remove if collecting over multiple hours
    # basket of cryptos SIGINT to exit
    ./work.py -t 1 -f /data/mltest.dat
    # t_display, read file and plot prod and side
    ./work.py -t 2 -f /data/mltest.dat
    # gemini websocket feed, SIGINT to exit
    # sudo pip3 install websocket
    # ./p3_async.py -f 6
}

guest_tflite_test()
{
    export TPU=/home/ml/google-coral/edgetpu
    
    # label objects in an image using tflite model
    # 0.919720: 653  military uniform, 0.017762: 907  Windsor tie
    I=$TPU/test_data/grace_hopper.bmp
    # I=/home/work/gstreamer/img-x.png
    # M=/home/edgetpu/test_data/inception_v2_224_quant.tflite
    # .89 653 military uniform
    M=$TPU/test_data/mobilenet_v2_1.0_224_quant.tflite
    # M=/home/edgetpu/test_data/mobilenet_v1_0.75_192_quant.tflite
    L=$TPU/test_data/imagenet_labels.txt
    echo "Run a tflite model ${M}"

    cd /home/ml/tensorflow/tensorflow/lite/examples/python
    python3 label_image.py \
	    -i ${I} \
	    -m ${M} \
	    -l ${L}
}

guest_tf2_images()
{
    export TPU=/home/ml/google-coral/edgetpu
    
    imglist="/data/TEST_IMAGES/strawberry.jpg \
             /data/TEST_IMAGES/dog-1210559_640.jpg \
	     /data/TEST_IMAGES/cat-2536662_640.jpg \
	     /data/TEST_IMAGES/siamese.cat-2068462_640.jpg \
	     /data/TEST_IMAGES/animals-2198994_640.jpg \
	     "
    
    M=$TPU/test_data/mobilenet_v2_1.0_224_quant.tflite
    # M=$TPU/test_data/inception_v2_224_quant.tflite
    L=$TPU/test_data/imagenet_labels.txt

    cd /home/ml/tensorflow/tensorflow/lite/examples/python
    for I in $imglist
    do
	echo "*** label $I"
	python3 label_image.py \
		-i ${I} \
    		-m ${M} \
		-l ${L} 2> /dev/null
    done
}

guest_tflite_image()
{
    export TPU=/home/ml/google-coral/edgetpu
    
    #I=/home/work/gstreamer/img-rgb.png
    #I=/data/TEST_IMAGES/strawberry.jpg
    I=/home/ml/google-coral/edgetpu/test_data/grace_hopper.bmp

    # only size-1 arrays can be converted to Python scalars
    #M=/data/GST_TEST/detect.tflite
    #L=/data/GST_TEST/labelmap.txt
    M=$TPU/test_data/mobilenet_v2_1.0_224_quant.tflite
    L=$TPU/test_data/imagenet_labels.txt

    #cd /home/ml/tensorflow/tensorflow/lite/examples/python
    cd /home/ref
    echo "*** label $I"
    python label_image.py \
	    -i ${I} \
    	    -m ${M} \
	    -l ${L}
}

guest_gst_plugin_test()
{
    echo "check custom python plugins"
    cd /home/work/gstreamer
    # all python plugins are under $PWD/plugins/python
    export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:$PWD/plugins
    
    gst-inspect-1.0 identity_py
    GST_DEBUG=python:4 gst-launch-1.0 fakesrc num-buffers=10 ! identity_py ! fakesink
}

guest_gpu_test()
{
    cd /home/ref

    echo "check CUDA version"
    cat /usr/local/cuda/version.txt
    
    #echo "check tools for devel images"
    #nvcc --version
    
    echo "Benchmark GPU"
    python ut_gpu.py

    python ut_ml.py
    python ut_tf.py
}

###########################################################
# 240106 Restart ML Docker Work
# * https://www.tensorflow.org/install/docker
#   Docker setup has become more complex
###########################################################
git_tag_work()
{
    git tag -a ML1 -m 'original tensorflow work from 220219'
}

new_host_cpu_b()
{
    export ML_IMG=tf240106

    docker images
    docker rmi $ML_IMG
    
    #docker pull $ML_IMG
    #docker images

    docker build -f ml.Dockerfile -t $ML_IMG .
}

new_quick_test()
{
    docker run -it --rm tensorflow/tensorflow \
	   python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

    # 2024-01-06 19:35:45.656896: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    # To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    # tf.Tensor(2039.9213, shape=(), dtype=float32)

}
	   
new_host_cpu_r()
{
    DIR_REF=$ML_WORKPATH
    DIR_PYTHON=$HOME/ggit/python.git
    DIR_DATA=$HOME/ML_DATA

    cd $DIR_REF
    
    printf "$PWD: starting\n$(docker images $ML_IMG)\n"
    t_prompt

    if [ -z "$ML_IMG" ]; then
	echo "\$ML_IMG needs to be set"	
	docker images
	exit -1
    fi

    # Copied from host_cpu_r
    echo "Starting $ML_IMG in $DIR_REF"
    echo "for non-root, su user1"
    docker run \
	   --env="DISPLAY" \
	   --device=/dev/video0:/dev/video0 \
	   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	   --volume="$PWD:/home/ref" \
	   --volume="$DIR_PYTHON:/home/work" \
	   --volume="$DIR_DATA:/data" \
	   --workdir=/home/ref \
	   --net=host \
	   --rm -it $ML_IMG

    # su user1
    #  /sbin/ldconfig.real: Can't create temporary cache file /etc/ld.so.cache~: Permission denied
    #  can replicate by `user1> ldconfig -v`
    #  SOLVED: this is caused when a shared library is added or modified, seems to be
    #  unimportant but can `sudo ldconfig -V` next
}

# Use docker.sh:conn_shell
new_test_container()
{
    # clean up root prompt in emacs inferior shell
    export PS1='\u:\!> '

    # additional package installs, remove after image update:
    # 1. copy to ml.Dockerfile and rebuild image
    # 2. 'docker commit' to new image (host_cpu_commit)
    apt install -y vim

    # upgrade pip
    python3 -m pip install --upgrade pip

    # for sklearn
    pip install scikit-learn
}

# 240106: some old tests don't work on new image
new_regtest()
{
    echo 240106: quick regression test in container user1

    cd /home/ref
    python ut_ml.py
    python ut_tf.py
    # flowers image matching, takes a long time to train
    python ut_hub.py
}

###########################################################
# Docker image management
###########################################################

# push to docker hub
# https://ropenscilabs.github.io/r-docker-tutorial/04-Dockerhub.html
# https://docs.docker.com/docker-hub/builds/
host_push()
{
    HUB_IMG=dturvene/tensorflow2_python3:hub_tfds

    # locate tag of stable image
    docker images
    docker tag 0b24e14db02d $HUB_IMG
    docker images

    echo "push $HUB_IMG... long time"
    docker push $HUB_IMG
}

save_image()
{
    cd /opt/distros/docker-images
    
    echo "save $ML_IMG.tar"
    docker save -o $ML_IMG.tar $ML_IMG

    # ~50% smaller
    gzip $ML_IMG.tar
}

load_image()
{
    cd /opt/distros/docker-images
    
    echo "load $ML_IMG.tar"
    gunzip $ML_IMG.tar.gz
    docker load -i $ML_IMG.tar

    docker images

    echo "regression test: host_cpu_run, su user1, guest_cpu_test"
}

###########################################
#  Main processing logic
###########################################
trap trapexit INT

parseargs $*

for func in $EXECFUNCS
do
    eval $func
done

