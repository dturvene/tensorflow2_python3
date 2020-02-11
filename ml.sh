#!/bin/bash
# bash function library for ML docker support

dstamp=$(date +%y%m%d)
tstamp='date +%H%M%S'
default_func=usage

# Defaults for options
M_DEF="placeholder"

usage() {

    printf "\nUsage: $0 [options] func [func]+"
    printf "\n Valid opts:\n"
    printf "\t-e M_DEF: override env var (default ${M_DEF})\n"
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
host_cpu_build()
{
    if [ -z "$ML_IMG" ]; then
	echo "\$ML_IMG needs to be set"
	docker images
	exit -1
    fi
    
    # need to be here for small directory cache size
    # and copy files to guest
    cd ~/GIT/tensorflow2_python3

    echo "Building $ML_IMG in $PWD"
    docker build -f ml.Dockerfile -t $ML_IMG .

    docker images
}

# start docker image
host_cpu_run()
{
    DIR_REF=$HOME/GIT/tensorflow2_python3
    DIR_WORK=$HOME/GIT/python
    DIR_DATA=$HOME/ML_DATA
    
    # host workspace, the git repo
    cd $DIR_REF

    if [ -z "$ML_IMG" ]; then
	echo "\$ML_IMG needs to be set"	
	docker images
	exit -1
    fi
    
    docker run \
	   --env="DISPLAY" \
	   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	   --volume="$PWD:/home/ref" \
	   --volume="$DIR_WORK:/home/work" \
	   --volume="$DIR_DATA:/data" \
	   --workdir=/home/ref \
	   --rm -it $ML_IMG
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
    DIR_REF=$HOME/GIT/tensorflow2_python3
    DIR_WORK=$HOME/GIT/python
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
# Guest setup and regression testing
###############################################################################
guest_cpu_test()
{
    echo "must be user1 for X11 display"
    
    # unit test python ML packages
    echo "Tensorflow regression testing..."
    python ut_ml.py
    
    python ut_tf.py

    # tensorflow, hub and pre-trained model
    # comprehensive image training and validation on flower dataset
    # this takes 20min to download and train....
    FLOWERS_PREDICT_FILES=/data/flowers.predict
    if [ -d $FLOWERS_PREDICT_FILES  ]; then
	echo '$FLOWERS_PREDICT_FILES exists so training'
	python ut_hub.py
    fi

    # label an image
    # get datafiles into /data/demo_files
    python3 label_image.py \
	    -i /data/demo_files/grace_hopper.bmp \
	    -m /data/demo_files/mobilenet_v1_1.0_224_quant.tflite \
	    -l /data/demo_files/imagenet_labels.txt

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

###########################################
#  Main processing logic
###########################################
trap trapexit INT

parseargs $*

for func in $EXECFUNCS
do
    eval $func
done

