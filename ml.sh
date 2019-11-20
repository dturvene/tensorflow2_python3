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
host_build()
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
host_run()
{
    # host workspace, the git repo
    cd ~/GIT/tensorflow2_python3/

    if [ -z "$ML_IMG" ]; then
	echo "\$ML_IMG needs to be set"	
	docker images
	exit -1
    fi
    
    docker run \
	   --env="DISPLAY" \
	   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	   --volume="$PWD:/home/work" \
	   --volume="$HOME/ML_DATA:/data" \
	   --workdir=/home/work \
	   --rm -it $ML_IMG
}

host_commit()
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

host_save()
{
    echo "docker save"
    echo "docker load"
    echo "docker commit"
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

###############################################################################
# Guest regression testing
###############################################################################
guest_test()
{
    # unit test python ML packages
    echo "Tensorflow regression testing..."
    python ut_ml.py
    
    # keras basic regression with seaborn plots
    # creates a png that is used in ut_ml.py
    echo "Keras basic regression using the MPG dataset"
    python ut_keras.py

    # tensorflow, hub and pre-trained model
    # comprehensive image training and validation on flower dataset
    # this takes 20min to download and train....
    FLOWERS_PREDICT_FILES=/data/flowers.predict
    if [ ! -d $FLOWERS_PREDICT_FILES  ]; then
	echo 'no $FLOWERS_PREDICT_FILES so cannot validate training'
	exit -1
    fi
    #python ut_hub.py
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

