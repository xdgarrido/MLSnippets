PWD=`pwd`
DOCKER_DIR=/`basename $PWD`
CTNRNAME=CudaContainer
# -u `id -u`:`id -g
IMAGE=nvcr.io/nvidia/tensorflow:20.03-tf1-py3
#IMAGE=tensorflow/tensorflow:2.2.0rc1-gpu-py3
docker run -it --gpus all --network=host --shm-size=16g  --ulimit memlock=-1 --ulimit stack=67108864 -v=`pwd`:$DOCKER_DIR -v /data:/data -w $DOCKER_DIR --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video  --security-opt seccomp=unconfined $IMAGE 


