PWD=`pwd`
DOCKER_DIR=/`basename $PWD`
CTNRNAME=ROCmDockerContainer

IMAGE="rocm/tensorflow:rocm3.3-tf1.15-dev"
#IMAGE="rocm/tensorflow:rocm3.1-tf1.15-ofed4.6-openmpi4.0.0-horovod"
#IMAGE="rocm/tensorflow-private:rocm3.3-tf2.1-ofed4.6-openmpi4.0.0-horovod"
# -u `id -u`:`id -g`
docker run -it --network=host  --ipc=host --shm-size 16G  -v=`pwd`:$DOCKER_DIR -v /data:/data -w $DOCKER_DIR --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined $IMAGE

