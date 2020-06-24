export PATH=$PATH:../bin/
function usage()
{
    echo "Usage:"
    echo ""
    echo "./run_attention.sh"
    echo "\t-h --help"
    echo "\t--iter=$COUNT (number of iterations) "
    echo "\t--vendor=$VENDOR (amd or nvidia)"
    echo "\t--mode=$MODE (benchmark or validation)"
    echo "\t--length=$LENGTH (sequence length) "
    echo "\t--size=$SIZE (size of attention head)"
    echo "\t--batch=$BATCH (batch size)"
    echo "\t--heads=$HEADS (number of attention heads)"
    echo "\t--profile=$PROFILE (true or false)"
    echo ""
}
secs_to_human() {
    if [[ -z ${1} || ${1} -lt 60 ]] ;then
        min=0 ; secs="${1}"
    else
        time_mins=$(echo "scale=2; ${1}/60" | bc)
        min=$(echo ${time_mins} | cut -d'.' -f1)
        secs="0.$(echo ${time_mins} | cut -d'.' -f2)"
        secs=$(echo ${secs}*60|bc|awk '{print int($1+0.5)}')
    fi
    echo "Time Elapsed : ${min} minutes and ${secs} seconds."
}

export CUDA_VISIBLE_DEVICES=0 # choose gpu
export HIP_VISIBLE_DEVICES=0 # choose gpu 

cp /MLSnippets/bin/bc /usr/bin

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --vendor)
            VENDOR=$VALUE
            ;;
        --iter)
            COUNT=$VALUE
            ;;
        --mode)
            MODE=$VALUE
            ;;
        --size)
            SIZE=$VALUE
            ;;
        --batch)
            BATCH=$VALUE
            ;;
        --length)
            LENGTH=$VALUE
            ;;
        --heads)
            HEADS=$VALUE
            ;;
        --profile)
            PROFILE=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

if [ -z "$VENDOR" ]
then
      VENDOR=amd
      echo " SET VENDOR=$VENDOR"
fi


if [ -z "$COUNT" ]
then
      COUNT=10
      echo " SET ITERATIONS=$COUNT"
fi


if [ -z "$MODE" ]
then
      MODE=benchmark
      echo " SET MODE=$MODE"
fi

if [ -z "$BATCH" ]
then
      BATCH=8
      echo " SET BATCH=$BATCH"
fi


if [ -z "$HEADS" ]
then
      HEADS=16
      echo " SET NUM_HEADS=$HEADS"
fi

if [ -z "$SIZE" ]
then
      SIZE=64
      echo " SET SIZE_ATTENTION_HEAD=$SIZE"
fi

if [ -z "$LENGTH" ]
then
      LENGTH=128
      echo " SET SEQ_LENGTH=$LENGTH"
fi

# rocblas trace
# export ROCBLAS_LAYER=3
# export ROCBLAS_LOG_TRACE_PATH=$TRAIN_DIR/ROCBLAS_LOG_TRACE.csv
# export ROCBLAS_LOG_BENCH_PATH=$TRAIN_DIR/ROCBLAS_LOG_BENCH.csv

# # miopen trace
# export MIOPEN_ENABLE_LOGGING=1
# export MIOPEN_LOG_LEVEL=6

# # hip trace
#export HIP_TRACE_API=2
#export HIP_TRACE_API_COLOR=none 
if [ "$PROFILE" = true ]
then
    export HCC_PROFILE=2 
    export HCC_PROFILE_VERBOSE=0x3f
    echo "Set profiling"
fi

starttime=$(date +%s)
# run attention
output=$(python3 attention.py --iter=$COUNT --seq_length=$LENGTH --batch=$BATCH --num_attention_heads=$HEADS --attention_head_size=$SIZE --mode=$MODE 2>&1 | tee log.txt)
if [ "$PROFILE" = true ]
then
    /opt/rocm/hcc/bin/rpt log.txt > hist.txt
fi

endtime=$(date +%s)
echo "VENDOR=$VENDOR MODE=$MODE ITER=$COUNT BATCH_SIZE=$BATCH SEQ_LENGTH=$LENGTH NUM_ATTENTION_HEADS=$HEADS SIZE_ATTENTION_HEAD=$SIZE" >> eval_results.txt
secs_to_human "$(($(date +%s) - ${starttime}))" >> eval_results.txt

