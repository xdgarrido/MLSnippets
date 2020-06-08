export PATH=$PATH:../bin/
function usage()
{
    echo "Usage:"
    echo ""
    echo "./run_optimizer.sh"
    echo "\t-h --help"
    echo "\t--iter=$COUNT (number of iterations) "
    echo "\t--vendor=$VENDOR (amd or nvidia)"
    echo "\t--mode=$MODE (benchmark or validation)"
    echo "\t--length=$LENGTH (sequence length) "
    echo "\t--batch=$BATCH (amd or nvidia)"
    echo "\t--heads=$HEADS (number of attention heads)"
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

# # hcc profile
#export HCC_PROFILE=2 

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
        --batch)
            BATCH=$VALUE
            ;;
        --length)
            LENGTH=$VALUE
            ;;
        --heads)
            HEADS=$VALUE
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
      COUNT=100
      echo " SET ITERATIONS=$COUNT"
fi


if [ -z "$MODE" ]
then
      MODE=benchmark
      echo " SET MODE=$MODE"
fi

if [ -z "$BATCH" ]
then
      BATCH=6
      echo " SET BATCH=$BATCH"
fi


if [ -z "$HEADS" ]
then
      HEADS=24
      echo " SET HEADS=$HEADS"
fi

if [ -z "$LENGTH" ]
then
      LENGTH=512
      echo " SET SEQ_LENGTH=$LENGTH"
fi

starttime=$(date +%s)
# run dropout
output=$(python3 dropout.py --iter=$COUNT --seq_length=$LENGTH --batch=$BATCH --attention_heads=$HEADS --mode=benchmark &> profile_dropout.txt)
#/opt/rocm/hcc/bin/rpt profile_droput.txt > profile_dropout_HIST.txt
endtime=$(date +%s)
echo "VENDOR=$VENDOR MODE=$MODE ITER=$COUNT BATCH_SIZE=$BATCH SEQ_LENGTH=$LENGTH NUM_ATTENTION_HEADS=$HEADS" >> eval_results.txt
secs_to_human "$(($(date +%s) - ${starttime}))" >> eval_results.txt

