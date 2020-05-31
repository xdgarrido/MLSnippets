export PATH=$PATH:../bin/
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

COUNT=$1 
starttime=$(date +%s)
# run dropout
output=$(python3 dropout.py --iter=$COUNT --mode=benchmark &> profile_dropout.txt)
#/opt/rocm/hcc/bin/rpt profile_droput.txt > profile_dropout_HIST.txt
endtime=$(date +%s)
secs_to_human "$(($(date +%s) - ${starttime}))" >> eval_results.txt

