#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models
mkdir -p $logs

num_threads=4
device=""

SECONDS=0

dropouts=("0.0" "0.2" "0.3" "0.5" "0.6" "0.7")

for dropout in "${dropouts[@]}"; do
    echo "Running model with dropout=$dropout"

    model_name=$(echo "$dropout" | tr '.' '_')

    (cd $tools/pytorch-examples/word_language_model &&
        CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/fitzgerald \
            --accel \
            --epochs 40 \
            --log-interval 100 \
            --emsize 200 --nhid 200 --dropout $dropout --tied \
            --save $models/model_dp_${dropout}.pt \
            --log-file "../../../logs/log_dp_${dropout}.tsv"
    )

done

echo "time taken:"
echo "$SECONDS seconds"