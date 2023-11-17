sample=$1
num_layers=$2
num_nonlinear=$3
hidden=$4
dataset=$5
alpha=$6

python train_cifar.py --seed 0 --alpha $alpha --eps 0.1 --dataset $dataset --init orthogonal --depth $num_layers --num_nonlinear $num_nonlinear --hidden $hidden --sample_size $sample

for layer_idx in {0..8}; 
    do
    python linear_probe.py \
        --depth $num_layers --hidden $hidden \
        --num_nonlinear $num_nonlinear \
        --sample_size $sample \
        --layer_idx $layer_idx \
        --load_path saved_linear/dataset_${dataset}_w${hidden}_d{args.depth}_nd${num_nonlinear}_init_orthogonal_eps0.1_sample${sample}/
done

python post_processing.py \
    --load_path saved_linear/dataset_${dataset}_w${hidden}_d{args.depth}_nd${num_nonlinear}_init_orthogonal_eps0.1_sample${sample}/ \
    --depth $num_layers \
    --num_nonlinear $num_nonlinear \
    --hidden $hidden