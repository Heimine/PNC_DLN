# Assumption 2
for depth in "1" "2" "3" "4" "5" "6"; do
    for hidden in "200" "300" "400" "500" "600" "700"; do
        for seed in {1..10}; do
            python train.py --hidden $hidden --depth $depth --eps 0.1 --lr 0.1 --seed $seed
        done
    done
done