#!/bin/bash
set -euo pipefail

for seed in 43 44 45
do
    for suffix in none image_robustness image_responsiveness
    do
        echo "cub ${seed} ${suffix}"
        python src/create_vectors.py \
            --algorithm tcav \
            --target zebra \
            --num_random_exp 3 \
            --images_per_folder 100 \
            --seed "${seed}" \
            --suffix "${suffix}" \
            --dataset cub
    done
done
