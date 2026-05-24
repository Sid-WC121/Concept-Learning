#!/bin/bash

for seed in 43 44 45
do
    for suffix in "" "model_robustness" "model_responsiveness" "image_robustness" "image_responsiveness"
    do
        if [ -z "$suffix" ]; then
            echo "$seed mnist"
            python src/models.py --algorithm vae --seed $seed --dataset mnist
        else
            echo "$seed mnist_$suffix"
            python src/models.py --algorithm vae --seed $seed --dataset mnist --suffix $suffix
        fi
    done
done
