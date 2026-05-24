#!/bin/bash

for seed in 43 44 45
do
    for suffix in "" "model_robustness" "model_responsiveness" "image_robustness" "image_responsiveness"
    do
        if [ -z "$suffix" ]; then
            echo "$seed mnist"
            python src/concept_vectors.py --algorithm model --dataset mnist --seed $seed
        else
            echo "$seed mnist_$suffix"
            python src/concept_vectors.py --algorithm model --dataset mnist --suffix $suffix --seed $seed
        fi
    done
done
