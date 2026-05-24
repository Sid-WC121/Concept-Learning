#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_DIR="${CONCEPT_DATASET_DIR:-$REPO_ROOT/dataset}"

# Ensure TensorFlow can find CUDA/cuDNN from the conda environment
export LD_LIBRARY_PATH="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null)/lib:${LD_LIBRARY_PATH:-}"

PYTHON="python"
SEEDS="43 44 45"
NUM_RANDOM_EXP=3
IMAGES_PER_FOLDER=100

create_variant() {
    local variant="$1"
    local suffix="$2"
    local flip_prob="$3"
    local img_dirname="$4"

    local base_train="${DATASET_DIR}/chexpert/preprocessed/train.pkl"
    local base_val="${DATASET_DIR}/chexpert/preprocessed/val.pkl"
    local out_train="${DATASET_DIR}/${variant}/chexpert/preprocessed/train.pkl"
    local out_val="${DATASET_DIR}/${variant}/chexpert/preprocessed/val.pkl"

    if [[ ! -f "$out_train" || ! -f "$out_val" ]]; then
        echo "  Creating pkls for ${suffix}..."
        mkdir -p "$(dirname "$out_train")" "$(dirname "$out_val")"
        $PYTHON -c "
import pickle, numpy, random, sys, os
sys.path.insert(0, 'src')
from dataset import Chexpert_Dataset, flip_concept_labels
d = Chexpert_Dataset()
for inp, out in [('${base_train}', '${out_train}'), ('${base_val}', '${out_val}')]:
    data = pickle.load(open(inp, 'rb'))
    new_data = flip_concept_labels(data, ${flip_prob}, d.fix_path, '_${suffix}')
    pickle.dump(new_data, open(out, 'wb'))
    print(f'  Wrote {out} ({len(new_data)} samples)')
" 2>/dev/null
    else
        echo "  pkls already exist for ${suffix}"
    fi

    local img_dir="${DATASET_DIR}/${img_dirname}"
    local src_img="${DATASET_DIR}/chexpert/images"
    if [[ -d "$img_dir/images" && ! -L "$img_dir/images" ]]; then
        echo "  Replacing real image dir with symlink (avoids incomplete processing)..."
        rm -rf "$img_dir/images"
    fi
    if [[ ! -d "$img_dir/images" ]]; then
        echo "  Creating image symlink ${img_dirname}/images -> chexpert/images..."
        mkdir -p "$img_dir"
        ln -sf "$src_img" "$img_dir/images"
    fi
}

echo "=== CheXpert TCAV Pipeline ==="
echo ""

if [[ ! -f "${DATASET_DIR}/chexpert/preprocessed/train.pkl" ]]; then
    echo "ERROR: Base CheXpert data not found at ${DATASET_DIR}/chexpert/preprocessed/train.pkl"
    echo "Run: python scripts/dataset_scripts/download_chexpert.py"
    exit 1
fi

echo "--- Ensuring robustness/responsiveness variants ---"
create_variant "robustness" "image_robustness" 0.01 "chexpert_image_robustness"
create_variant "responsiveness" "image_responsiveness" 0.5 "chexpert_image_responsiveness"
echo ""

echo "--- Running TCAV ---"
for seed in $SEEDS; do
    for suffix in none image_robustness image_responsiveness; do
        echo ""
        echo "chexpert seed=${seed} suffix=${suffix}"
        $PYTHON src/create_vectors.py \
            --algorithm tcav \
            --dataset chexpert \
            --target zebra \
            --num_random_exp "${NUM_RANDOM_EXP}" \
            --images_per_folder "${IMAGES_PER_FOLDER}" \
            --seed "${seed}" \
            --suffix "${suffix}"
    done
done

echo ""
echo "=== Done ==="
