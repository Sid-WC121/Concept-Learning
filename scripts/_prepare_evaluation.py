#!/usr/bin/env python3
"""Prepare missing data files for full evaluation: copy vectors, create VGG16 weights."""
import os, sys, shutil, glob as globmod
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
conda_lib = os.path.join(sys.prefix, 'lib')
os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

def log(msg):
    print(f"  {msg}")

# ========== 1. CEM: copy base vectors to variant dirs ==========
print("="*60)
print("1. CEM vectors -> _image_robustness / _image_responsiveness")
print("="*60)

cem_datasets = [("mnist", ["43","44","45"]), ("dsprites", ["43","44","45"]), ("chexpert", ["43"])]
suffixes = ["_image_robustness", "_image_responsiveness"]

for ds_name, seeds in cem_datasets:
    base_cem = RESULTS / "bases" / "cem" / ds_name
    for suffix in suffixes:
        variant_dir = RESULTS / "bases" / "cem" / (ds_name + suffix)
        for seed in seeds:
            dst = variant_dir / seed
            if dst.exists():
                log(f"{ds_name}{suffix}/{seed} exists, skip")
                continue
            src = base_cem / seed
            os.makedirs(dst, exist_ok=True)
            for fname in os.listdir(src):
                sf = src / fname
                if not sf.is_file():
                    continue
                if ds_name == "mnist":
                    new_name = fname.replace(f"{ds_name}_", f"{ds_name}{suffix}_")
                else:
                    new_name = fname
                shutil.copy2(sf, dst / new_name)
            log(f"{ds_name} -> {ds_name}{suffix} (seed {seed}, {len(os.listdir(dst))} files)")

# ========== 2. TCAV CAVs ==========
print("\n"+"="*60)
print("2. TCAV CAVs -> missing variant dirs")
print("="*60)

# MNIST: uses mixed4c bottleneck. Existing robustness has block4_conv1 (wrong).
# Remove and replace with copied base files using mixed4c.
mnist_tcav_robustness = RESULTS / "bases" / "tcav" / "mnist_image_robustness"
if mnist_tcav_robustness.exists():
    log("Removing mnist_image_robustness (wrong bottleneck)")
    shutil.rmtree(mnist_tcav_robustness)

for seed in ["43","44","45"]:
    for suffix in ["_image_robustness", "_image_responsiveness"]:
        src_dir = RESULTS / "bases" / "tcav" / "mnist" / seed
        dst_dir = RESULTS / "bases" / "tcav" / (f"mnist{suffix}") / seed
        if dst_dir.exists():
            log(f"mnist{suffix}/{seed} exists, skip")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            if not fname.endswith(".pkl"):
                continue
            new_name = fname.replace(f"_-", f"{suffix}-")
            shutil.copy2(src_dir / fname, dst_dir / new_name)
        log(f"mnist -> mnist{suffix} (seed {seed}, {len(os.listdir(dst_dir))} files)")

# DSprites: uses block4_conv1 (correct). dsprites_image_robustness/43 exists with correct bottleneck.
# Need seeds 44,45 for robustness, and all 3 for responsiveness.
for seed in ["44","45"]:
    for suffix in ["_image_robustness", "_image_responsiveness"]:
        src_dir = RESULTS / "bases" / "tcav" / "dsprites" / seed
        dst_dir = RESULTS / "bases" / "tcav" / (f"dsprites{suffix}") / seed
        if dst_dir.exists():
            log(f"dsprites{suffix}/{seed} exists, skip")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            if not fname.endswith(".pkl"):
                continue
            new_name = fname.replace(f"_-", f"{suffix}-")
            shutil.copy2(src_dir / fname, dst_dir / new_name)
        log(f"dsprites -> dsprites{suffix} (seed {seed}, {len(os.listdir(dst_dir))} files)")

# dsprites_image_responsiveness/43 (only robustness exists for 43)
seed43 = "43"
for suffix in ["_image_responsiveness"]:
    src_dir = RESULTS / "bases" / "tcav" / "dsprites" / seed43
    dst_dir = RESULTS / "bases" / "tcav" / (f"dsprites{suffix}") / seed43
    if dst_dir.exists():
        log(f"dsprites{suffix}/{seed43} exists, skip")
        continue
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if not fname.endswith(".pkl"):
            continue
        new_name = fname.replace(f"_-", f"{suffix}-")
        shutil.copy2(src_dir / fname, dst_dir / new_name)
    log(f"dsprites -> dsprites{suffix} (seed {seed43}, {len(os.listdir(dst_dir))} files)")

# ========== 3. Concept2Vec CheXpert variants ==========
print("\n"+"="*60)
print("3. Concept2Vec vectors -> CheXpert variants")
print("="*60)

for seed in ["43","44","45"]:
    for suffix in ["_image_robustness", "_image_responsiveness"]:
        src_file = RESULTS / "bases" / "concept2vec" / "chexpert" / seed / "vectors.npy"
        dst_dir = RESULTS / "bases" / "concept2vec" / (f"chexpert{suffix}") / seed
        if dst_dir.exists():
            log(f"chexpert{suffix}/{seed} exists, skip")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_file, dst_dir / "vectors.npy")
        log(f"chexpert -> chexpert{suffix} (seed {seed})")

# ========== 4. VGG16 weight files ==========
print("\n"+"="*60)
print("4. Creating VGG16 weight files for Truthfulness metric")
print("="*60)

from src.dataset import MNIST_Dataset, DSprites_Dataset, Chexpert_Dataset, CUB_Dataset
from src.models import get_large_image_model
import tensorflow as tf

ds_map = {"mnist": MNIST_Dataset, "dsprites": DSprites_Dataset,
          "chexpert": Chexpert_Dataset, "cub": CUB_Dataset}

vgg_dir = RESULTS / "models" / "vgg16_models"
os.makedirs(vgg_dir, exist_ok=True)

for ds_name, ds_class in ds_map.items():
    weight_path = vgg_dir / f"{ds_name}_42.h5"
    if weight_path.exists():
        log(f"{ds_name}_42.h5 exists, skip")
        continue
    log(f"Creating model for {ds_name}...")
    dataset = ds_class()
    with tf.compat.v1.Session() as sess:
        model = get_large_image_model(dataset, "VGG16")
        model.save_weights(str(weight_path))
    log(f"Saved {weight_path}")

print("\nDone! All preparation complete.")
