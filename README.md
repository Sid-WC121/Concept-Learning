# Understanding Inter-Concept Relationships in Concept-Based Models

This repository contains the reproducibility challenge implementation for the paper "Understanding Inter-Concept Relationships in Concept-Based Models", published at ICML 2024.

This paper was done by [Naveen Raman](https://naveenraman.com/), [Mateo Espinosa](https://hairyballtheorem.com/), and [Mateja Jamnik](https://www.cl.cam.ac.uk/~mj201/). 

Paper link: [https://arxiv.org/abs/2405.18217](https://arxiv.org/abs/2405.18217)

Reproducibility challenge implementation done by [Sidharth Padmanabhan](https://sid-wc121.github.io/), [Laura Jimena Tagle Muñoz](https://github.com/jimenatagle), Qinyou Wang, [Fabian Bosshard](https://fabianbosshard.github.io/)

Our Report: [BAIC Report](https://github.com/Sid-WC121/Concept-Learning/blob/main/BAIC%20Report.pdf)

#### TL;DR
We construct concept bases, a way to study inter relationships in concept models. This allows us to understand the types of inter-concept relationships captured by existing concept-based models. Additionally, well-constructed concept bases can assist with downstream applications such as concept intervention. 

We provide code here to perform the following operations: 
1. Extract concept bases and concept vectors
2. Evaluate concept bases
3. Employ concept bases for concept intervention

## Reproduction Runbook

Run all commands from the repository root.

### 1. Create Environment

Windows:
```powershell
conda env create -f environment.yaml
conda activate concepts
```

Linux/Modal:
```bash
conda env create -f environment-linux.yaml
conda activate concepts

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH. # for gpu connection issue
```

Check GPU visibility:
```powershell
python -c "import torch, tensorflow as tf; print('torch', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('tf', tf.config.list_physical_devices('GPU'))"
```

If TensorFlow prints missing `cudart64_110.dll`, `cublas64_11.dll`, or `cudnn64_8.dll`, TCAV will run on CPU. PyTorch/CEM can still use GPU. For exact full runs, Linux/Modal GPU is preferred.

Optional path overrides:
```powershell
$env:CONCEPT_DATASET_DIR="$PWD\dataset"
$env:CONCEPT_RESULTS_DIR="$PWD\results"
```

### 2. Download Frozen Models

```powershell
python scripts/dataset_scripts/download_models.py
```

This creates:
```text
dataset/models/inception5h/tensorflow_inception_graph.pb
dataset/models/inception5h/imagenet_comp_graph_label_strings.txt
dataset/models/mobilenet_v2_1.0_224/*
```

`dataset/models/` contains neural network weights. `dataset/imagenet/` contains TCAV image folders such as `random500_0` and `zebra`.

### 3. Download And Prepare Datasets

```powershell
python scripts/dataset_scripts/download_mnist.py
python scripts/dataset_scripts/download_dsprites.py
python scripts/dataset_scripts/download_cub.py
```

Expected prepared files:
```text
dataset/colored_mnist/preprocessed/train.pkl
dataset/colored_mnist/preprocessed/val.pkl
dataset/dsprites/preprocessed/train.pkl
dataset/dsprites/preprocessed/val.pkl
dataset/dsprites/preprocessed/test.pkl
dataset/CUB/preprocessed/train.pkl
dataset/CUB/preprocessed/val.pkl
dataset/CUB/preprocessed/test.pkl
dataset/CUB/metadata/attributes.txt
dataset/CUB/metadata/classes.txt
```

Verify:
```powershell
python -c "from src.dataset import MNIST_Dataset, DSprites_Dataset, CUB_Dataset; m=MNIST_Dataset(); d=DSprites_Dataset(); c=CUB_Dataset(); print('MNIST', len(m.get_data()), len(m.get_attributes())); print('dSprites', len(d.get_data()), len(d.get_attributes())); print('CUB', len(c.get_data()), len(c.get_data(train=False)), len(c.get_attributes()), len(c.class_names))"
```

Current expected counts:
```text
MNIST train 60000, attributes 20
dSprites train 2500, attributes 18
CUB train 4796, val 1198, attributes 312, classes 200
```

CheXpert requires accepting Stanford's terms via Kaggle (free account). The script auto-downloads once credentials are set.

**Option A — environment variable (easiest for any collaborator):**
```powershell
# 1. Go to https://www.kaggle.com -> Settings -> API -> Create New Token
#    Kaggle shows a token like: KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 2. Set it and run:
pip install kaggle
$env:KAGGLE_API_TOKEN = "KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python scripts/dataset_scripts/download_chexpert.py
```

Linux/macOS equivalent:
```bash
export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
python scripts/dataset_scripts/download_chexpert.py
```

**Option B — kaggle.json file (classic):**
Download `kaggle.json` from kaggle.com → Settings → API and place it at `C:\Users\<you>\.kaggle\kaggle.json`, then run the script with no flags.

**Option C — manual download (no API):**
```powershell
# Download and unzip CheXpert-v1.0-small from https://www.kaggle.com/datasets/ashery/chexpert, then:
python scripts/dataset_scripts/download_chexpert.py --source-dir C:\data\CheXpert-v1.0-small
```

All options produce:
```text
dataset/chexpert/images/<patient>/study1/<view>.jpg
dataset/chexpert/preprocessed/train.pkl
dataset/chexpert/preprocessed/val.pkl
dataset/chexpert/preprocessed/test.pkl   (same as val; no held-out test split)
```

Verify:
```powershell
python -c "from src.dataset import Chexpert_Dataset; d=Chexpert_Dataset(); print('chexpert', len(d.get_data()), len(d.get_attributes()))"
```

### 4. Running Each Method By Dataset

Below are the estimated wall-clock times **per seed** on a system with a single GPU and SSD storage. Multiply by 3 (seeds 43, 44, 45) for full paper runs. Times are rough — actual duration depends on GPU model, CPU, disk speed, and system load.

The **Label** baseline is instant (no training — one-hot concept vectors).

#### MNIST (20 attributes, 10 classes, 60K train / 10K val)

| Method | Time (per seed) | Notes |
|--------|-----------------|-------|
| Label | <1s | Hardcoded one-hot vectors |
| Concept2Vec | ~1 min | 1000 samples, 5 epochs, skipgram |
| TCAV | ~20-30 min | 20 attributes × 3 random experiments; uses GoogleNet |
| CEM | ~10-15 min | ResNet34 backbone on 28×28 images |
| VAE | ~5-10 min | 5% training data (3K), 30 epochs, small conv net |

Commands:
```powershell
# All methods (Label, Concept2Vec, TCAV, CEM) — paper default:
python scripts/run_mnist_pipeline.py --dataset mnist

# VAE:
python src/models.py --algorithm vae --seed 43 --dataset mnist

# VAE with concept alignment:
python src/models.py --algorithm vae_concept --seed 43 --dataset mnist

# Robustness/responsiveness variants for all 5 datasets:
bash scripts/bash_scripts/create_vae_mnist.sh
bash scripts/bash_scripts/create_model_mnist.sh

# Evaluate only (after vectors exist):
python scripts/run_mnist_pipeline.py --dataset mnist --eval-only
```

Robustness/responsiveness dataset variants must be created first:
```powershell
python -c "from src.dataset import MNIST_Dataset; d=MNIST_Dataset(); d.create_robustness(); d.create_responsiveness()"
```

#### dSprites (18 attributes, 100 classes, 2500 train / 750 val)

| Method | Time (per seed) | Notes |
|--------|-----------------|-------|
| Label | <1s | Hardcoded one-hot |
| Concept2Vec | ~1 min | Same 1000-sample skipgram |
| TCAV | ~10-15 min | 18 attributes × 3 random experiments |
| CEM | ~10-15 min | ResNet34 backbone; SLURM wall-time 15 min |
| VAE | ~3-5 min | Small dataset, 30 epochs |

Commands:
```powershell
# All four base methods:
python scripts/run_mnist_pipeline.py --dataset dsprites

# TCAV individually (with robustness/responsiveness variants):
bash scripts/bash_scripts/create_tcav_dsprites.sh

# CEM individually:
python -m scripts.cem_scripts.extract_cem_concepts --experiment_name dsprites --num_gpus 1 --num_epochs 50 --validation_epochs 25 --seed 43

# Evaluate only:
python scripts/run_mnist_pipeline.py --dataset dsprites --eval-only
```

#### CUB (312 attributes, 200 classes, 4796 train / 1198 val)

| Method | Time (per seed) | Notes |
|--------|-----------------|-------|
| Label | <1s | Hardcoded one-hot |
| Concept2Vec | ~1-2 min | Still 1000 samples; 312 concepts make skipgram slightly larger |
| TCAV | ~5-10 hrs | **312 attributes × 3 random experiments; runs VGG16**. Run overnight |
| CEM | ~3-4 hrs | 312 concepts, ResNet34 backbone, 50 epochs; SLURM wall-time 4 hrs |
| VAE | ~15-30 min | Full CUB dataset (64×64 images), 30 epochs |

Commands:
```powershell
# Concept2Vec:
python -c "from src.dataset import CUB_Dataset; from src.create_vectors import create_concept2vec; d=CUB_Dataset(); [create_concept2vec(d, '', seed=s, embedding_size=32, num_epochs=5, dataset_size=1000, initial_embedding=None) for s in [43,44,45]]"

# TCAV (slow — 5-10 hrs per seed):
bash scripts/bash_scripts/create_tcav_cub.sh

# CEM:
python -m scripts.cem_scripts.extract_cem_concepts --experiment_name cub --num_gpus 1 --num_epochs 50 --validation_epochs 25 --seed 43 --concept_pair_loss_weight 0
python -m scripts.cem_scripts.extract_cem_concepts --experiment_name cub --num_gpus 1 --num_epochs 50 --validation_epochs 25 --seed 44 --concept_pair_loss_weight 0
python -m scripts.cem_scripts.extract_cem_concepts --experiment_name cub --num_gpus 1 --num_epochs 50 --validation_epochs 25 --seed 45 --concept_pair_loss_weight 0

# VAE:
python src/models.py --algorithm vae --seed 43 --dataset cub

# VAE with concept alignment (latent_dim = 312):
python src/models.py --algorithm vae_concept --seed 43 --dataset cub
```

CUB full metrics and paper tables are in `scripts/Evaluate Hierarchies.ipynb`.
That notebook also uses robustness/responsiveness variants and Shapley/model vectors; see the notebook notes below before running every cell.

Robustness/responsiveness dataset variants:
```powershell
python -c "from src.dataset import CUB_Dataset; d=CUB_Dataset(); d.create_robustness(); d.create_responsiveness()"
```

#### CheXpert (13 attributes, 2 classes)

| Method | Time (per seed) | Notes |
|--------|-----------------|-------|
| Label | <1s | Hardcoded one-hot |
| Concept2Vec | ~1 min | Same skipgram procedure |
| TCAV | ~10-15 min | 13 attributes |
| CEM | ~20-30 min | SLURM wall-time 30 min |

Commands:
```powershell
# Concept2Vec:
python -c "from src.dataset import Chexpert_Dataset; from src.create_vectors import create_concept2vec; d=Chexpert_Dataset(); [create_concept2vec(d, '', seed=s) for s in [43,44,45]]"

# TCAV:
bash scripts/bash_scripts/create_tcav_chexpert.sh

# CEM:
python -m scripts.cem_scripts.extract_cem_concepts --experiment_name chexpert --num_gpus 1 --num_epochs 50 --validation_epochs 25 --seed 43
```

### 5. Full Pipeline (All Methods At Once)

The pipeline script handles Label, Concept2Vec, TCAV, and CEM for MNIST and dSprites with one command:

```powershell
python scripts/run_mnist_pipeline.py --dataset mnist
python scripts/run_mnist_pipeline.py --dataset dsprites
```

CEM defaults (overridable via flags):
```text
50 epochs
validation every 25 epochs
ResNet34 extractor
concept_pair_loss_weight=0
1 GPU
```

Partial runs (skip methods):
```powershell
python scripts/run_mnist_pipeline.py --dataset mnist --skip-cem --skip-tcav
python scripts/run_mnist_pipeline.py --dataset mnist --skip-concept2vec --eval-only
```

Seeds used:
```text
43, 44, 45
```

### 6. Total Estimated Run Times (Full Paper — All 3 Seeds)

| Step | MNIST | dSprites | CUB | CheXpert |
|------|-------|----------|-----|----------|
| Dataset creation | 5-10 min | 5 min | 10 min | 5-10 min |
| Label | <1s | <1s | <1s | <1s |
| Concept2Vec | 3 min | 3 min | 5 min | 3 min |
| TCAV | 1-1.5 hr | 30-45 min | 15-30 hrs | 30-45 min |
| CEM | 30-45 min | 30-45 min | 9-12 hrs | 1-1.5 hr |
| VAE | 15-30 min | 10-15 min | 45-90 min | — |
| Pipeline + eval | 2-3 hrs | 1-2 hrs | — | — |
| **Total** | **~3-5 hrs** | **~2-3 hrs** | **~25-45 hrs** | **~2-3 hrs** |

TCAV on CUB dominates the total (312 attributes × 3 random experiments × 3 seeds). Run it overnight or on a GPU cluster. The `--skip-tcav` flag lets you evaluate the other methods while TCAV runs separately.

### 7. Outputs

Basis vectors:
```text
results/bases/concept2vec/<dataset>/<seed>/vectors.npy
results/bases/tcav/<dataset>/<seed>/*.pkl
results/bases/cem/<dataset>/<seed>/*_active.npy
results/bases/cem/<dataset>/<seed>/manifest.json
```

Evaluation and plots:
```text
results/evaluation/ablation/distance_<dataset>.json
results/evaluation/ablation/agreement_<dataset>.json
results/evaluation/<dataset>_hierarchies/*.txt
results/figures/<dataset>/*_hierarchy.png
```

What the commands above cover:
```text
MNIST: Label, Concept2Vec, TCAV, CEM bases plus distance/agreement/hierarchy outputs.
dSprites: Label, Concept2Vec, TCAV, CEM bases plus distance/agreement/hierarchy outputs.
CUB: Concept2Vec, TCAV, CEM bases. Full CUB tables are computed in the notebook.
```

What is not produced by the default runbook:
```text
CheXpert data/results.
Shapley, VAE, and model-vector bases.
Intervention JSON/PKL files.
Extra ablation files under results/extra_evaluation and results/evaluation/ois.
```

### 8. Notebooks And Plotting

Use these after vectors exist:
```text
scripts/Evaluate Hierarchies.ipynb
scripts/Plotting.ipynb
scripts/cem_scripts/CEM Intervention Experiments.ipynb
```

Important: these are paper notebooks, not clean push-button scripts. Run them from a Jupyter server started in the repository root or `scripts/`, and keep the first `os.chdir('../')` cell consistent with that working directory. If imports fail after that cell, restart the kernel from the repository root and skip the `os.chdir('../')` cell.

What each does:
```text
scripts/run_mnist_pipeline.py
  Reproducible script for MNIST/dSprites basis generation, KNN/top-k agreement, hierarchy text, dendrogram PNGs.

scripts/Evaluate Hierarchies.ipynb
  Main paper evaluation notebook. Computes robustness, responsiveness, stability, truthfulness, ablation JSONs, and cross-method hierarchy distances.

scripts/Plotting.ipynb
  Builds paper plots from files under results/evaluation.

src/metrics.py
  KNN/top-k logic: get_top_k_pairs(), embedding_distance(), compute_all_metrics().

src/hierarchy.py
  Concept distance and hierarchy construction: get_concept_distances(), create_ward_hierarchy(), create_hierarchy().

src/plots.py
  Helper plotting functions for dendrograms, PCA, t-SNE, images.
```

Notebook coverage after the default runbook:
```text
Evaluate Hierarchies.ipynb
  MNIST/dSprites/CUB cells for Label, Concept2Vec, TCAV, and CEM can run only after the matching base vectors exist for seeds 43, 44, 45.
  Cells that call compute_all_metrics need _image_robustness and _image_responsiveness vectors too.
  Cells using Chexpert_Dataset need CheXpert prepared manually.
  Cells using load_shapley_vectors_simple need Shapley vectors and TensorFlow model weights under results/models/.

Plotting.ipynb
  Reads text/JSON files produced by Evaluate Hierarchies.ipynb and intervention experiments.
  Later cells need results/evaluation/ois, results/extra_evaluation, results/intervention, and figures/.

CEM Intervention Experiments.ipynb
  Needs the pretrained CUB CEM model/config in models/.
  Needs exported intervention hierarchy arrays named concept_vectors/<method>_<seed>.npy.
  The default basis run does not create those files; export them with src.util.save_concept_vectors first.
```

To create robustness/responsiveness datasets used by the metric notebook:
```powershell
python -c "from src.dataset import MNIST_Dataset, DSprites_Dataset, CUB_Dataset; [getattr(d, m)() for d in [MNIST_Dataset(), DSprites_Dataset(), CUB_Dataset()] for m in ['create_robustness', 'create_responsiveness']]"
```

### 9. Replacing KNN Or Averaging

Do not change vector training first. Reuse saved bases and replace evaluation logic.

Current distance path:
```text
loader in src/concept_vectors.py
  -> src/hierarchy.py:get_concept_distances()
  -> pairwise distances averaged across all vectors for two concepts
  -> src/metrics.py:get_top_k_pairs()
  -> src/metrics.py:embedding_distance()
```

To replace KNN/top-k:
1. Add a new metric function in `src/metrics.py`.
2. Call it from `scripts/run_mnist_pipeline.py` or `scripts/Evaluate Hierarchies.ipynb`.
3. Save results to a new JSON name under `results/evaluation/ablation/`, so paper metrics stay comparable.

To replace pairwise averaging:
1. Edit or wrap `src/hierarchy.py:get_concept_distances()`.
2. Keep the loader API unchanged: `method(attribute, dataset, suffix, seed)` returns a matrix of vectors.
3. Save new outputs under a new method/metric name.

To add a new basis method:
1. Add `load_<method>_vectors_simple(attribute, dataset, suffix, seed)` in `src/concept_vectors.py`.
2. Add it to `METHODS` in `scripts/run_mnist_pipeline.py` or the notebook method list.
3. Keep outputs under `results/bases/<method>/<dataset>/<seed>/`.

