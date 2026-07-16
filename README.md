# WEARLYZE

WEARLYZE is a collaborative computer-vision prototype for detecting, segmenting,
classifying, and retrieving fashion items. The repository brings together YOLOv8,
CLIP, FAISS, DeepFashion/DeepFashion2 data tooling, and an interactive Gradio UI.

## What is in this repository

- Dataset loaders, transformations, sampling, and optional S3-backed data access
- YOLOv8 detection and segmentation components for fashion categories
- CLIP embeddings, FAISS similarity search, classifiers, and ensemble utilities
- DeepFashion2 conversion, training, evaluation, mask generation, and visualization
- Training configuration, checkpointing, scheduling, and experiment utilities
- A Gradio interface for inspecting segmentation results

This is a research and development codebase. Datasets, trained weights, and other
large artifacts are not included in the repository.

## Repository map

```text
config/        Core configuration
configs/       DeepFashion2 training configurations
data/          Dataset, loader, transform, and S3 utilities
models/        Detection, classification, embedding, ensemble, and retrieval code
training/      Training loop, scheduler, checkpoint, and experiment utilities
utils/         Evaluation metrics and visualizations
ui/            Gradio segmentation interface
train.py       Unified training entry point
```

## Setup

Python 3.8 or newer is required. A CUDA-capable environment is recommended for
training, but the exact PyTorch installation should match the target machine.

```bash
git clone https://github.com/NishantSaiChalla/WEARLYZE.git
cd WEARLYZE

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Entry points

Inspect the unified trainer options:

```bash
python train.py --help
```

Inspect the DeepFashion2-specific trainer options:

```bash
python train_deepfashion2.py --help
```

After installing the UI-specific dependencies and providing compatible local
model artifacts, launch the Gradio interface:

```bash
python -m pip install -r ui/requirements.txt
python ui/fashion_segmentation_ui.py
```

See [ui/README.md](ui/README.md), [data/README.md](data/README.md), and
[models/README.md](models/README.md) for subsystem notes. Some experiments need
local dataset paths, pretrained weights, or service credentials before they can
run.

## Contributions and ownership

WEARLYZE is a team project. [CONTRIBUTIONS.md](CONTRIBUTIONS.md) records a
verified contribution slice for Kunwarbir Singh Padda based on Git authorship;
it is not an exhaustive allocation of team ownership.

## Contributing

1. Create a focused branch.
2. Keep generated datasets, weights, credentials, and local outputs out of Git.
3. Describe the data and model assumptions needed to reproduce the change.
4. Run the smallest relevant test or smoke check and include the result in the PR.
