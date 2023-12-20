# phase-aware-rt-calibration
Code repository for paper "Calibrating Wireless Ray Tracing for Digital Twinning using Local Phase Error Estimates"


## Project structure

- `src/`: main codebase. Implements synthetic channel observations and material parameters calibration using any of the presented schemes, which comprise:
  - the proposed Phase Error-Aware Calibration (PEAC) scheme;
  - the Phase Error-Oblivious Calibration (PEOC) baseline;
  - and the Uniform Phase Error Calibration (UPEC) baseline.
- `study/`: codebase for launching the experiments presented in the paper. Note that the different experimental configurations presented in the paper can be selected by setting the variable `STUDY_EXPERIMENT_VERSION` in `.env`.
- `blender/`: source files of the 3D models used in the experiments.
- `assets/ `: Mitsuba exports of the Blender source files.
- `logs/`: experimental data is stored in this folder by default. Can be set to a custom folder by setting the environment variable `LOGS_FOLDER` in `.env`.
- `scripts/`: bash utils.

## Setup environment

### Pre-requisites

- Download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CPU-based: download and install [LLVM](https://llvm.org/)
- GPU-based: follow [TensorFlow GPU support tutorial](https://www.tensorflow.org/install/pip) for the required drivers.\
  Note that a script is provided in `./scripts/create-gpu-env-linux.sh` for Linux-based systems.

### Create conda environment
- Install conda environment by running the command:\
  ``conda env create -f ./environment.yaml``
- Source the environment by running:\
  ``conda activate phase-aware-rt-calibration``

### Update python environment

``conda env update -f ./environment.yaml``

### Install notebook kernel

Add the installed conda environment to Jupyter by setting a notebook kernel.\
Inside the `phase-aware-rt-calibration` environment, run:\
``python -m ipykernel install --user --name phase-aware-rt-calibration``


Run jupyter: ``jupyter notebook``


## Miscellaneous

### Extract Mitsuba scene

Prerequisites:
- Download and install Blender
- Install Mitsuba add-on for Blender

Generate Mitsuba export:
- Open .blend scene file in ``./blender/``
- In Blender, go to ``File > Export > Mitsuba (.xml)``
- Save file with ``Forward: Y Forward`` and ``Up: Z Up``

### Extract Bezier curves from Blender files

Drawing curves inside Blender can be useful to easily define a set of coordinates (e.g., Rx positions) to be used during simulation.
These can be extracted from the `.blend` source file as follows:
- Prerequisite: Add Blender installation folder to environment PATH
- Run:\
  ``blender --background <BLENDER_SCENE_PATH> --python ./blender/extract_curves.py --output-path=<JSON_OUTPUT_PATH>"``

Example:
``blender --background ./blender/the_strand/the_strand.blend --python ./blender/extract_curves.py --output-path="./assets/the_strand/curves.json"``