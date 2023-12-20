# Calibrating Wireless Ray Tracing using Local Phase Error Estimates

Code repository for paper "[Calibrating Wireless Ray Tracing for Digital Twinning using Local Phase Error Estimates](https://arxiv.org/abs/2312.12625)"

If you use this software, please cite it as
```bibtex
@article{ruah2023calibrating,
  title={Calibrating Wireless Ray Tracing for Digital Twinning using Local Phase Error Estimates},
  author={Ruah, Clement and Simeone, Osvaldo and Hoydis, Jakob and Al-Hashimi, Bashir},
  journal={arXiv preprint arXiv:2312.12625},
  year={2023},
  online={https://arxiv.org/abs/2312.12625}
}
```

## Project structure

- `src/`: main codebase. Implements synthetic channel observations and material parameters calibration using any of the presented schemes, which comprise:
  - the proposed Phase Error-Aware Calibration (PEAC) scheme;
  - the Phase Error-Oblivious Calibration (PEOC) baseline;
  - and the Uniform Phase Error Calibration (UPEC) baseline.
- `study/`: codebase containing the experiments presented in the paper.
- `blender/`: source files of the 3D models used in the experiments.
- `assets/ `: Mitsuba exports of the Blender source files.
- `logs/`: experimental data is stored in this folder by default. Can be set to a custom folder by setting the environment variable `LOGS_FOLDER` in `.env`.
- `scripts/`: bash utils and scripts to launch the experiments.


## Setup environment

### Pre-requisites

- Download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CPU-based: download and install [LLVM](https://llvm.org/)
- GPU-based: follow [TensorFlow GPU support tutorial](https://www.tensorflow.org/install/pip) for the required drivers.\
  Note: for Linux-based systems, an installation script is provided in `./scripts/create-gpu-env-linux.sh`.

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


## Run Calibration and Coverage Map Experiments

By default, experimental data is stored in `./logs` and plots are stored in `./logs/saved_plots/`.

### Urban scenario "The Strand Campus"

#### Run calibration

```bash
bash scripts/the_strand/run_calibration_phase_noise_concentration.sh
bash scripts/the_strand/run_calibration_phase_noise_snr.sh
bash scripts/the_strand/run_calibration_pos_noise.sh
```

#### Calibration Plots

```bash
bash scripts/the_strand/run_plots.sh
```

#### Compute Coverage Maps

```bash
bash scripts/the_strand/run_coverage_maps.sh
```

#### Coverage Map Plots

```bash
bash scripts/the_strand/run_coverage_maps_plots.sh
```


### Toy Example

#### Run calibration

```bash
bash scripts/toy_example/run_calibration.sh
```

#### Calibration Plots

```bash
bash scripts/toy_example/run_plots.sh
```


### Experiments with FDTD-generated data

#### Run calibration

```bash
bash scripts/toy_example_maxwell/run_calibration.sh
```

#### Calibration Plots

```bash
bash scripts/toy_example_maxwell/run_plots.sh
```


## Run FDTD Experiments

Instructions to run the FDTD simulations of Maxwell's equations can be found in a separate [README.md](study%2Fmaxwell_simulation%2FREADME.md).


## Miscellaneous

### Extract Mitsuba scene

Prerequisites:
- Download and install [Blender](https://www.blender.org/)
- Install [Mitsuba add-on for Blender](https://github.com/mitsuba-renderer/mitsuba-blender)

Generate Mitsuba export:
- Open the .blend scene file in ``./blender/``
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