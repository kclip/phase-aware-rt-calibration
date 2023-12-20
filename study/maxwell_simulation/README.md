# FDTD Simulation of Maxwell Equations
Simulate exact solutions to Maxwell equations using [gprMax](https://www.gprmax.com/).


## Setup

### CPU (All platforms)

- Follow install and build instructions from the [gprMax documentation](https://docs.gprmax.com/en/latest/include_readme.html#installation) 

### GPU (Linux)

Setup the variables `CUDA_RUN_FILEPATH` and `CUDNN_TAR_FILEPATH` in the bash file [create-gpu-env-linux.sh](scripts%2Fcreate-gpu-env-linux.sh) and run it.

## Run Simulation

### Activate env

```bash
conda activate gprMax
```

### Run

- Run experiments from this folder:
```bash
cd study/maxwell_simulation
```

- CPU:
```bash
python -m gprMax scenarios/toy_example_maxwell.in
```

- GPU:
```bash
python -m gprMax scenarios/toy_example_maxwell.in -gpu
```

### Logs

The result of the simulation will be saved in the file `<PROJECT_FOLDER>/study/maxwell_simulation/logs/toy_example_maxwell.out`.
The data can be used [as-is](https://docs.gprmax.com/en/latest/output.html), or as part of the calibration pipeline by copy/pasting it in `<PROJECT_FOLDER>/assets/maxwell_simulation/toy_example_maxwell.out`.