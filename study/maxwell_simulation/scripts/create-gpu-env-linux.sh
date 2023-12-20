#!/bin/bash

# Prerequisites
# - Download CUDA-Toolkit v12.2 .run file at: https://developer.nvidia.com/cuda-12-2-0-download-archive
# - Download cuDNN v8.9 .tar file at: https://developer.nvidia.com/cudnn | https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/


# Parameters (modify for custom envs)
CUDA_FILES_FOLDER="$HOME/cuda-install"
CUDA_RUN_FILEPATH="$CUDA_FILES_FOLDER/cuda_12.2.0_535.54.03_linux.run"
CUDNN_TAR_FILEPATH="$CUDA_FILES_FOLDER/cudnn-linux-x86_64-8.9.4.25_cuda12-archive.tar.xz"
CACHE_DIR="$HOME/cuda-cache"
LIBS_DIR="$HOME/libs"
CUDA_INSTALL_DIR="$LIBS_DIR/cuda-12-2"
GPRMAX_INSTALL_DIR="$LIBS_DIR/gprMax"

CONDA_ENV_FILEPATH="./environment-gpu.yaml"
CONDA_ENV_NAME="gprMax"
# -------


# Install NVIDIA toolkits and drivers
if [ -d "$CUDA_INSTALL_DIR" ];
then
    echo "CUDA-toolkit install already found in $CUDA_INSTALL_DIR, skipping install of CUDA and cuDNN..."
else
    # Install CUDA
    echo "Installing CUDA-toolkit in $CUDA_INSTALL_DIR" &&
    sh $CUDA_RUN_FILEPATH --silent --toolkit --installpath=$CUDA_INSTALL_DIR --tmpdir=$CACHE_DIR &&
    echo "CUDA-toolkit installed !"
    # Extract and install cuDNN
    CUDNN_EXTRACT_DIR="$CACHE_DIR/cuDNN_extract"
    if [ -d "$CUDNN_EXTRACT_DIR" ];
    then
        echo "Please remove pre-existing cuDNN extract folder in $CUDNN_EXTRACT_DIR"
        exit 1
    else
      echo "Extracting cuDNN in $CUDNN_EXTRACT_DIR" &&
      mkdir $CUDNN_EXTRACT_DIR &&
      tar -xvf $CUDNN_TAR_FILEPATH --directory $CUDNN_EXTRACT_DIR &&
      cp "$CUDNN_EXTRACT_DIR/cudnn-"*"-archive/include/cudnn"*".h" "$CUDA_INSTALL_DIR/include" &&
      cp -P "$CUDNN_EXTRACT_DIR/cudnn-"*"-archive/lib/libcudnn"* "$CUDA_INSTALL_DIR/lib64" &&
      chmod a+r "$CUDA_INSTALL_DIR/include/cudnn"*".h" "$CUDA_INSTALL_DIR/lib64/libcudnn"* &&
      rm -rf $CUDNN_EXTRACT_DIR &&
      echo "cuDNN installed !"
    fi
fi


# Install python env
conda env create -f $CONDA_ENV_FILEPATH &&
conda activate $CONDA_ENV_NAME &&

# Build gprMax
if [ -d "$GPRMAX_INSTALL_DIR" ];
then
    echo "gprMax install already found, skipping git clone"
else
  cd $LIBS_DIR &&
  git clone https://github.com/gprMax/gprMax.git
fi
cd $GPRMAX_INSTALL_DIR &&
python setup.py build &&
python setup.py install &&

# Configure the python env to use PATH and LD_LIBRARY_PATH variable pointed to the installed versions of CUDA and cuDNN (and their python libraries)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d &&
echo "export PATH=$CUDA_INSTALL_DIR/bin:\$PATH" >| $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh &&
echo "export LD_LIBRARY_PATH=$CUDA_INSTALL_DIR/lib64:\$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh &&
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh &&
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh &&
echo "export XLA_FLAGS=\"--xla_gpu_cuda_data_dir='$CUDA_INSTALL_DIR'\"" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh &&

# Source env (check)
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh