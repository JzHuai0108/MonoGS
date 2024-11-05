# Install monogs in a virtual env on whu hpc

The hpc os is centOS7.
···
conda create -n monogs python=3.12
conda activate monogs

conda install -c conda-forge open3d # This will install open3d 0.18.0
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge cxx-compiler=1.5.2

# to be completed
pip install evo==1.19.1
# and other pip packages from environment.yml

# python TORCH_CUDA_ARCH_LIST="YOUR_GPUs_CC+PTX" setup.py install
# for V100, CC=7.0 can be found at https://developer.nvidia.com/cuda-gpus
cd simple-knn
TORCH_CUDA_ARCH_LIST="7.0+PTX" python setup.py install
cd diff-gaussian-rasterization
TORCH_CUDA_ARCH_LIST="7.0+PTX" python setup.py install

# Then in MonoGS folder, run
sbatch -p gpu -A youraccount RGBDtest.sbatch

···