# 切换到 base 加速求解
conda update -n base -y conda
conda install -n base -y conda-libmamba-solver
conda config --set solver libmamba
conda config --set channel_priority strict

# 在项目环境安装 11.8 的 nvcc 和最小运行库头文件
conda install -n cdformer -y -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8

# 激活并设置环境
conda activate cdformer
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# 验证
nvcc --version
python - <<'PY'
import torch, os
print("torch.version.cuda =", torch.version.cuda)
print("torch.cuda.is_available =", torch.cuda.is_available())
from torch.utils.cpp_extension import CUDA_HOME
print("CUDA_HOME (detected) =", CUDA_HOME)
PY

# 编译
pip install -U ninja
python setup.py build_ext --inplace
