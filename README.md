# IceCube

## Environment setting
1. Windows
```shell
conda create -n icecube python=3.7
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install transformers
pip install -r requirements.txt
```

2. Linux  
linux install miniconda：
```shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```
```shell
conda create -n icecube python=3.7
conda activate icecube
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install -r requirements.txt
```

3. MacOS
```shell
conda create --name icecube python=3.8
conda activate icecube
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0%2Bcpu.html
pip install transformers
pip install -r requirements.txt
```
