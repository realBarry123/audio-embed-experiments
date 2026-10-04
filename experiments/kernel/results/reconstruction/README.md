# Reconstruction Test Results

**Some clips may be loud at certain parts, listen on very low volume first!**

Each audio clip is the same clip played three times: 
1. The first time is the original clip. 
2. The second time is the clip reconstructed by the VAE. 
3. The third time is the clip reconstructed by the modified VAE. 

`./ablation` contains reconstructions from removing the first snake layer of the VAE (output of `experiments/kernel/ablation.py`). 

`./linearization` contains reconstructions from replacing the first snake layer with an element-wise affine transform (output of `experiments/kernel/snakelin_valid.py`). 