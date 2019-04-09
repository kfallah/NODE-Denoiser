# NODE-Denoiser
Experiments on Neural ODEs for Denoising and Other Problems

## Installation
The Anaconda environment used during development has been provided as a .yml file:
```
conda env create -f NODE_env.yml
source activate NODE
```

Afterwards, make sure you install torchdiffeq:
```
git clone https://github.com/kfallah/NODE-Denoiser.git
cd NODE-Denoiser/torchdiffeq
pip install -e .
```

### References
[1] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. "Neural Ordinary Differential Equations." *Advances in Neural Processing Information Systems.* 2018. [[arxiv]](https://arxiv.org/abs/1806.07366)

[2] Emilien Dupont, Arnaud Doucet, Yee Whye Teh. "Augmented Neural ODEs." 2019. [[arxiv]](https://arxiv.org/abs/1904.01681)

[3] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, Lei Zhang. "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising." 2016. [[arxiv]](https://arxiv.org/abs/1608.03981)
