# A Universal Class of Sharpness-Aware Minimization Algorithms

This codebase accompanies the ICML 2024 paper [Universal Class of Sharpness-Aware Minimization Algorithms](https://arxiv.org/html/2406.03682v1).
It was created by cloning the [codebase](https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization) for [Make Sharpness-Aware Minimization Stronger: A Sparsified Perturbation Approach](https://arxiv.org/abs/2210.05177#) and adapting it as needed.

## Installation
- Clone this repo
```bash
git clone https://github.com/dbahri/universal_sam.git
cd universal_sam
```

- Create a virtual environment and install dependencies
```bash
python3 -m venv universal_sam_venv
source bin/activate/universal_sam_venv
python3 -m pip install -r requirements.txt
# possibly nuke out LD_LIBRARY_PATH
# LD_LIBRARY_PATH=
```

## Train on CIFAR10
Commands write a pickle file 'all.pickle' to the 'save_dir'. 'load_results.ipynb' shows how to read this pickle.

For other datasets, replace 'CIFAR10_base' with 'CIFAR100_base' or 'SVHN_base'. For noisy label experiments, change 'frac_corrupt' from 0 to 0.2. For limited data experiments, change 'frac_samples' from 1. to 0.1. Update other hyper-parameters according to paper.


Frob-SAM
```bash
python train.py --milestone '50|100|150' --model resnet18 --dataset 'CIFAR10_base' --opt 'frosam-sgd' --lr 0.1 --lr_scheduler 'MultiStepLRscheduler' --epochs 200 --seed 101 --weight_decay 5e-4 --isam_rho 0.005 --isam_lam 0.005 --isam_n_samples 2 --frac_samples 1. --frac_corrupt 0. --gamma 0.1 --datadir /tmp/cifar10_data --device 'cuda:0' --save_dir /tmp/cifar10_frobsam
```
Det-SAM
```bash
python train.py --milestone '50|100|150' --model resnet18 --dataset 'CIFAR10_base' --opt 'isam-sgd' --isam_n_samples 1 --isam_setting 'det' --isam_lam 1. --isam_rho 1.0 --isam_half_cube_len 0.01 --lr 0.1 --lr_scheduler 'MultiStepLRscheduler' --epochs 200 --seed 101 --weight_decay 5e-4 --frac_samples 1. --frac_corrupt 0. --gamma 0.1 --datadir /tmp/cifar10_data --device 'cuda:0' --save_dir /tmp/cifar10_detsam
```

Trace-SAM
```bash
python train.py --milestone '50|100|150' --model resnet18 --dataset 'CIFAR10_base' --opt 'isam-sgd' --isam_n_samples 1 --isam_setting 'trace' --isam_lam 1. --isam_rho 0.01 --lr 0.1 --lr_scheduler 'MultiStepLRscheduler' --epochs 200 --seed 101 --weight_decay 5e-4 --frac_samples 1. --frac_corrupt 0. --gamma 0.1 --datadir /tmp/cifar10_data --device 'cuda:0' --save_dir /tmp/cifar10_tracesam
```

ASAM
```bash
python train.py --milestone '50|100|150' --model resnet18 --dataset 'CIFAR10_base' --opt 'sam-sgd' --rho 0.5  --sam_variant 'asam' --asam_eta 0.01 --lr 0.1 --lr_scheduler 'MultiStepLRscheduler' --epochs 200 --seed 101 --weight_decay 5e-4 --frac_samples 1. --frac_corrupt 0. --gamma 0.1 --datadir /tmp/cifar10_data --device 'cuda:0' --save_dir /tmp/cifar10_asam
```

SAM
```bash
python train.py --milestone '50|100|150' --model resnet18 --dataset 'CIFAR10_base' --opt 'sam-sgd'  --rho 0.05 --lr 0.1 --lr_scheduler 'MultiStepLRscheduler' --epochs 200 --seed 101 --weight_decay 5e-4 --frac_samples 1. --frac_corrupt 0. --gamma 0.1 --datadir /tmp/cifar10_data --device 'cuda:0' --save_dir /tmp/cifar10_sam
```

SGD
```bash
python train.py --milestone '50|100|150' --model resnet18 --dataset 'CIFAR10_base' --opt 'sgd' --lr 0.1 --lr_scheduler 'MultiStepLRscheduler' --epochs 200 --seed 101 --weight_decay 5e-4 --frac_samples 1. --frac_corrupt 0. --gamma 0.1 --datadir /tmp/cifar10_data --device 'cuda:0' --save_dir /tmp/cifar10_sgd
```

SSAM
```bash
python train.py --milestone '50|100|150' --model resnet18 --dataset 'CIFAR10_base' --opt 'ssamf-sgd' --rho 0.1 --sparsity 0.5 --num_samples 16 --update_freq 1 --lr 0.1 --lr_scheduler 'MultiStepLRscheduler' --epochs 200 --seed 101 --weight_decay 5e-4 --frac_samples 1. --frac_corrupt 0. --gamma 0.1 --datadir /tmp/cifar10_data --device 'cuda:0' --save_dir /tmp/cifar10_ssam
```

## Train on MNIST

```bash
python train.py --model simple_mnist --dataset 'MNIST_base' --opt 'frosam-sgd' --lr 0.001 --lr_scheduler 'Constant' --epochs 20 --seed 101 --weight_decay 0 --isam_rho 0.01 --isam_lam 0.005 --isam_n_samples 2 --hessian_n_samples 1280 --hessian_every_n_epochs 1 --hessian_calc_frobenius --datadir /tmp/mnist_data --device 'cuda:0' --save_dir /tmp/mnist_frosam
```
