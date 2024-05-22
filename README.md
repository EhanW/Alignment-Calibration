# Alignment Calibration
This is the PyTorch implementation of the paper "Alignment Calibration: Machine Unlearning for Contrastive Learning under Auditing" which is under review for NeurIPS 2024.
This code can reproduce the results of baseline unlearning methods and our proposed Alignment Calibration (AC) on CIFAR-10/100 in our paper.

## Prerequisites
1. Prepare the conda environment:
```
conda env create -f env.yml
conda activate alignment-calibration
```
2. We use WandB to log the results. Please create an account on [WandB](https://wandb.ai/) and login to your account:
```
pip install wandb
wandb login
```
3. Download the CIFAR-10/100 datasets in the ```./data``` directory:






## Pretrain Encoders
Before unlearning, we need to pretrain the encoder using contrastive learning algorithms (SimCLR, MoCo):
```
python pretrain.py \
        --dataset [cifar10/cifar100] \
        --seed 0 \
        --cl-alg [simclr/moco] \
        --cl-epochs 800 \
```
WandB logger will automatically save the final encoder as ```logs/alignment-calibration/xx/xx.ckpt ```.

## Alignment Calibration
```
python main.py \
    --dataset [cifar10/cifar100] \
    --num-unlearn-samples 4500 \
    --seed 0 \
    --cl-alg [simclr/moco] \
    [--enable-checkpointing] \
    --cl-epochs 10 \
    --cl-lr ${LR} \
    --ckpt-path load_encoder_path.ckpt \
    --alpha ${ALPHA} \
    --beta ${BETA} \
```





## Other Mehods
### Retrain
The gold standard method for white-box evluation is Retrain:
```
python forget.py \
    --dataset [cifar10/cifar100] \
    --unlearn-mode rt \
    --num-unlearn-samples 4500 \
    --seed 0 \
    --cl-alg [simclr/moco] \
    --enable-scheduler \
    [--enable-checkpointing] \
    --cl-epochs 800 
```
### Fine-tuning/Gradient Ascent/NegGrad/l1-Sparsity
Starting from a pretrained encoder (```load_encoder_path.ckpt```), we can apply fine-tuning, gradient ascent, NegGrad, and l1-Sparsity:
```
python forget.py \
    --dataset [cifar10/cifar100] \
    --unlearn-mode [ft/ga/l1/ng] \
    --num-unlearn-samples 4500 \
    --seed 0 \
    --cl-alg [simclr/moco] \
    [--enable-checkpointing] \
    --cl-epochs ${EPOCHS} \
    --cl-lr ${LR} \
    [--l1-reg ${L1_REG}] \
    --ckpt-path load_encoder_path.ckpt
```

