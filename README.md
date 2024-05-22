# Self-Supervised Unlearning

## Pre-trained Models: ```pretrain.py```

```
python pretrain.py \
        --project pretrain \
        --seed 0 \
        --gpu 1 \
        --cl-alg [simclr/moco/mocov1] \
        --cl-epochs 800 \
        --cl-lr 0.5 \
        --cl-momentum 0.9 \
        --cl-weight-decay 0.0001 \
        [--log-model]
```

## Unlearning Baselines: ```forget.py```
### Retraining
```
python forget.py \
    --project unlearn \
    --unlearn-mode rt \
    --num-unlearn-samples 4500 \
    --seed 0 \
    --gpu 1 \
    --cl-alg [simclr/moco/mocov1] \
    --enable-scheduler \
    [--enable-checkpointing] \
    --cl-epochs 800 \
    --cl-lr 0.5 \
    --cl-momentum 0.9 \
    --cl-weight-decay 0.0001 
```
### Fine-tuning/Gradient Ascent/Hiding
First add correct checkpoint paths in the file ```ckpt_paths.yml```. Then run the following command:
```
python forget.py \
    --project unlearn \
    --unlearn-mode [ft/ga/hiding] \
    --num-unlearn-samples 4500 \
    --seed 0 \
    --gpu 1 \
    --cl-alg [simclr/moco/mocov1] \
    [--enable-checkpointing] \
    --cl-epochs ${EPOCHS} \
    --cl-lr ${LR} \
```


<!-- ## EncoderMI attack: ```mia.py```
Apply white-box EncoderMI attack on a "unlearned" model.
Add correct checkpoint paths of "unlearned" models in the file ```ckpt_paths.yml```. Then run the following command:
```
python mia.py \
    --project mia \
    --unlearn-mode [pt/rt/ft/ga/hiding] \
    --num-unlearn-samples 4500 \
    --seed 0 \
    --gpu 1 \
    --cl-alg [simclr/moco] \
    [--enable-checkpointing] \
    --mia-epochs 300 \
    --mia-lr 0.0001 \
    --mia-weigth-decay 0.000001 \
``` -->
