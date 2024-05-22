import os
import yaml
import torch
import argparse
import numpy as np
from time import time
from collections import OrderedDict
from torch.utils.data import DataLoader, ConcatDataset
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from algs import Classifier, SimCLRModel, MocoModel
from dataset import CIFAR10Triad, CIFAR100Triad
from mia import SVC_MIA, get_membership_features, encodermi_threshold, encodermi_threshold_test
from utils import get_indices, get_cl_augs, get_sl_augs, get_test_augs, cls_test, args_check, setup_seed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unlearn-mode', '-m', default='ac')
    parser.add_argument('--num-unlearn-samples', '-n', type=int, default=4500)
    parser.add_argument('--project', '-p', type=str, default='alignment-calibration')
    parser.add_argument('--log-model', '-l', action='store_true')    
    parser.add_argument('--backbone', '-b', default='resnet18', choices=['resnet18', 'resnet50']) 

    parser.add_argument('--cl-alg', '-a', choices=['moco', 'simclr'], default='moco')
    parser.add_argument('--cl-epochs', type=int, default=10)
    parser.add_argument('--cl-lr', type=float, default=0.06)
    parser.add_argument('--cl-momentum', type=float, default=0.9)
    parser.add_argument('--cl-weight-decay', type=float, default=5e-4)

    parser.add_argument('--fc-epochs', type=int, default=100)
    parser.add_argument('--fc-lr', type=float, default=1.0)    
    parser.add_argument('--fc-momentum', type=float, default=0.9)
    parser.add_argument('--feature-dim', type=int, default=512)

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes in the dataset')
    parser.add_argument('--size', type=int, default=32, help='size of the image')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--num-total-samples', type=int, default=50000, help='number of total samples in the dataset')
    parser.add_argument('--num-test-samples', type=int, default=10000, help='number of test samples in the dataset')

    parser.add_argument('--mi-augs', type=int, default=10)

    parser.add_argument('--gpu-id', '-g', type=int, default=0, help='the gpu id')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--enable-checkpointing', action='store_true', default=False)
    parser.add_argument('--enable-scheduler', action='store_true', default=False)

    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--index-shift', type=int, default=0, help='apply negative alignment calibration, 0 means randomly construct negative pairs, otherwise shift the index')
    parser.add_argument('--alpha', type=float, default=1, help='the hyper parameter alpha and gamma (alpha=gamma) in paper that controls negative calibration; default is alpha=gamma=1')
    parser.add_argument('--beta', type=float, default=0, help='the hyper parameter beta in paper that controls positive calibration')
    
    parser.add_argument('--l1-reg', type=float, default=None)
    parser.add_argument('--precision', type=int, default=None)
    parser.add_argument('--no-eval', action='store_true', default=False)
    return parser.parse_args()

def make_dataloaders():
    num_workers = args.num_workers
    batch_size = args.batch_size

    ### Define the augmentations
    imagenet = False
    train_classifier_transforms = get_sl_augs(size=args.size, imagenet=imagenet)
    test_transforms = get_test_augs(imagenet=imagenet)

    ### Get the indices for the dataset
    validation_indices, retain_indices, unlearn_indices = get_indices(
            seed=args.seed, num_total_samples=args.num_total_samples, num_unlearn_samples=args.num_unlearn_samples)

    if args.dataset == 'cifar10':
        DatasetTriad = CIFAR10Triad
    elif args.dataset == 'cifar100':
        DatasetTriad = CIFAR100Triad
    else:
        raise ValueError("Invalid value for dataset")

    ### Dataloaders for training feature extractor and then linear probing classifier
    imagenet = False
    train_extractor_transforms = get_cl_augs(strength=args.strength, size=args.size, imagenet=imagenet)
    train_retain_dataset = DatasetTriad(root=args.data, pair=False, triad=True, train=True, transform=train_extractor_transforms, selected_indices=retain_indices)
    train_unlearn_dataset = DatasetTriad(root=args.data, pair=False, triad=True, train=True, transform=train_extractor_transforms, 
                                    selected_indices=unlearn_indices, index_shift=args.index_shift)
    
    train_extractor_dataset = ConcatDataset([train_retain_dataset, train_unlearn_dataset])
    train_extractor_dataloader = DataLoader(train_extractor_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    
    train_classifier_dataset = DatasetTriad(
        root=args.data, train=True, pair=False, triad=False, transform=train_classifier_transforms, selected_indices=retain_indices)
    train_classifier_dataloader = DataLoader(train_classifier_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

    train_cmia_dataset = DatasetTriad(
            root=args.data, train=True, pair=False, triad=False, transform=test_transforms, selected_indices=retain_indices[:args.num_test_samples])
    train_cmia_dataloader = DataLoader(train_cmia_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)


    ### Dataloaders for evaluation after linear probing
    evaluation_retain_dataset = DatasetTriad(
        root=args.data, train=True, pair=False, triad=False, transform=test_transforms, selected_indices=retain_indices)
    evaluation_validation_dataset = DatasetTriad(
        root=args.data, train=True, pair=False, triad=False, transform=test_transforms, selected_indices=validation_indices)
    evaluation_test_dataset = DatasetTriad(
        root=args.data, train=False, pair=False, triad=False, transform=test_transforms)
    evaluation_unlearn_dataset = DatasetTriad(
        root=args.data, train=True, pair=False, triad=False, transform=test_transforms, selected_indices=unlearn_indices)
    evaluation_retain_dataloader = DataLoader(evaluation_retain_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    evaluation_validation_dataloader = DataLoader(evaluation_validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    evaluation_test_dataloader = DataLoader(evaluation_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    evaluation_unlearn_dataloader = DataLoader(evaluation_unlearn_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    ### Return the dataloaders as OrderedDict
    train_dataloaders = OrderedDict(train_extractor=train_extractor_dataloader, train_classifier=train_classifier_dataloader)
    evaluation_dataloaders = OrderedDict(retain=evaluation_retain_dataloader, validation=evaluation_validation_dataloader, test=evaluation_test_dataloader, unlearn=evaluation_unlearn_dataloader)
    return train_dataloaders, evaluation_dataloaders, train_cmia_dataloader

def make_emia_features(model):
    num_workers = args.num_workers
    batch_size = args.batch_size

    ### Define the augmentations
    imagenet = False
    feature_transforms = get_cl_augs(strength=1.0, size=args.size, imagenet=imagenet)

    ### Get the indices for the dataset
    _, retain_indices, unlearn_indices = get_indices(
            seed=args.seed, num_total_samples=args.num_total_samples, num_unlearn_samples=args.num_unlearn_samples)

    ### Non-membership samples are args.num_test_samples test data
    ### To construct a balanced dataset, select args.num_test_samples samples for training the membership classifier
    membership_indices = retain_indices[:args.num_test_samples]
    evaluation_unlearn_indices = unlearn_indices ### non-membership samples
    
    if args.dataset == 'cifar10':
        DatasetTriad = CIFAR10Triad
    elif args.dataset == 'cifar100':
        DatasetTriad = CIFAR100Triad
    else:
        raise ValueError("Invalid value for dataset")
    
    ### Dataloaders of (non-)membership features for training a binary classifier
    train_membership_dataset = DatasetTriad(
        root=args.data, train=True, pair=False, triad=False, transform=feature_transforms, selected_indices=membership_indices)
    train_nonmembership_dataset = DatasetTriad(
        root=args.data, train=False, pair=False, triad=False, transform=feature_transforms)
    train_membership_dataloader = DataLoader(train_membership_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    train_nonmembership_dataloader = DataLoader(train_nonmembership_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    train_membership_features = get_membership_features(model=model, dataloader=train_membership_dataloader, args=args)
    train_nonmembership_features = get_membership_features(model=model, dataloader=train_nonmembership_dataloader, args=args)

    train_features = OrderedDict(membership=train_membership_features, nonmembership=train_nonmembership_features)

    ### Dataloaders for evaluation 
    evaluation_unlearn_dataset = DatasetTriad(
        root=args.data, train=True, pair=False, triad=False, transform=feature_transforms, selected_indices=evaluation_unlearn_indices)
    evaluation_unlearn_dataloader = DataLoader(evaluation_unlearn_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    evaluation_unlearn_features = get_membership_features(model=model, dataloader=evaluation_unlearn_dataloader, args=args)

    evaluation_features = OrderedDict(unlearn=evaluation_unlearn_features)
    return train_features, evaluation_features


def main():
    ## make dataloaders and models
    train_dataloaders, evaluation_dataloaders, train_cmia_dataloader = make_dataloaders()

    if args.cl_alg == 'simclr':
        model = SimCLRModel(max_epochs= args.cl_epochs, lr=args.cl_lr, momentum=args.cl_momentum, weight_decay=args.cl_weight_decay, args=args, enable_scheduler=args.enable_scheduler, negative_loss=False)
    elif args.cl_alg == 'moco':
        model = MocoModel(max_epochs= args.cl_epochs, lr=args.cl_lr, momentum=args.cl_momentum, weight_decay=args.cl_weight_decay, args=args, enable_scheduler=args.enable_scheduler, negative_loss=False)
    else:
        raise ValueError("Invalid value for cl-alg")
        
    ## load the pretrained checkpoint
    # with open('ckpt_paths.yml', 'r') as f:
    #     ckpt_path = yaml.load(f, Loader=yaml.FullLoader)['pt'][args.dataset][args.cl_alg][args.seed]
    state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)

    ## configure the logger
    save_dir = 'logs/'
    log_name = f'{args.dataset}-{args.unlearn_mode}-{args.cl_alg}-{args.num_unlearn_samples}-seed={args.seed}-lr={args.cl_lr}-alpha={args.alpha}' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    wandb_logger = WandbLogger(project=args.project, 
                               name= log_name, 
                               log_model=args.log_model, 
                               save_dir=save_dir,
                               config=vars(args)
                               )
    wandb_logger.watch(model, log="all")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    columns = ['EMIA', 'RA', 'TA', 'UA', "CMIA"]
    results = []
    time_columns = ['Unlearn_time', 'EMIA_time', 'Linear_probing_time', 'CMIA_time']
    time_results = []

    start_time = time()
    ## Train the feature extractor
    trainer = Trainer(max_epochs=args.cl_epochs, devices=[args.gpu_id], accelerator="gpu", 
                        logger=wandb_logger, callbacks=[lr_monitor], enable_checkpointing=args.enable_checkpointing, precision=args.precision)
    trainer.fit(model, train_dataloaders['train_extractor'])
    time_results.append(time() - start_time)
    start_time = time()
    if args.no_eval:
        time_columns = ['Unlearn_time']
        wandb_logger.log_text(key="time", columns=time_columns, data=[time_results])
        return

    ## EncoderMI-Threshold
    emia_train_features, emia_evluation_features = make_emia_features(model)
    threshold = encodermi_threshold(
        features=np.concatenate([emia_train_features['membership'], emia_train_features['nonmembership']], axis=0),
        labels=np.concatenate([np.ones(emia_train_features['membership'].shape[0]), np.zeros(emia_train_features['nonmembership'].shape[0])])
    )

    features = emia_evluation_features['unlearn']
    if features is None:
        results.append(None)
    else:
        EMIA_acc = encodermi_threshold_test(features=features, labels=np.zeros(features.shape[0]), threshold=threshold)
        results.append(EMIA_acc)
    time_results.append(time() - start_time)
    start_time = time()

    ## train the linear classifier
    classifier = Classifier(backbone=model.backbone, max_epochs=args.fc_epochs, 
                            lr=args.fc_lr, momentum=args.fc_momentum, num_classes=args.num_classes, val_names=list(evaluation_dataloaders.keys()),
                            feature_dim=args.feature_dim)
    trainer = Trainer(max_epochs=args.fc_epochs, devices=[args.gpu_id], accelerator="gpu", 
                         logger=wandb_logger, callbacks=[lr_monitor], enable_checkpointing=args.enable_checkpointing)
    trainer.fit(classifier, train_dataloaders['train_classifier'], val_dataloaders=evaluation_dataloaders)

    ## evaluation by a linear classifier
    for key in ['retain', 'test', 'unlearn']:
        cls_acc = cls_test(net=classifier, test_data_loader=evaluation_dataloaders[key], args=args)
        results.append(cls_acc)
    time_results.append(time() - start_time)
    start_time = time()

    ## CMIA evaluation
    CMIA_acc = SVC_MIA(model=classifier, shadow_train=train_cmia_dataloader, shadow_test=evaluation_dataloaders['test'], target_train=None, target_test=evaluation_dataloaders['unlearn'], args=args)
    results.append(CMIA_acc)
    time_results.append(time() - start_time)
    start_time = time()
    wandb_logger.log_text(key="evluation", columns=columns, data=[results])
    wandb_logger.log_text(key="time", columns=time_columns, data=[time_results])

if __name__ == "__main__":
    args = get_args()
    args = args_check(args)
    setup_seed(args.seed)
    main()