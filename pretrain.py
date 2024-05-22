import os
import argparse
from time import time
from collections import OrderedDict
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from algs import Classifier, SimCLRModel, MocoModel
from dataset import CIFAR10Triad, CIFAR100Triad
from utils import get_indices, get_cl_augs, get_sl_augs, get_test_augs, cls_test, args_check, setup_seed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', type=str, default='conul')
    parser.add_argument('--log-model', '-l', action='store_true')    
    parser.add_argument('--unlearn-mode', '-m', default='pt')
    parser.add_argument('--num-unlearn-samples', '-n', default=0)
    parser.add_argument('--backbone', '-b', default='resnet18', choices=['resnet18', 'resnet50']) 

    parser.add_argument('--cl-alg', '-a', choices=['simclr', 'moco'], default='moco')
    parser.add_argument('--cl-epochs', type=int, default=800)
    parser.add_argument('--cl-lr', type=float, default=0.06)
    parser.add_argument('--cl-momentum', type=float, default=0.9)
    parser.add_argument('--cl-weight-decay', type=float, default=5e-4)
    parser.add_argument('--cl-temperature', type=float, default=0.1)

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

    parser.add_argument('--gpu-id', '-g', type=int, default=0, help='the gpu id')
    parser.add_argument('--seed', '-s', type=int, default=0)

    parser.add_argument('--alpha', type=float, default=0.0, help='regularization parameter for the unlearning loss')
    parser.add_argument('--gamma', type=float, default=None)
    
    parser.add_argument('--l1-reg', type=float, default=None)
    parser.add_argument('--precision', type=int, default=None) 
    return parser.parse_args()

def make_dataloaders():
    num_workers = args.num_workers
    batch_size = args.batch_size

    ### Define the augmentations
    imagenet = False
    train_extractor_transforms = get_cl_augs(strength=1.0, size=args.size, imagenet=imagenet)
    train_classifier_transforms = get_sl_augs(size=args.size, imagenet=imagenet)
    test_transforms = get_test_augs(imagenet=imagenet)

    ### Get the indices for the dataset
    validation_indices, retain_indices, _ = get_indices(
        seed=args.seed, num_total_samples=args.num_total_samples, num_unlearn_samples=0)
    
    if args.dataset == 'cifar10':
        DatasetTriad = CIFAR10Triad
    elif args.dataset == 'cifar100':
        DatasetTriad = CIFAR100Triad
    else:
        raise ValueError("Invalid value for dataset")
    
    ### Dataloaders for training feature extractor and then linear probing classifier
    train_extractor_dataset = DatasetTriad(
        root=args.data, train=True, pair=True, triad=False, transform=train_extractor_transforms, selected_indices=retain_indices)
    train_classifier_dataset = DatasetTriad(
        root=args.data, train=True, pair=False, triad=False, transform=train_classifier_transforms, selected_indices=retain_indices)
    train_extractor_dataloader = DataLoader(train_extractor_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    train_classifier_dataloader = DataLoader(train_classifier_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

    ### Dataloaders for evaluation after linear probing
    evaluation_train_dataset = DatasetTriad(
        root=args.data, train=True, pair=False, triad=False, transform=test_transforms, selected_indices=retain_indices)
    evaluation_validation_dataset = DatasetTriad(
        root=args.data, train=True, pair=False, triad=False, transform=test_transforms, selected_indices=validation_indices)
    evaluation_test_dataset = DatasetTriad(
        root=args.data, train=False, pair=False, triad=False, transform=test_transforms)
    evaluation_train_dataloader = DataLoader(evaluation_train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    evaluation_validation_dataloader = DataLoader(evaluation_validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    evaluation_test_dataloader = DataLoader(evaluation_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    ### Return the dataloaders as OrderedDict
    train_dataloaders = OrderedDict(train_extractor=train_extractor_dataloader, train_classifier=train_classifier_dataloader)
    evaluation_dataloaders = OrderedDict(train=evaluation_train_dataloader, validation=evaluation_validation_dataloader, test=evaluation_test_dataloader)
    return train_dataloaders, evaluation_dataloaders
    
def main():
    ## make dataloaders and models
    train_dataloaders, evaluation_dataloaders = make_dataloaders()
    if args.cl_alg == 'simclr':
        model = SimCLRModel(max_epochs= args.cl_epochs, lr=args.cl_lr, momentum=args.cl_momentum, weight_decay=args.cl_weight_decay, args=args, enable_scheduler=True, negative_loss=False)
    elif args.cl_alg == 'moco':
        model = MocoModel(max_epochs= args.cl_epochs, lr=args.cl_lr, momentum=args.cl_momentum, weight_decay=args.cl_weight_decay, args=args, enable_scheduler=True, negative_loss=False)
    else:
        raise ValueError("Invalid value for cl-alg")

    ## configure the logger
    save_dir = 'logs/'
    log_name = f'{args.dataset}-pretrain-{args.cl_alg}-seed={args.seed}-lr={args.cl_lr}' 
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
    columns = ['metrics'] + list(evaluation_dataloaders.keys())
    results = []
    time_columns = ['training_time', 'cls_time']
    time_results = []

    start_time = time()
    ## train the feature extractor
    trainer = Trainer(max_epochs=args.cl_epochs, devices=[args.gpu_id], accelerator="gpu", 
                         logger=wandb_logger, callbacks=[lr_monitor], enable_checkpointing=True, precision=args.precision)
    trainer.fit(model, train_dataloaders['train_extractor'])
    time_results.append(time() - start_time)
    start_time = time()

    ## train the linear classifier
    classifier = Classifier(model.backbone, args.fc_epochs, args.fc_lr, args.fc_momentum, args.num_classes, val_names=list(evaluation_dataloaders.keys()), 
                            feature_dim=args.feature_dim)
    trainer = Trainer(max_epochs=args.fc_epochs, devices=[args.gpu_id], accelerator="gpu", 
                         logger=wandb_logger, callbacks=[lr_monitor], enable_checkpointing=True)
    trainer.fit(classifier, train_dataloaders['train_classifier'], val_dataloaders=evaluation_dataloaders)

    ## evaluation by a linear classifier
    cls_results = ['cls_acc']
    for key, dataloader in evaluation_dataloaders.items():
        cls_acc = cls_test(net=classifier, test_data_loader=dataloader, args=args)
        cls_results.append(cls_acc)
    results.append(cls_results)
    time_results.append(time() - start_time)
    start_time = time()
    
    wandb_logger.log_text(key="evluation", columns=columns, data=results)
    wandb_logger.log_text(key="time", columns=time_columns, data=[time_results])

if __name__ == "__main__":
    args = get_args()
    args = args_check(args)
    setup_seed(args.seed)
    main()