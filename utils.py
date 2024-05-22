import torch
from torchvision import transforms
from lightly.transforms import utils
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import random 
from lightly.transforms import GaussianBlur

def get_cl_augs(strength=1.0, size=32, imagenet=False):
    s = strength
    if imagenet:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.1, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigmas=(0.1, 2.0), prob=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=utils.IMAGENET_NORMALIZE["mean"],
                std=utils.IMAGENET_NORMALIZE["std"],)
            ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(1 - 0.9 * s, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)],
                p=0.8 * s),
            transforms.RandomGrayscale(p=0.2 * s),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=utils.IMAGENET_NORMALIZE["mean"],
                std=utils.IMAGENET_NORMALIZE["std"],)
            ])
    return transform

def get_test_augs(imagenet=False):
    if imagenet:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=utils.IMAGENET_NORMALIZE["mean"],
                std=utils.IMAGENET_NORMALIZE["std"],)
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=utils.IMAGENET_NORMALIZE["mean"],
                std=utils.IMAGENET_NORMALIZE["std"],)
            ])
    return transform

def get_sl_augs(size=32, imagenet=False):
    if imagenet:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.1, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=utils.IMAGENET_NORMALIZE["mean"],
                std=utils.IMAGENET_NORMALIZE["std"],)
            ])

    else:
        transform = transforms.Compose([
            transforms.RandomCrop(size, padding=int(size/8)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=utils.IMAGENET_NORMALIZE["mean"],
                std=utils.IMAGENET_NORMALIZE["std"],)
            ])
    return transform

def get_indices(seed, num_total_samples=50000,num_unlearn_samples=4500):
    rs = np.random.RandomState(seed)
    permutation = rs.permutation(num_total_samples)
    num_validation_samples = int(num_total_samples * 0.1)

    assert num_unlearn_samples < num_total_samples - num_validation_samples

    validation_indices = permutation[:num_validation_samples]
    unlearn_indices = permutation[num_validation_samples:num_validation_samples+num_unlearn_samples]
    retain_indices = permutation[num_validation_samples+num_unlearn_samples:]
    return validation_indices, retain_indices, unlearn_indices


def cls_test(net, test_data_loader, args):
    net.eval()
    device = torch.device(f'cuda:{args.gpu_id}')
    net.to(device)
    total_top1, total_top5, total_num, = 0.0, 0.0, 0
    with torch.no_grad():
        test_bar = tqdm(test_data_loader)
        for data, target, _ in test_bar:
            data, target = data.to(device), target.to(device)
            pred = net(data).argmax(dim=1)
            total_num += data.size(0)
            total_top1 += (pred == target).float().sum().item()
            test_bar.set_description('Test: Acc@1:{:.2f}%'.format(total_top1 / total_num * 100))
    return total_top1 / total_num * 100

def args_check(args):
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.size = 32
        args.num_test_samples = 10000
        args.num_total_samples = 50000
    elif args.dataset == 'cifar100':
        args.num_total_samples = 50000
        args.num_classes = 100
        args.size = 32
        args.num_test_samples = 10000
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    if args.backbone == 'resnet18':
        args.feature_dim = 512
    elif args.backbone == 'resnet50':
        args.feature_dim = 2048
    else:
        raise ValueError('backbone not supported: {}'.format(args.backbone))
    return args

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        if param.requires_grad:
            params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)