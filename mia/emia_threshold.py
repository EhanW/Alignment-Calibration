import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CosineSimilarity
from sklearn.metrics import accuracy_score


def get_membership_features(model, dataloader, args):
    if len(dataloader) == 0:
        return None
    device = torch.device(f'cuda:{args.gpu_id}')
    model = model.to(device)
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(args.mi_augs):
            features = []
            for batch in dataloader:
                x, _, _ = batch
                x = x.to(device)
                features.append(model.backbone(x).flatten(start_dim=1).cpu())
            features = torch.cat(features)
            all_features.append(features)
    all_features = torch.stack(all_features, dim=1) # 10000x10x512  
    similarity = CosineSimilarity(dim=1, eps=1e-6)
    membership_features = []
    for i in range(args.mi_augs):
        for j in range(i+1, args.mi_augs):
            x = all_features[:, i].to(device)
            y = all_features[:, j].to(device)
            sim = similarity(x, y).cpu()
            membership_features.append(sim)
    membership_features = torch.stack(membership_features, dim=1) # 10000x45
    # membership_features = torch.sort(membership_features, dim=1).values
    membership_features = membership_features.mean(dim=1)
    # print(membership_features.size(), membership_features.mean(), membership_features.std())
    return membership_features.cpu().numpy()

def get_train_feature_dataloader(model, train_dataloader, test_dataloader, args):
    train_membership_features = get_membership_features(model, train_dataloader)
    test_membership_features = get_membership_features(model, test_dataloader)
    train_targets = torch.ones(train_membership_features.size(0), dtype=torch.int64)
    test_targets = torch.zeros(test_membership_features.size(0), dtype=torch.int64)
    features = torch.cat([train_membership_features, test_membership_features])
    targets = torch.cat([train_targets, test_targets])
    dataset = TensorDataset(features, targets)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    return dataloader

def get_single_feature_dataloader(model, dataloader, membership, args):
    features = get_membership_features(model, dataloader)
    if membership:
        targets = torch.ones(features.size(0), dtype=torch.int64)
    else:
        targets = torch.zeros(features.size(0), dtype=torch.int64)
    dataset = TensorDataset(features, targets)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    return dataloader


def encodermi_threshold(features, labels):
    bins = np.histogram_bin_edges(None, range=(0, 1), bins=500)
    accuracy = []
    for threshold in bins:
        pred_attack = []
        for i in range(len(features)):
            if features[i]>=threshold:
                pred_attack.append(1)
            else:
                pred_attack.append(0)
        pred_attack = np.array(pred_attack)
        accuracy.append(accuracy_score(labels, pred_attack))    
    threshold_index = np.argmax(accuracy)
    threshold = bins[threshold_index]
    return threshold

def encodermi_threshold_test(features, labels, threshold):
    pred_attack = []
    for i in range(len(features)):
        if features[i]>=threshold:
            pred_attack.append(1)
        else:
            pred_attack.append(0)
    pred_attack = np.array(pred_attack)
    accuracy = accuracy_score(labels, pred_attack)
    return accuracy*100
