from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from PIL import Image

class CIFAR10Triad(CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False,  transform_2=None, pair=False, triad=False, selected_indices=None, index_shift=None):
        super().__init__(root, train=train, transform=transform, download=download)
        self.transform_2 = transform_2
        self.index_shift = index_shift
        self.pair = pair
        self.triad = triad
        
        self.selected_indices = selected_indices
        if selected_indices is not None:
            self.data = self.data[selected_indices]
            self.targets = [self.targets[i] for i in selected_indices]
        else:
            self.selected_indices = np.arange(len(self.data))

    def __getitem__(self, index):
        # anchor and pasitive pairs (img1, img2)
        img = Image.fromarray(self.data[index])
        img_anchor = self.transform(img)
        label = self.targets[index]
        if not self.pair and not self.triad:
            return img_anchor, label, self.selected_indices[index]

        img_pos = self.transform(img)
        if not self.triad:
            return (img_anchor, img_pos), label, self.selected_indices[index]

        # negative pair (img1, img3)
        if self.index_shift is not None:
            if self.index_shift > 0:
                index_np = (index + self.index_shift) % len(self.data)
            else:
                index_np = np.random.randint(len(self.data))
        else:
            index_np = index
        img_np = Image.fromarray(self.data[index_np])
        img_np = self.transform_2(img_np) if self.transform_2 is not None else self.transform(img_np)
        return (img_anchor, img_np, img_pos), label, self.selected_indices[index] 




class CIFAR100Triad(CIFAR100):
    def __init__(self, root, train=True, transform=None, download=False,  transform_2=None, pair=False, triad=False, selected_indices=None, index_shift=None):
        super().__init__(root, train=train, transform=transform, download=download)
        self.transform_2 = transform_2
        self.index_shift = index_shift
        self.pair = pair
        self.triad = triad
        
        self.selected_indices = selected_indices
        if selected_indices is not None:
            self.data = self.data[selected_indices]
            self.targets = [self.targets[i] for i in selected_indices]
        else:
            self.selected_indices = np.arange(len(self.data))

    def __getitem__(self, index):
        # anchor and pasitive pairs (img1, img2)
        img = Image.fromarray(self.data[index])
        img_anchor = self.transform(img)
        label = self.targets[index]
        if not self.pair and not self.triad:
            return img_anchor, label, self.selected_indices[index]

        img_pos = self.transform(img)
        if not self.triad:
            return (img_anchor, img_pos), label, self.selected_indices[index]

        # negative pair (img1, img3)
        if self.index_shift is not None:
            if self.index_shift > 0:
                index_np = (index + self.index_shift) % len(self.data)
            else:
                index_np = np.random.randint(len(self.data))
        else:
            index_np = index
        img_np = Image.fromarray(self.data[index_np])
        img_np = self.transform_2(img_np) if self.transform_2 is not None else self.transform(img_np)
        return (img_anchor, img_np, img_pos), label, self.selected_indices[index] 
