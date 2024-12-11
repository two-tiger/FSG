"""
An utils code for loading dataset
"""
import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from datetime import datetime
import utils.imagenet_utils as imagenet_utils
import utils.CIFAR10_utils as CIFAR10_utils

imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_mean_and_std(dataset, n_channels=3):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(n_channels):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_dataloader(dataset_name, split, batch_size, shuffle = True, num_workers=4):

    print ('[%s] Loading %s from %s' %(datetime.now(), split, dataset_name))

    if dataset_name == 'CIFAR10':

        data_root_list = ['/home/chenxinquan/MetaQuant/CIFAR10',
                          '/home/chenxinquan/MetaQuant/datasets/CIFAR10']

        for data_root in data_root_list:
            if os.path.exists(data_root):
                print('Found %s in %s' %(dataset_name, data_root))
                break

        if split == 'train':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = CIFAR10_utils.CIFAR10(root=data_root, train=True, download=True,
                                                    transform=transform_train)
            print ('Number of training instances used: %d' %(len(trainset)))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        elif split == 'test' or split == 'val':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                                   transform=transform_test)
            loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    elif dataset_name == 'CIFAR100':

        data_root_list = ['/home/chenxinquan/MetaQuant/CIFAR100', '/home/chenxinquan/MetaQuant/datasets/CIFAR100']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        if split == 'train':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            trainset = CIFAR10_utils.CIFAR100(root=data_root, train=True, download=True,
                                                    transform=transform_train)
            print ('Number of training instances used: %d' %(len(trainset)))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        elif split == 'test' or split == 'val':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True,
                                                   transform=transform_test)
            loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    elif dataset_name == 'ImageNet':

        # data_root_list = ['/data/imagenet', '/mnt/public/imagenet']
        # for data_root in data_root_list:
        #     if os.path.exists(data_root):
        #         break

        # traindir = ('./train_imagenet_list.pkl', './classes.pkl', './classes-to-idx.pkl','%s/train' %data_root)
        # valdir = ('./val_imagenet_list.pkl', './classes.pkl', './classes-to-idx.pkl', '%s/val-pytorch' %data_root)

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # if split == 'train':
        #     trainDataset = imagenet_utils.ImageFolder(traindir, transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))
        #     print ('Number of training data used: %d' %(len(trainDataset)))
        #     loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, \
        #                                          num_workers = num_workers, pin_memory=True)

        # elif split == 'val' or split == 'test':
        #     valDataset = imagenet_utils.ImageFolder(valdir, transforms.Compose([
		#         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))
        #     loader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=True, \
        #                                          num_workers = num_workers, pin_memory=True)
        
        imagenet_dir = '/root/pubdatasets/ILSVRC2012'
        # imagenet_dir = '/home/bingxing2/public/imagenet2012/ImageNet_ILSVRC2012'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        crop_scale = 0.08
        lighting_param = 0.1
        
        if split == 'train':
            traindir = os.path.join(imagenet_dir, 'train')
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
                Lighting(lighting_param),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])

            train_dataset = datasets.ImageFolder(
                traindir,
                transform=train_transforms)

            loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True)

        elif split == 'test' or split == 'val':
            valdir = os.path.join(imagenet_dir, 'val')
            loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)

    else:
        raise Exception('%s is not implemented yet' %(dataset_name))

    print ('[DATA LOADING] Loading from %s-%s finish. Number of images: %d, Number of batches: %d' \
           %(dataset_name, split, len(loader.dataset), len(loader)))

    return loader

if __name__ == '__main__':

    dataloader = get_dataloader('CIFAR10', 'train', batch_size=128)
    for (inputs, targets) in dataloader:
        break