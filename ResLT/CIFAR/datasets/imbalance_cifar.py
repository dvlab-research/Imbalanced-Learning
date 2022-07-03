import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets

import numpy as np
from PIL import Image
import random
import math

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    data_path="/mnt/proj56/jqcui/Data/cifar10"
    def __init__(self, root=None, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, class_balance=False):
        root=self.data_path
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

        self.class_balance = class_balance
        if class_balance:
           self.class_data=[ [] for i in range(self.cls_num) ]
           for i in range(len(self.targets)):
              self.class_data[self.targets[i]].append(i)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
          if self.class_balance:
             sample_class = random.randint(0, self.cls_num - 1)
             index = random.choice(self.class_data[sample_class])
             img, target = self.data[index], sample_class 
          else:
             img, target = self.data[index], self.targets[index]
          img = Image.fromarray(img)
          if self.transform is not None:
            img = self.transform(img)

          return img, target 


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100
    data_path="/mnt/proj56/jqcui/Data/cifar100"


class CIFAR10V2(object):
    def __init__(self, batch_size=128, class_balance=False, imb_factor=None):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean, std)
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        trainset = IMBALANCECIFAR10(root="/mnt/proj56/jqcui/Data/cifar10", train=True, transform=transform_train, download=False, imb_factor=imb_factor, class_balance=class_balance)
        testset = datasets.CIFAR10(root='/mnt/proj56/jqcui/Data/cifar10', train=False, transform=transform_test, download=False)

        self.train = torch.utils.data.DataLoader(
            trainset,
            batch_size = batch_size, shuffle = True,
            num_workers = 8, pin_memory = True, drop_last=True)

        self.test = torch.utils.data.DataLoader(
            testset,
            batch_size = batch_size, shuffle = False,
            num_workers = 4, pin_memory = True)

class CIFAR10V2_auto(object):
    def __init__(self, batch_size=128, class_balance=False, imb_factor=None):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean, std)
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                normalize,
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        trainset = IMBALANCECIFAR10(root="/mnt/proj56/jqcui/Data/cifar10", train=True, transform=transform_train, download=False, imb_factor=imb_factor, class_balance=class_balance)
        testset = datasets.CIFAR10(root='/mnt/proj56/jqcui/Data/cifar10', train=False, transform=transform_test, download=False)

        self.train = torch.utils.data.DataLoader(
            trainset,
            batch_size = batch_size, shuffle = True,
            num_workers = 8, pin_memory = True, drop_last=True)

        self.test = torch.utils.data.DataLoader(
            testset,
            batch_size = batch_size, shuffle = False,
            num_workers = 4, pin_memory = True)

class CIFAR100V2(object):
    def __init__(self, batch_size=128, class_balance=False, dual_sampler=False, imb_factor=None):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean, std)
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        trainset = IMBALANCECIFAR100(root='/mnt/proj56/jqcui/Data/cifar100', train=True, transform=transform_train, download=False, imb_factor=imb_factor, class_balance=class_balance)
        testset = datasets.CIFAR100(root='/mnt/proj56/jqcui/Data/cifar100', train=False, transform=transform_test, download=False)

        self.train = torch.utils.data.DataLoader(
            trainset,
            batch_size = batch_size, shuffle = True,
            num_workers = 8, pin_memory = True, drop_last=True)

        self.test = torch.utils.data.DataLoader(
            testset,
            batch_size = batch_size, shuffle = False,
            num_workers = 4, pin_memory = True)

class CIFAR100V2_auto(object):
    def __init__(self, batch_size=128, class_balance=False, dual_sampler=False, imb_factor=None):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean, std)
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                normalize,
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        trainset = IMBALANCECIFAR100(root='/mnt/proj56/jqcui/Data/cifar100', train=True, transform=transform_train, download=False, imb_factor=imb_factor, class_balance=class_balance)
        testset = datasets.CIFAR100(root='/mnt/proj56/jqcui/Data/cifar100', train=False, transform=transform_test, download=False)

        self.train = torch.utils.data.DataLoader(
            trainset,
            batch_size = batch_size, shuffle = True,
            num_workers = 8, pin_memory = True, drop_last=True)

        self.test = torch.utils.data.DataLoader(
            testset,
            batch_size = batch_size, shuffle = False,
            num_workers = 4, pin_memory = True)


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()
