import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from models import resnet_cifar
from models import resnet_imagenet

from datasets.txt_dataset import SiameseLongTailedDataset, LongTailedDataset
from datasets.cifar_lt import ImbalanceCIFAR10, ImbalanceCIFAR100

from logger import _C as config
from logger import update_config, get_logger

from utils import AverageMeter, ProgressMeter, accuracy

import augment.randaugment # For large-scale datasets
import augment.autoaugment # For CIFAR

import rescom.loader
import rescom.builder
import losses




model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='ResCom training')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    config.dist_url = "tcp://127.0.0.1:{}".format(random.randint(0, 20000) % 20000 + 3415)

    # debug mode (single GPU)
    if config.debug:
        config.batch_size = 32
        config.workers = 0
        config.gpu = 0
        config.rank = 0
        config.world_size = 1
        config.seed = 0

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    if config.debug:
        config.distributed = True
        config.multiprocessing_distributed = False



    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    
    config.gpu = gpu
    logger = get_logger(config, resume=False, is_rank0=(gpu==0))
    logger.info(str(config))

    # suppress printing if not master
    if config.multiprocessing_distributed and config.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    
    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    # create model
    logger.info("=> creating model '{}'".format(config.arch))
    if config.dataset == 'cifar10' or config.dataset == 'cifar100':
        model = rescom.builder.ResCom(
                    getattr(resnet_cifar, config.arch), dim_feat=config.dim_feat, 
                    batch_size=config.batch_size // ngpus_per_node, num_classes=config.num_classes, dim=config.dim_con, 
                    queue_size_per_cls=config.queue_size_per_cls, select_num_pos=config.select_num_pos, select_num_neg=config.select_num_neg)
    else:
        model = rescom.builder.ResCom(
                    getattr(resnet_imagenet, config.arch), dim_feat=config.dim_feat, 
                    batch_size=config.batch_size // ngpus_per_node, num_classes=config.num_classes, dim=config.dim_con, 
                    queue_size_per_cls=config.queue_size_per_cls, select_num_pos=config.select_num_pos, select_num_neg=config.select_num_neg)

    logger.info(model)
    if config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])

            if config.resume:
                state_dict = model.state_dict()
                state_dict_resume = torch.load(config.resume, map_location=torch.device('cpu'))['state_dict']

                unloaded_modules = []
                for key in state_dict.keys():
                    if key in state_dict_resume.keys() and state_dict[key].shape == state_dict_resume[key].shape:
                        state_dict[key ]= state_dict_resume[key]
                    else:
                        unloaded_modules.append(key)
                if len(unloaded_modules) == 0:
                    logger.info("*****************Model loading success*****************")
                    model.load_state_dict(state_dict)
                else:
                    logger.info("*****************Loaded model mismatched*****************")
                    logger.info(unloaded_modules)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")


    optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            if config.gpu is None:
                checkpoint = torch.load(config.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config.gpu)
                checkpoint = torch.load(config.resume, map_location=loc)
            config.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(config.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    cudnn.benchmark = True

    if config.dataset == 'inat':
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
    elif config.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif config.dataset == 'cifar10' or config.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    else:
        raise NotImplementedError("The dataset is not supported.")
    
    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)
    
    if config.dataset == 'cifar10' or config.dataset == 'cifar100':
        augmentation = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                augment.autoaugment.CIFAR10Policy(),
                transforms.ToTensor(),
                augment.autoaugment.Cutout(n_holes=1, length=16),
                normalize,
            ]
    else:
        augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0),
                transforms.RandomGrayscale(p=0.2),
                augment.randaugment.rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
                transforms.ToTensor(),
                normalize,
            ]
    
    transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
   
    if config.dataset == 'cifar10':
        train_dataset = ImbalanceCIFAR10(root=config.data_dir, imb_type='exp', imb_factor=config.imb_factor, 
                                         rand_number=0, train=True, download=True, transform=transform_train)
    elif config.dataset == 'cifar100':
        train_dataset = ImbalanceCIFAR100(root=config.data_dir, imb_type='exp', imb_factor=config.imb_factor, 
                                         rand_number=0, train=True, download=True, transform=transform_train)
    else:
        train_dataset = SiameseLongTailedDataset(img_root_path=config.data_dir, txt_path=config.train_txt_path, 
                                                 num_classes=config.num_classes, transform=transform_train)
    
    if config.dataset == 'cifar10' or config.dataset == 'cifar100':
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    else:
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    if config.dataset == 'cifar10':
        val_dataset = torchvision.datasets.CIFAR10(root=config.data_dir, train=False, download=True, transform=transform_val)
    elif config.dataset == 'cifar100':
        val_dataset = torchvision.datasets.CIFAR100(root=config.data_dir, train=False, download=True, transform=transform_val)
    else:
        val_dataset = LongTailedDataset(img_root_path=config.data_dir, txt_path=config.val_txt_path, transform=transform_val)


    logger.info(f'===> Training data length {len(train_dataset)}')
    logger.info(f'===> Evaluation data length {len(val_dataset)}')

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size * 4, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = losses.SiamBS_SPM(train_dataset.cls_num_list, balsfx_n=config.balsfx_n, 
                                 queue_size_per_cls=config.queue_size_per_cls, temperature=config.temperature, 
                                 con_weight=config.con_weight, effective_num_beta=config.effective_num_beta).cuda(config.gpu)
    
    if config.evaluate:
        logger.info(" ******************** start evaualteion (ONLY) ******************** ")
        validate(val_loader, model, logger, config)
        return

    best_acc1 = 0
    acc1 = 0
    is_best = False
    
    for epoch in range(config.start_epoch, config.epochs + 1):
        if config.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(config, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger, config)
        logger.info(" ")
        
        if epoch >= config.start_eval_epoch:
            acc1 = validate(val_loader, model, logger, config)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            logger.info(output_best)
        
        if not config.multiprocessing_distributed or \
                (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best, config.model_dir)



def train(train_loader, model, criterion, optimizer, epoch, logger, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_cls = AverageMeter('Cls_L', ':.3f')
    losses_cont = AverageMeter('Con_L', ':.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [losses_cls, losses_cont, losses, top1],
        logger,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, (images_org, images_con, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end) 

        if config.gpu is not None:
            images_org = images_org.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)
            images_con = images_con.cuda(config.gpu, non_blocking=True)

        img = torch.cat([images_org, images_con], dim=0)
        target = torch.cat([target, target], dim=0)
        # compute output
        sim_con, labels_con, logits_cls = model(img=img, labels=target)
        loss_cls, loss_con, loss = criterion(sim_con, labels_con, logits_cls, target)

        acc1, _ = accuracy(logits_cls, target, topk=(1, 5))
        losses_cls.update(loss_cls.item(), logits_cls.size(0))
        losses_cont.update(loss_con.item(), logits_cls.size(0))
        losses.update(loss.item(), logits_cls.size(0))
        top1.update(acc1[0], logits_cls.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, logger, config):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [top1, top5],
        logger, 
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            output = model(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + '/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/model_best.pth.tar')


def adjust_learning_rate(config, optimizer, epoch):
    if config.dataset == 'cifar10' or config.dataset == 'cifar100':
        """Sets the learning rate to the initial LR decayed by 10 at the 320th and 360th epoch"""
        lr = config.lr
        if epoch < config.warm_epochs:
            lr = lr / config.warm_epochs * epoch
        elif epoch > 360:
            lr = lr * 0.01
        elif epoch > 320:
            lr = lr * 0.1
        else:
            lr = lr
    else:
        """Cosine learning rate schedule"""
        lr = config.lr
        lr_decay_rate = 0.1
        eta_min = lr * (lr_decay_rate ** 5)
        # eta_min = 0.0
        if epoch < config.warm_epochs:
            lr = lr / config.warm_epochs * epoch
        else:
            lr = eta_min + (lr - eta_min) * (
                    1 + math.cos(math.pi * (epoch - config.warm_epochs) / (config.epochs - config.warm_epochs))) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    main()
