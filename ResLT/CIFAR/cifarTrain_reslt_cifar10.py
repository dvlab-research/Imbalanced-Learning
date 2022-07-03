import argparse
import torch
import torch.nn as nn
import sys
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from networks import nets
from datasets import dataset
import time
import math

model_names=['ResLTResNet32']
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-mark',type=str,default='')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--finetune', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--imb_factor', default=None, type=float)
parser.add_argument('--scale', default=1, type=int)

###cosine
parser.add_argument('-cosine', default=False, type=bool)
parser.add_argument('-Ti', default=10, type=int)
parser.add_argument('-Ti_mul', default=2, type=int)
parser.add_argument('-lr_min', default=1e-5, type=float)
parser.add_argument('-lr_max', default=0.1, type=float)
###dataset
parser.add_argument('-dataset',default='CIFAR10', type=str)
parser.add_argument('-num_classes',default=10, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')

### random seed
parser.add_argument('-seed',default=None, type=int)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam1 = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam1 * x + (1-lam1) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam1

def mixup_criterion(criterion, pred, y_a, y_b, lam1):
    return lam1 * criterion(pred, y_a) + (1-lam1) * criterion(pred, y_b)

def crossEntropy(softmax, logit, label, weight):
    target = F.one_hot(label, args.num_classes)
    loss = - (weight * (target * torch.log(softmax(logit)+1e-7)).sum(dim=1)).sum()
    return loss

def Train():

    ### SEED
    if args.seed is not None:
       SEED=args.seed
       torch.manual_seed(SEED)
       torch.cuda.manual_seed(SEED)
       np.random.seed(SEED)
       torch.backends.cudnn.deterministic=True

    #####
    data=getattr(dataset,args.dataset)(batch_size=args.batch_size, imb_factor=args.imb_factor)
  
    #####
    model=getattr(nets,args.arch)(num_classes=args.num_classes, scale=args.scale)
    model = nn.DataParallel(model)
    model=model.cuda()

    optimizer=torch.optim.SGD(model.parameters(), lr=args.lr_max if args.cosine else args.lr, momentum=args.momentum,weight_decay=args.weight_decay, nesterov=True)
    ce=torch.nn.CrossEntropyLoss().cuda()
    mse=torch.nn.MSELoss().cuda()
    kl=nn.KLDivLoss(size_average=False)
    softmax=nn.Softmax(dim=1)
    ##### lr schedule cosine
    Ti=args.Ti
    Ti_mul=args.Ti_mul
    lr_min=args.lr_min
    lr_max=args.lr_max
    last_restart_epoch=0
    ##### cosine
    def lr_schedule_cosine(epoch,Ti,last_restart_epoch):
        T_cur=epoch-last_restart_epoch
        if T_cur<Ti:
           rate=T_cur/Ti*3.1415926
           lr=lr_min+0.5*(lr_max-lr_min)*(1.0+math.cos(rate))
        else:
           last_restart_epoch=epoch
           Ti=int(Ti*Ti_mul+0.5)
           T_cur=epoch-last_restart_epoch
           rate=T_cur/Ti*3.1415926
           lr=lr_min+0.5*(lr_max-lr_min)*(1.0+math.cos(rate))
        for param_group in optimizer.param_groups:
                param_group['lr']=lr
        return lr,Ti,last_restart_epoch

    ##### lr multi step schedule
    step=[160, 180, 200]
    def lr_schedule_multistep(epoch):
        if epoch<5:
           factor=(epoch+1)/5.0
           lr= args.lr * ( 1/3.0 *(1-factor) + factor )
        elif epoch<160 * 1 :
              lr=args.lr
        elif epoch<180 * 1 :
              lr=args.lr * 0.1
        elif epoch<200 * 1 :
              lr=args.lr * 0.1 * 0.1
        for param_group in optimizer.param_groups:
                param_group['lr']=lr
        return lr

    #####Train
    start_epoch=0
    end_epoch=args.epochs
    best_acc=0.0
    for epoch in range(start_epoch,end_epoch):
        #####adjust learning rate every epoch begining
        if args.cosine:
           lr,Ti,last_restart_epoch=lr_schedule_cosine(epoch,Ti,last_restart_epoch)
        else:
           lr=lr_schedule_multistep(epoch)
        #####

        model.train()
        train_loss=0.0
        total=0.0
        correct=0.0
        num=0
        for i,(inputs,target) in enumerate(data.train):
            input, target = inputs.cuda(),target.cuda()
            logitH, logitM, logitT = model(input)
            ### ResLT 
            labelH = F.one_hot(target, args.num_classes).sum(dim=1)
            labelM = F.one_hot(target, args.num_classes)[:,2:10].sum(dim=1)
            labelT = F.one_hot(target, args.num_classes)[:,4:10].sum(dim=1)
            loss_ice = (crossEntropy(softmax, logitH, target, labelH) + crossEntropy(softmax, logitM, target, labelM) \
                       + crossEntropy(softmax, logitT, target, labelT)) / (labelH.sum() + labelM.sum() + labelT.sum())

            logit = (logitH + logitM + logitT)
            loss_fce=ce(logit,target)
            loss = loss_ice * args.beta + (1-args.beta) * loss_fce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            _,predicted = logit.max(1)
            total += target.size(0)
            num+=1
            correct += predicted.eq(target).sum().item()
            if i % args.print_freq == 0:
                acc_n=logit.max(dim=1)[1].eq(target).sum().item()
                open("Logs/CIFAR10/"+args.mark+".log","a+").write('Train loss %.5f  loss_ice %.5f loss_fce %.5f acc_n %.5f lr %.5f\n'%(loss.item(), loss_ice.item(), loss_fce.item(), acc_n/input.size(0), lr))
        open("Logs/CIFAR10/"+args.mark+".log","a+").write("Train epoch=%d loss=%.5f  acc=%.5f\n"%(epoch,train_loss/num,correct/total))

        model.eval()
        test_loss=0.0
        total=0.0
        class_num=torch.zeros(args.num_classes).cuda()
        correct=torch.zeros(args.num_classes).cuda()
        num=0
        for i,(inputs,target) in enumerate(data.test):
            input, target = inputs.cuda(),target.cuda()
            logitH, logitM, logitT= model(input)
            logit = logitH  + logitM + logitT
            loss=ce(logit, target)
            test_loss+=loss.item()
            _,predicted = logit.max(1)
            total += target.size(0)
            num+=1
            target_one_hot=F.one_hot(target,args.num_classes)
            predict_one_hot=F.one_hot(predicted,args.num_classes)
            class_num=class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct=correct + (target_one_hot + predict_one_hot==2).sum(dim=0).to(torch.float)

        acc=correct.sum()/total
        acc_classes=correct/class_num
        head_acc=acc_classes[:3].mean()
        medium_acc=acc_classes[3:6].mean()
        tail_acc=acc_classes[6:10].mean()

        if best_acc<acc:
           best_acc=acc
           torch.save(model.state_dict(),"checkpoints/CIFAR10/"+args.mark+"_"+"best.pth")

        open("Logs/CIFAR10/"+args.mark+".log","a+").write("Test epoch=%d loss=%.5f  acc=%.5f best_acc=%.5f\n"%(epoch,test_loss/num,correct.sum()/total, best_acc))
        open("Logs/CIFAR10/"+args.mark+".log","a+").write("Test "+str(correct/class_num)+"\n") 
        open("Logs/CIFAR10/"+args.mark+".log","a+").write("Test head acc:"+str(head_acc)+" medium acc "+str(medium_acc)+" tail acc "+str(tail_acc)+"\n")

if __name__=='__main__':
   args = parser.parse_args()
   Train()
