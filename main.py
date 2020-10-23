import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import importlib
from utils import AverageMeter
from tensorboardX import SummaryWriter

from data import get_dataloader
from model import get_model

import sys
sys.path.append('configs')

import argparse

np.random.seed(1024)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--config', type=str,
                        default='drone')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--bs', dest='batch_size', type=int, default=256)
parser.add_argument('--num_train', type=int, default=10)
parser.add_argument('--num_test', type=int, default=5)
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01)
parser.add_argument('--lambda1', dest='_lambda1', type=float, default=0.1)
parser.add_argument('--lambda2', dest='_lambda2', type=float, default=0.1)
parser.add_argument('--alpha', dest='alpha', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr_step', type=int, default=5)
parser.add_argument('--pretrained', type=str)
parser.add_argument('--data_file_train', type=str)
parser.add_argument('--data_file_eval', type=str)
parser.add_argument('--log', type=str)

args = parser.parse_args()

os.system('cp *.py '+args.log)
os.system('cp -r configs/ '+args.log)
os.system('cp -r examples/ '+args.log)

np.random.seed(1024)

config = importlib.import_module('config_'+args.config)

model, forward = get_model(config.num_dim_input, config.num_dim_output)
if args.use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every * epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filename = args.log + '/' + filename
    torch.save(state, filename)

def hinge_loss_function(LHS, RHS):
    res = LHS - RHS + args.alpha
    res[res<0] = 0
    return res.mean()

global_step = 0

def trainval(epoch, dataloader, writer, training):
    global global_step
    loss = AverageMeter()
    hinge_loss = AverageMeter()
    volume_loss = AverageMeter()
    l2_loss = AverageMeter()
    prec = AverageMeter()

    result = [[],[],[],[],[]] # for plotting

    if training:
        model.train()
    else:
        model.eval()
    end = time.time()
    for step, (X0, R, Xi0, Xi1, T) in enumerate(dataloader):
        batch_size = X0.size(0)
        time_str = 'data time: %.3f s\t'%(time.time()-end)
        end = time.time()
        if args.use_cuda:
            X0 = X0.cuda()
            R = R.cuda()
            Xi0 = Xi0.cuda()
            Xi1 = Xi1.cuda()
            T = T.cuda()
        # import ipdb; ipdb.set_trace()
        TransformMatrix = forward(torch.cat([config.observe_for_input(X0),config.observe_for_input(Xi0),R,T], dim=1))
        time_str += 'forward time: %.3f s\t'%(time.time()-end)
        end = time.time()

        DXi = config.observe_for_output(Xi1 - Xi0)
        LHS = ((torch.matmul(TransformMatrix, DXi.view(batch_size,-1,1)).view(batch_size,-1)) ** 2).sum(dim=1)
        RHS = torch.ones(LHS.size()).type(DXi.type())

        _hinge_loss = hinge_loss_function(LHS, RHS)
        eps = 0.001
        # _volume_loss = (-torch.log((TransformMatrix + eps * torch.eye(TransformMatrix.shape[-1]).unsqueeze(0).type(X0.type())).det())).mean()
        _volume_loss = -torch.log((TransformMatrix + eps * torch.eye(TransformMatrix.shape[-1]).unsqueeze(0).type(X0.type())).det().abs()).mean()
        _l2_loss = F.mse_loss(LHS, RHS)

        # _volume_loss = torch.zeros([1]).cuda()
        # print(_hinge_loss, _volume_loss)
        _loss = _hinge_loss + args._lambda1 * _volume_loss + args._lambda2 * _l2_loss

        loss.update(_loss.item(), batch_size)
        prec.update((LHS.detach().cpu().numpy() <= (RHS.detach().cpu().numpy())).sum() / batch_size, batch_size)
        hinge_loss.update(_hinge_loss.item(), batch_size)
        volume_loss.update(_volume_loss.item(), batch_size)
        l2_loss.update(_l2_loss.item(), batch_size)

        if writer is not None and training:
            writer.add_scalar('loss', loss.val, global_step)
            writer.add_scalar('prec', prec.val, global_step)
            writer.add_scalar('Volume_loss', volume_loss.val, global_step)
            writer.add_scalar('Hinge_loss', hinge_loss.val, global_step)
            writer.add_scalar('L2_loss', l2_loss.val, global_step)

        time_str += 'other time: %.3f s\t'%(time.time()-end)
        c = time.time()
        if training:
            global_step += 1
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
        time_str += 'backward time: %.3f s'%(time.time()-c)
        end = time.time()
        #print(time_str)

    print('Loss: %.3f, PREC: %.3f, HINGE_LOSS: %.3f, VOLUME_LOSS: %.3f, L2_loss: %.3f'%(loss.avg, prec.avg, hinge_loss.avg, volume_loss.avg, l2_loss.avg))

    if writer is not None and not training:
        writer.add_scalar('loss', loss.avg, global_step)
        writer.add_scalar('prec', prec.avg, global_step)
        writer.add_scalar('Volume_loss', volume_loss.avg, global_step)
        writer.add_scalar('Hinge_loss', hinge_loss.avg, global_step)
        writer.add_scalar('L2_loss', l2_loss.avg, global_step)

    return result, loss.avg, prec.avg

# train_loader, val_loader = get_dataloader(30, 5, 4096)
train_loader, val_loader = get_dataloader(config, args.num_train, args.num_test, args.batch_size, [args.data_file_train, args.data_file_eval])

train_writer = SummaryWriter(args.log+'/train')
val_writer = SummaryWriter(args.log+'/val')

best_loss = np.inf
best_prec = 0

model = torch.nn.DataParallel(model)
if args.use_cuda:
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch)
    # train for one epoch
    print('Epoch %d'%(epoch))
    _, _, _ = trainval(epoch, train_loader, writer=train_writer, training=True)
    result_train, _, _ = trainval(epoch, train_loader, writer=None, training=False)
    result_val, loss, prec = trainval(epoch, val_loader, writer=val_writer, training=False)
    epoch += 1
    # if prec > best_prec:
    if loss < best_loss:
        best_loss = loss
        # best_prec = prec
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict()})
