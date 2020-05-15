# from dreal import *
import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from utils import AverageMeter

from data import get_dataloader

from tensorboardX import SummaryWriter

from model import get_model, projection, num_dim_projected

use_cuda = True

np.random.seed(1024)

# num_epochs = 350
# num_epochs = 13
num_epochs = 100
# learning_rate = 0.001
learning_rate = 0.01
miscoverage_rate = 0.001

num_dim = 9
model, forward = get_model(num_dim)
if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every * epochs"""
    # lr = learning_rate * (0.1 ** (epoch // 5))
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    filename = 'log/' + filename
    torch.save(state, filename)

# L2_loss_weight = 1.0
L2_loss_weight = 0.1
convergence_reg_weight = 1.0
alpha = 1e-1
def hinge_loss_function(LHS, RHS):
    res = LHS - RHS + alpha
    res[res<0] = 0
    return res.mean()

# def separator_loss(LHS, RHS, margin):
#     res = RHS - LHS
#     loss = margin + lambda_1 * (-1.0 * res) + lambda_2 * (res - margin)
#     return res.mean()

def pol2cart(rho, theta):
    # rho, theta: Nx1
    rho = rho.view(-1,1)
    theta = theta.view(-1,1)
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    return torch.cat((x, y), dim=1)

global_step = 0

def calc_half_conformal_interval_width(calibration_loader):
    result, _ = trainval(0, calibration_loader, None, training=False)
    # from IPython import embed; embed()
    LHS = np.concatenate(result[3], axis=0)
    RHS = np.concatenate(result[4], axis=0)
    residuals = np.abs(LHS - RHS).reshape(-1).tolist()
    residuals.sort()
    d = residuals[int(np.ceil((len(residuals)+1) * (1-miscoverage_rate)))-1]
    return d, residuals

def show_ellipsoid(P, samples):
    thetas = np.arange(0,1,0.01) * 2 * np.pi
    points = []
    for i, theta in enumerate(thetas):
        point = np.array([np.cos(theta), np.sin(theta)])
        points.append(point)
    points = np.array(points)
    points = np.linalg.inv(P).dot(points.T)
    plt.scatter(points[0,:], points[1,:], s=0.1, marker='o', color='r')
    plt.scatter(samples[0,:], samples[1,:], s=0.1, marker='o', color='g')
    plt.show()

K = 1024
def loss_zero_matrix(A):
    # import ipdb; ipdb.set_trace()
    # A: bs x d x d
    # z: K x d
    z = torch.randn(A.size(-1), K)
    if use_cuda:
        z = z.cuda()
    z = z / z.norm(dim=0, keepdim=True)
    Az = A.matmul(z)
    return Az.norm(dim=1).mean()

def trainval(epoch, dataloader, writer, training):
    global global_step
    loss = AverageMeter()
    hinge_loss = AverageMeter()
    L2_loss = AverageMeter()
    prec = AverageMeter()
    convergence_reg = AverageMeter()

    result = [[],[],[],[],[]] # for plotting

    if training:
        model.train()
    else:
        model.eval()
    end = time.time()
    for step, (X0, R, Xi0, Xi1, T) in enumerate(dataloader):
    # _iter = iter(dataloader)
    # (X0, R, Xi0, Xi1, T) = _iter.next()
    # for step in range(len(dataloader)):
        # print(R)
        # import pdb; pdb.set_trace()
        # import ipdb; ipdb.set_trace()
        batch_size = X0.size(0)
        time_str = 'data time: %.3f s\t'%(time.time()-end)
        end = time.time()
        if use_cuda:
            X0 = X0.cuda()
            R = R.cuda()
            Xi0 = Xi0.cuda()
            Xi1 = Xi1.cuda()
            T = T.cuda()
        # import ipdb; ipdb.set_trace()
        TransformMatrix = forward(torch.cat([X0,Xi0,R,T], dim=1))
        time_str += 'forward time: %.3f s\t'%(time.time()-end)
        end = time.time()

        # result[0].append(X.detach().cpu().numpy())
        # result[1].append(R.detach().cpu().numpy())
        # result[2].append(T.detach().cpu().numpy())
        # result[3].append(LHS.detach().cpu().numpy())
        # result[4].append(RHS.detach().cpu().numpy())

        # TransformMatrix = TransformMatrix.view(-1,num_dim_projected,num_dim_projected)
        # eps = 1e-5
        # TransformMatrix = torch.inverse(W+eps*torch.eye(W.shape[-1]).unsqueeze(0).type(W.type()))
        # TransformMatrix = model.module.P
        DXi = projection(Xi1-Xi0)
        # # DXi_norm2 = (DXi ** 2).sum(dim=1)
        # DXi_inv_weights = ((2 * torch.matmul(torch.matmul(DXi.view(batch_size, 1, num_dim), torch.transpose(TransformMatrix, 1,2)),TransformMatrix))**2).view(batch_size,num_dim).sum(dim=1)# + 1e-3
        LHS = ((torch.matmul(TransformMatrix, DXi.view(batch_size,num_dim_projected,1)).view(batch_size,num_dim_projected)) ** 2).sum(dim=1)# / DXi_inv_weights
        # RHS = 1.0 / DXi_inv_weights
        RHS = torch.ones(LHS.size()).type(DXi.type())

        # TransformMatrix = (model.module.P + torch.transpose(model.module.P, 1,2))
        # DXi = Xi1-Xi0
        # from IPython import embed;embed()
        # DXi_norm2 = (DXi ** 2).sum(dim=1)
        # DXi_inv_weights = ((2 * torch.matmul(DXi.view(batch_size, 1, num_dim), TransformMatrix))**2).view(batch_size,num_dim).sum(dim=1).detach()# + 1e-3
        # LHS = torch.matmul(torch.matmul(DXi.view(batch_size, 1, num_dim), TransformMatrix), DXi.view(batch_size, num_dim, 1)).view(-1)# / DXi_inv_weights
        # RHS = torch.ones(LHS.size()).cuda()
        # RHS = 1.0 / DXi_inv_weights

        # import ipdb; ipdb.set_trace()
        _hinge_loss = hinge_loss_function(LHS, RHS)
        _L2_loss = F.mse_loss(LHS, RHS)
        # _convergence_reg = loss_zero_matrix(torch.inverse(model(torch.cat([X0,Xi0,torch.zeros(R.size()).type(R.type()),T], dim=1)).view(-1,num_dim_projected,num_dim_projected)))
        # _loss = _hinge_loss + L2_loss_weight * _L2_loss + convergence_reg_weight * _convergence_reg
        _loss = _hinge_loss + L2_loss_weight * _L2_loss# + convergence_reg_weight * _convergence_reg
        # _loss = L2_loss_weight * _L2_loss# + convergence_reg_weight * _convergence_reg

        loss.update(_loss.item(), batch_size)
        prec.update((LHS.detach().cpu().numpy() <= (RHS.detach().cpu().numpy())).sum() / batch_size, batch_size)
        hinge_loss.update(_hinge_loss.item(), LHS.size(0))
        L2_loss.update(_L2_loss.item(), batch_size)
        # convergence_reg.update(_convergence_reg.item(), batch_size)

        if writer is not None and training:
            writer.add_scalar('loss', loss.val, global_step)
            writer.add_scalar('prec', prec.val, global_step)
            writer.add_scalar('L2_loss', L2_loss.val, global_step)
            writer.add_scalar('Hinge_loss', hinge_loss.val, global_step)
            writer.add_scalar('convergence_reg', convergence_reg.val, global_step)

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

    print('Loss: %.3f, PREC: %.3f, HINGE_LOSS: %.3f, MARGIN: %.3f, CONVER: %.3f'%(loss.avg, prec.avg, hinge_loss.avg, L2_loss.avg, convergence_reg.avg))

    if writer is not None and not training:
        writer.add_scalar('loss', loss.avg, global_step)
        writer.add_scalar('prec', prec.avg, global_step)
        writer.add_scalar('L2_loss', L2_loss.avg, global_step)
        writer.add_scalar('Hinge_loss', hinge_loss.avg, global_step)
        writer.add_scalar('convergence_reg', convergence_reg.avg, global_step)

    # result = [np.concatenate(r) for r in result]
    # import ipdb; ipdb.set_trace()
    # if epoch>45:
        # from IPython import embed; embed()
    # from IPython import embed; embed()
    return result, loss.avg, prec.avg

# from IPython import embed;embed()

# train_loader, val_loader = get_dataloader(30, 5, 4096)
# train_loader, val_loader = get_dataloader(30, 5, 4096)
train_loader, val_loader = get_dataloader(10, 5, 256)
# calibration_loader, _ = get_dataloader(30, 5, 4096)
# calibration_loader, _ = get_dataloader(30, 5, 4096)
# calibration_loader, _ = get_dataloader(25, 5, 256)

train_writer = SummaryWriter('log/train')
val_writer = SummaryWriter('log/val')

# best_loss = np.inf
best_prec = 0

model = torch.nn.DataParallel(model).cuda()

torch.backends.cudnn.benchmark = True

os.system('cp data.py main.py log/')

for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch)
    # train for one epoch
    print('Epoch %d'%(epoch))
    _, _, _ = trainval(epoch, train_loader, writer=train_writer, training=True)
    result_train, _, _ = trainval(epoch, train_loader, writer=None, training=False)
    result_val, loss, prec = trainval(epoch, val_loader, writer=val_writer, training=False)
    epoch += 1
    if prec > best_prec:
    # if loss < best_loss
        # best_loss = loss
        best_prec = prec
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict()})

# d, residuals = calc_half_conformal_interval_width(calibration_loader)
# filename = 'log/checkpoint.pth.tar'
# checkpoint = torch.load(filename)
# checkpoint['d'] = d
# checkpoint['residuals'] = residuals
# save_checkpoint(checkpoint)
