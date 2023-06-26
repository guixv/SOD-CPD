import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from datetime import datetime
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import math

from model.PVT import pvt_tiny, pvt_small, pvt_medium, pvt_large
from model.pvt_v2 import pvt_v2b2
from data import get_loader
from utils import clip_gradient, adjust_lr


def train(train_loader, model, optimizer, epoch, scheduler, min_loss):
    model.train()
    total_loss = 0
    for i, pack in enumerate(tqdm(train_loader), start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        dets = model(images)
        loss = CE(dets, gts)
        loss.backward()
        total_loss += loss.data

        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        scheduler.step()  # 更新学习率

        if i == 400 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, total_loss))

    save_path = opt.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 25 == 0:
        torch.save(model.state_dict(), save_path + 'epoch_%d' % epoch + '_CPD.pth')
    if total_loss < min_loss:
        min_loss = total_loss
        torch.save(model.state_dict(), save_path + 'minloss_%d' % epoch + '_CPD.pth')
    return min_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--lrf", type=int, default=0.1)
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.05, help='decay rate of learning rate')
    # parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--save_path', type=str, default='models/PVTv2/', help='save path')
    parser.add_argument("--image_path", type=str, default='data/DUTS-TR/DUTS-TR-Image/')
    parser.add_argument("--mask_path", type=str, default='data/DUTS-TR/DUTS-TR-Mask/')
    opt = parser.parse_args()

    print('Learning Rate: {} '.format(opt.lr))
    # build models
    model = pvt_v2b2()

    model.cuda()
    params = model.parameters()
    # optimizer = torch.optim.Adam(params, opt.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.9999))

    total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(total_paramters / 1e6) + "M")

    image_root = opt.image_path
    gt_root = opt.mask_path
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    print("using {} step for training.".format(total_step))  # 用于打印总的训练集数量和验证集数量

    CE = torch.nn.BCEWithLogitsLoss()

    epoch = opt.epoch

    lrf = opt.lrf
    lf = lambda x: ((1 + math.cos(x * math.pi / (epoch + 1))) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 学习率变化

    min_loss = 999

    print("Let's go!")

    for epoch in range(1, opt.epoch + 1):
        # adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        min_loss = train(train_loader, model, optimizer, epoch, scheduler, min_loss)
    print("min loss = %d" % min_loss)
    print("======end===========")
