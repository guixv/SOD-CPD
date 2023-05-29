import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime
from tqdm import tqdm

from model.CPD_models import CPD_VGG
from model.CPD_convnext import CPD_convnext
from model.CPD_ResNet_models import CPD_ResNet
from data import get_loader
from utils import clip_gradient, adjust_lr


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(tqdm(train_loader), start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        atts, dets = model(images)
        loss1 = CE(atts, gts)
        loss2 = CE(dets, gts)
        loss = loss1 + loss2
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i == 400 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data))

    save_path = opt.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 25 == 0:
        torch.save(model.state_dict(), save_path+ 'epoch_%d' % epoch + '_CPD.pth' )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=4e-3, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.05, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--save_path', type=str, default='models/CPD_convnext/', help='save path')
    parser.add_argument("--image_path", type=str, default='data/DUTS-TR/DUTS-TR-Image/')
    parser.add_argument("--mask_path", type=str, default='data/DUTS-TR/DUTS-TR-Mask/')
    opt = parser.parse_args()

    print('Learning Rate: {} '.format(opt.lr))
    # build models
    model = CPD_convnext()

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
    print("using {} images for training.".format(total_step))  # 用于打印总的训练集数量和验证集数量

    CE = torch.nn.BCEWithLogitsLoss()

    print("Let's go!")

    for epoch in range(1, opt.epoch+1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
