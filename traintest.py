import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from datetime import datetime
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import math
from metric import *

from model.CPD_models import CPD_VGG
from model.CPD_convnext import CPD_convnext
from model.CPD_ResNet_models import CPD_ResNet
from model.PVT import pvt_tiny
from data import get_loader
from utils import clip_gradient, adjust_lr


def train(train_loader, model, optimizer, epoch, scheduler):
    model.train()
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

        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        scheduler.step()  # 更新学习率

        if i == 400 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))

    save_path = opt.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 25 == 0:
        torch.save(model.state_dict(), save_path + 'epoch_%d' % epoch + '_CPD.pth')

def test(epoch):
    # torch.nn.DataParallel(model)
    model.eval()
    # model_sub.eval()
    # test
    sum_loss=0
    cal_fm = CalFM(num=test_step)  # cal是一个对象
    cal_mae = CalMAE(num=test_step)
    # cal_sm = CalSM(num=test_step)
    for step, packs in enumerate(tqdm(test_loader), start=1):

        image, target = packs  # 这个得增加gt的部分

        images = Variable(image)
        target = Variable(target)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        target = target.unsqueeze(1)
        # target = torch.as_tensor(target, dtype=torch.float).cuda()

        # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
        with torch.no_grad():

            output = model(images)  # 模型

            output = torch.unsqueeze(output, 1)
            sum_loss += CE(output.detach(), target)
            output = torch.squeeze(output, 1)


        # assert image_index.type is list

        predict_rgb = output.sigmoid().cpu().detach().numpy()
        # print(predict_rgb.shape)
        # saveimg(predict_rgb, img_name)
        target = target.data.cpu().squeeze(1).numpy()
        for i in range(target.shape[0]):
            max_pred_array = predict_rgb[i].max()
            min_pred_array = predict_rgb[i].min()
            if max_pred_array == min_pred_array:
                predict_rgb[i] = predict_rgb[i] / 255
            else:
                predict_rgb[i] = (predict_rgb[i] - min_pred_array) / (max_pred_array - min_pred_array)

            cal_fm.update(predict_rgb[i], target[i])
            cal_mae.update(predict_rgb[i], target[i])
            # cal_sm.update(predict_rgb[i], target[i])

    _, maxf, mmf, _, _ = cal_fm.show()  # 这里其实有maxf
    mae = cal_mae.show()
    # sm = cal_sm.show()
    return mmf, mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--lrf", type=int, default=0.1)
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=224, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    # parser.add_argument('--decay_rate', type=float, default=0.05, help='decay rate of learning rate')
    # parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--save_path', type=str, default='models/PVT/', help='save path')
    parser.add_argument("--image_path", type=str, default='data/DUTS-TR/DUTS-TR-Image/')
    parser.add_argument("--mask_path", type=str, default='data/DUTS-TR/DUTS-TR-Mask/')
    parser.add_argument("--test_image_path", type=str, default='path/dataset/DUTS-TE/DUTS-TE-Image/')
    parser.add_argument("--test_mask_path", type=str, default='path/dataset/DUTS-TE/DUTS-TE-Mask/')
    opt = parser.parse_args()

    print('Learning Rate: {} '.format(opt.lr))
    # build models
    model = pvt_tiny()

    model.cuda()
    params = model.parameters()
    # optimizer = torch.optim.Adam(params, opt.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.9999))

    total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(total_paramters / 1e6) + "M")

    output_path = opt.save_path
    if os.path.exists(output_path) is True:
        inputting = input("路径已经存在，是否要覆盖并继续Y:")
        if inputting != "Y":
            quit()
    if os.path.exists(output_path) is not True:
        os.makedirs(output_path)
    log_eval_path = output_path + "/log_eval.txt"
    if os.path.exists(log_eval_path):  # 如果log_eval.txt在存储之前存在则删除，防止后续内容冲突
        os.remove(log_eval_path)

    log_list_path = output_path + "/log_list.txt"
    if os.path.exists(log_list_path):  # 如果log_eval.txt在存储之前存在则删除，防止后续内容冲突
        os.remove(log_list_path)

    image_root = opt.image_path
    gt_root = opt.mask_path
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    test_loader = get_loader(opt.test_image_path,opt.test_mask_path, batchsize=opt.batchsize, trainsize=opt.trainsize)
    test_step = len(test_loader)
    print("using {} step for training.".format(total_step))  # 用于打印总的训练集数量和验证集数量

    CE = torch.nn.BCEWithLogitsLoss()

    epoch = opt.epoch

    lrf = opt.lrf
    lf = lambda x: ((1 + math.cos(x * math.pi / (epoch + 1))) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 学习率变化

    print("Let's go!")

    with open(log_list_path, "a") as fp:
        fp.write("epoch,mmf, mae\n")
    for epoch in range(1, opt.epoch + 1):
        # adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, scheduler)
        if epoch % 10 == 1:
            mmf, mae=test(epoch)
            print(mmf, mae)
            with open(log_eval_path, "a") as fp:
                fp.write("======\n")
                fp.write("test  Epoch({}): F-measure:{} MAE:{} \n".format(epoch, mmf, mae))
            with open(log_list_path, "a") as fp:
                fp.write("{},{},{}\n".format(epoch, mmf, mae))

    print("======end===========")


