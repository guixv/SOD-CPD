import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
import scipy
import imageio


from model.PVT import pvt_tiny,pvt_small,pvt_medium,pvt_large
from model.pvt_v2 import pvt_v2b2
from data import test_dataset
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=224, help='testing size')
    opt = parser.parse_args()

    dataset_path = 'path/dataset/'

    model = pvt_v2b2()
    model.load_state_dict(torch.load('epoch_300_CPD.pth'))


    model.cuda()
    model.eval()

    test_datasets = ['DUTS-TE']

    for dataset in test_datasets:
        save_path = './results/pvtv2_300/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/DUTS-TE-Image/'
        gt_root = dataset_path + dataset + '/DUTS-TE-Mask/'
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        # pbar = tqdm(total=test_loader.size)
        for i in tqdm(range(test_loader.size)):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255.0).astype('uint8')
            imageio.imsave(save_path+name, res)
        #     pbar.update(1)
        # pbar.close()
