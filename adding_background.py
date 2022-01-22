#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
import skvideo.io
import skvideo.datasets
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

def vis_parsing_maps(im, parsing_anno, stride, backimg_path='./data/background.jpg'):

    backimg = Image.open(backimg_path)
    backimg = np.array(backimg).astype(np.uint8)

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    #vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    pi = 0
    index = np.where(vis_parsing_anno == pi)
    vis_im[index[0], index[1], :] = backimg[index[0], index[1], :]
    return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    #net.cuda()
    device = torch.device('cpu')
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth, map_location=device))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    outputvideo = []

    with torch.no_grad():
        count = 0
        for image_path in os.listdir(dspth):
            image = Image.open(osp.join(dspth, image_path))
            # to speed up the running time just untag the next 2 lines
            # img = Image.open(osp.join(dspth, image_path))
            #image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            #img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            count +=1
            #parsing = parsing.resize(Image.NEAREST)
            x = vis_parsing_maps(image, parsing, stride=1)
            outputvideo.append(x)
            print("frame %d done" % count)
    print("all frames are done")
    outputvideo = np.array(outputvideo)
    #output = np.ndarray(outputvideo)
    skvideo.io.vwrite("./data/outputvideo2.mp4", outputvideo.astype(np.uint8))
if __name__ == "__main__":
    evaluate(dspth='./data/ilyassCloseUp', cp='79999_iter.pth')