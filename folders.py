import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import cv2 as cv


class LIVEFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        live_txt_path = os.path.join(root, 'LIVE.txt')
        live_data = np.genfromtxt(live_txt_path, delimiter=',', dtype=str)
        imgpath = []
        imgscore = []
        for i in range(live_data.shape[0] - 1):
            imgpath.append(live_data[i + 1][0])
            imgscore.append(float(live_data[i + 1][3]))
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join('../IQA_dataset/data', imgpath[item]), imgscore[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        # 当使用索引时自动运行，获取图片和标签
        path, target = self.samples[index]
        sample1 = pil_loader(path)
        sample2 = HSV_loader(path)
        sample = Image.blend(sample1, sample2, 0.2)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        # 返回数据长度
        length = len(self.samples)
        return length


class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        livec_txt_path = os.path.join(root, 'LIVE_Challenge.txt')
        livec_data = np.genfromtxt(livec_txt_path, delimiter=',', dtype=str)
        imgpath = []
        imgscore = []
        for i in range(livec_data.shape[0] - 1):
            imgpath.append(livec_data[i + 1][0])
            imgscore.append(float(livec_data[i + 1][2]))
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join('../IQA_dataset/data', imgpath[item]), imgscore[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample1 = pil_loader(path)
        sample2 = HSV_loader(path)
        sample = Image.blend(sample1, sample2, 0.2)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class CSIQFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        csiq_txt_path = os.path.join(root, 'CSIQ.txt')
        csiq_data = np.genfromtxt(csiq_txt_path, delimiter=',', dtype=str)
        imgpath = []
        imgscore = []
        for i in range(csiq_data.shape[0] - 1):
            imgpath.append(csiq_data[i + 1][0])
            imgscore.append(float(csiq_data[i + 1][3]))
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join('../IQA_dataset/data', imgpath[item]), imgscore[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample1 = pil_loader(path)
        sample2 = HSV_loader(path)
        sample = Image.blend(sample1, sample2, 0.2)
        sample = self.transform(sample)
        return sample, target
    
    def __len__(self):
        length = len(self.samples)
        return length


class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        Koniq_10k_txt_path = os.path.join(root, 'KonIQ-10k.txt')
        Koniq_10k_data = np.genfromtxt(Koniq_10k_txt_path, delimiter=',', dtype=str)
        imgpath = []
        imgscore = []
        for i in range(Koniq_10k_data.shape[0] - 1):
            imgpath.append(Koniq_10k_data[i + 1][0])
            imgscore.append(float(Koniq_10k_data[i + 1][2]))
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join('../IQA_dataset/data', imgpath[item]), imgscore[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample1 = pil_loader(path)
        sample2 = HSV_loader(path)
        sample = Image.blend(sample1, sample2, 0.2)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class TID2013Folder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        TID2013_txt_path = os.path.join(root, 'TID2013.txt')
        TID2013_data = np.genfromtxt(TID2013_txt_path, delimiter=',', dtype=str)
        imgpath = []
        imgscore = []
        for i in range(TID2013_data.shape[0] - 1):
            imgpath.append(TID2013_data[i + 1][0])
            imgscore.append(float(TID2013_data[i + 1][3]))
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join('../IQA_dataset/data', imgpath[item]), imgscore[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample1 = pil_loader(path)
        sample2 = HSV_loader(path)
        sample = Image.blend(sample1, sample2, 0.2)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length
  
    
class CID2013Folder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        CID2013_txt_path = os.path.join(root, 'CID2013.txt')
        CID2013_data = np.genfromtxt(CID2013_txt_path, delimiter=',', dtype=str)
        imgpath = []
        imgscore = []
        for i in range(CID2013_data.shape[0] - 1):
            imgpath.append(CID2013_data[i + 1][0])
            imgscore.append(float(CID2013_data[i + 1][2]))
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join('../IQA_dataset/data', imgpath[item]), imgscore[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample1 = pil_loader(path)
        sample2 = HSV_loader(path)
        sample = Image.blend(sample1, sample2, 0.2)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


# 从PIL编程RGB格式
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def HSV_loader(path):
    img_HSV = cv.imread(path)
    img_HSV = cv.cvtColor(img_HSV, cv.COLOR_BGR2HSV)

    H, S, V  = cv.split(img_HSV)
    imgZeros = np.zeros_like(img_HSV)
    imgZeros[:,:,0] = 240
    imgZeros[:,:,1]=S
    imgZeros[:,:,2]=V
    img = cv.cvtColor(imgZeros, cv.COLOR_HSV2RGB)

    img_HSV = Image.fromarray(img)

    return img_HSV
