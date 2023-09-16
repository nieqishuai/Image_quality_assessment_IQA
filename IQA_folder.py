import warnings
import torch
import torchvision
import models
from PIL import Image
import numpy as np
import os
import cv2 as cv


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

def cls(path, device):
    print('device:{} is runing'.format(device))
    model_NQS=torch.load('./pretrained/livec_model.pkl', map_location=device)
    model_NQS = models.SelfAdapt_Net(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_NQS.train(False)
    # 载入预训练好的模型：koniq-10k
    model_NQS.load_state_dict((torch.load('./pretrained/koniq_pretrained.pkl', map_location=device)))
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))])

    floder_score = []
    all_images = os.listdir(path)

    for img_p in all_images:

        img_path = os.path.join(path, img_p)

        # 随机裁剪25个斑块，计算平均质量分数
        pred_scores = []
        for i in range(25):
            RGB_img = pil_loader(img_path)
            HSV_img = HSV_loader(img_path)
            img = Image.blend(RGB_img, HSV_img, 0.2)
            
            img = transforms(img)
            img = torch.tensor(img.to(device)).unsqueeze(0)
            paras = model_NQS(img)

            # 建立目标网络
            model_target = models.TargetNet(paras).to(device)
            for param in model_target.parameters():
                param.requires_grad = False

            # 质量预测
            pred = model_target(paras['target_in_vec'])
            # 'paras['target_in_vec']' 是质量预测网络的输入
            pred_scores.append(float(pred.item()))
        score = np.mean(pred_scores)
        # 质量得分在0-100之间，得分越高，说明质量越好
        floder_score.append(float(score))
    # print(floder_score)
    print("评价完成")

    file = open(path + '_Score.txt', 'w')
    file.write('img_name,img_score\n')
    for i in range(len(all_images)):
        file.write(all_images[i] + ',' + str(floder_score[i]) + '\n')
    file.close()


if __name__ == "__main__":
    # 忽略警告信息
    warnings.filterwarnings("ignore")
    self_device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
    cls(path="selfdata", device=self_device)
