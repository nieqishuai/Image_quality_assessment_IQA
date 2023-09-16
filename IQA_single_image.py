import torch
import torchvision
import models
from PIL import Image
import numpy as np
import warnings
import time
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



# 忽略警告信息
warnings.filterwarnings("ignore")

start_time = time.time()
img_path = './selfdata/40blur.jpg'
device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')
print('device:{} is runing'.format(device))

model_NQS = models.SelfAdapt_Net(16, 112, 224, 112, 56, 28, 14, 7).to(device=device)
model_NQS.train(False)
# 在koniq-10k数据集上加载我们预先训练好的模型
model_NQS.load_state_dict((torch.load('./pretrained/livec_model.pkl', map_location=device)))


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 384)),
    torchvision.transforms.RandomCrop(size=224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])

# 随机裁剪25个斑块，计算平均质量分数
pred_scores = []
for i in range(25):
    RGB_img = pil_loader(img_path)
    HSV_img = HSV_loader(img_path)
    img = Image.blend(RGB_img, HSV_img, 0.2)

    img = transforms(img)
    img = torch.as_tensor(img.to(device=device)).unsqueeze(0)
    # paras'包含传达给目标网络的网络权重
    paras = model_NQS(img)
    # 建立目标网络
    model_target = models.TargetNet(paras).to(device=device)
    for param in model_target.parameters():
        param.requires_grad = False

    # 质量预测
    # paras['target_in_vec']是对目标网的输入
    pred = model_target(paras['target_in_vec'])
    pred_scores.append(float(pred.item()))
score = np.mean(pred_scores)
# 质量评分范围为0~100，分数越高说明质量越好
print('Predicted quality score: %.2f' % score)
end_time = time.time()
print("运行时间：{}s".format(round(end_time - start_time, 2)))
