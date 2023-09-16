import torch
import torchvision
import folders


class DataLoader(object):
    # 数据加载
    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):

        self.batch_size = batch_size
        # 训练集和测试集的transform不一样
        self.istrain = istrain

        if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'tid2013') | (dataset == 'livec') | (dataset == 'cid2013'):
            # 训练集 transforms
            # 不同数据集使用不同的transform
            # 对图像进行预处理和数据增强
            if istrain:
                transforms = torchvision.transforms.Compose([
                    # 随机水平翻转（默认p=0.5）
                    torchvision.transforms.RandomHorizontalFlip(),
                    # 在一个随机位置上对图像进行裁剪
                    torchvision.transforms.RandomCrop(size=patch_size),
                    # 将PIL Image或numpy.ndarray转为pytorch的Tensor，并会将像素值由[0, 255]变为[0, 1]之间
                    torchvision.transforms.ToTensor(),
                    # 用均值和标准差对Tensor进行归一化处理
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
            # 测试集 transforms
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
        elif dataset == 'koniq-10k':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])

        if dataset == 'live':
            self.data = folders.LIVEFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'livec':
            self.data = folders.LIVEChallengeFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'csiq':
            self.data = folders.CSIQFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'koniq-10k':
            self.data = folders.Koniq_10kFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'cid2013':
            self.data = folders.CID2013Folder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)

    def get_data(self):
        # 训练集使用批量图片同时训练，测试集使用单个图片测试
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader
