import torch
from scipy import stats
import numpy as np
import models
import data_loader
import time
from torch.utils.tensorboard import SummaryWriter


class NQSIQASolver(object):
    # 用于训练和测试hyperIQA的求解器
    def __init__(self, config, path, train_idx, test_idx, pretrained):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')

        # 1.数据加载
        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size,
                                              config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num,
                                             istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        # 2.模型参数载入
        self.model_NQS = models.SelfAdapt_Net(16, 112, 224, 112, 56, 28, 14, 7).to(device=self.device)
        self.model_NQS.train(True)
        if pretrained:
            save_model = torch.load('pretrained/koniq_pretrained.pkl',  map_location=self.device)
            model_dict = self.model_NQS.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model_NQS.load_state_dict(model_dict)

        # 3.损失函数
        self.l1_loss = torch.nn.SmoothL1Loss().to(device=self.device)

        # 4.优化器
        backbone_params = list(map(id, self.model_NQS.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_NQS.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_NQS.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

    def train(self,i):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTrain_PLCC\tTest_SRCC\tTest_PLCC\tEpoch_Time')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []  # 真实分数
            start_time = time.time()

            for img, label in self.train_data:
                img = torch.tensor(img.to(device=self.device))
                label = torch.tensor(label.to(device=self.device))

                self.solver.zero_grad()

                # 为目标网络生成权重
                paras = self.model_NQS(img)  # paras'包含传达给目标网络的网络权重
                # 建立目标网络
                model_target = models.TargetNet(paras).to(device=self.device)
                for param in model_target.parameters():
                    param.requires_grad = False

                # 质量预测
                pred = model_target(paras['target_in_vec'])
                # 其中'paras['target_in_vec']'是目标网的输入。
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
            
            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                torch.save(self.model_NQS, 'pretrained/'+'第{}次'.format(i+1)+'第{}轮'.format(t+1)+'model.pkl')

            writer = SummaryWriter('./log')
            writer.add_scalar(tag='epoch_loss', scalar_value=round(sum(epoch_loss) / len(epoch_loss), 3), global_step=t, walltime=None)
            end_time = time.time()
            run_time = end_time - start_time
            print('{}\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}s'.format(t + 1, round(sum(epoch_loss) / len(epoch_loss), 3),
                                                                 round(train_srcc, 4), round(train_plcc, 4),
                                                                 round(test_srcc, 4), round(test_plcc, 4),
                                                                 round(run_time, 2)))

            # 更新优化器
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_NQS.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_NQS.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in data:
            # Data.
            img = torch.tensor(img.to(device=self.device))
            label = torch.tensor(label.to(device=self.device))

            paras = self.model_NQS(img)
            model_target = models.TargetNet(paras).to(device=self.device)
            model_target.train(False)
            pred = model_target(paras['target_in_vec'])

            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_NQS.train(True)
        return test_srcc, test_plcc
