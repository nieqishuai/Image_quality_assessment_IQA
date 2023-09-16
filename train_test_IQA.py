import os
import argparse
import random
import numpy as np
import datetime
from NQSIQASolver import NQSIQASolver

import warnings


# 主程序，用来训练
def main(config):
    # 忽略警告信息
    warnings.filterwarnings("ignore")

    # 使用第一片GPU加速
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 数据集路径
    folder_path = {
        'live': '../IQA_dataset/data/LIVE/',
        'csiq': '../IQA_dataset/data/CSIQ/',
        'tid2013': '../IQA_dataset/data/TID2013/',
        'livec': '../IQA_dataset/data/LIVE_Challenge/',
        'koniq-10k': '../IQA_dataset/data/KonIQ-10k/',
        'cid2013': '../IQA_dataset/data/CID2013/',
    }

    # 数据集中图片数量
    img_num = {
        'live': list(range(0, 779)),
        'csiq': list(range(0, 866)),
        'tid2013': list(range(0, 3000)),
        'livec': list(range(0, 1161)),
        'koniq-10k': list(range(0, 10073)),
        'cid2013': list(range(0,474))
    }
    # 选择的数据集的图片个数
    sel_num = img_num[config.dataset]

    # 存放Spearman秩相关系数（SROCC，衡量预测的单调性）
    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    # 存放线性相关系数（PLCC，衡量预测的准确性）
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)

    print('Training and testing on %s dataset for %d rounds...' %
          (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))
        # 随机选择80%的图片用于训练，剩下的用于测试
        # 打乱图片顺序
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

        solver = NQSIQASolver(
            config, folder_path[config.dataset], train_index, test_index, pretrained=True)
        srcc_all[i], plcc_all[i] = solver.train(i)

    # print(srcc_all)
    # print(plcc_all)

    # 计算SROCC和PLCC的中位数
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' %
          (srcc_med, plcc_med))


if __name__ == '__main__':
    # 创建一个参数解析实例
    parser = argparse.ArgumentParser()
    # 添加参数，同时注明解析标志，解析后名称，类型，默认值，提示信息
    parser.add_argument('--dataset', dest='dataset', type=str, default='live',
                        help='Support datasets: livec|cid2013|koniq-10k|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25,
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay',
                        type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=96, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=10, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num',
                        type=int, default=3, help='Train-test times')
    # 解析参数，使用 parse_args() 解析添加的参数
    config = parser.parse_args()

    all_start_time = datetime.datetime.now()
    main(config)
    all_end_time = datetime.datetime.now()
    print("训练时间：{}".format(all_end_time - all_start_time))
