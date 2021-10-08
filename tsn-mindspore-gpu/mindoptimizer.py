# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train tsn"""
import os
import copy
import argparse
import datetime
import numpy as np
import argparse

import mindspore.nn as nn
import mindspore.dataset as ds

from mindspore.train.model import Model
from mindspore import context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.train.callback import TimeMonitor,ModelCheckpoint,CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.communication.management import init, get_group_size, get_rank

from src.dataset import create_dataset
from src.models import TSN
from src.tsn_for_train import TrainOneStepCellWithGradClip
# from src.tsn_for_train_new import TrainOneStepCellWithGradClip # 改成new
from src.util import load_backbone, process_trainable_params, get_lr, EvalCallBack, DistAccuracy, ClassifyCorrectCell, EmaEvalCallBack
from src.config import tsn_flow, tsn_rgb, tsn_rgb_diff

set_seed(1)

parser = argparse.ArgumentParser(description="MindSpore implementation of Temporal Segment Networks")
# parser.add_argument('--platform', type=str, default='Ascend', choices=['Ascend'],
#                     help='Running platform, only support Ascend now. Default is Ascend.')
parser.add_argument('--platform', type=str, default='GPU', choices=['Ascend'],
                    help='Running platform, only support Ascend now. Default is Ascend.')
# parser.add_argument('--dataset_path', type=str, default='/data/Fancyshun/TSN/dataset/tvl1/')
parser.add_argument('--dataset_path', type=str, default='/home/shh_ucf/data/data_extracted/ucf101/tvl1')
parser.add_argument('--dataset', type=str, default='ucf101',choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('--modality', type=str, default='Flow',choices=['RGB', 'Flow', 'RGBDiff'])
# parser.add_argument('--train_list', type=str, default="/data/Fancyshun/TSN/dataset/ucf101_train_split_1_rawframes.txt")
parser.add_argument('--train_list', type=str, default="/home/shh_ucf/data/data_extracted/ucf101/ucf101_train_split_1_rawframes.txt")
# parser.add_argument('--val_list', type=str, default="/data/Fancyshun/TSN/dataset/ucf101_val_split_1_rawframes.txt")
parser.add_argument('--val_list', type=str, default="/home/shh_ucf/data/data_extracted/ucf101/ucf101_val_split_1_rawframes.txt")
# parser.add_argument('--pretrained_path', type=str, default="/data/Fancyshun/TSN/tsn-pytorch/bninception_flow.npy")
parser.add_argument('--pretrained_path', type=str, default="/home/shh_ucf/tsn_flow.ckpt")
parser.add_argument('--device_id', default=0, type=int)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg', choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--dropout', default=0.3, type=float, help='dropout ratio (default: 0.5)')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=450, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--learning_rate', default=0.005, type=float, help='initial learning rate')
parser.add_argument('--gamma', default=0.07, type=float, help='dacay rate of learning rate')

parser.add_argument('--lr_steps', default=110, type=int, help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--clip_gradient', default=20.0, type=float, help='weight decay (default: 5e-4)')

# ========================= Eval Configs ==========================
#interval=1, eval_start_epoch=1
parser.add_argument('--eval_model', default=True, type=bool, help='')
parser.add_argument('--eval_start_epoch', default=100, type=int, help='')

# ========================= EMA Configs ==========================
parser.add_argument('--ema_decay', default=0.99, type=float, help='')
parser.add_argument('--ckpt_save_epoch', default=0, type=int, help='')
parser.add_argument('--interval', default=2, type=int, help='') # 每2个epoch保存一次ema的ckpt
parser.add_argument('--dataset_sink_mode', default=False, type=bool, help='') # 训练的时候一般设成true

# ========================= Monitor Configs ==========================
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
parser.add_argument('--run_modelarts', type=bool, default=False, help='Run on modelarts')
# parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--workers', default=1, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--save_check_point', type=bool, default=True)
parser.add_argument('--ckpt_save_dir', type=str, default="/home/tsn_checkpoint")
parser.add_argument('--snapshot_pref', type=str, default="ucf101_bninception_")

parser.add_argument('--flow_prefix', default="flow_", type=str)


def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds, dataset_sink_mode=False)
    return res[metrics_name]

if __name__ == '__main__':
    args = parser.parse_args()

    # """print 4 test"""
    # print(args)

    
    train_start = datetime.datetime.now()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, device_id=args.device_id, save_graphs=False)
    # PYNATIVE_MODE 用于调试
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=args.platform, device_id=args.device_id,
    #                     save_graphs=False)

    if args.run_distribute:
        # device_id = int(os.getenv('DEVICE_ID'))
        # device_num = int(os.getenv('RANK_SIZE'))
        # rank = int(os.environ.get("RANK_ID"))
        init("nccl")  # GPU通信协议
        device_id = 0
        device_num = get_group_size()
        rank = get_rank()

        context.set_context(device_id=device_id)
        # init()

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
    else:
        #device_id = int(os.getenv('DEVICE_ID'))
        device_id = args.device_id
        context.set_context(device_id=device_id)
        device_num = 1
        rank = 0

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    if args.modality == 'Flow':
        cfg = tsn_flow
        args.learning_rate = cfg.learning_rate
        args.epochs = cfg.epochs
        args.lr_steps = cfg.lr_steps
        args.gamma = cfg.gamma
        args.dropout = cfg.dropout
        args.flow_prefix = "flow_"
        args.eval_start_epoch = 300
    elif args.modality == 'RGB':
        cfg = tsn_rgb
        args.learning_rate = cfg.learning_rate
        args.epochs = cfg.epochs
        args.lr_steps = cfg.lr_steps
        args.gamma = cfg.gamma
        args.dropout = cfg.dropout
        args.flow_prefix = ""
    elif args.modality == 'RGBDiff':
        cfg = tsn_rgb_diff
        args.learning_rate = cfg.learning_rate
        args.epochs = cfg.epochs
        args.lr_steps = cfg.lr_steps
        args.gamma = cfg.gamma
        args.dropout = cfg.dropout
        args.flow_prefix = ""
        args.clip_gradient *= 2
    else:
        raise ValueError('Unknown modality ' + args.modality)

    print(args)

    net = TSN(num_class, args.num_segments, args.modality, base_model=args.arch,\
         consensus_type=args.consensus_type, dropout=args.dropout)

    # 在这里加一个预训练判断？
    net = load_backbone(net, args.pretrained_path)

    crop_size = net.crop_size
    scale_size = net.scale_size
    input_mean = net.input_mean
    input_std = net.input_std

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = create_dataset(root_path=args.dataset_path, list_file=args.train_list, batch_size=args.batch_size, num_segments=args.num_segments,\
         new_length=data_length, modality=args.modality, image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",\
              input_size=crop_size, roll=args.arch == 'BNInception', input_mean=input_mean, input_std=input_std, div=args.arch != 'BNInception',\
                   test_crops=1, scale_size=scale_size, test_mode=0, run_distribute=args.run_distribute, worker=args.workers, num_shards=device_num, shard_id=rank)
    data_size = train_loader.get_dataset_size()

    lr = get_lr(learning_rate=args.learning_rate, gamma=args.gamma, epochs=args.epochs, steps_per_epoch=data_size, lr_steps=args.lr_steps)
    #lr = nn.exponential_decay_lr(learning_rate=args.learning_rate, decay_rate=args.gamma, total_step=args.epochs*data_size, step_per_epoch=data_size, decay_epoch=args.lr_steps)
    #lr = np.array(lr)
    base_lr = 5 if args.modality == 'Flow' else 1
    group1, group2, group3, group4, group5 = process_trainable_params(net.trainable_params())
    group_params = [{'params': group1, 'lr': lr*base_lr, 'weight_decay': args.weight_decay},
                        {'params': group2, 'lr': lr*base_lr*2, 'weight_decay': 0},
                        {'params': group3, 'lr': lr*1, 'weight_decay': args.weight_decay},
                        {'params': group4, 'lr': lr*2, 'weight_decay': 0},
                        {'params': group5, 'lr': lr*1, 'weight_decay': 0}]

    # define loss function (criterion) and optimizer

    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    #define callbacks
    loss_cb = LossMonitor()
    time_cb = TimeMonitor(data_size=data_size)
    callbacks = [loss_cb, time_cb]

    #process net for train
    optimizer = nn.SGD(group_params, momentum=args.momentum)
    #model = Model(net, loss_fn=criterion, optimizer=optimizer, amp_level="O3")
    dist_eval_network = ClassifyCorrectCell(net) if device_num>1 else None
    metrics = {"acc"}
    if device_num>1:
        metrics = {'acc': DistAccuracy(batch_size=2, device_num=device_num)}
    # network = nn.WithLossCell(net, criterion)
    ### network = TrainOneStepCellWithGradClip(network, optimizer, args.clip_gradient)
    ## network = TrainOneStepCellWithGradClip(network, optimizer)

    # model = Model(network, loss_fn=criterion, optimizer=optimizer, eval_network=dist_eval_network, metrics=metrics)
    model = Model(net, loss_fn=criterion, optimizer=optimizer, eval_network=dist_eval_network, metrics=metrics)

    # model = Model(network, eval_network=dist_eval_network, metrics=metrics)

    if args.save_check_point and (device_num == 1 or device_id == 0):
        # config_ck = CheckpointConfig(save_checkpoint_steps=data_size*args.epochs, keep_checkpoint_max=1)
        config_ck = CheckpointConfig(save_checkpoint_steps=data_size, keep_checkpoint_max=10) # 只保留最新的10个ckpt文件

        if args.run_modelarts:
            ckpoint_cb = ModelCheckpoint(prefix=args.snapshot_pref+args.modality, directory=args.ckpt_save_dir, config=config_ck)
        else:
            ckpoint_cb = ModelCheckpoint(prefix=args.snapshot_pref+args.modality, directory=args.ckpt_save_dir, config=config_ck)
        callbacks += [ckpoint_cb]

    if args.eval_model:
        val_dataset = create_dataset(root_path=args.dataset_path, list_file=args.val_list, batch_size=2, num_segments=args.num_segments,\
            new_length=data_length, modality=args.modality, image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg", random_shift=False,\
                input_size=crop_size, roll=args.arch == 'BNInception', input_mean=input_mean, input_std=input_std, div=args.arch != 'BNInception',\
                    test_crops=1, scale_size=scale_size, test_mode=1, run_distribute=args.run_distribute, worker=args.workers, num_shards=device_num, shard_id=rank)

        net_ema = copy.deepcopy(net) # 这里对net进行了深拷贝
        net_ema.set_train(False) # 关闭训练
        assert args.ema_decay > 0, "EMA should be used in tinynet training."

        ema_cb = EmaEvalCallBack(network=net,
                                ema_network=net_ema,
                                loss_fn=criterion,
                                eval_dataset=val_dataset,
                                decay=args.ema_decay,
                                save_epoch=1,
                                dataset_sink_mode=args.dataset_sink_mode,
                                start_epoch=args.ckpt_save_epoch,
                                interval=args.interval,
                                ckpt_path=args.ckpt_save_dir)
        callbacks.append(ema_cb)
        """
        eval_param_dict = {"model": model, "dataset": val_dataset, "metrics_name": "acc"}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=args.interval,
                               eval_start_epoch=args.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=args.ckpt_save_dir, besk_ckpt_name="best.ckpt",
                               metrics_name='acc')
        callbacks.append(eval_cb)
        """

    ds.config.set_enable_shared_mem(False)
    # 这里正式跑训练
    model.train(args.epochs, train_loader, callbacks=callbacks, dataset_sink_mode=False)

    train_end = datetime.datetime.now()
    time_cost = train_end - train_start
    print("Total train time:", str(time_cost).split('.')[0])
