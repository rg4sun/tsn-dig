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
"""test tsn"""
import time
import datetime
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import mindspore.ops as ops

from mindspore import dtype as mstype
from mindspore import Tensor,context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from numpy.core.fromnumeric import reshape

from src.dataset import create_dataset
from src.models import TSN

parser = argparse.ArgumentParser(description="Standard video-level testing")
# parser.add_argument('--weights', type=str, default="/data/Fancyshun/TSN/tsn-pytorch/train_parallel0/ema-336.ckpt")
parser.add_argument('--weights', type=str, default="/home/tsn_checkpoint/ucf101_bninception_Flow-11_299.ckpt")
parser.add_argument('--dataset', type=str, default="ucf101", choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('--modality', type=str, default="Flow", choices=['RGB', 'Flow', 'RGBDiff'])
# parser.add_argument('--test_list', type=str, default="/data/Fancyshun/TSN/dataset/ucf101_val_split_1_rawframes.txt")
parser.add_argument('--test_list', type=str, default="/home/shh_ucf/data/data_extracted/ucf101/ucf101_val_split_1_rawframes.txt")
# parser.add_argument('--dataset_path', type=str, default="/data/Fancyshun/TSN/dataset/tvl1/")
parser.add_argument('--dataset_path', type=str, default="/home/shh_ucf/data/data_extracted/ucf101/tvl1")
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--arch', type=str, default="BNInception")
# parser.add_argument('--save_scores', type=str, default="./score/score_warmup", help="./score/score_warmup_flow_2_")
parser.add_argument('--save_scores', type=str, default="/home/score/score_warmup", help="./score/score_warmup_flow_2_")
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--flow_prefix', type=str, default='flow_')


args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args.device_id)

test_start = datetime.datetime.now()

if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
else:
    raise ValueError('Unknown dataset '+args.dataset)

model = TSN(num_class, 1, args.modality, base_model=args.arch,\
        consensus_type=args.crop_fusion_type, dropout=args.dropout)

param_dict = load_checkpoint(args.weights)
load_param_into_net(model, param_dict)

crop_size = model.crop_size
scale_size = model.scale_size
input_mean = model.input_mean
input_std = model.input_std

if args.modality == 'RGB':
    data_length = 1
    args.flow_prefix = ''
    args.dropout = 0.2
elif args.modality in ['Flow', 'RGBDiff']:
    data_length = 5

data_loader = create_dataset(root_path=args.dataset_path, list_file=args.test_list, batch_size=1, num_segments=args.test_segments,\
        new_length=data_length, modality=args.modality, image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",\
            input_size=crop_size, roll=args.arch == 'BNInception', input_mean=input_mean, input_std=input_std, div=args.arch != 'BNInception',\
                test_crops=args.test_crops, scale_size=scale_size, test_mode=2, random_shift=False, run_distribute=False)


total_num = data_loader.get_dataset_size()

output = []

proc_start_time = time.time()

reshape = ops.Reshape()
cast = ops.Cast()

def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)
    input_var = reshape(data, (-1, length, data.shape[2], data.shape[3]))

    if args.modality == 'RGBDiff':
        reverse = list(range(data_length, 0, -1))
        input_c = 3
        input = np.array(input_var)
        input_view = input.reshape((-1, args.test_segments, data_length + 1, input_c,) + input.shape[2:])

        new_data = input_view[:, :, 1:, :, :, :].copy()
        for x in reverse:
            new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
        input_var = Tensor(new_data, mstype.float32)
    input_var = cast(input_var, mstype.float32)
    rst = model(input_var).asnumpy().copy()
    rst = rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

    return i, rst, label.asnumpy().tolist()

for i, data in enumerate(data_loader.create_dict_iterator()):
    step_start = time.time()
    res = eval_video((i, data['input'], data['label']))
    output.append(res[1:])
    step_end = time.time()
    cnt_time = step_end - proc_start_time
    this_step = step_end - step_start    
    print('step: {} , total: {}/{}, time used: {:.2f} , average {:.2f} sec/step'.format((i+1), (i+1),
                                                                    total_num, this_step,
                                                                    float(cnt_time) / (i+1)))
"""video_pred = []
for x in output:
    for item in x[0]:
        video_pred.append(np.argmax(np.mean(item, axis=0)))
video_labels = []
for x in output:
    video_labels.extend(x[1])"""
video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

video_labels = [x[1] for x in output]

cf = confusion_matrix(video_labels, video_pred).astype(float)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

#print("cls_cnt:", cls_cnt, "   cls_hit:", cls_hit)

cls_acc = cls_hit / cls_cnt

print('Accuracy {:.01f}%'.format(np.mean(cls_acc) * 100))

test_end = datetime.datetime.now()
time_cost = test_end - test_start
print("Total test time:", str(time_cost).split('.')[0])

if args.save_scores is not None:

    # reorder before saving
    """name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}
    length = 0
    out = []
    for i, item in enumerate(output):
        length += output[i][0].shape[0]
        for j, x in enumerate(item[0]):
            out.append((x, item[1][j]))
    reorder_output = [None] * length
    reorder_label = [None] * length

    for i in range(length):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = out[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores+args.modality, scores=reorder_output, labels=reorder_label)"""
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores+args.modality, scores=reorder_output, labels=reorder_label)
