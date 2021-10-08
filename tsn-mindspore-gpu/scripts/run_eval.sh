#!/bin/bash
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

if [ $# != 4 ]; then
  echo "Usage: sh run_distribute_train.sh  [DATASET_PATH] [DATASET] [VAL_LIST] [DEVICE_ID]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

dataset_path=$(get_real_path $1)
dataset=$2
val_list=$(get_real_path $3)
device_id=$4

if [ ! -d $dataset_path ]; then
  echo "error: DATASET_PATH=$dataset_path is not a directory"
  exit 1
fi

if [ ! -f $val_list ]; then
  echo "error: VAL_LIST=$val_list is not a file"
  exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_SIZE=1

export DEVICE_ID=$device_id
export RANK_ID=$device_id
echo "start training for rank $RANK_ID, device $DEVICE_ID"
python ../eval.py --device_id=$device_id \
 --dataset_path=$dataset_path \
 --dataset=$dataset \
 --test_list=$val_list > ../result.log 2>&1 &

