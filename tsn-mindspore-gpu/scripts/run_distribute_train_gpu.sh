DATA_DIR=/home/shh_ucf/data/data_extracted/ucf101/tvl1

CUDA_VISIBLE_DEVICES=4,5,6,7 mpirun --allow-run-as-root -n 4 --output-filename log_output --merge-stderr-to-stdout \
  python ./train.py --is_distributed --platform 'GPU' --dataset_path $DATA_DIR > tsn_train.log 2>&1 &